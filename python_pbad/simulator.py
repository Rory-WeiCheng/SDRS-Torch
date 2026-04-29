"""ConvexHullPBAD simulator: analytic g/H, Schur LM, IFT backward. World Y-up."""

import math
import torch
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict
from robot import Robot

# Unified finite-difference step for all debug_* routines (DE/DDE, IFT, θ g/H).
DEBUG_FD_EPS = 1e-8


def _autograd_functional_jvp(*args, **kwargs):
    from torch.autograd.functional import jvp as _jvp
    return _jvp(*args, **kwargs)




def barrier_eval(d: torch.Tensor, x0: float, d0_offset: float = 0.0,
                 mode: str = 'truncated',
                 ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Two support modes; 'truncated' (default) preserves prior behaviour."""
    x = d - d0_offset
    clamp_min = 1e-15 if x.dtype == torch.float64 else 1e-12
    x_s = torch.clamp(x, min=clamp_min)

    if mode == 'log':
        # Global -log barrier (active for all x > 0).
        val  = -torch.log(x_s)
        grad = -1.0 / x_s
        hess = 1.0 / (x_s ** 2)
        active = x > 0
    elif mode == 'truncated':
        # Original: polynomial × log, finite support (0, x0).
        u = x0 / x_s
        L = torch.log(x_s / x0)
        val  = -L * (x_s - x0) ** 2 / x_s
        grad = -(1.0 - u) ** 2 - L * (1.0 - u ** 2)
        hess = (-(x_s - x0) * (x_s + 3.0 * x0) - 2.0 * x0 ** 2 * L) / (x_s ** 3)
        active = (x > 0) & (x < x0)
    else:
        raise ValueError(f"barrier_eval: mode must be 'truncated' or 'log', got {mode!r}")

    zero = torch.zeros_like(x)
    val  = torch.where(active, val,  zero)
    grad = torch.where(active, grad, zero)
    hess = torch.where(active, hess, zero)
    val  = torch.where(x <= 0, torch.full_like(x, 1e8), val)
    return val, grad, hess



@dataclass
class ContactManifold:
    link_a: int
    link_b: int
    p: torch.Tensor = None
    # Friction auxiliary: ``u[0:2]`` = 2D tangent coeffs, ``u[2]`` = ω (scalar).
    u: torch.Tensor = None
    g_p: torch.Tensor = None
    H_pp: torch.Tensor = None
    H_theta_p: torch.Tensor = None
    g_u: torch.Tensor = None
    H_uu: torch.Tensor = None
    H_theta_u: torch.Tensor = None


@dataclass
class GradInfo:
    dtheta_dtheta_t:   torch.Tensor        # [n, n]
    dtheta_dtheta_tm1: torch.Tensor        # [n, n]
    dtheta_dpd:        torch.Tensor        # [n, n]
    H_bar:             torch.Tensor        # [n, n]  Schur complement (used for descent)
    dtheta_dd:         torch.Tensor = None  # [n, L*M*3] or None
    H_theta_O:         torch.Tensor = None  # [n, n]  ∇_θθ Ō (raw, no Schur correction)


def ift_jacobian_theta_tp1_wrt_theta_t(gi: GradInfo) -> torch.Tensor:
    """``∂θ_{t+1}/∂θ_t`` from IFT (same tensor as ``gi.dtheta_dtheta_t``). Shape ``[n, n]``."""
    return gi.dtheta_dtheta_t


def ift_jacobian_theta_tp1_wrt_theta_tm1(gi: GradInfo) -> torch.Tensor:
    """``∂θ_{t+1}/∂θ_{t-1}``. Shape ``[n, n]``."""
    return gi.dtheta_dtheta_tm1


def ift_jacobian_theta_tp1_wrt_pd_target(gi: GradInfo) -> torch.Tensor:
    """``∂θ_{t+1}/∂θ_pd`` w.r.t. PD set-point ``pd_target``. Shape ``[n, n]``."""
    return gi.dtheta_dpd


def ift_jacobian_theta_tp1_wrt_convex_d(gi: GradInfo) -> Optional[torch.Tensor]:
    """``∂θ_{t+1}/∂d`` w.r.t. flattened convex local vertices ``d`` (``[n, L*M*3]``), or ``None``."""
    return gi.dtheta_dd


class SimStepFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, pd_target, theta_t, theta_tm1, local_verts, sim, kp, kd):
        def _req(t):
            return isinstance(t, torch.Tensor) and t.requires_grad

        ctx._req_pd = _req(pd_target)
        ctx._req_theta_t = _req(theta_t)
        ctx._req_theta_tm1 = _req(theta_tm1)
        ctx._req_lv = _req(local_verts)

        theta_tp1, grad_info, manifolds = sim.step(
            theta_t.detach(), theta_tm1.detach(), pd_target.detach(), kp, kd)
        ctx.grad_info = grad_info
        ctx.save_for_backward(theta_tp1)
        # No autograd graph on θ*; sensitivities come from custom backward + IFT.
        return theta_tp1.clone()

    @staticmethod
    def backward(ctx, grad_output):
        gi = ctx.grad_info
        g = grad_output.detach()
        d_pd = (gi.dtheta_dpd.T @ g) if ctx._req_pd else None
        d_theta_t = (gi.dtheta_dtheta_t.T @ g) if ctx._req_theta_t else None
        d_theta_tm1 = (gi.dtheta_dtheta_tm1.T @ g) if ctx._req_theta_tm1 else None
        if ctx._req_lv and gi.dtheta_dd is not None:
            d_d = gi.dtheta_dd.T @ g
        else:
            d_d = None
        return d_pd, d_theta_t, d_theta_tm1, d_d, None, None, None


class ConvexHullPBADSimulator:

    def __init__(self, robot: Robot, dt: float = 0.01,
                 barrier_x0: float = 0.01,
                 coef_barrier: float = 1e-2,
                 lm_gamma: float = 1e0,
                 friction: float = 0.8, gravity: float = -9.81,
                 max_newton_iter: int = 1000, gtol: float = 1e-4,
                 device='cuda', _output: bool = True,
                 reject_step_energy_above: Optional[float] = 1e5,
                 use_joint_limit_barrier: bool = False,
                 barrier_d0: float = 1e-3,
                 implicit: bool = True,
                 use_friction: bool = True,
                 contact_exclude_tree_distance: int = 4):
        self.robot = robot
        self.dt = dt
        self.x0 = barrier_x0
        self.coef_barrier = coef_barrier
        self.lm_gamma = lm_gamma
        self.friction = friction
        self.gravity = gravity
        self.max_iter = max_newton_iter
        self.gtol = gtol
        self.device = device
        self.dtype = torch.float64
        self.n_dof = robot.n_dof
        self._output = _output
        self._reject_step_energy_above = (
            None if reject_step_energy_above is None
            else float(reject_step_energy_above))
        self._use_joint_limit_barrier = bool(use_joint_limit_barrier)
        self.barrier_d0 = float(barrier_d0)
        self._barrier_d0_half = self.barrier_d0 * 0.5
        self._implicit = bool(implicit)
        self._use_friction = bool(use_friction)
        self._contact_exclude_tree_distance = max(1, int(contact_exclude_tree_distance))
        self._friction_eps_s = 1e-4
        self._friction_eps_n = 1e-8
        self._alpha = 1e-6
        self._last_manifolds = {}
        self._last_step_wv_curr: Optional[torch.Tensor] = None
        self._last_step_wv_last: Optional[torch.Tensor] = None

        self.joint_mask = torch.zeros(self.n_dof, device=device, dtype=self.dtype)
        if self.n_dof > 6:
            self.joint_mask[6:] = 1.0

        for link in self.robot.links:
            link.local_vertices = link.local_vertices.to(dtype=self.dtype, device=device)
        for j in self.robot.joints:
            j.origin = j.origin.to(dtype=self.dtype, device=device)
            j.axis   = j.axis.to(dtype=self.dtype, device=device)
        if self.robot.ground is not None:
            self.robot.ground.vertices = self.robot.ground.vertices.to(
                dtype=self.dtype, device=device)
        if self.robot.initial_pos is not None:
            self.robot.initial_pos = self.robot.initial_pos.to(
                dtype=self.dtype, device=device)
        self.robot.dtype = self.dtype

        self._precompute_cache()
        self._precompute_contact_pairs()

    def reset_episode_state(self) -> None:
        """Reset LM damping and contact warm-start (same spirit as after ``__init__``).

        Call between independent rollouts (e.g. each training epoch) so prior
        trajectory does not leave ``_alpha`` / ``_pair_manifolds`` in a stale state.
        """
        self._alpha = 1e-6
        self._last_manifolds = {}
        self._last_step_wv_curr = None
        self._last_step_wv_last = None
        self._precompute_contact_pairs()

    def _precompute_cache(self):
        dev, dty = self.device, self.dtype
        L = len(self.robot.links)
        n = self.n_dof
        self._L = L

        # --- Variable vertex counts: pad to max_M with mask ---
        vert_counts = [lk.local_vertices.shape[0] for lk in self.robot.links]
        max_M = max(vert_counts)
        self._max_M = max_M
        self._vert_counts = torch.tensor(vert_counts, device=dev, dtype=torch.long)

        padded = []
        for lk in self.robot.links:
            v = lk.local_vertices                       # [M_i, 3]
            if v.shape[0] < max_M:
                pad = torch.zeros(max_M - v.shape[0], 3, device=dev, dtype=dty)
                v = torch.cat([v, pad], dim=0)
            padded.append(v)
        self._local_verts = torch.stack(padded)          # [L, max_M, 3]

        # Boolean mask: True for real vertices
        mask = torch.zeros(L, max_M, device=dev, dtype=torch.bool)
        for i, c in enumerate(vert_counts):
            mask[i, :c] = True
        self._vert_mask = mask                            # [L, max_M]
        self._vert_mask_float = mask.to(dty)              # [L, max_M]

        # Mass-per-vertex (per-link, using actual vertex count)
        self._rho = torch.tensor(
            [lk.mass / lk.local_vertices.shape[0]
             for lk in self.robot.links], device=dev, dtype=dty)   # [L]
        self._rho_3d = self._rho.view(L, 1, 1)                     # [L,1,1]
        self._rho_2d = self._rho.view(L, 1)                        # [L,1]

        # Ground homogeneous [Mg, 4]
        gv = self.robot.ground.vertices
        self._ground_h = torch.cat(
            [gv, torch.ones(gv.shape[0], 1, device=dev, dtype=dty)], 1)
        # Cached identity matrices and constants
        self._eye3 = torch.eye(3, device=dev, dtype=dty)
        self._eye4 = torch.eye(4, device=dev, dtype=dty)
        self._eye_n = torch.eye(n, device=dev, dtype=dty)
        self._sqrt_friction_eps_s = math.sqrt(self._friction_eps_s)
        # Hinge joint limits (batched)
        h_off, h_lo, h_hi = [], [], []
        for j in self.robot.joints:
            if j.jtype == 'hinge':
                h_off.append(j.dof_offset)
                h_lo.append(j.limit_lower)
                h_hi.append(j.limit_upper)
        if h_off:
            self._hinge_idx = torch.tensor(h_off, device=dev, dtype=torch.long)
            self._hinge_lo  = torch.tensor(h_lo, device=dev, dtype=dty)
            self._hinge_hi  = torch.tensor(h_hi, device=dev, dtype=dty)
        else:
            self._hinge_idx = None

        # Descendant map: joint_idx → sorted list of descendant link indices
        self._descendants = {}
        for ji, joint in enumerate(self.robot.joints):
            desc = set()
            queue = [joint.child_link]
            while queue:
                lk = queue.pop()
                desc.add(lk)
                for jj, j2 in enumerate(self.robot.joints):
                    if j2.parent_link == lk:
                        queue.append(j2.child_link)
            self._descendants[ji] = sorted(desc)

        # Ancestor map: joint_idx → list of ancestor joint indices (root→joint)
        self._ancestors = {}
        for ji in range(len(self.robot.joints)):
            anc = []
            cur_link = self.robot.joints[ji].parent_link
            while cur_link >= 0:
                for jj, j2 in enumerate(self.robot.joints):
                    if j2.child_link == cur_link:
                        anc.append(jj)
                        cur_link = j2.parent_link
                        break
                else:
                    break
            anc.reverse()
            self._ancestors[ji] = anc


    @torch.no_grad()
    def _init_separating_planes_batch(
            self,
            va_list: List[torch.Tensor],
            vb_list: List[torch.Tensor],
    ) -> torch.Tensor:
        """GPU-batched separating plane for all convex pairs.

        Returns ``p_batch`` [P, 4] with ``p = [n*s, d*s]``.
        """
        dev, dty = self.device, self.dtype
        P = len(va_list)
        if P == 0:
            return torch.zeros(0, 4, device=dev, dtype=dty)

        na_list = [va.shape[0] for va in va_list]
        nb_list = [vb.shape[0] for vb in vb_list]
        max_na = max(na_list)
        max_nb = max(nb_list)

        Va = torch.zeros(P, max_na, 3, device=dev, dtype=dty)
        Vb = torch.zeros(P, max_nb, 3, device=dev, dtype=dty)
        mask_a = torch.zeros(P, max_na, device=dev, dtype=dty)
        mask_b = torch.zeros(P, max_nb, device=dev, dtype=dty)
        for k in range(P):
            Va[k, :na_list[k]] = va_list[k]
            Vb[k, :nb_list[k]] = vb_list[k]
            mask_a[k, :na_list[k]] = 1.0
            mask_b[k, :nb_list[k]] = 1.0

        count_a = torch.tensor(na_list, device=dev, dtype=dty)
        count_b = torch.tensor(nb_list, device=dev, dtype=dty)
        BIG = 1e30

        lo_a = (Va + (1.0 - mask_a.unsqueeze(2)) * BIG).amin(dim=1)
        hi_a = (Va - (1.0 - mask_a.unsqueeze(2)) * BIG).amax(dim=1)
        lo_b = (Vb + (1.0 - mask_b.unsqueeze(2)) * BIG).amin(dim=1)
        hi_b = (Vb - (1.0 - mask_b.unsqueeze(2)) * BIG).amax(dim=1)

        # Six candidates: gap_pos[ax] = B above A, gap_neg[ax] = A above B
        gap_pos = lo_b - hi_a
        gap_neg = lo_a - hi_b
        g_stack = torch.cat([gap_pos, gap_neg], dim=1)
        best_gap, best_j = g_stack.max(dim=1)
        separated = best_gap > 0

        axis = (best_j % 3).to(torch.long)
        sign_pos = best_j < 3

        hi_a_ax = hi_a.gather(1, axis.unsqueeze(1)).squeeze(1)
        lo_a_ax = lo_a.gather(1, axis.unsqueeze(1)).squeeze(1)
        hi_b_ax = hi_b.gather(1, axis.unsqueeze(1)).squeeze(1)
        lo_b_ax = lo_b.gather(1, axis.unsqueeze(1)).squeeze(1)

        mid_pos = 0.5 * (hi_a_ax + lo_b_ax)
        mid_neg = 0.5 * (hi_b_ax + lo_a_ax)
        d_axis = torch.where(sign_pos, -mid_pos, mid_neg)

        sign_axis = torch.where(sign_pos,
                                torch.ones(P, device=dev, dtype=dty),
                                -torch.ones(P, device=dev, dtype=dty))
        n_axis = torch.zeros(P, 3, device=dev, dtype=dty)
        n_axis.scatter_(1, axis.unsqueeze(1), sign_axis.unsqueeze(1))

        # --- Centroid fallback when no axis has positive AABB gap ---
        ca = (Va * mask_a.unsqueeze(2)).sum(1) / count_a.unsqueeze(1).clamp(min=1)
        cb = (Vb * mask_b.unsqueeze(2)).sum(1) / count_b.unsqueeze(1).clamp(min=1)
        diff = cb - ca
        n_fb = diff / diff.norm(dim=1, keepdim=True).clamp(min=1e-12)
        proj_a = torch.einsum('pmi,pi->pm', Va, n_fb)
        proj_b = torch.einsum('pmi,pi->pm', Vb, n_fb)
        a_max = (proj_a - (1.0 - mask_a) * BIG).amax(dim=1)
        b_min = (proj_b + (1.0 - mask_b) * BIG).amin(dim=1)
        d_fb = -0.5 * (a_max + b_min)

        n = torch.where(separated.unsqueeze(1), n_axis, n_fb)
        d = torch.where(separated, d_axis, d_fb)

        # Match legacy sign check: mean(n·x + d) on A should be ≤ 0
        proj_a_n = torch.einsum('pmi,pi->pm', Va, n) + d.unsqueeze(1)
        mean_a = (proj_a_n * mask_a).sum(1) / count_a.clamp(min=1)
        flip = mean_a > 0
        n[flip] = -n[flip]
        d[flip] = -d[flip]

        s = 0.8
        return torch.cat([n * s, (d * s).unsqueeze(1)], dim=1)

    @staticmethod
    def _excluded_link_pairs_floyd(robot: Robot, max_edge_dist: int) -> frozenset:
        """Undirected link tree: exclude link–link contact if graph distance ≤ ``max_edge_dist``.

        ``max_edge_dist=1`` → only direct parent/child (legacy). Larger values skip
        contacts along the same limb (e.g. 4 covers torso→tip on the Gym ant chain).
        """
        L = len(robot.links)
        md = max(1, int(max_edge_dist))
        inf = L + 100
        dist = [[inf] * L for _ in range(L)]
        for i in range(L):
            dist[i][i] = 0
        for jt in robot.joints:
            a, b = jt.parent_link, jt.child_link
            if a >= 0:
                dist[a][b] = 1
                dist[b][a] = 1
        for k in range(L):
            for i in range(L):
                if dist[i][k] >= inf:
                    continue
                dik = dist[i][k]
                for j in range(L):
                    nd = dik + dist[k][j]
                    if nd < dist[i][j]:
                        dist[i][j] = nd
        out = []
        for i in range(L):
            for j in range(i + 1, L):
                if dist[i][j] <= md:
                    out.append((i, j))
        return frozenset(out)

    def _precompute_contact_pairs(self):
        dev, dty = self.device, self.dtype
        L = self._L

        excluded = ConvexHullPBADSimulator._excluded_link_pairs_floyd(
            self.robot, self._contact_exclude_tree_distance)

        theta0 = self.robot.default_theta().to(dtype=dty, device=dev)
        with torch.no_grad():
            wv0 = self._get_wv_stacked(theta0)       # [L, max_M, 3]

        self._excluded_link_pairs = excluded
        vc = self._vert_counts
        gv = self.robot.ground.vertices

        # --- Collect all vertex pairs for batched SVM ---
        va_list: List[torch.Tensor] = []
        vb_list: List[torch.Tensor] = []
        pair_tags: List[tuple] = []     # ('link', i, j) | ('ground', i)

        for i in range(L):
            for j in range(i + 1, L):
                if (i, j) in excluded:
                    continue
                va_list.append(wv0[i, :vc[i]])
                vb_list.append(wv0[j, :vc[j]])
                pair_tags.append(('link', i, j))

        for i in range(L):
            va_list.append(wv0[i, :vc[i]])
            vb_list.append(gv)
            pair_tags.append(('ground', i))

        # --- Compute all separating planes in one batched GPU pass ---
        p_all = self._init_separating_planes_batch(va_list, vb_list)

        # --- Distribute into pair info & warm-start dict ---
        self._contact_pair_info = []
        self._ground_pair_info = []
        self._pair_manifolds = {}
        u0 = torch.zeros(3, device=dev, dtype=dty)

        for idx, tag in enumerate(pair_tags):
            p = p_all[idx]
            if tag[0] == 'link':
                _, i, j = tag
                self._contact_pair_info.append((i, j, p.clone()))
                self._pair_manifolds[(i, j)] = (p.clone(), u0.clone())
            else:
                _, i = tag
                self._ground_pair_info.append((i, p.clone()))
                self._pair_manifolds[(i, -1)] = (p.clone(), u0.clone())

        if self._output:
            print(
                f"  [Contact pairs] link-link={len(self._contact_pair_info)}  "
                f"link-ground={len(self._ground_pair_info)}  "
                f"excluded link pairs (tree dist ≤ "
                f"{self._contact_exclude_tree_distance})={len(excluded)}"
            )

    def _get_wv_stacked(self, theta: torch.Tensor) -> torch.Tensor:
        transforms = self.robot.forward_kinematics(theta)
        T = torch.stack(transforms)           # [L, 4, 4]
        R = T[:, :3, :3]                      # [L, 3, 3]
        t = T[:, :3, 3]                       # [L, 3]
        # world[l,m,k] = Σ_j local[l,m,j] * R[l,k,j] + t[l,k]
        return torch.einsum('lmj,lkj->lmk', self._local_verts, R) + t.unsqueeze(1)

    def _friction_tangent_basis(self, n_hat: torch.Tensor):
        """Return orthonormal ``t0, t1 ⟂ n_hat``. [K,3] each."""
        K = n_hat.shape[0]
        dev, dty = n_hat.device, n_hat.dtype
        ref = torch.zeros(K, 3, device=dev, dtype=dty)
        ref[:, 2] = 1.0
        t0 = torch.cross(n_hat, ref, dim=1)
        bad = t0.norm(dim=1) < 1e-10
        if bad.any():
            ref2 = torch.zeros_like(ref)
            ref2[:, 0] = 1.0
            t0[bad] = torch.cross(n_hat[bad], ref2[bad], dim=1)
        t0 = t0 / (t0.norm(dim=1, keepdim=True) + 1e-30)
        t1 = torch.cross(n_hat, t0, dim=1)
        return t0, t1

    def _unified_u_to_xyz_omega(
            self, u_unified: torch.Tensor, n_hat: torch.Tensor):
        """``u_unified`` [K,3] → embedded tangent ``u_xyz`` [K,3], ``ω`` [K]."""
        t0, t1 = self._friction_tangent_basis(n_hat)
        u_xyz = (u_unified[:, 0:1] * t0 + u_unified[:, 1:2] * t1)
        omega = u_unified[:, 2]
        return u_xyz, omega, t0, t1

    @staticmethod
    def _friction_J_lift(t0: torch.Tensor, t1: torch.Tensor) -> torch.Tensor:
        """``J`` [K,4,3]: ``(u_x,u_y,u_z, ω) = J @ (α, β, ω)`` for g/H reduction."""
        K = t0.shape[0]
        dev, dty = t0.device, t0.dtype
        J = torch.zeros(K, 4, 3, device=dev, dtype=dty)
        J[:, :3, 0] = t0
        J[:, :3, 1] = t1
        J[:, 3, 2] = 1.0
        return J

    @staticmethod
    def _friction_reduce_g(g_xyz: torch.Tensor, g_om: torch.Tensor,
                           J: torch.Tensor) -> torch.Tensor:
        """``[K,3]`` + ``[K]`` → ``[K,3]`` intrinsic gradient."""
        g4 = torch.cat([g_xyz, g_om.unsqueeze(1)], dim=1)
        return torch.einsum('kji,kj->ki', J, g4)

    @staticmethod
    def _friction_reduce_H(H4: torch.Tensor, J: torch.Tensor) -> torch.Tensor:
        """``[K,4,4]`` → ``[K,3,3]`` intrinsic Hessian."""
        return torch.matmul(J.transpose(1, 2), torch.matmul(H4, J))

    @staticmethod
    def _friction_reduce_H_theta_u(Htu4: torch.Tensor, J: torch.Tensor):
        """``[K,n,4]`` → ``[K,n,3]``."""
        return torch.einsum('knj,kji->kni', Htu4, J)

    @staticmethod
    def _friction_reduce_dgu(dgu4: torch.Tensor, J: torch.Tensor):
        """``[K,4,n]`` → ``[K,3,n]``."""
        return torch.einsum('kji,kjn->kin', J, dgu4)

    def _friction_plane_list_from_snap(
            self, manifolds: List[ContactManifold],
            friction_plane_snap: Dict[Tuple[int, int], torch.Tensor],
            ) -> List[torch.Tensor]:
        """
        Friction plane ``p`` for each manifold, aligned with C++
        ``ConvHullPBDSimulator`` / ``tangentEnergy(..., manifoldsLast)``.

        At step start, ``detectLastContact`` copies separating planes from the
        **current** pose ``θ_t`` into ``_manifoldsLast``; inner Newton/LM keeps
        those planes for **friction** while **contact** barriers re-detect
        ``m.p`` at the iterate ``θ_{t+1}``.  Call only when
        ``friction_plane_snap`` is not ``None``; otherwise use the caller's
        ``p_normals`` list unchanged (legacy).
        """
        out: List[torch.Tensor] = []
        for m in manifolds:
            key = (m.link_a, m.link_b if m.link_b >= 0 else -1)
            t = friction_plane_snap.get(key)
            if t is not None:
                out.append(t)
            else:
                out.append(m.p.detach().clone())
        return out

    def _friction_plane_snap_dict(
            self, manifolds: List[ContactManifold],
            ) -> Dict[Tuple[int, int], torch.Tensor]:
        """Snap dict ``{(link_a, link_b): p}`` from a manifold list."""
        return {(m.link_a, m.link_b if m.link_b >= 0 else -1): m.p.detach().clone()
                for m in manifolds}

    def _build_energy(self, theta_var, pd_target, theta_t_ref,
                      wv_next, wv_curr, wv_last,
                      p_list, u_list, p_normals,
                      manifold_link_ids, manifold_link_b_ids,
                      kp, kd,
                      early_exit_above: Optional[float] = None):
        dev, dty = self.device, self.dtype
        dt  = self.dt
        coef = self.coef_barrier
        x0  = self.x0
        K   = len(p_list)
        vmask = self._vert_mask_float                     # [L, max_M]

        E = (theta_var * 0).sum()

        # 1. Inertial
        accel = wv_next - 2.0 * wv_curr + wv_last
        E = E + (0.5 / (dt * dt)) * (self._rho_3d * vmask.unsqueeze(2) * accel.pow(2)).sum()

        # 2. Gravity
        E = E - (self._rho_2d * self.gravity * wv_next[:, :, 1] * vmask).sum()

        # 3. PD controller
        mask = self.joint_mask
        pos_err = (theta_var - pd_target) * mask
        vel_err = (theta_var - theta_t_ref) * mask
        E = E + 0.5 * kp * pos_err.pow(2).sum() + 0.5 * kd * vel_err.pow(2).sum()

        # 4. Joint limits
        if self._use_joint_limit_barrier and self._hinge_idx is not None:
            q = theta_var[self._hinge_idx]
            vl, _, _ = barrier_eval(q - self._hinge_lo, 0.3)
            vu, _, _ = barrier_eval(self._hinge_hi - q, 0.3)
            E = E + (vl + vu).sum() * 10.0

        if K == 0:
            return E

        if early_exit_above is not None and E.item() > early_exit_above:
            return E

        # --- Stack manifold data ---
        p_stack  = torch.stack(p_list)        # [K, 4]
        u_stack  = torch.stack(u_list)        # [K, 3]  (α, β, ω)
        pn_stack = torch.stack(p_normals)     # [K, 4]
        lid_a = torch.tensor(manifold_link_ids, device=dev, dtype=torch.long)
        lid_b = torch.tensor(manifold_link_b_ids, device=dev, dtype=torch.long)
        is_ground = (lid_b < 0)               # [K]
        gnd_idx = is_ground.nonzero(as_tuple=True)[0]
        lnk_idx = (~is_ground).nonzero(as_tuple=True)[0]

        max_M = self._max_M
        va_next = wv_next[lid_a]
        va_curr = wv_curr[lid_a]
        link_mask_a = vmask[lid_a]
        ones_KM1 = torch.ones(K, max_M, 1, device=dev, dtype=dty)

        # 5a. Normal contact — d_a (link A vertices, all manifolds)
        va_h = torch.cat([va_next, ones_KM1], dim=2)
        d_a = -torch.einsum('kmi,ki->km', va_h, p_stack)
        val_a, _, _ = barrier_eval(d_a, x0, self._barrier_d0_half)
        val_a = val_a * link_mask_a
        norm_p = torch.norm(p_stack[:, :3], dim=1)
        val_n, _, _ = barrier_eval(1.0 - norm_p, x0)
        E = E + coef * (val_n.sum() + val_a.sum())
        if early_exit_above is not None and E.item() > early_exit_above:
            return E

        # 5b. Normal contact — d_b (ground contacts)
        if gnd_idx.numel() > 0:
            p_gnd = p_stack[gnd_idx]
            d_b_gnd = torch.einsum('gi,ki->kg', self._ground_h, p_gnd)
            val_b_gnd, _, _ = barrier_eval(d_b_gnd, x0, self._barrier_d0_half)
            E = E + coef * val_b_gnd.sum()

        # 5c. Normal contact — d_b (link-link contacts)
        if lnk_idx.numel() > 0:
            lid_b_lnk = lid_b[lnk_idx]
            p_lnk = p_stack[lnk_idx]
            vb_next_lnk = wv_next[lid_b_lnk]
            mask_b_lnk = vmask[lid_b_lnk]
            ones_lnk = torch.ones(lnk_idx.numel(), max_M, 1, device=dev, dtype=dty)
            vb_h_lnk = torch.cat([vb_next_lnk, ones_lnk], dim=2)
            d_b_lnk = torch.einsum('kmi,ki->km', vb_h_lnk, p_lnk)
            val_b_lnk, _, _ = barrier_eval(d_b_lnk, x0, self._barrier_d0_half)
            val_b_lnk = val_b_lnk * mask_b_lnk
            E = E + coef * val_b_lnk.sum()

        if early_exit_above is not None and E.item() > early_exit_above:
            return E

        # 6. Friction
        if self._use_friction:
            E = E + self._friction_energy_total_pn(
                pn_stack, u_stack, wv_next, wv_curr,
                lid_a, lid_b, lnk_idx, coef, x0, self._barrier_d0_half)

        return E

    def _friction_energy_total_pn(
            self,
            pn_stack: torch.Tensor,
            u_stack: torch.Tensor,
            wv_next: torch.Tensor,
            wv_curr: torch.Tensor,
            lid_a: torch.Tensor,
            lid_b: torch.Tensor,
            lnk_idx: torch.Tensor,
            coef: float,
            x0: float,
            d0h: float,
            ) -> torch.Tensor:
        """
        Scalar friction energy (C++ ``energyP`` / ``energyN`` structure).

        - Slip uses ``wv_next`` (candidate ``θ_{t+1}``) and ``wv_curr`` (``θ_t``):
          tangential velocity ``Proj @ (v_next - v_last)/dt`` minus manifold ``u``.
        - Normal load ``A_m`` uses barrier depth from **``wv_curr``** (``θ_t``)
          against **``pn_stack``** (friction plane; frozen at step start when using
          ``friction_plane_snap`` in :meth:`step`).
        """
        dev, dty = self.device, self.dtype
        dt = self.dt
        K = pn_stack.shape[0]
        max_M = self._max_M
        vmask = self._vert_mask_float
        va_next = wv_next[lid_a]
        va_curr = wv_curr[lid_a]
        link_mask_a = vmask[lid_a]
        ones_KM1 = torch.ones(K, max_M, 1, device=dev, dtype=dty)
        _eps_n = self._friction_eps_n
        _eps_s = self._friction_eps_s

        n_vecs = pn_stack[:, :3]
        norm_n = torch.norm(n_vecs, dim=1, keepdim=True) + _eps_n
        n_hat = n_vecs / norm_n
        Proj = self._eye3.unsqueeze(0) - n_hat.unsqueeze(2) * n_hat.unsqueeze(1)
        u_xyz, omega_stack, _, _ = self._unified_u_to_xyz_omega(u_stack, n_hat)

        vel_a = (va_next - va_curr) / dt
        vel = vel_a.clone()
        if lnk_idx.numel() > 0:
            lid_b_lnk = lid_b[lnk_idx]
            vb_next_f = wv_next[lid_b_lnk]
            vb_curr_f = wv_curr[lid_b_lnk]
            vel[lnk_idx] = vel[lnk_idx] - (vb_next_f - vb_curr_f) / dt

        tan_vel = torch.einsum('kij,kmj->kmi', Proj, vel)
        r_nx = torch.cross(n_hat.unsqueeze(1), va_curr, dim=2)
        omega_term = omega_stack.view(K, 1, 1) * r_nx
        rel_vel = tan_vel - u_xyz.unsqueeze(1) - omega_term

        va_h_fn = torch.cat([va_curr, ones_KM1], dim=2)
        d_fn = -torch.einsum('kmi,ki->km', va_h_fn, pn_stack)
        _, bg_fn, _ = barrier_eval(d_fn, x0, d0h)
        pn3 = pn_stack[:, :3]
        f_vec = coef * bg_fn.unsqueeze(2) * pn3.unsqueeze(1)
        fn_sq = f_vec.pow(2).sum(dim=2)
        A_m = torch.sqrt(fn_sq + _eps_s) - self._sqrt_friction_eps_s

        s_norm = torch.sqrt(rel_vel.pow(2).sum(dim=2) + _eps_s)
        return (self.friction * dt * (A_m * s_norm * link_mask_a)).sum()

    def _reject_trial_state(
            self, E_trial: float, theta_trial: torch.Tensor,
            manifolds: List[ContactManifold]) -> Tuple[bool, str]:
        """
        Returns ``(True, reason)`` if LM / Newton must reject this trial
        regardless of descent / rho.
        """
        if self._reject_step_energy_above is not None:
            if math.isfinite(E_trial) and E_trial > self._reject_step_energy_above:
                return True, (
                    f"E_trial>{self._reject_step_energy_above:g}")
        return False, ""

    def _directional_deriv_energy_autograd(
            self,
            theta: torch.Tensor,
            theta_t: torch.Tensor,
            theta_tm1: torch.Tensor,
            pd_target: torch.Tensor,
            kp: float,
            kd: float,
            manifolds: List[ContactManifold],
            p_normals,
            u_tangents,
            direction_dx: torch.Tensor,
            wv_curr_cache=None,
            wv_last_cache=None) -> torch.Tensor:
        """
        Scalar directional derivative ``(∇_θ E)^T dx`` via PyTorch autograd.

        Uses the same scalar energy as :meth:`_build_energy` with
        ``wv_next = FK(θ)``, fixed ``wv_curr`` / ``wv_last``, and frozen
        manifold ``p`` / ``u`` / ``p_normals`` / ``u_tangents`` (detached),
        matching the DE probe in :meth:`debug_energy`.
        """
        dev, dty = self.device, self.dtype
        theta_ad = theta.to(device=dev, dtype=dty).detach().clone().requires_grad_(True)
        pd_t = pd_target.to(device=dev, dtype=dty).detach()
        th_t = theta_t.to(device=dev, dtype=dty).detach()
        th_tm1 = theta_tm1.to(device=dev, dtype=dty).detach()
        dx = direction_dx.to(device=dev, dtype=dty)

        with torch.no_grad():
            wv_curr = wv_curr_cache if wv_curr_cache is not None \
                else self._get_wv_stacked(th_t)
            wv_last = wv_last_cache if wv_last_cache is not None \
                else self._get_wv_stacked(th_tm1)

        wv_next = self._get_wv_stacked(theta_ad)
        p_list, u_list, mid_a, mid_b = self._manifold_energy_lists(
            manifolds, u_tangents)

        E = self._build_energy(
            theta_ad, pd_t, th_t, wv_next, wv_curr, wv_last,
            p_list, u_list, p_normals, mid_a, mid_b, kp, kd)

        (g_ad,) = torch.autograd.grad(
            E, theta_ad, create_graph=False, retain_graph=False,
            allow_unused=False)
        return (g_ad * dx).sum().detach()

    def _manifold_energy_lists(self, manifolds, u_tangents):
        """Return ``(p_list, u_list, mid_a, mid_b)`` for :meth:`_build_energy`."""
        K = len(manifolds)
        if K == 0:
            return [], [], [], []
        p_list = [m.p.detach() for m in manifolds]
        u_list = list(u_tangents)
        mid_a = [m.link_a for m in manifolds]
        mid_b = [m.link_b for m in manifolds]
        return p_list, u_list, mid_a, mid_b

    @staticmethod
    def _pair_p_u(entry):
        """``_pair_manifolds`` value: ``(p, u)`` with ``u`` shape ``[3]``.

        Legacy triple ``(p, u_xyz_free, omega)`` is coerced to unified
        ``[0, 0, ω]`` (tangential warm-start dropped).
        """
        if len(entry) == 2:
            return entry[0], entry[1]
        p, _u_xyz, o = entry
        dev, dty = p.device, p.dtype
        z2 = torch.zeros(2, device=dev, dtype=dty)
        if o.dim() == 0:
            o1 = o.unsqueeze(0)
        else:
            o1 = o.flatten()[:1].to(dtype=dty, device=dev)
        return p, torch.cat([z2, o1], dim=0)

    def _hvp_energy_theta_dir_autograd(
            self,
            theta: torch.Tensor,
            theta_t: torch.Tensor,
            theta_tm1: torch.Tensor,
            pd_target: torch.Tensor,
            kp: float,
            kd: float,
            manifolds: List[ContactManifold],
            p_normals,
            u_tangents,
            direction_dx: torch.Tensor,
            wv_curr_cache=None,
            wv_last_cache=None) -> torch.Tensor:
        """Autograd Hessian-vector product ``(∇²_θ E) @ dx`` (same E as DE)."""
        dev, dty = self.device, self.dtype
        th = theta.detach().to(dty).clone().requires_grad_(True)
        pd_t = pd_target.to(dty).detach()
        th_t = theta_t.to(dty).detach()
        th_tm1 = theta_tm1.to(dty).detach()
        dxv = direction_dx.to(dty)
        with torch.no_grad():
            wv_curr = wv_curr_cache if wv_curr_cache is not None \
                else self._get_wv_stacked(th_t)
            wv_last = wv_last_cache if wv_last_cache is not None \
                else self._get_wv_stacked(th_tm1)
        p_list, u_list, mid_a, mid_b = self._manifold_energy_lists(
            manifolds, u_tangents)
        wv_next = self._get_wv_stacked(th)
        E = self._build_energy(
            th, pd_t, th_t, wv_next, wv_curr, wv_last,
            p_list, u_list, p_normals, mid_a, mid_b, kp, kd)
        (g,) = torch.autograd.grad(
            E, th, create_graph=True, retain_graph=True)
        (hvp,) = torch.autograd.grad(
            (g * dxv).sum(), th, retain_graph=False, allow_unused=False)
        return hvp.detach()

    def _jvp_grad_theta_wrt_theta_t_autograd(
            self,
            theta: torch.Tensor,
            theta_t: torch.Tensor,
            theta_tm1: torch.Tensor,
            pd_target: torch.Tensor,
            kp: float,
            kd: float,
            manifolds: List[ContactManifold],
            p_normals,
            u_tangents,
            direction_dx: torch.Tensor,
            wv_last_cache=None) -> torch.Tensor:
        """``(∂(∇_θ E)/∂θ_t) @ dx`` at fixed ``θ`` (matches DDE-L FD)."""
        dev, dty = self.device, self.dtype
        pd_t = pd_target.to(dty).detach()
        th_tm1 = theta_tm1.to(dty).detach()
        with torch.no_grad():
            wv_last = wv_last_cache if wv_last_cache is not None \
                else self._get_wv_stacked(th_tm1)
        p_list, u_list, mid_a, mid_b = self._manifold_energy_lists(
            manifolds, u_tangents)
        tt0 = theta_t.detach().to(dty).clone().requires_grad_(True)
        dxv = direction_dx.to(dty)

        def g_fn(tt):
            th_fix = theta.detach().to(dty).clone().requires_grad_(True)
            wv_c = self._get_wv_stacked(tt)
            wv_n = self._get_wv_stacked(th_fix)
            E = self._build_energy(
                th_fix, pd_t, tt, wv_n, wv_c, wv_last,
                p_list, u_list, p_normals, mid_a, mid_b, kp, kd)
            # retain_graph=True: ``functional.jvp`` backprops through g_fn twice;
            # retain_graph=False frees E's graph before the outer JVP finishes.
            (g_out,) = torch.autograd.grad(
                E, th_fix, create_graph=True, retain_graph=True)
            return g_out

        _, jv = _autograd_functional_jvp(g_fn, (tt0,), (dxv,))
        return jv.detach()

    def _jvp_grad_theta_wrt_theta_tm1_autograd(
            self,
            theta: torch.Tensor,
            theta_t: torch.Tensor,
            theta_tm1: torch.Tensor,
            pd_target: torch.Tensor,
            kp: float,
            kd: float,
            manifolds: List[ContactManifold],
            p_normals,
            u_tangents,
            direction_dx: torch.Tensor,
            wv_curr_cache=None) -> torch.Tensor:
        """``(∂(∇_θ E)/∂θ_{t-1}) @ dx`` (matches DDE-LL FD)."""
        dev, dty = self.device, self.dtype
        pd_t = pd_target.to(dty).detach()
        th_t = theta_t.to(dty).detach()
        with torch.no_grad():
            wv_curr = wv_curr_cache if wv_curr_cache is not None \
                else self._get_wv_stacked(th_t)
        p_list, u_list, mid_a, mid_b = self._manifold_energy_lists(
            manifolds, u_tangents)
        tm0 = theta_tm1.detach().to(dty).clone().requires_grad_(True)
        dxv = direction_dx.to(dty)

        def g_fn(tm1):
            th_fix = theta.detach().to(dty).clone().requires_grad_(True)
            wv_n = self._get_wv_stacked(th_fix)
            wv_l = self._get_wv_stacked(tm1)
            E = self._build_energy(
                th_fix, pd_t, th_t, wv_n, wv_curr, wv_l,
                p_list, u_list, p_normals, mid_a, mid_b, kp, kd)
            (g_out,) = torch.autograd.grad(
                E, th_fix, create_graph=True, retain_graph=True)
            return g_out

        _, jv = _autograd_functional_jvp(g_fn, (tm0,), (dxv,))
        return jv.detach()

    def _jvp_grad_theta_wrt_pd_autograd(
            self,
            theta: torch.Tensor,
            theta_t: torch.Tensor,
            theta_tm1: torch.Tensor,
            pd_target: torch.Tensor,
            kp: float,
            kd: float,
            manifolds: List[ContactManifold],
            p_normals,
            u_tangents,
            direction_dx: torch.Tensor,
            wv_curr_cache=None,
            wv_last_cache=None) -> torch.Tensor:
        """``(∂(∇_θ E)/∂p_d) @ dx`` for PD target (matches DTDP / ``dg_dpd`` direction)."""
        dev, dty = self.device, self.dtype
        pd0 = pd_target.detach().to(dty).clone().requires_grad_(True)
        th_t = theta_t.to(dty).detach()
        th_tm1 = theta_tm1.to(dty).detach()
        dxv = direction_dx.to(dty)
        with torch.no_grad():
            wv_curr = wv_curr_cache if wv_curr_cache is not None \
                else self._get_wv_stacked(th_t)
            wv_last = wv_last_cache if wv_last_cache is not None \
                else self._get_wv_stacked(th_tm1)
        p_list, u_list, mid_a, mid_b = self._manifold_energy_lists(
            manifolds, u_tangents)

        def g_fn(pdv):
            th_fix = theta.detach().to(dty).clone().requires_grad_(True)
            wv_n = self._get_wv_stacked(th_fix)
            E = self._build_energy(
                th_fix, pdv, th_t, wv_n, wv_curr, wv_last,
                p_list, u_list, p_normals, mid_a, mid_b, kp, kd)
            (g_out,) = torch.autograd.grad(
                E, th_fix, create_graph=True, retain_graph=True)
            return g_out

        _, jv = _autograd_functional_jvp(g_fn, (pd0,), (dxv,))
        return jv.detach()

    def _dg_cross_blocks_for_ift(
            self,
            theta_star: torch.Tensor,
            theta_t: torch.Tensor,
            theta_tm1: torch.Tensor,
            manifolds: List[ContactManifold],
            wv_curr: torch.Tensor,
            p_normals,
            u_tangents,
            kp: float,
            kd: float):
        """Analytic ``∂g/∂θ_t``, ``∂g/∂θ_{t-1}``, ``∂g/∂p_d`` + FK results.

        Returns (dg_dt, dg_dtm1, dg_dpd, wv_star, J_star, wv_t, J_t, wv_tm1, J_tm1).
        """
        dt = self.dt
        vmask = self._vert_mask_float
        w_rho = self._rho_2d * vmask
        with torch.no_grad():
            wv_star, J_star, _, _ = self._compute_fk_jacobian_analytic(theta_star)
            wv_t, J_t, _, _ = self._compute_fk_jacobian_analytic(theta_t)
            wv_tm1, J_tm1, _, _ = self._compute_fk_jacobian_analytic(theta_tm1)
        B_rho = self._L * self._max_M
        n = self.n_dof
        w_sqrt = w_rho.reshape(B_rho).sqrt()
        Js_w = (J_star.reshape(B_rho, 3, n)
                * w_sqrt.unsqueeze(1).unsqueeze(2)).reshape(B_rho * 3, n)
        Jt_w = (J_t.reshape(B_rho, 3, n)
                * w_sqrt.unsqueeze(1).unsqueeze(2)).reshape(B_rho * 3, n)
        dg_dt = ((-2.0 / (dt * dt)) * torch.mm(Js_w.T, Jt_w)
                 - kd * torch.diag(self.joint_mask ** 2))
        dg_dt = dg_dt + self._friction_cross_theta_t(
            wv_star, wv_curr, J_star, J_t, manifolds, p_normals,
            u_tangents=u_tangents)
        Jtm1_w = (J_tm1.reshape(B_rho, 3, n)
                   * w_sqrt.unsqueeze(1).unsqueeze(2)).reshape(B_rho * 3, n)
        dg_dtm1 = (1.0 / (dt * dt)) * torch.mm(Js_w.T, Jtm1_w)
        dg_dpd = -kp * torch.diag(self.joint_mask ** 2)
        return dg_dt, dg_dtm1, dg_dpd, wv_star, J_star, wv_t, J_t, wv_tm1, J_tm1

    @staticmethod
    def _ift_rhs_vector_theta_t_u(
            b_top_n: torch.Tensor,
            dgu_dt: Optional[torch.Tensor],
            dx: torch.Tensor,
            n: int,
            K: int) -> torch.Tensor:
        """Directional RHS for ``∂/∂θ_t`` KKT row: top ``n`` + ``∂g_u/∂θ_t dx``."""
        dev, dty = b_top_n.device, b_top_n.dtype
        dim = n + K * 4 + K * 3
        R = torch.zeros(dim, device=dev, dtype=dty)
        R[:n] = b_top_n
        if K > 0 and dgu_dt is not None:
            # dgu_dt: [K, 3, n]  →  (dgu_dt @ dx).reshape(K * 3)
            R[n + K * 4:n + K * 4 + K * 3] = (dgu_dt @ dx).reshape(-1)
        return R

    def _jvp_grad_theta_wrt_local_verts_autograd(
            self,
            theta: torch.Tensor,
            theta_t: torch.Tensor,
            theta_tm1: torch.Tensor,
            pd_target: torch.Tensor,
            kp: float,
            kd: float,
            manifolds: List[ContactManifold],
            p_normals,
            u_tangents,
            dc_flat: torch.Tensor,
            verts0: torch.Tensor,
            wv_curr_cache=None,
            wv_last_cache=None) -> torch.Tensor:
        """``(∂(∇_θ E)/∂(vec d)) @ dc`` for local hull vertices (DDE-XL).

        ``wv_curr`` / ``wv_last`` are recomputed **inside** the JVP closure so
        that they also flow through ``d`` (local verts).  This matches C++ where
        perturbing a local vertex affects wv at **all** three timesteps.
        ``wv_curr_cache`` / ``wv_last_cache`` are ignored.
        """
        dev, dty = self.device, self.dtype
        L, mM = self._L, self._max_M
        pd_t = pd_target.to(dty).detach()
        th_t = theta_t.to(dty).detach()
        th_tm1 = theta_tm1.to(dty).detach()
        p_list, u_list, mid_a, mid_b = self._manifold_energy_lists(
            manifolds, u_tangents)
        v0 = verts0.reshape(-1).detach().to(dty).clone().requires_grad_(True)
        dc_v = dc_flat.to(dty).reshape(-1)
        saved_lv = self._local_verts

        def g_fn(vf):
            self._local_verts = vf.reshape(L, mM, 3)
            th_fix = theta.detach().to(dty).clone().requires_grad_(True)
            wv_n = self._get_wv_stacked(th_fix)
            wv_c = self._get_wv_stacked(th_t)
            wv_l = self._get_wv_stacked(th_tm1)
            E = self._build_energy(
                th_fix, pd_t, th_t, wv_n, wv_c, wv_l,
                p_list, u_list, p_normals, mid_a, mid_b, kp, kd)
            (g_out,) = torch.autograd.grad(
                E, th_fix, create_graph=True, retain_graph=True)
            return g_out

        try:
            _, jv = _autograd_functional_jvp(g_fn, (v0,), (dc_v,))
        finally:
            self._local_verts = saved_lv
        return jv.detach()

    def _compute_energy(self, theta, theta_t, theta_tm1, pd_target,
                        kp, kd, manifolds,
                        analytic_derivs=False, p_normals=None, u_tangents=None,
                        wv_curr_cache=None, wv_last_cache=None,
                        friction_plane_snap=None):
        dev, dty = self.device, self.dtype
        n = self.n_dof
        K = len(manifolds)
        dt = self.dt
        coef = self.coef_barrier
        x0 = self.x0
        vmask = self._vert_mask_float
        mM = self._max_M

        if p_normals is None:
            p_normals = [m.p for m in manifolds]
        if friction_plane_snap is not None:
            pn_friction = self._friction_plane_list_from_snap(
                manifolds, friction_plane_snap)
        else:
            pn_friction = p_normals
        if u_tangents is None:
            u_tangents = [m.u for m in manifolds]
        manifold_link_ids = [m.link_a for m in manifolds]
        theta_d = theta.detach().to(dty)
        pd_t = pd_target.to(dty)
        th_t = theta_t.to(dty)

        with torch.no_grad():
            wv_curr = wv_curr_cache if wv_curr_cache is not None \
                else self._get_wv_stacked(th_t)
            wv_last = wv_last_cache if wv_last_cache is not None \
                else self._get_wv_stacked(theta_tm1.to(dty))

        T_all_cache, dR_cache = None, None
        if analytic_derivs:
            wv_next, J, T_all_cache, dR_cache = self._compute_fk_jacobian_analytic(theta_d)
        else:
            with torch.no_grad():
                wv_next = self._get_wv_stacked(theta_d)
            J = None
        early_cap = None
        if (not analytic_derivs) and self._reject_step_energy_above is not None:
            early_cap = self._reject_step_energy_above

        with torch.no_grad():
            manifold_link_b_ids = [m.link_b for m in manifolds]
            E_val = self._build_energy(
                theta_d, pd_t, th_t, wv_next, wv_curr, wv_last,
                [m.p for m in manifolds], u_tangents,
                pn_friction, manifold_link_ids, manifold_link_b_ids, kp, kd,
                early_exit_above=early_cap)

        if not analytic_derivs:
            return E_val.item(), None, None

        # Vertex-level gradient g_v
        accel = wv_next - 2.0 * wv_curr + wv_last
        g_v = ((1.0 / (dt * dt)) * (self._rho_2d * vmask).unsqueeze(2)
               * accel)
        g_v[:, :, 1] = (g_v[:, :, 1]
                        - self._rho_2d * self.gravity * vmask)

        if K > 0:
            p_stack = torch.stack([m.p.detach() for m in manifolds])
            pn_stack = torch.stack([t.detach() for t in pn_friction])
            u_stack = torch.stack(u_tangents)
            lid = torch.tensor(manifold_link_ids, device=dev,
                               dtype=torch.long)
            lid_b = torch.tensor(manifold_link_b_ids, device=dev,
                                 dtype=torch.long)
            is_ground = (lid_b < 0)
            gnd_idx = is_ground.nonzero(as_tuple=True)[0]
            lnk_idx = (~is_ground).nonzero(as_tuple=True)[0]

            link_mask = vmask[lid]
            va_next = wv_next[lid]
            va_curr = wv_curr[lid]
            ones_K1 = torch.ones(K, mM, 1, device=dev, dtype=dty)
            p3 = p_stack[:, :3]
            _eps_n = self._friction_eps_n
            _eps_s = self._friction_eps_s

            va_h = torch.cat([va_next, ones_K1], 2)
            d_a = -torch.einsum('kmi,ki->km', va_h, p_stack)
            _, bg_a, bh_a = barrier_eval(d_a, x0, self._barrier_d0_half)
            bg_m = bg_a * link_mask
            g_v.index_add_(0, lid,
                           -coef * bg_m.unsqueeze(2) * p3.unsqueeze(1))

            if lnk_idx.numel() > 0:
                lid_b_lnk = lid_b[lnk_idx]
                mask_b_lnk = vmask[lid_b_lnk]
                vb_next_lnk = wv_next[lid_b_lnk]
                ones_lnk = torch.ones(lnk_idx.numel(), mM, 1, device=dev, dtype=dty)
                vb_h_lnk = torch.cat([vb_next_lnk, ones_lnk], 2)
                d_b_lnk = torch.einsum('kmi,ki->km', vb_h_lnk, p_stack[lnk_idx])
                _, bg_b_lnk, _ = barrier_eval(d_b_lnk, x0, self._barrier_d0_half)
                bg_b_lnk_m = bg_b_lnk * mask_b_lnk
                p3_lnk = p3[lnk_idx]
                g_v.index_add_(0, lid_b_lnk,
                               coef * bg_b_lnk_m.unsqueeze(2) * p3_lnk.unsqueeze(1))

            if self._use_friction:
                n_vecs = pn_stack[:, :3]
                norm_n = torch.norm(n_vecs, dim=1, keepdim=True) + _eps_n
                n_hat = n_vecs / norm_n
                Proj = (self._eye3.unsqueeze(0)
                        - n_hat.unsqueeze(2) * n_hat.unsqueeze(1))
                vel_a = (va_next - va_curr) / dt
                if lnk_idx.numel() > 0:
                    lid_b_lnk = lid_b[lnk_idx]
                    vb_next_f = wv_next[lid_b_lnk]
                    vb_curr_f = wv_curr[lid_b_lnk]
                    vel = vel_a.clone()
                    vel[lnk_idx] = vel[lnk_idx] - (vb_next_f - vb_curr_f) / dt
                else:
                    vel = vel_a
                tan_vel = torch.einsum('kij,kmj->kmi', Proj, vel)
                u_xyz_fd, omega_stack_fd, t0_fd, t1_fd = (
                    self._unified_u_to_xyz_omega(u_stack, n_hat))
                r_nx_fd = torch.cross(n_hat.unsqueeze(1), va_curr, dim=2)
                omega_term_fd = omega_stack_fd.view(K, 1, 1) * r_nx_fd
                rel_vel = tan_vel - u_xyz_fd.unsqueeze(1) - omega_term_fd

                va_h_fn = torch.cat([va_curr, ones_K1], 2)
                d_fn = -torch.einsum('kmi,ki->km', va_h_fn, pn_stack)
                _, bg_fn, _ = barrier_eval(d_fn, x0, self._barrier_d0_half)
                pn3_fd = pn_stack[:, :3]
                f_vec_fd = coef * bg_fn.unsqueeze(2) * pn3_fd.unsqueeze(1)
                fn_sq_fd = f_vec_fd.pow(2).sum(dim=2)
                A_m_fd = torch.sqrt(fn_sq_fd + _eps_s) - self._sqrt_friction_eps_s

                s_norm = torch.sqrt(rel_vel.pow(2).sum(2) + _eps_s)
                inv_s = 1.0 / (s_norm + 1e-30)
                w_fric = (A_m_fd * link_mask).unsqueeze(2)
                rel_over_s = rel_vel * inv_s.unsqueeze(2)
                proj_rs = torch.einsum('kij,kmj->kmi', Proj, rel_over_s)
                g_fric_s = self.friction * w_fric * proj_rs
                g_v.index_add_(0, lid, g_fric_s)
                if lnk_idx.numel() > 0:
                    lid_b_lnk = lid_b[lnk_idx]
                    g_v.index_add_(0, lid_b_lnk, -g_fric_s[lnk_idx])

        B = self._L * mM
        g_theta = torch.einsum('bci,bc->i',
                               J.reshape(B, 3, n),
                               g_v.reshape(B, 3))
        mask = self.joint_mask
        g_theta = (g_theta
                   + kp * mask.pow(2) * (theta_d - pd_t)
                   + kd * mask.pow(2) * (theta_d - th_t))

        jl_val_lo = jl_val_hi = jl_bg_lo = jl_bg_hi = jl_bh_lo = jl_bh_hi = None
        if self._use_joint_limit_barrier and self._hinge_idx is not None:
            q = theta_d[self._hinge_idx]
            jl_val_lo, jl_bg_lo, jl_bh_lo = barrier_eval(q - self._hinge_lo, 0.3)
            jl_val_hi, jl_bg_hi, jl_bh_hi = barrier_eval(self._hinge_hi - q, 0.3)
            g_lim = torch.zeros(n, device=dev, dtype=dty)
            g_lim[self._hinge_idx] = 10.0 * (jl_bg_lo - jl_bg_hi)
            g_theta = g_theta + g_lim

        H_theta = self._assemble_H_theta(
            J, theta_d, wv_next, wv_curr, wv_last,
            manifolds, pn_friction, pd_t, th_t, kp, kd,
            g_v=g_v, u_tangents=u_tangents,
            T_all=T_all_cache, dR_cache=dR_cache,
            jl_bh=(jl_bh_lo, jl_bh_hi))

        if K > 0:
            self._compute_manifold_hessians(
                J, wv_next, wv_curr, manifolds, pn_friction, u_tangents)

        return E_val.item(), g_theta.detach(), H_theta

    # FK JACOBIAN (analytic): J = d(wv)/d(θ)  [L, max_M, 3, n]
    def _fk_j_wv(self, theta: torch.Tensor, lv: torch.Tensor):
        """Returns ``(wv, J, T_all, dR_cache)``.

        ``T_all`` and ``dR_cache`` are cached so that
        ``_compute_fk_correction_analytic`` can reuse them.
        """
        n = self.n_dof
        dev, dty = lv.device, lv.dtype
        L, mM = self._L, self._max_M

        with torch.no_grad():
            transforms = self.robot.forward_kinematics(theta.detach())
            T_all = torch.stack(transforms)
            R_all = T_all[:, :3, :3].to(device=dev, dtype=dty)
            t_all = T_all[:, :3, 3].to(device=dev, dtype=dty)
        wv = torch.einsum('lmj,lkj->lmk', lv, R_all) + t_all.unsqueeze(1)
        J = torch.zeros(L, mM, 3, n, device=dev, dtype=dty)

        dR_cache = {}

        def _embed_dof_column(col_lm3: torch.Tensor, dof_i: int) -> torch.Tensor:
            if dof_i > 0:
                z0 = torch.zeros(
                    L, mM, 3, dof_i, device=dev, dtype=col_lm3.dtype)
            else:
                z0 = col_lm3.new_zeros(L, mM, 3, 0)
            tail = n - dof_i - 1
            if tail > 0:
                z1 = torch.zeros(
                    L, mM, 3, tail, device=dev, dtype=col_lm3.dtype)
            else:
                z1 = col_lm3.new_zeros(L, mM, 3, 0)
            return torch.cat(
                [z0, col_lm3.unsqueeze(-1), z1], dim=-1)

        def _scatter_links_to_l(desc_t: torch.Tensor, src_d_m3: torch.Tensor
                                ) -> torch.Tensor:
            D = desc_t.numel()
            idx = desc_t[:, None, None].expand(D, mM, 3)
            base = torch.zeros(L, mM, 3, device=dev, dtype=src_d_m3.dtype)
            return base.scatter_add(0, idx, src_d_m3)

        for ji, joint in enumerate(self.robot.joints):
            off = joint.dof_offset
            parent = joint.parent_link
            desc_links = self._descendants[ji]
            if not desc_links:
                continue
            desc = torch.tensor(desc_links, device=dev, dtype=torch.long)
            wv_desc = wv[desc]
            D = len(desc_links)

            R_par = (torch.eye(3, device=dev, dtype=dty)
                     if parent < 0 else T_all[parent, :3, :3])
            t_par = (torch.zeros(3, device=dev, dtype=dty)
                     if parent < 0 else T_all[parent, :3, 3])

            if joint.jtype == 'hinge':
                a_norm = joint.axis / (joint.axis.norm() + 1e-12)
                a_w = R_par @ a_norm
                o_w = R_par @ joint.origin + t_par
                diff = wv_desc - o_w.reshape(1, 1, 3)
                cross_val = torch.linalg.cross(
                    a_w.reshape(1, 1, 3).expand_as(diff), diff)
                col = _scatter_links_to_l(desc, cross_val)
                J = J + _embed_dof_column(col, off)

            elif joint.jtype == 'free':
                R_exp = R_par.reshape(1, 1, 3, 3).expand(D, mM, -1, -1)
                for ddx in range(3):
                    col = _scatter_links_to_l(desc, R_exp[:, :, :, ddx])
                    J = J + _embed_dof_column(col, off + ddx)

                roll = theta[off + 3].detach()
                pitch = theta[off + 4].detach()
                yaw = theta[off + 5].detach()
                cr, sr = torch.cos(roll), torch.sin(roll)
                cp, sp = torch.cos(pitch), torch.sin(pitch)
                cy, sy = torch.cos(yaw), torch.sin(yaw)
                z = torch.zeros_like(cr)

                dR_stack = torch.stack([
                    torch.stack([
                        torch.stack([z, cy*sp*cr + sy*sr, -cy*sp*sr + sy*cr]),
                        torch.stack([z, sy*sp*cr - cy*sr, -sy*sp*sr - cy*cr]),
                        torch.stack([z, cp*cr, -cp*sr])]),
                    torch.stack([
                        torch.stack([-cy*sp, cy*cp*sr, cy*cp*cr]),
                        torch.stack([-sy*sp, sy*cp*sr, sy*cp*cr]),
                        torch.stack([-cp, -sp*sr, -sp*cr])]),
                    torch.stack([
                        torch.stack([-sy*cp, -sy*sp*sr - cy*cr, -sy*sp*cr + cy*sr]),
                        torch.stack([cy*cp, cy*sp*sr - sy*cr, cy*sp*cr + sy*sr]),
                        torch.stack([z, z, z])])
                ])

                dR_cache[ji] = (cr, sr, cp, sp, cy, sy, z, dR_stack)

                child = joint.child_link
                R_child = T_all[child, :3, :3]
                t_child = T_all[child, :3, 3]
                q = torch.einsum(
                    'ij,dmj->dmi', R_child.T,
                    wv_desc - t_child.reshape(1, 1, 3))
                RdR = torch.einsum('ij,ejk->eik', R_par, dR_stack)
                J_euler = torch.einsum('eij,dmj->edmi', RdR, q)
                for e in range(3):
                    col = _scatter_links_to_l(desc, J_euler[e])
                    J = J + _embed_dof_column(col, off + 3 + e)

        return wv, J, T_all, dR_cache

    def _compute_fk_jacobian_analytic(self, theta: torch.Tensor):
        with torch.no_grad():
            return self._fk_j_wv(theta, self._local_verts)

    # FK CORRECTION: Σ_v g_v · d²(wv)/dθ² (geometric, no AD)
    def _compute_fk_correction_analytic(self, J, g_v, theta, wv,
                                       T_all=None, dR_cache=None):
        dev, dty = self.device, self.dtype
        n = self.n_dof
        vmask = self._vert_mask_float
        H_FK = torch.zeros(n, n, device=dev, dtype=dty)

        if T_all is None:
            with torch.no_grad():
                transforms = self.robot.forward_kinematics(theta.detach())
                T_all = torch.stack(transforms)
        if dR_cache is None:
            dR_cache = {}

        axes_w = [None] * len(self.robot.joints)
        for ji, jt in enumerate(self.robot.joints):
            if jt.jtype != 'hinge':
                continue
            par = jt.parent_link
            R_p = torch.eye(3, device=dev, dtype=dty) if par < 0 \
                else T_all[par, :3, :3]
            axes_w[ji] = R_p @ (jt.axis / (jt.axis.norm() + 1e-12))

        free_ji, f_off, R_eu, M_stk = None, None, None, None
        cr = sr = cp = sp = cy = sy = z = None
        for ji, jt in enumerate(self.robot.joints):
            if jt.jtype != 'free':
                continue
            free_ji = ji
            f_off = jt.dof_offset
            R_eu = T_all[jt.child_link, :3, :3]
            if ji in dR_cache:
                cr, sr, cp, sp, cy, sy, z, dR_stk = dR_cache[ji]
            else:
                roll  = theta[f_off + 3].detach()
                pitch = theta[f_off + 4].detach()
                yaw   = theta[f_off + 5].detach()
                cr, sr = torch.cos(roll),  torch.sin(roll)
                cp, sp = torch.cos(pitch), torch.sin(pitch)
                cy, sy = torch.cos(yaw),   torch.sin(yaw)
                z = torch.zeros_like(cr)
                dR_stk = torch.stack([
                    torch.stack([
                        torch.stack([z, cy*sp*cr+sy*sr, -cy*sp*sr+sy*cr]),
                        torch.stack([z, sy*sp*cr-cy*sr, -sy*sp*sr-cy*cr]),
                        torch.stack([z, cp*cr, -cp*sr])]),
                    torch.stack([
                        torch.stack([-cy*sp, cy*cp*sr, cy*cp*cr]),
                        torch.stack([-sy*sp, sy*cp*sr, sy*cp*cr]),
                        torch.stack([-cp, -sp*sr, -sp*cr])]),
                    torch.stack([
                        torch.stack([-sy*cp, -sy*sp*sr-cy*cr, -sy*sp*cr+cy*sr]),
                        torch.stack([ cy*cp,  cy*sp*sr-sy*cr,  cy*sp*cr+sy*sr]),
                        torch.stack([z, z, z])])])
            M_stk = torch.einsum('rij,kj->rik', dR_stk, R_eu)
            break

        # ---- Hinge-Hinge & Euler-Hinge FK correction ----
        for ki, jt_k in enumerate(self.robot.joints):
            if jt_k.jtype != 'hinge':
                continue
            k_off = jt_k.dof_offset
            desc_k = self._descendants[ki]
            if not desc_k:
                continue
            dt_ = torch.tensor(desc_k, device=dev, dtype=torch.long)
            J_k   = J[dt_, :, :, k_off]              # [D,mM,3]
            g_d   = g_v[dt_]                          # [D,mM,3]
            mk    = vmask[dt_].unsqueeze(2)           # [D,mM,1]

            S_k = (torch.linalg.cross(J_k, g_d) * mk).sum(dim=(0, 1))

            # Self (hinge k vs k)
            H_FK[k_off, k_off] = axes_w[ki] @ S_k

            # Ancestors
            W_k = None  # lazily computed for euler ancestors
            for ji in self._ancestors[ki]:
                jt_j = self.robot.joints[ji]
                j_off = jt_j.dof_offset
                if jt_j.jtype == 'hinge':
                    val = axes_w[ji] @ S_k
                    H_FK[j_off, k_off] = val
                    H_FK[k_off, j_off] = val
                elif jt_j.jtype == 'free' and M_stk is not None:
                    if W_k is None:
                        g_f = (g_d * mk).reshape(-1, 3)
                        j_f = (J_k * mk).reshape(-1, 3)
                        W_k = g_f.T @ j_f             # [3,3]
                    vals_eh = (M_stk * W_k.unsqueeze(0)).sum(dim=(1, 2))
                    eo = torch.arange(3, device=dev, dtype=torch.long)
                    H_FK[j_off + 3 + eo, k_off] = vals_eh
                    H_FK[k_off, j_off + 3 + eo] = vals_eh

        # ---- Euler-Euler FK correction ----
        if free_ji is not None and z is not None:
            t_eu = T_all[self.robot.joints[free_ji].child_link, :3, 3]
            diff = wv - t_eu.reshape(1, 1, 3)
            q_all = torch.einsum('ij,lmj->lmi', R_eu.T, diff)
            g_fl = (g_v * vmask.unsqueeze(2)).reshape(-1, 3)
            q_fl = (q_all * vmask.unsqueeze(2)).reshape(-1, 3)
            Q = g_fl.T @ q_fl                         # [3,3]

            d2R = self._compute_d2R_single(cr, sr, cp, sp, cy, sy, z)
            vals_ee = (d2R * Q.unsqueeze(0)).sum(dim=(1, 2))
            ri = torch.tensor([0, 0, 0, 1, 1, 2], device=dev, dtype=torch.long)
            si = torch.tensor([0, 1, 2, 1, 2, 2], device=dev, dtype=torch.long)
            fo = f_off + 3
            H_FK[fo + ri, fo + si] = vals_ee
            H_FK[fo + si, fo + ri] = vals_ee

        return H_FK

    @staticmethod
    def _compute_d2R_single(cr, sr, cp, sp, cy, sy, z):
        """Single-env d²R: scalar inputs, returns [6, 3, 3] for upper-tri (r,s)."""
        def s3(a, b, c):
            return torch.stack([a, b, c])
        def m3(r0, r1, r2):
            return torch.stack([r0, r1, r2], dim=1)
        d00 = m3(s3(z, -cy*sp*sr+sy*cr, -cy*sp*cr-sy*sr),
                 s3(z, -sy*sp*sr-cy*cr, -sy*sp*cr+cy*sr),
                 s3(z, -cp*sr, -cp*cr))
        d01 = m3(s3(z, cy*cp*cr, -cy*cp*sr),
                 s3(z, sy*cp*cr, -sy*cp*sr),
                 s3(z, -sp*cr, sp*sr))
        d02 = m3(s3(z, -sy*sp*cr+cy*sr, sy*sp*sr+cy*cr),
                 s3(z, cy*sp*cr+sy*sr, -cy*sp*sr+sy*cr),
                 s3(z, z, z))
        d11 = m3(s3(-cy*cp, -cy*sp*sr, -cy*sp*cr),
                 s3(-sy*cp, -sy*sp*sr, -sy*sp*cr),
                 s3(sp, -cp*sr, -cp*cr))
        d12 = m3(s3(sy*sp, -sy*cp*sr, -sy*cp*cr),
                 s3(-cy*sp, cy*cp*sr, cy*cp*cr),
                 s3(z, z, z))
        d22 = m3(s3(-cy*cp, -cy*sp*sr+sy*cr, -cy*sp*cr-sy*sr),
                 s3(-sy*cp, -sy*sp*sr-cy*cr, -sy*sp*cr+cy*sr),
                 s3(z, z, z))
        return torch.stack([d00, d01, d02, d11, d12, d22], dim=0)

    # ASSEMBLE H_theta = J^T H_vv J + friction + FK_correction + PD + limits
    def _assemble_H_theta(self, J, theta, wv_next, wv_curr, wv_last,
                          manifolds, p_normals, pd_target, theta_t,
                          kp, kd, g_v=None,
                          u_tangents=None,
                          T_all=None, dR_cache=None,
                          jl_bh=None):
        dev, dty = self.device, self.dtype
        n = self.n_dof
        L, mM = self._L, self._max_M
        K = len(manifolds)
        dt = self.dt
        coef, x0 = self.coef_barrier, self.x0
        vmask = self._vert_mask_float
        B = L * mM

        # ---- Inertial H_vv, g_v (batch: h_in.unsqueeze(0).expand(N,-1,-1)) ----
        accel = wv_next - 2.0 * wv_curr + wv_last
        rho_vm = (self._rho_2d * vmask).unsqueeze(2)
        g_v = ((1.0 / (dt * dt)) * rho_vm * accel)
        g_v[:, :, 1] = g_v[:, :, 1] - self._rho_2d * self.gravity * vmask
        h_in = self._rho_2d * vmask / (dt * dt)
        H_vv = h_in.unsqueeze(2).unsqueeze(3) * self._eye3

        H_fric = None
        va_h = d_a = bg_a = bh_a = bg_m = bh_m = p3 = pp = None
        vb_h_lnk = bg_b_lnk_m = bh_b_lnk_m = d_b_lnk = vmask_b = None
        n_hat = Proj = rel_vel = inv_s = inv_s3 = A_m_ht = vel = None

        if K > 0:
            p_stack = torch.stack([m.p.detach() for m in manifolds])
            pn_stack = torch.stack(p_normals)
            u_stack = torch.stack(u_tangents) if u_tangents is not None \
                else torch.stack([m.u.detach() for m in manifolds])
            lid = torch.tensor([m.link_a for m in manifolds], device=dev, dtype=torch.long)
            lid_b = torch.tensor([m.link_b for m in manifolds], device=dev, dtype=torch.long)
            is_ground = (lid_b < 0)
            gnd_idx = is_ground.nonzero(as_tuple=True)[0]
            lnk_idx = (~is_ground).nonzero(as_tuple=True)[0]

            vmask_a = vmask[lid]
            va = wv_next[lid]
            va_curr = wv_curr[lid]
            ones = torch.ones(K, mM, 1, device=dev, dtype=dty)
            p3 = p_stack[:, :3]

            va_h = torch.cat([va, ones], 2)
            d_a = -torch.einsum('kmi,ki->km', va_h, p_stack)
            _, bg_a, bh_a = barrier_eval(d_a, x0, self._barrier_d0_half)
            link_mask_a = vmask_a
            bg_m = bg_a * link_mask_a
            bh_m = bh_a * link_mask_a

            # Contact g_v, H_vv (batch: scatter_a_flat)
            g_v_contact_a = -coef * bg_m.unsqueeze(2) * p3.unsqueeze(1)
            g_v.index_add_(0, lid, g_v_contact_a)

            pp = p3.unsqueeze(2) * p3.unsqueeze(1)
            H_contact_a = coef * bh_m.unsqueeze(2).unsqueeze(3) * pp.unsqueeze(1)
            H_vv.index_add_(0, lid, H_contact_a)

            if lnk_idx.numel() > 0:
                lid_b_lnk = lid_b[lnk_idx]
                vb = wv_next[lid_b_lnk]
                vmask_b = vmask[lid_b_lnk]
                ones_l = torch.ones(lnk_idx.numel(), mM, 1, device=dev, dtype=dty)
                vb_h_lnk = torch.cat([vb, ones_l], 2)
                d_b_lnk = torch.einsum('kmi,ki->km', vb_h_lnk, p_stack[lnk_idx])
                _, bg_b_lnk, bh_b_lnk = barrier_eval(d_b_lnk, x0, self._barrier_d0_half)
                link_mask_b = vmask_b
                bg_b_lnk_m = bg_b_lnk * link_mask_b
                bh_b_lnk_m = bh_b_lnk * link_mask_b
                g_v_contact_b = coef * bg_b_lnk_m.unsqueeze(2) * p3[lnk_idx].unsqueeze(1)
                g_v.index_add_(0, lid_b_lnk, g_v_contact_b)
                pp_lnk = pp[lnk_idx]
                H_contact_b = coef * bh_b_lnk_m.unsqueeze(2).unsqueeze(3) * pp_lnk.unsqueeze(1)
                H_vv.index_add_(0, lid_b_lnk, H_contact_b)

            if self._use_friction:
                _eps_n = self._friction_eps_n
                _eps_s = self._friction_eps_s
                n_vecs = pn_stack[:, :3]
                norm_n = torch.norm(n_vecs, dim=1, keepdim=True) + _eps_n
                n_hat = n_vecs / norm_n
                Proj = self._eye3.unsqueeze(0) - n_hat.unsqueeze(2) * n_hat.unsqueeze(1)
                vel_a = (va - va_curr) / dt
                if lnk_idx.numel() > 0:
                    vel = vel_a.clone()
                    vel[lnk_idx] = vel[lnk_idx] - (wv_next[lid_b_lnk] - wv_curr[lid_b_lnk]) / dt
                else:
                    vel = vel_a
                tan_vel = torch.einsum('kij,kmj->kmi', Proj, vel)
                u_xyz_ht, omega_stack_ht, _, _ = self._unified_u_to_xyz_omega(
                    u_stack, n_hat)
                r_nx_ht = torch.cross(n_hat.unsqueeze(1), va_curr, dim=2)
                rel_vel = (tan_vel - u_xyz_ht.unsqueeze(1)
                           - omega_stack_ht.view(K, 1, 1) * r_nx_ht)

                va_h_fn = torch.cat([va_curr, ones], 2)
                d_fn = -torch.einsum('kmi,ki->km', va_h_fn, pn_stack)
                _, bg_fn, _ = barrier_eval(d_fn, x0, self._barrier_d0_half)
                pn3_ht = pn_stack[:, :3]
                f_vec_ht = coef * bg_fn.unsqueeze(2) * pn3_ht.unsqueeze(1)
                fn_sq_ht = f_vec_ht.pow(2).sum(dim=2)
                A_m_ht = torch.sqrt(fn_sq_ht + _eps_s) - self._sqrt_friction_eps_s

                s_norm = torch.sqrt(rel_vel.pow(2).sum(2) + _eps_s)
                inv_s = 1.0 / (s_norm + 1e-30)
                inv_s3 = inv_s.pow(3)

                rel_over_s = rel_vel * inv_s.unsqueeze(2)
                proj_rs = torch.einsum('kij,kmj->kmi', Proj, rel_over_s)
                w_ht = (A_m_ht * link_mask_a).unsqueeze(2)
                g_fric_s = self.friction * w_ht * proj_rs
                g_v.index_add_(0, lid, g_fric_s)
                if lnk_idx.numel() > 0:
                    g_v.index_add_(0, lid_b_lnk, -g_fric_s[lnk_idx])

                weight = self.friction / dt * A_m_ht * link_mask_a
                PMP = (Proj.unsqueeze(1) * inv_s.unsqueeze(2).unsqueeze(3)
                       - rel_vel.unsqueeze(-1) * rel_vel.unsqueeze(-2)
                         * inv_s3.unsqueeze(2).unsqueeze(3))
                H_fric = weight.unsqueeze(2).unsqueeze(3) * PMP
                H_vv.index_add_(0, lid, H_fric)
                if lnk_idx.numel() > 0:
                    H_vv.index_add_(0, lid_b_lnk, H_fric[lnk_idx])

        # ---- H_theta = J^T H_vv J  (bmm + single GEMM) ----
        J2 = J.reshape(B, 3, n)
        H2 = H_vv.reshape(B, 3, 3)
        JtH = torch.bmm(J2.transpose(1, 2), H2)                   # [B, n, 3]
        H_theta = torch.mm(
            JtH.permute(0, 2, 1).reshape(B * 3, n).T,
            J2.reshape(B * 3, n))                                   # [n, n]

        # Friction cross terms (link-link, bmm + single GEMM)
        if K > 0 and lnk_idx.numel() > 0 and H_fric is not None:
            la = lid[lnk_idx]
            lb = lid_b[lnk_idx]
            Ja = J[la]
            Jb = J[lb]
            Hf = H_fric[lnk_idx]
            Kll = lnk_idx.numel()
            Bll = Kll * mM
            Ja2 = Ja.reshape(Bll, 3, n)
            Hf2 = Hf.reshape(Bll, 3, 3)
            Jb2 = Jb.reshape(Bll, 3, n)
            JaHf = torch.bmm(Ja2.transpose(1, 2), Hf2)            # [Bll, n, 3]
            cross_sum = torch.mm(
                JaHf.permute(0, 2, 1).reshape(Bll * 3, n).T,
                Jb2.reshape(Bll * 3, n))                           # [n, n]
            H_theta = H_theta - cross_sum - cross_sum.T

        H_theta = H_theta + self._compute_fk_correction_analytic(
            J, g_v, theta, wv_next, T_all=T_all, dR_cache=dR_cache)

        mask = self.joint_mask
        H_theta = H_theta + torch.diag((kp + kd) * mask * mask)
        if self._use_joint_limit_barrier and self._hinge_idx is not None:
            if jl_bh is not None:
                bh_lo, bh_hi = jl_bh
            else:
                q = theta[self._hinge_idx].detach()
                _, _, bh_lo = barrier_eval(q - self._hinge_lo, 0.3)
                _, _, bh_hi = barrier_eval(self._hinge_hi - q, 0.3)
            diag_lim = torch.zeros(n, device=dev, dtype=dty)
            diag_lim[self._hinge_idx] = 10.0 * (bh_lo + bh_hi)
            H_theta = H_theta + torch.diag(diag_lim)

        H_theta = 0.5 * (H_theta + H_theta.T)
        return H_theta.detach()

    # MANIFOLD HESSIANS: H_pp, g_p, H_theta_p (contact); H_uu, g_u, H_theta_u (+ friction)
    def _compute_manifold_hessians(self, J, wv_next, wv_curr,
                                   manifolds, p_normals, u_tangents=None):
        """``p_normals``: friction plane 4-vectors (same convention as energy)."""
        dev, dty = self.device, self.dtype
        n = self.n_dof
        mM = self._max_M
        dt = self.dt
        coef = self.coef_barrier
        x0 = self.x0
        vmask = self._vert_mask_float
        K = len(manifolds)
        if K == 0:
            return

        _eps_n = self._friction_eps_n
        _eps_s = self._friction_eps_s

        p_stack = torch.stack([m.p.detach() for m in manifolds])
        pn_stack = torch.stack(p_normals)
        u_stack = torch.stack(u_tangents) if u_tangents is not None \
            else torch.stack([m.u.detach() for m in manifolds])
        lid = torch.tensor([m.link_a for m in manifolds], device=dev, dtype=torch.long)
        lid_b = torch.tensor([m.link_b for m in manifolds], device=dev, dtype=torch.long)
        is_ground = (lid_b < 0)
        gnd_idx = is_ground.nonzero(as_tuple=True)[0]
        lnk_idx = (~is_ground).nonzero(as_tuple=True)[0]

        vmask_a = vmask[lid]
        va = wv_next[lid]
        va_curr = wv_curr[lid]
        ones = torch.ones(K, mM, 1, device=dev, dtype=dty)
        p3 = p_stack[:, :3]

        va_h = torch.cat([va, ones], 2)
        d_a = -torch.einsum('kmi,ki->km', va_h, p_stack)
        _, bg_a, bh_a = barrier_eval(d_a, x0, self._barrier_d0_half)
        link_mask_a = vmask_a
        bg_m = bg_a * link_mask_a
        bh_m = bh_a * link_mask_a

        # H_pp [K, 4, 4] — batch: H_pp_a = einsum('npm,npmi,npmj->npij', bh_m, va_h, va_h)
        H_pp_a = torch.einsum('km,kmi,kmj->kij', bh_m, va_h, va_h)

        H_pp_b = torch.zeros(K, 4, 4, device=dev, dtype=dty)
        bg_b_gnd = bh_b_gnd = None
        if gnd_idx.numel() > 0:
            d_b_gnd = torch.einsum('gi,ki->gk', self._ground_h, p_stack[gnd_idx])
            _, bg_b_gnd, bh_b_gnd = barrier_eval(d_b_gnd, x0, self._barrier_d0_half)
            amask_gnd = torch.ones(gnd_idx.numel(), device=dev, dtype=dty)
            H_pp_b[gnd_idx] = torch.einsum(
                'kg,gi,gj->kij', bh_b_gnd.T * amask_gnd.unsqueeze(1),
                self._ground_h, self._ground_h)

        vb_h_lnk = bg_b_lnk_m = bh_b_lnk_m = d_b_lnk = vmask_b = None
        if lnk_idx.numel() > 0:
            lid_b_lnk = lid_b[lnk_idx]
            vmask_b = vmask[lid_b_lnk]
            vb = wv_next[lid_b_lnk]
            ones_l = torch.ones(lnk_idx.numel(), mM, 1, device=dev, dtype=dty)
            vb_h_lnk = torch.cat([vb, ones_l], 2)
            d_b_lnk = torch.einsum('kmi,ki->km', vb_h_lnk, p_stack[lnk_idx])
            _, bg_b_lnk, bh_b_lnk = barrier_eval(d_b_lnk, x0, self._barrier_d0_half)
            link_mask_b = vmask_b
            bg_b_lnk_m = bg_b_lnk * link_mask_b
            bh_b_lnk_m = bh_b_lnk * link_mask_b
            H_pp_b[lnk_idx] = torch.einsum(
                'km,kmi,kmj->kij', bh_b_lnk_m, vb_h_lnk, vb_h_lnk)

        np3 = torch.norm(p3, dim=1, keepdim=True) + _eps_n
        n_hat_p = p3 / np3
        s_p = 1.0 - np3.squeeze(1)
        _, bg_n, bh_n = barrier_eval(s_p, x0)
        nn_p = n_hat_p.unsqueeze(2) * n_hat_p.unsqueeze(1)
        I_nn_p = self._eye3.unsqueeze(0) - nn_p
        H_pp_n33 = (bh_n.unsqueeze(1).unsqueeze(2) * nn_p
                    - bg_n.unsqueeze(1).unsqueeze(2) * I_nn_p / np3.unsqueeze(2))
        H_pp_n = torch.zeros(K, 4, 4, device=dev, dtype=dty)
        amask_f = torch.ones(K, device=dev, dtype=dty)
        H_pp_n[:, :3, :3] = H_pp_n33 * amask_f.unsqueeze(1).unsqueeze(2)

        H_pp_all = coef * (H_pp_a + H_pp_b + H_pp_n)
        H_pp_all = 0.5 * (H_pp_all + H_pp_all.transpose(1, 2))

        # ``H_pp`` / ``g_p`` below are **contact-barrier only** (same as
        # :meth:`batch_simulator.BatchSimulator._compute_manifold_hessians_batch`).
        # Friction enters the KKT via ``H_uu``, ``g_u``, ``H_theta_u``, and
        # ``H_theta`` / ``g_v`` — not via autograd on ``p``.
        if self._use_friction:
            n_vecs = pn_stack[:, :3]
            norm_n = torch.norm(n_vecs, dim=1, keepdim=True) + _eps_n
            n_hat = n_vecs / norm_n
            Proj = self._eye3.unsqueeze(0) - n_hat.unsqueeze(2) * n_hat.unsqueeze(1)
            vel_a = (va - va_curr) / dt
            if lnk_idx.numel() > 0:
                vel = vel_a.clone()
                vel[lnk_idx] = vel[lnk_idx] - (wv_next[lid_b_lnk] - wv_curr[lid_b_lnk]) / dt
            else:
                vel = vel_a
            tan_vel = torch.einsum('kij,kmj->kmi', Proj, vel)
            u_xyz_mh, omega_stack_mh, t0_mh, t1_mh = self._unified_u_to_xyz_omega(
                u_stack, n_hat)
            r_nx_mh = torch.cross(n_hat.unsqueeze(1), va_curr, dim=2)
            rel_vel = (tan_vel - u_xyz_mh.unsqueeze(1)
                       - omega_stack_mh.view(K, 1, 1) * r_nx_mh)
            va_h_fn = torch.cat([va_curr, ones], 2)
            d_fn = -torch.einsum('kmi,ki->km', va_h_fn, pn_stack)
            _, bg_fn, _ = barrier_eval(d_fn, x0, self._barrier_d0_half)
            pn3_mh = pn_stack[:, :3]
            f_vec_mh = coef * bg_fn.unsqueeze(2) * pn3_mh.unsqueeze(1)
            fn_sq_mh = f_vec_mh.pow(2).sum(dim=2)
            A_m_mh = torch.sqrt(fn_sq_mh + _eps_s) - self._sqrt_friction_eps_s
            s_norm = torch.sqrt(rel_vel.pow(2).sum(2) + _eps_s)
            inv_s = 1.0 / (s_norm + 1e-30)
            inv_s3 = inv_s.pow(3)

            c_mh = self.friction * dt * A_m_mh * link_mask_a
            alpha_uu = (c_mh * inv_s).sum(dim=1)
            v_w = rel_vel * (c_mh * inv_s3).sqrt().unsqueeze(2)
            H_uu_33 = (alpha_uu.reshape(K, 1, 1) * self._eye3
                       - torch.bmm(v_w.transpose(1, 2), v_w))
            h_mh = (rel_vel * r_nx_mh).sum(dim=2)
            t_mix = (r_nx_mh / s_norm.unsqueeze(2)
                     - (h_mh / s_norm.pow(2)).unsqueeze(2)
                     * rel_vel / s_norm.unsqueeze(2))
            col_mh = (c_mh.unsqueeze(2) * t_mix).sum(dim=1)
            rnx_sq_mh = r_nx_mh.pow(2).sum(dim=2)
            H_ww_mh = (c_mh * (rnx_sq_mh / s_norm
                                - h_mh.pow(2) / s_norm.pow(3))).sum(dim=1)
            H_uu_all = torch.zeros(K, 4, 4, device=dev, dtype=dty)
            H_uu_all[:, :3, :3] = 0.5 * (H_uu_33 + H_uu_33.transpose(1, 2))
            H_uu_all[:, :3, 3] = col_mh
            H_uu_all[:, 3, :3] = col_mh
            H_uu_all[:, 3, 3] = H_ww_mh
            H_uu_all = 0.5 * (H_uu_all + H_uu_all.transpose(1, 2))
            rel_over_s = rel_vel * inv_s.unsqueeze(2)
            g_u3_mh = -(c_mh.unsqueeze(2) * rel_over_s).sum(dim=1)
            g_om_mh = -(c_mh * h_mh / s_norm).sum(dim=1)
            J_mh = self._friction_J_lift(t0_mh, t1_mh)
            H_uu_all = self._friction_reduce_H(H_uu_all, J_mh)
            g_u_all = self._friction_reduce_g(g_u3_mh, g_om_mh, J_mh)
        else:
            H_uu_all = torch.zeros(K, 3, 3, device=dev, dtype=dty)
            g_u_all = torch.zeros(K, 3, device=dev, dtype=dty)

        # g_p, g_u (batch formulas)
        g_p_a = -coef * torch.einsum('km,kmi->ki', bg_m, va_h)
        g_p_b = torch.zeros(K, 4, device=dev, dtype=dty)
        if gnd_idx.numel() > 0 and bg_b_gnd is not None:
            amask_gnd = torch.ones(gnd_idx.numel(), device=dev, dtype=dty)
            g_p_b[gnd_idx] = coef * torch.einsum(
                'kg,gi->ki', bg_b_gnd.T * amask_gnd.unsqueeze(1), self._ground_h)
        if lnk_idx.numel() > 0 and bg_b_lnk_m is not None:
            g_p_b[lnk_idx] = coef * torch.einsum('km,kmi->ki', bg_b_lnk_m, vb_h_lnk)
        g_p_n = torch.zeros(K, 4, device=dev, dtype=dty)
        g_p_n[:, :3] = -coef * bg_n.unsqueeze(1) * n_hat_p * amask_f.unsqueeze(1)
        g_p_all = g_p_a + g_p_b + g_p_n

        # H_theta_p [K, n, 4] — batch: cross_1, cross_2, cross_a
        J_a_all = J[lid]
        cross_1 = torch.einsum('ki,kmj,km->kmij', p3, va_h, bh_m)
        cross_2 = torch.zeros(K, mM, 3, 4, device=dev, dtype=dty)
        cross_2[:, :, :3, :3] = -bg_m.unsqueeze(2).unsqueeze(3) * self._eye3.unsqueeze(0).unsqueeze(0)
        cross_a = coef * (cross_1 + cross_2)
        H_tp_all = torch.einsum('kmci,kmcj->kij', J_a_all, cross_a)

        if lnk_idx.numel() > 0 and bh_b_lnk_m is not None:
            J_b_lnk = J[lid_b_lnk]
            p3_lnk = p3[lnk_idx]
            cross_b1 = torch.einsum('ki,kmj,km->kmij', p3_lnk, vb_h_lnk, bh_b_lnk_m)
            cross_b2 = torch.zeros(lnk_idx.numel(), mM, 3, 4, device=dev, dtype=dty)
            bg_b2_m = bg_b_lnk_m
            cross_b2[:, :, :3, :3] = bg_b2_m.unsqueeze(2).unsqueeze(3) * self._eye3.unsqueeze(0).unsqueeze(0)
            cross_b = coef * (cross_b1 + cross_b2)
            H_tp_b = torch.einsum('kmci,kmcj->kij', J_b_lnk, cross_b)
            H_tp_all[lnk_idx] = H_tp_all[lnk_idx] + H_tp_b

        # H_theta_u [K, n, 4] — friction coupling to (u, ω)
        if self._use_friction:
            M_wu = (Proj.unsqueeze(1) * inv_s.unsqueeze(2).unsqueeze(3)
                    - rel_vel.unsqueeze(-1) * rel_vel.unsqueeze(-2)
                      * inv_s3.unsqueeze(2).unsqueeze(3))
            w_wu = -(self.friction * A_m_mh * link_mask_a)
            cross_wu = w_wu.unsqueeze(2).unsqueeze(3) * M_wu
            H_tu_u = torch.einsum('kmci,kmcj->kij', J_a_all, cross_wu)
            if lnk_idx.numel() > 0:
                J_b_lnk = J[lid_b_lnk]
                H_tu_u[lnk_idx] = H_tu_u[lnk_idx] + torch.einsum(
                    'kmci,kmcj->kij', J_b_lnk, -cross_wu[lnk_idx])
            inner_om = (-r_nx_mh / s_norm.unsqueeze(2)
                          + (h_mh / s_norm.pow(2)).unsqueeze(2)
                          * rel_vel / s_norm.unsqueeze(2))
            cross_wom = (self.friction * (A_m_mh * link_mask_a).unsqueeze(2)
                         * inner_om)
            H_tu_om = torch.einsum('kmci,kmc->ki', J_a_all, cross_wom)
            if lnk_idx.numel() > 0:
                H_tu_om[lnk_idx] = (H_tu_om[lnk_idx]
                                    - torch.einsum(
                                        'kmci,kmc->ki',
                                        J[lid_b_lnk], cross_wom[lnk_idx]))
            H_tu_4 = torch.cat([H_tu_u, H_tu_om.unsqueeze(2)], dim=2)
            H_tu_all = self._friction_reduce_H_theta_u(H_tu_4, J_mh)
        else:
            H_tu_all = torch.zeros(K, n, 3, device=dev, dtype=dty)

        g_p_store = g_p_all.detach()
        g_u_store = g_u_all.detach()
        for ki, m in enumerate(manifolds):
            m.g_p = g_p_store[ki].detach()
            m.g_u = g_u_store[ki].detach()
            m.H_pp = H_pp_all[ki].detach()
            m.H_uu = H_uu_all[ki].detach()
            m.H_theta_p = H_tp_all[ki].detach()
            m.H_theta_u = H_tu_all[ki].detach()

    # STEP (LM solver)
    def step(self, theta_t: torch.Tensor, theta_tm1: torch.Tensor,
             pd_target: torch.Tensor, kp: float = 100.0, kd: float = 10.0,
             theta_init: Optional[torch.Tensor] = None,
             initial_manifolds: Optional[List[ContactManifold]] = None,
             ) -> Tuple[torch.Tensor, GradInfo, List[ContactManifold]]:
        dev, dty = self.device, self.dtype
        theta_t   = theta_t.to(dtype=dty, device=dev)
        theta_tm1 = theta_tm1.to(dtype=dty, device=dev)
        pd_target = pd_target.to(dtype=dty, device=dev)

        if theta_init is None:
            theta = theta_t.clone().detach()
        else:
            ti = theta_init.to(dtype=dty, device=dev)
            if ti.shape != theta_t.shape:
                raise ValueError(
                    f"theta_init shape {tuple(ti.shape)} != theta_t "
                    f"{tuple(theta_t.shape)}")
            theta = ti.clone().detach()

        if initial_manifolds is not None and len(initial_manifolds) > 0:
            manifolds = self._clone_manifolds(initial_manifolds)
            for m in manifolds:
                key = (m.link_a, m.link_b) if m.link_b >= 0 else (m.link_a, -1)
                self._pair_manifolds[key] = (
                    m.p.detach().clone(), m.u.detach().clone())
            p_normals = [m.p.detach().clone() for m in manifolds]
            u_tangents = [m.u.detach().clone() for m in manifolds]
        else:
            manifolds = self._detect_contacts(theta)
            p_normals = [m.p.detach().clone() for m in manifolds]
            u_tangents = [m.u.detach().clone() for m in manifolds]

        friction_plane_snap = self._friction_plane_snap_dict(manifolds)

        ctx = dict(theta_t=theta_t, theta_tm1=theta_tm1,
                   pd_target=pd_target, kp=kp, kd=kd,
                   friction_plane_snap=friction_plane_snap)

        # Entire forward solve + IFT blocks are analytic; no autograd on θ/p/u.
        with torch.no_grad():
            wv_curr = self._get_wv_stacked(theta_t)
            wv_last = self._get_wv_stacked(theta_tm1)
            self._last_step_wv_curr = wv_curr.clone()
            self._last_step_wv_last = wv_last.clone()
            ctx['wv_curr'] = wv_curr
            ctx['wv_last'] = wv_last

            theta, manifolds, H_bar = self._solve_lm(
                    theta, manifolds, p_normals, ctx, u_tangents=u_tangents)

            for m in manifolds:
                key = (m.link_a, m.link_b) if m.link_b >= 0 else (m.link_a, -1)
                self._pair_manifolds[key] = (
                    m.p.detach().clone(), m.u.detach().clone())

            grad_info = self._compute_backward(
                theta, ctx['theta_t'], ctx['theta_tm1'], ctx['pd_target'],
                ctx['kp'], ctx['kd'], manifolds, H_bar,
                wv_curr=ctx['wv_curr'], wv_last=ctx['wv_last'],
                p_normals=[m.p.detach().clone() for m in manifolds],
                u_tangents=[m.u.detach().clone() for m in manifolds],
                friction_plane_snap=ctx['friction_plane_snap'])
        return theta, grad_info, manifolds

    def _schur_damp_pu(self, alpha: float) -> float:
        """p/u LM damping ``α·lm_gamma`` (C++ SchurUpdate)."""
        return alpha * float(self.lm_gamma)

    # LM trust-region solver
    def _solve_theta_direct(self, g_theta, H_theta, alpha):
        """Solve ``(H_theta + alpha * I) delta = -g_theta`` without Schur."""
        H_reg = H_theta + alpha * self._eye_n
        H_reg = 0.5 * (H_reg + H_reg.T)
        try:
            delta = -torch.linalg.solve(H_reg, g_theta)
        except Exception:
            delta = -g_theta * 0.01
        return delta, H_reg.detach()

    def _solve_lm(self, theta, manifolds, p_normals, ctx, u_tangents=None):
        nu = 2.0
        alpha_max = 1e20
        alpha_min = 1e-6
        use_schur = self._implicit

        fps = ctx.get('friction_plane_snap')
        E, g_theta, H_theta = self._compute_energy(
            theta, ctx['theta_t'], ctx['theta_tm1'], ctx['pd_target'],
            ctx['kp'], ctx['kd'], manifolds, analytic_derivs=True,
            p_normals=p_normals, u_tangents=u_tangents,
            wv_curr_cache=ctx['wv_curr'], wv_last_cache=ctx['wv_last'],
            friction_plane_snap=fps)

        H_bar_last = None
        it_accepted = 0
        mode_tag = "Schur" if use_schur else "explicit"
        if self._output:
            print(f"  [LM] contacts={len(manifolds)}  E0={E:.6e}  "
                  f"max|g_θ|={g_theta.abs().max().item():.3e}  "
                  f"alpha={self._alpha:.2e}  ({mode_tag})")

        for _ in range(self.max_iter * 10):
            if use_schur:
                delta_theta, g_bar, H_bar = self._schur_update(
                    g_theta, H_theta, manifolds, self._alpha)
            else:
                delta_theta, H_bar = self._solve_theta_direct(
                    g_theta, H_theta, self._alpha)
                g_bar = g_theta
            H_bar_last = H_bar

            theta_trial = theta + delta_theta

            pair_snap = self._snapshot_pair_manifolds()
            if use_schur:
                manifolds_trial = self._clone_manifolds(manifolds)
                self._back_substitute(manifolds_trial, delta_theta, self._alpha)
                for m in manifolds_trial:
                    key = (m.link_a, m.link_b) if m.link_b >= 0 \
                        else (m.link_a, -1)
                    self._pair_manifolds[key] = (
                        m.p.detach().clone(), m.u.detach().clone())

            manifolds2 = self._detect_contacts(theta_trial)
            pn2 = [m.p.detach().clone() for m in manifolds2]
            ut2 = [m.u.detach().clone() for m in manifolds2]
            E_trial_v, _, _ = self._compute_energy(
                theta_trial, ctx['theta_t'], ctx['theta_tm1'],
                ctx['pd_target'], ctx['kp'], ctx['kd'],
                manifolds2, analytic_derivs=False,
                p_normals=pn2, u_tangents=ut2,
                wv_curr_cache=ctx['wv_curr'],
                wv_last_cache=ctx['wv_last'],
                friction_plane_snap=fps)
            E_trial = float(E_trial_v)

            reject_phys, rsn_phys = self._reject_trial_state(
                E_trial, theta_trial, manifolds2)
            if reject_phys or not math.isfinite(E_trial):
                self._restore_pair_manifolds(pair_snap)
                self._alpha *= nu
                nu *= 2.0
                if self._output:
                    tag = (f"  [REJECT bad_state: {rsn_phys}]"
                           if reject_phys else "  [REJECT non-finite E]")
                    print(f"    attempt: E_trial={E_trial:.6e}  "
                          f"alpha={self._alpha:.2e}  nu={nu:.0f}{tag}")
                if self._alpha >= alpha_max:
                    if self._output:
                        print("    FAIL: alpha >= alpha_max")
                    break
                continue

            if use_schur:
                H_schur = H_bar - self._alpha * self._eye_n
                pred = (delta_theta @ (g_bar + H_schur @ delta_theta * 0.5)).item()
            else:
                pred = (delta_theta @ (g_theta + H_theta @ delta_theta * 0.5)).item()
            rho = (E_trial - E) / pred if abs(pred) > 1e-30 else 0.0

            if E_trial < E and rho > 0:
                self._alpha *= max(1.0 / 3.0, 1.0 - (2.0 * rho - 1.0) ** 3)
                self._alpha = max(min(self._alpha, alpha_max), alpha_min)
                nu = 2.0
                theta = theta_trial.detach()
                manifolds = manifolds2
                p_normals = pn2
                u_tangents = ut2

                E, g_theta, H_theta = self._compute_energy(
                    theta, ctx['theta_t'], ctx['theta_tm1'],
                    ctx['pd_target'], ctx['kp'], ctx['kd'],
                    manifolds, analytic_derivs=True,
                    p_normals=p_normals, u_tangents=u_tangents,
                    wv_curr_cache=ctx['wv_curr'],
                    wv_last_cache=ctx['wv_last'],
                    friction_plane_snap=fps)

                it_accepted += 1
                g_max = g_theta.abs().max().item()
                if self._output:
                    print(f"    iter {it_accepted:2d}: E={E:.6e}  "
                          f"max|g_θ|={g_max:.3e}  alpha={self._alpha:.2e}  "
                          f"rho={rho:.4f}  K={len(manifolds)}  [ACCEPT]")
                if g_max < self.gtol:
                    if self._output:
                        print(f"    converged: max|g_θ|={g_max:.2e} "
                              f"< gtol={self.gtol:.1e}")
                    break
                if it_accepted >= self.max_iter:
                    break
            else:
                self._restore_pair_manifolds(pair_snap)
                self._alpha *= nu
                nu *= 2.0
                if self._output:
                    tag = (f"  [REJECT bad_state: {rsn_phys}]"
                           if reject_phys else "  [REJECT]")
                    print(f"    attempt: E_trial={E_trial:.6e}  "
                          f"alpha={self._alpha:.2e}  nu={nu:.0f}{tag}")
                if self._alpha >= alpha_max:
                    if self._output:
                        print("    FAIL: alpha >= alpha_max")
                    break

        if H_bar_last is not None and g_theta is not None:
            if use_schur:
                _, _, H_bar_last = self._schur_update(
                    g_theta, H_theta, manifolds, self._alpha)
            else:
                _, H_bar_last = self._solve_theta_direct(
                    g_theta, H_theta, self._alpha)
        return theta, manifolds, H_bar_last

    def _snapshot_pair_manifolds(self):
        """Deep copy of _pair_manifolds (p, u) per pair."""
        out = {}
        for k, v in self._pair_manifolds.items():
            p, u = self._pair_p_u(v)
            out[k] = (p.clone(), u.clone())
        return out

    def _restore_pair_manifolds(self, snap) -> None:
        """Restore snapshot from :meth:`_snapshot_pair_manifolds` (in-place)."""
        if snap is None:
            return
        self._pair_manifolds = {}
        for k, v in snap.items():
            p, u = self._pair_p_u(v)
            self._pair_manifolds[k] = (p.clone(), u.clone())

    # DETECT CONTACTS (energy-based, warm-start p/u) — batched barrier_eval
    def _detect_contacts(self, theta: torch.Tensor) -> List[ContactManifold]:
        dev, dty = self.device, self.dtype
        manifolds = []
        x0 = self.x0
        d0h = self._barrier_d0_half
        with torch.no_grad():
            wv = self._get_wv_stacked(theta.to(dty))
        vmask = self._vert_mask_float
        mM = self._max_M

        # Link-link pairs (batched)
        P_ll = len(self._contact_pair_info)
        if P_ll > 0:
            lid_a = torch.tensor([t[0] for t in self._contact_pair_info],
                                 device=dev, dtype=torch.long)
            lid_b = torch.tensor([t[1] for t in self._contact_pair_info],
                                 device=dev, dtype=torch.long)
            p_u_ll = [self._pair_p_u(self._pair_manifolds[(t[0], t[1])])
                      for t in self._contact_pair_info]
            p_ll = torch.stack([pu[0] for pu in p_u_ll])
            u_ll = torch.stack([pu[1] for pu in p_u_ll])

            ones_ll = torch.ones(P_ll, mM, 1, device=dev, dtype=dty)
            va_h = torch.cat([wv[lid_a], ones_ll], 2)
            d_a = -torch.einsum('pmi,pi->pm', va_h, p_ll)
            val_a, _, _ = barrier_eval(d_a, x0, d0h)

            vb_h = torch.cat([wv[lid_b], ones_ll], 2)
            d_b = torch.einsum('pmi,pi->pm', vb_h, p_ll)
            val_b, _, _ = barrier_eval(d_b, x0, d0h)

            energy_ll = ((val_a * vmask[lid_a]).sum(1)
                         + (val_b * vmask[lid_b]).sum(1))
            for idx in (energy_ll > 0).nonzero(as_tuple=True)[0]:
                ii = idx.item()
                manifolds.append(ContactManifold(
                    link_a=lid_a[ii].item(), link_b=lid_b[ii].item(),
                    p=p_ll[ii].clone(), u=u_ll[ii].clone()))

        # Link-ground pairs (batched)
        gv = self.robot.ground.vertices if self.robot.ground is not None else None
        P_gnd = len(self._ground_pair_info)
        if gv is not None and P_gnd > 0:
            gv_h = torch.cat([gv,
                              torch.ones(gv.shape[0], 1, device=dev, dtype=dty)], 1)
            lid_a_g = torch.tensor([t[0] for t in self._ground_pair_info],
                                   device=dev, dtype=torch.long)
            p_u_gnd = [self._pair_p_u(self._pair_manifolds[(t[0], -1)])
                       for t in self._ground_pair_info]
            p_gnd = torch.stack([pu[0] for pu in p_u_gnd])
            u_gnd = torch.stack([pu[1] for pu in p_u_gnd])

            ones_g = torch.ones(P_gnd, mM, 1, device=dev, dtype=dty)
            va_h_g = torch.cat([wv[lid_a_g], ones_g], 2)
            d_a_g = -torch.einsum('pmi,pi->pm', va_h_g, p_gnd)
            val_a_g, _, _ = barrier_eval(d_a_g, x0, d0h)

            d_b_g = torch.einsum('gi,pi->pg', gv_h, p_gnd)
            val_b_g, _, _ = barrier_eval(d_b_g, x0, d0h)

            energy_gnd = ((val_a_g * vmask[lid_a_g]).sum(1)
                          + val_b_g.sum(1))
            for idx in (energy_gnd > 0).nonzero(as_tuple=True)[0]:
                ii = idx.item()
                manifolds.append(ContactManifold(
                    link_a=lid_a_g[ii].item(), link_b=-1,
                    p=p_gnd[ii].clone(), u=u_gnd[ii].clone()))

        return manifolds

    # SCHUR UPDATE: Schur-complement elimination of p/u blocks
    def _schur_update(self, g_theta, H_theta, manifolds, alpha):
        dev, dty = self.device, self.dtype
        n = self.n_dof
        K = len(manifolds)

        g_bar = g_theta.clone()
        H_bar = H_theta.clone()
        reg_pu = self._schur_damp_pu(alpha)

        # --- Eliminate p  (batched solve) ---
        valid_p = [m for m in manifolds
                   if m.g_p is not None and m.H_pp is not None
                   and m.H_theta_p is not None]
        if valid_p:
            Kp = len(valid_p)
            Hpp_batch = torch.stack([m.H_pp for m in valid_p]) \
                        + reg_pu * self._eye4.unsqueeze(0)   # [Kp,4,4]
            gp_batch  = torch.stack([m.g_p for m in valid_p])        # [Kp,4]
            Htp_batch = torch.stack([m.H_theta_p for m in valid_p])  # [Kp,n,4]

            # Batched RHS:  [Kp, 4, 1+n]  (g_p | H_tp^T)
            rhs = torch.cat([gp_batch.unsqueeze(2),
                             Htp_batch.transpose(1, 2)], dim=2)      # [Kp,4,1+n]
            sol = torch.linalg.solve(Hpp_batch, rhs)                 # [Kp,4,1+n]
            inv_gp  = sol[:, :, 0]                                   # [Kp,4]
            inv_Htp = sol[:, :, 1:]                                  # [Kp,4,n]

            g_bar = g_bar - torch.einsum('kni,ki->n', Htp_batch, inv_gp)
            H_bar = H_bar - torch.einsum('kni,kij->nj', Htp_batch, inv_Htp)

        # --- Eliminate u  (batched solve) ---
        valid_u = [m for m in manifolds
                   if m.g_u is not None and m.H_uu is not None
                   and m.H_theta_u is not None]
        if valid_u:
            Ku = len(valid_u)
            Huu_batch = torch.stack([m.H_uu for m in valid_u]) \
                        + reg_pu * self._eye3.unsqueeze(0)   # [Ku,3,3]
            gu_batch  = torch.stack([m.g_u for m in valid_u])        # [Ku,3]
            Htu_batch = torch.stack([m.H_theta_u for m in valid_u])  # [Ku,n,3]

            rhs = torch.cat([gu_batch.unsqueeze(2),
                             Htu_batch.transpose(1, 2)], dim=2)      # [Ku,3,1+n]
            sol = torch.linalg.solve(Huu_batch, rhs)
            inv_gu  = sol[:, :, 0]
            inv_Htu = sol[:, :, 1:]

            g_bar = g_bar - torch.einsum('kni,ki->n', Htu_batch, inv_gu)
            H_bar = H_bar - torch.einsum('kni,kij->nj', Htu_batch, inv_Htu)

        # θ block: +α on diagonal only (C++ SchurUpdate ~L909), not α·γ
        H_bar = H_bar + alpha * self._eye_n
        H_bar = 0.5 * (H_bar + H_bar.T)

        try:
            delta_theta = -torch.linalg.solve(H_bar, g_bar)
        except Exception:
            delta_theta = -g_bar * 0.01

        return delta_theta, g_bar.detach(), H_bar.detach()

    # BACK-SUBSTITUTION from Δθ → Δp, Δu
    def _back_substitute(self, manifolds, delta_theta, alpha):
        dev, dty = self.device, self.dtype
        n = self.n_dof
        reg_pu = self._schur_damp_pu(alpha)

        valid_p = [i for i, m in enumerate(manifolds)
                   if m.g_p is not None and m.H_pp is not None
                   and m.H_theta_p is not None]
        if valid_p:
            Hpp_batch = torch.stack([manifolds[i].H_pp for i in valid_p]) \
                        + reg_pu * self._eye4.unsqueeze(0)
            gp_batch  = torch.stack([manifolds[i].g_p for i in valid_p])
            Htp_batch = torch.stack([manifolds[i].H_theta_p for i in valid_p])
            rhs = gp_batch + torch.einsum('kni,n->ki', Htp_batch, delta_theta)
            dp = torch.linalg.solve(Hpp_batch, rhs)
            p_new = torch.stack([manifolds[i].p for i in valid_p]) - dp
            for j, mi in enumerate(valid_p):
                manifolds[mi].p.copy_(p_new[j])

        valid_u = [i for i, m in enumerate(manifolds)
                   if m.g_u is not None and m.H_uu is not None
                   and m.H_theta_u is not None]
        if valid_u:
            Huu_batch = torch.stack([manifolds[i].H_uu for i in valid_u]) \
                        + reg_pu * self._eye3.unsqueeze(0)
            gu_batch  = torch.stack([manifolds[i].g_u for i in valid_u])
            Htu_batch = torch.stack([manifolds[i].H_theta_u for i in valid_u])
            rhs = gu_batch + torch.einsum('kni,n->ki', Htu_batch, delta_theta)
            du = torch.linalg.solve(Huu_batch, rhs)
            u_new = torch.stack([manifolds[i].u for i in valid_u]) - du
            for j, mi in enumerate(valid_u):
                manifolds[mi].u.copy_(u_new[j])

    # FULL-SYSTEM ASSEMBLY: (n+4K+3K) KKT from sub-blocks
    def _assemble_full_system(self, g_theta, H_theta, manifolds, alpha):
        dev, dty = self.device, self.dtype
        n = self.n_dof
        K = len(manifolds)
        dim = n + K * 4 + K * 3

        H_full = torch.zeros(dim, dim, device=dev, dtype=dty)
        g_full = torch.zeros(dim, device=dev, dtype=dty)
        reg_pu = self._schur_damp_pu(alpha)
        # Block layout: no ∂²E/(∂p∂u) coupling (``H_pu``) — by formulation.

        # θ: H + αI (C++ commented KKT L851); p/u: H + αγI (L852)
        H_full[:n, :n] = H_theta + alpha * self._eye_n
        g_full[:n] = g_theta

        all_populated = K > 0 and all(
            m.H_pp is not None and m.H_uu is not None and
            m.H_theta_p is not None and m.H_theta_u is not None and
            m.g_p is not None and m.g_u is not None
            for m in manifolds)

        if all_populated:
            H_pp_b = torch.stack([m.H_pp for m in manifolds])
            H_uu_b = torch.stack([m.H_uu for m in manifolds])
            H_tp_b = torch.stack([m.H_theta_p for m in manifolds])
            H_tu_b = torch.stack([m.H_theta_u for m in manifolds])
            g_p_b = torch.stack([m.g_p for m in manifolds])
            g_u_b = torch.stack([m.g_u for m in manifolds])

            # Off-diagonal θ–p and θ–u blocks
            H_full[:n, n:n+K*4] = H_tp_b.transpose(0, 1).reshape(n, K*4)
            H_full[n:n+K*4, :n] = H_full[:n, n:n+K*4].T
            H_full[:n, n+K*4:dim] = H_tu_b.transpose(0, 1).reshape(n, K*3)
            H_full[n+K*4:dim, :n] = H_full[:n, n+K*4:dim].T

            # Gradient vectors
            g_full[n:n+K*4] = g_p_b.reshape(K*4)
            g_full[n+K*4:dim] = g_u_b.reshape(K*3)

            # Block-diagonal p blocks [K, 4, 4]
            ki_idx = torch.arange(K, device=dev, dtype=torch.long)
            r4 = torch.arange(4, device=dev, dtype=torch.long)
            H_pp_reg = H_pp_b + reg_pu * self._eye4.unsqueeze(0)
            rows_p = (n + ki_idx.view(K, 1, 1) * 4
                      + r4.view(1, 4, 1)).expand(K, 4, 4)
            cols_p = (n + ki_idx.view(K, 1, 1) * 4
                      + r4.view(1, 1, 4)).expand(K, 4, 4)
            H_full[rows_p.reshape(-1), cols_p.reshape(-1)] = H_pp_reg.reshape(-1)

            # Block-diagonal u blocks [K, 3, 3]
            r3 = torch.arange(3, device=dev, dtype=torch.long)
            H_uu_reg = H_uu_b + reg_pu * self._eye3.unsqueeze(0)
            base_u = n + K * 4
            rows_u = (base_u + ki_idx.view(K, 1, 1) * 3
                      + r3.view(1, 3, 1)).expand(K, 3, 3)
            cols_u = (base_u + ki_idx.view(K, 1, 1) * 3
                      + r3.view(1, 1, 3)).expand(K, 3, 3)
            H_full[rows_u.reshape(-1), cols_u.reshape(-1)] = H_uu_reg.reshape(-1)
        elif K > 0:
            for ki in range(K):
                m = manifolds[ki]
                pi = n + ki * 4
                ui = n + K * 4 + ki * 3
                if m.H_pp is not None:
                    H_full[pi:pi+4, pi:pi+4] = m.H_pp + reg_pu * self._eye4
                if m.H_uu is not None:
                    H_full[ui:ui+3, ui:ui+3] = m.H_uu + reg_pu * self._eye3
                if m.H_theta_p is not None:
                    H_full[:n, pi:pi+4] = m.H_theta_p
                    H_full[pi:pi+4, :n] = m.H_theta_p.T
                if m.H_theta_u is not None:
                    H_full[:n, ui:ui+3] = m.H_theta_u
                    H_full[ui:ui+3, :n] = m.H_theta_u.T
                if m.g_p is not None:
                    g_full[pi:pi+4] = m.g_p
                if m.g_u is not None:
                    g_full[ui:ui+3] = m.g_u

        H_full = 0.5 * (H_full + H_full.T)
        return H_full, g_full

    # d²E/(dθ dθ_t) friction contribution
    def _friction_cross_theta_t(self, wv_star, wv_curr,
                                J_star, J_t, manifolds, p_normals,
                                u_tangents=None):
        """Compute friction contribution to d²E/(dθ dθ_t).  [n, n]"""
        dev, dty = self.device, self.dtype
        n = self.n_dof
        mM = self._max_M
        dt = self.dt
        vmask = self._vert_mask_float
        K = len(manifolds)
        if K == 0 or not self._use_friction:
            return torch.zeros(n, n, device=dev, dtype=dty)

        coef = self.coef_barrier
        pn_stack = torch.stack(p_normals)
        u_stack = torch.stack(u_tangents) if u_tangents is not None \
            else torch.stack([m.u.detach() for m in manifolds])
        lid = torch.tensor([m.link_a for m in manifolds],
                           device=dev, dtype=torch.long)
        lid_b = torch.tensor([m.link_b for m in manifolds],
                             device=dev, dtype=torch.long)
        is_ground = (lid_b < 0)
        lnk_idx = (~is_ground).nonzero(as_tuple=True)[0]

        vmask_a = vmask[lid]
        va_next = wv_star[lid]
        va_curr_k = wv_curr[lid]
        ones_K1 = torch.ones(K, mM, 1, device=dev, dtype=dty)

        _eps_n = self._friction_eps_n
        _eps_s = self._friction_eps_s
        n_vecs = pn_stack[:, :3]
        norm_n = torch.norm(n_vecs, dim=1, keepdim=True) + _eps_n
        n_hat = n_vecs / norm_n
        Proj = self._eye3.unsqueeze(0) - n_hat.unsqueeze(2) * n_hat.unsqueeze(1)
        vel_a = (va_next - va_curr_k) / dt
        lid_b_lnk = None
        if lnk_idx.numel() > 0:
            lid_b_lnk = lid_b[lnk_idx]
            vel = vel_a.clone()
            vel[lnk_idx] = (vel[lnk_idx]
                            - (wv_star[lid_b_lnk] - wv_curr[lid_b_lnk]) / dt)
        else:
            vel = vel_a
        tan_vel = torch.einsum('kij,kmj->kmi', Proj, vel)
        u_xyz_fc, omega_stack_fc, _, _ = self._unified_u_to_xyz_omega(
            u_stack, n_hat)
        r_nx_fc = torch.cross(n_hat.unsqueeze(1), va_curr_k, dim=2)
        rel_vel = (tan_vel - u_xyz_fc.unsqueeze(1)
                   - omega_stack_fc.view(K, 1, 1) * r_nx_fc)

        va_h_fn = torch.cat([va_curr_k, ones_K1], 2)
        d_fn = -torch.einsum('kmi,ki->km', va_h_fn, pn_stack)
        _, bg_fn, _ = barrier_eval(d_fn, self.x0, self._barrier_d0_half)
        pn3_fc = pn_stack[:, :3]
        f_vec_fc = coef * bg_fn.unsqueeze(2) * pn3_fc.unsqueeze(1)
        fn_sq_fc = f_vec_fc.pow(2).sum(dim=2)
        A_m_fc = torch.sqrt(fn_sq_fc + _eps_s) - self._sqrt_friction_eps_s

        s_norm = torch.sqrt(rel_vel.pow(2).sum(2) + _eps_s)
        inv_s = 1.0 / (s_norm + 1e-30)
        inv_s3 = inv_s.pow(3)

        weight_fric = self.friction / dt * A_m_fc * vmask_a
        PMP = (Proj.unsqueeze(1) * inv_s.unsqueeze(2).unsqueeze(3)
               - rel_vel.unsqueeze(-1) * rel_vel.unsqueeze(-2)
                 * inv_s3.unsqueeze(2).unsqueeze(3))
        H_fric = weight_fric.unsqueeze(2).unsqueeze(3) * PMP

        def _jthj_sum(Jl, Hm, Jr):
            """``Σ_{k,m} Jl[k,m]^T @ Hm[k,m] @ Jr[k,m]`` via bmm+mm."""
            Bf = Jl.shape[0] * Jl.shape[1]
            Jl2 = Jl.reshape(Bf, 3, n)
            Hm2 = Hm.reshape(Bf, 3, 3)
            Jr2 = Jr.reshape(Bf, 3, n)
            tmp = torch.bmm(Jl2.transpose(1, 2), Hm2)
            return torch.mm(
                tmp.permute(0, 2, 1).reshape(Bf * 3, n).T,
                Jr2.reshape(Bf * 3, n))

        J_star_a = J_star[lid]
        J_t_a = J_t[lid]
        Hx_fric = -_jthj_sum(J_star_a, H_fric, J_t_a)

        if lnk_idx.numel() > 0:
            J_star_b = J_star[lid_b_lnk]
            J_t_b = J_t[lid_b_lnk]
            Hf_ll = H_fric[lnk_idx]
            Js_a_ll = J_star_a[lnk_idx]
            Jt_a_ll = J_t_a[lnk_idx]
            Hx_fric -= _jthj_sum(J_star_b, Hf_ll, J_t_b)
            Hx_fric += _jthj_sum(Js_a_ll, Hf_ll, J_t_b)
            Hx_fric += _jthj_sum(J_star_b, Hf_ll, Jt_a_ll)
        return Hx_fric

    # dg_u/dθ_t — friction velocity sensitivity for Schur correction
    def _compute_dgu_dtheta_t(self, wv_star, wv_curr,
                              J_t, manifolds, p_normals,
                              u_tangents=None):
        """Compute dg_u/dtheta_t.  Returns [K, 3, n] or None if K=0."""
        dev, dty = self.device, self.dtype
        n = self.n_dof
        mM = self._max_M
        dt = self.dt
        coef = self.coef_barrier
        vmask = self._vert_mask_float
        K = len(manifolds)
        if K == 0 or not self._use_friction:
            return None

        pn_stack = torch.stack(p_normals)
        u_stack = torch.stack(u_tangents) if u_tangents is not None \
            else torch.stack([m.u.detach() for m in manifolds])
        lid = torch.tensor([m.link_a for m in manifolds],
                           device=dev, dtype=torch.long)
        lid_b = torch.tensor([m.link_b for m in manifolds],
                             device=dev, dtype=torch.long)
        is_ground = (lid_b < 0)
        gnd_idx = is_ground.nonzero(as_tuple=True)[0]
        lnk_idx = (~is_ground).nonzero(as_tuple=True)[0]

        vmask_a = vmask[lid]
        va_next = wv_star[lid]
        va_curr_k = wv_curr[lid]
        ones_K1 = torch.ones(K, mM, 1, device=dev, dtype=dty)

        _eps_n = self._friction_eps_n
        _eps_s = self._friction_eps_s
        n_vecs = pn_stack[:, :3]
        norm_n = torch.norm(n_vecs, dim=1, keepdim=True) + _eps_n
        n_hat = n_vecs / norm_n
        Proj = self._eye3.unsqueeze(0) - n_hat.unsqueeze(2) * n_hat.unsqueeze(1)
        vel_a = (va_next - va_curr_k) / dt
        vel = vel_a.clone()
        lid_b_lnk = None
        if lnk_idx.numel() > 0:
            lid_b_lnk = lid_b[lnk_idx]
            vel[lnk_idx] = (vel[lnk_idx]
                            - (wv_star[lid_b_lnk] - wv_curr[lid_b_lnk]) / dt)
        tan_vel = torch.einsum('kij,kmj->kmi', Proj, vel)
        u_xyz_dgu, omega_stack_dgu, t0_dgu, t1_dgu = (
            self._unified_u_to_xyz_omega(u_stack, n_hat))
        r_nx_dgu = torch.cross(n_hat.unsqueeze(1), va_curr_k, dim=2)
        rel_vel = (tan_vel - u_xyz_dgu.unsqueeze(1)
                   - omega_stack_dgu.view(K, 1, 1) * r_nx_dgu)

        va_h_fn = torch.cat([va_curr_k, ones_K1], 2)
        d_fn = -torch.einsum('kmi,ki->km', va_h_fn, pn_stack)
        _, bg_fn, _ = barrier_eval(d_fn, self.x0, self._barrier_d0_half)
        pn3_dgu = pn_stack[:, :3]
        f_vec_dgu = coef * bg_fn.unsqueeze(2) * pn3_dgu.unsqueeze(1)
        fn_sq_dgu = f_vec_dgu.pow(2).sum(dim=2)
        A_m_dgu = torch.sqrt(fn_sq_dgu + _eps_s) - self._sqrt_friction_eps_s

        s_norm = torch.sqrt(rel_vel.pow(2).sum(2) + _eps_s)
        inv_s = 1.0 / (s_norm + 1e-30)
        inv_s3 = inv_s.pow(3)

        MP = (Proj.unsqueeze(1) * inv_s.unsqueeze(2).unsqueeze(3)
              - rel_vel.unsqueeze(-1) * rel_vel.unsqueeze(-2)
                * inv_s3.unsqueeze(2).unsqueeze(3))

        J_t_a = J_t[lid]
        J_t_eff = J_t_a.clone()
        if lnk_idx.numel() > 0:
            J_t_eff[lnk_idx] = J_t_eff[lnk_idx] - J_t[lid_b_lnk]

        c_vel = self.friction * A_m_dgu * vmask_a
        c_fm = self.friction * dt * A_m_dgu * vmask_a
        dgu_3 = torch.einsum('km,kmcd,kmdj->kcj', c_vel, MP, J_t_eff)

        h_dgu = (rel_vel * r_nx_dgu).sum(dim=2)
        omv = omega_stack_dgu.view(K, 1, 1)
        grad_h = (-r_nx_dgu / dt
                  + torch.cross(rel_vel, n_hat.unsqueeze(1), dim=2)
                  + omv * torch.cross(n_hat.unsqueeze(1), r_nx_dgu, dim=2))
        rel_over_s_dgu = rel_vel / s_norm.unsqueeze(2)
        grad_s_va = (-rel_over_s_dgu / dt
                      + omv * torch.cross(
                          n_hat.unsqueeze(1), rel_over_s_dgu, dim=2))
        grad_h_over_s = (grad_h / s_norm.unsqueeze(2)
                         - (h_dgu / s_norm.pow(2)).unsqueeze(2) * grad_s_va)
        d_gom_dva = -c_fm.unsqueeze(2) * grad_h_over_s

        rnx_over_dt = r_nx_dgu / dt
        grad_vb_h = rnx_over_dt
        grad_vb_s = rel_over_s_dgu / dt
        grad_h_over_s_vb = (grad_vb_h / s_norm.unsqueeze(2)
                            - (h_dgu / s_norm.pow(2)).unsqueeze(2)
                            * grad_vb_s)
        d_gom_dvb = -c_fm.unsqueeze(2) * grad_h_over_s_vb
        if gnd_idx.numel() > 0:
            d_gom_dvb[gnd_idx] = 0.0

        dgu_om = torch.einsum('kmc,kmcj->kj', d_gom_dva, J_t_a)
        if lnk_idx.numel() > 0:
            dgu_om[lnk_idx] = (dgu_om[lnk_idx]
                               + torch.einsum('kmc,kmcj->kj',
                                                d_gom_dvb[lnk_idx],
                                                J_t[lid_b_lnk]))

        dgu_4 = torch.zeros(K, 4, n, device=dev, dtype=dty)
        dgu_4[:, :3, :] = dgu_3
        dgu_4[:, 3, :] = dgu_om
        J_dgu = self._friction_J_lift(t0_dgu, t1_dgu)
        return self._friction_reduce_dgu(dgu_4, J_dgu)

    # IFT BACKWARD: dθ*/d(θ_t, θ_{t-1}, pd, d)
    def _compute_backward(self, theta_star, theta_t, theta_tm1, pd_target,
                          kp, kd, manifolds, H_bar,
                          wv_curr=None, wv_last=None, p_normals=None,
                          u_tangents=None,
                          friction_plane_snap=None):
        n = self.n_dof
        dev, dty = self.device, self.dtype
        L, mM = self._L, self._max_M
        K = len(manifolds)

        if K == 0 and H_bar is None:
            I = self._eye_n
            z_dd = torch.zeros(n, L * mM * 3, device=dev, dtype=dty)
            return GradInfo(I, torch.zeros_like(I), torch.zeros_like(I),
                            I, z_dd, I.clone())

        if wv_curr is None:
            wv_curr = self._get_wv_stacked(theta_t.detach())
        if wv_last is None:
            wv_last = self._get_wv_stacked(theta_tm1.detach())
        if p_normals is None:
            p_normals = [m.p.detach().clone() for m in manifolds]
        if u_tangents is None:
            u_tangents = [m.u.detach().clone() for m in manifolds]
        if friction_plane_snap is not None:
            pn_fric = self._friction_plane_list_from_snap(
                manifolds, friction_plane_snap)
        else:
            pn_fric = p_normals

        (dg_dt, dg_dtm1, dg_dpd,
         wv_star, J_star, wv_t, J_t, wv_tm1, J_tm1
         ) = self._dg_cross_blocks_for_ift(
            theta_star, theta_t, theta_tm1,
            manifolds, wv_curr, pn_fric, u_tangents, kp, kd)

        dg_dd = self._compute_HThetaD(
            theta_star, theta_t, theta_tm1,
            wv_star, wv_t, wv_tm1,
            J_star, pd_target, kp, kd, manifolds,
            u_tangents=u_tangents)
        dg_dd = dg_dd.reshape(n, L * mM * 3)

        if self._implicit:
            gi_n = self._backward_normal(
                dg_dt, dg_dtm1, dg_dpd, dg_dd,
                manifolds, J_t, J_star, wv_star, wv_curr, pn_fric,
                theta_star, theta_t, pd_target,
                kp, kd, wv_last=wv_last,
                u_tangents=u_tangents)
            H_stored = H_bar if H_bar is not None else gi_n.H_bar
            return GradInfo(
                gi_n.dtheta_dtheta_t, gi_n.dtheta_dtheta_tm1, gi_n.dtheta_dpd,
                H_stored, gi_n.dtheta_dd, gi_n.H_theta_O)
        else:
            return self._backward_explicit(
                dg_dt, dg_dtm1, dg_dpd, dg_dd, H_bar,
                J_star, wv_star, theta_star, theta_t,
                wv_curr, wv_last,
                manifolds, pn_fric, pd_target, kp, kd,
                u_tangents=u_tangents)

    def _backward_normal(self, dg_dt, dg_dtm1, dg_dpd, dg_dd,
                         manifolds, J_t, J_star, wv_star, wv_curr, p_normals,
                         theta_star, theta_t, pd_target,
                         kp, kd, wv_last=None,
                         u_tangents=None):
        """IFT backward via full (n+7K) system solve."""
        dev, dty = self.device, self.dtype
        n = self.n_dof
        L, mM = self._L, self._max_M
        K = len(manifolds)
        dim = n + K * 4 + K * 3

        H_theta = self._assemble_H_theta(
            J_star, theta_star, wv_star, wv_curr, wv_last,
            manifolds, p_normals, pd_target, theta_t,
            kp, kd, u_tangents=u_tangents)

        g_dummy = torch.zeros(n, device=dev, dtype=dty)
        H_full, _ = self._assemble_full_system(
            g_dummy, H_theta, manifolds, 0.0)

        H_bar_theta = H_full[:n, :n].clone()

        H_full = H_full + 1e-6 * torch.eye(dim, device=dev, dtype=dty)
        H_full = 0.5 * (H_full + H_full.T)

        dgu_dt = self._compute_dgu_dtheta_t(
            wv_star, wv_curr, J_t, manifolds, p_normals,
            u_tangents=u_tangents)

        n_rhs = n + n + n + L * mM * 3
        rhs_full = torch.zeros(dim, n_rhs, device=dev, dtype=dty)
        rhs_full[:n, :n] = dg_dt
        rhs_full[:n, n:2*n] = dg_dtm1
        rhs_full[:n, 2*n:3*n] = dg_dpd
        rhs_full[:n, 3*n:] = dg_dd

        if K > 0 and dgu_dt is not None:
            rhs_full[n + K*4 : n + K*4 + K*3, :n] = dgu_dt.reshape(K*3, n)

        try:
            sol = -torch.linalg.solve(H_full, rhs_full)
            dtheta_dt = sol[:n, :n]
            dtheta_dtm1 = sol[:n, n:2*n]
            dtheta_dpd = sol[:n, 2*n:3*n]
            dtheta_dd = sol[:n, 3*n:]
        except Exception:
            I = self._eye_n
            dtheta_dt = I.clone()
            dtheta_dtm1 = torch.zeros_like(I)
            dtheta_dpd = torch.zeros_like(I)
            dtheta_dd = torch.zeros(n, L * mM * 3, device=dev, dtype=dty)

        return GradInfo(dtheta_dt, dtheta_dtm1, dtheta_dpd,
                        H_bar_theta, dtheta_dd, H_theta)

    def _backward_explicit(self, dg_dt, dg_dtm1, dg_dpd, dg_dd, H_bar,
                           J_star, wv_star, theta_star, theta_t,
                           wv_curr, wv_last,
                           manifolds, p_normals, pd_target, kp, kd,
                           u_tangents=None):
        """IFT backward with fixed p/u: ``dtheta*/dx = -(H_theta)^{-1} @ (dg_theta/dx)``."""
        dev, dty = self.device, self.dtype
        n = self.n_dof
        L, mM = self._L, self._max_M

        H_theta = self._assemble_H_theta(
            J_star, theta_star, wv_star, wv_curr, wv_last,
            manifolds, p_normals, pd_target, theta_t,
            kp, kd, u_tangents=u_tangents)

        H_reg = H_theta + 1e-6 * self._eye_n
        H_reg = 0.5 * (H_reg + H_reg.T)

        n_rhs = n + n + n + L * mM * 3
        rhs = torch.zeros(n, n_rhs, device=dev, dtype=dty)
        rhs[:, :n] = dg_dt
        rhs[:, n:2*n] = dg_dtm1
        rhs[:, 2*n:3*n] = dg_dpd
        rhs[:, 3*n:] = dg_dd

        try:
            sol = -torch.linalg.solve(H_reg, rhs)
            dtheta_dt = sol[:, :n]
            dtheta_dtm1 = sol[:, n:2*n]
            dtheta_dpd = sol[:, 2*n:3*n]
            dtheta_dd = sol[:, 3*n:]
        except Exception:
            I = self._eye_n
            dtheta_dt = I.clone()
            dtheta_dtm1 = torch.zeros_like(I)
            dtheta_dpd = torch.zeros_like(I)
            dtheta_dd = torch.zeros(n, L * mM * 3, device=dev, dtype=dty)

        H_stored = H_bar if H_bar is not None else H_theta
        return GradInfo(dtheta_dt, dtheta_dtm1, dtheta_dpd,
                        H_stored, dtheta_dd, H_theta)

    # HThetaD: ∂²E/(∂θ·∂d) vertex design cross-Hessian
    def _compute_HThetaD(self, theta_star, theta_t, theta_tm1,
                         wv_star, wv_t, wv_tm1,
                         J_star, pd_target, kp, kd, manifolds,
                         u_tangents=None):
        dev, dty = self.device, self.dtype
        n = self.n_dof
        L, mM = self._L, self._max_M
        dt = self.dt
        coef = self.coef_barrier
        x0 = self.x0
        vmask = self._vert_mask_float
        w_rho = self._rho_2d * vmask           # [L, M]
        K = len(manifolds)

        with torch.no_grad():
            T_star = torch.stack(self.robot.forward_kinematics(theta_star.detach()))
            T_t = torch.stack(self.robot.forward_kinematics(theta_t.detach()))
            T_tm1 = torch.stack(self.robot.forward_kinematics(theta_tm1.detach()))
        R_star = T_star[:, :3, :3]
        R_t    = T_t[:, :3, :3]
        R_tm1  = T_tm1[:, :3, :3]
        deltaR = R_star - 2.0 * R_t + R_tm1    # [L, 3, 3]

        # ==================  A) Inertial direct  ==================
        dg_v_dd_inertial = (1.0 / (dt * dt)) * deltaR
        HThetaD = torch.einsum(
            'lmci, lm, lcj -> ilmj', J_star, w_rho, dg_v_dd_inertial)

        # ==================  B) Contact barrier direct  ==================
        if K > 0:
            p_stack = torch.stack([m.p.detach() for m in manifolds])
            p3 = p_stack[:, :3]
            lid_a = torch.tensor([m.link_a for m in manifolds],
                                 device=dev, dtype=torch.long)
            lid_b = torch.tensor([m.link_b for m in manifolds],
                                 device=dev, dtype=torch.long)
            is_ground = (lid_b < 0)
            lnk_idx = (~is_ground).nonzero(as_tuple=True)[0]

            va_h = torch.cat([wv_star[lid_a],
                              torch.ones(K, mM, 1, device=dev, dtype=dty)], 2)
            d_a = -torch.einsum('kmi, ki -> km', va_h, p_stack)
            _, bg_a, bh_a = barrier_eval(d_a, x0, self._barrier_d0_half)
            mask_a = vmask[lid_a]
            R_A = R_star[lid_a]
            Rtp3 = torch.einsum('kcj, kc -> kj', R_A, p3)
            w_contact_A = coef * bh_a * mask_a
            J_A = J_star[lid_a]
            Jp3 = torch.einsum('kmci, kc -> kmi', J_A, p3)
            contrib_A = torch.einsum(
                'km, kmi, kj -> ikmj', w_contact_A, Jp3, Rtp3)
            HThetaD.index_add_(1, lid_a, contrib_A)

            if lnk_idx.numel() > 0:
                lid_b_lnk = lid_b[lnk_idx]
                p3_lnk = p3[lnk_idx]
                vb_h = torch.cat([wv_star[lid_b_lnk],
                                  torch.ones(lnk_idx.numel(), mM, 1,
                                             device=dev, dtype=dty)], 2)
                d_b = torch.einsum('kmi, ki -> km', vb_h,
                                   p_stack[lnk_idx])
                _, bg_b, bh_b = barrier_eval(d_b, x0, self._barrier_d0_half)
                mask_b = vmask[lid_b_lnk]
                R_B = R_star[lid_b_lnk]
                Rtp3_b = torch.einsum('kcj, kc -> kj', R_B, p3_lnk)
                w_contact_B = coef * bh_b * mask_b
                J_B = J_star[lid_b_lnk]
                Jp3_b = torch.einsum('kmci, kc -> kmi', J_B, p3_lnk)
                contrib_B = torch.einsum(
                    'km, kmi, kj -> ikmj', w_contact_B, Jp3_b, Rtp3_b)
                HThetaD.index_add_(1, lid_b_lnk, contrib_B)

        # ==================  C) Friction velocity direct  ==================
        if K > 0 and self._use_friction:
            pn_stack_f = torch.stack([m.p.detach().clone() for m in manifolds])
            u_stack_f = (torch.stack(u_tangents) if u_tangents is not None
                         else torch.stack([m.u.detach() for m in manifolds]))
            lid_a_f = torch.tensor([m.link_a for m in manifolds],
                                   device=dev, dtype=torch.long)
            lid_b_f = torch.tensor([m.link_b for m in manifolds],
                                   device=dev, dtype=torch.long)
            is_ground_f = (lid_b_f < 0)
            lnk_idx_f = (~is_ground_f).nonzero(as_tuple=True)[0]

            vmask_a_f = vmask[lid_a_f]
            va_next_f = wv_star[lid_a_f]
            va_curr_f = wv_t[lid_a_f]
            ones_K1_f = torch.ones(K, mM, 1, device=dev, dtype=dty)

            _eps_n = self._friction_eps_n
            _eps_s = self._friction_eps_s
            n_vecs_f = pn_stack_f[:, :3]
            norm_n_f = torch.norm(n_vecs_f, dim=1, keepdim=True) + _eps_n
            n_hat_f = n_vecs_f / norm_n_f
            Proj_f = (self._eye3.unsqueeze(0)
                      - n_hat_f.unsqueeze(2) * n_hat_f.unsqueeze(1))
            vel_a_f = (va_next_f - va_curr_f) / dt
            vel_f = vel_a_f.clone()
            lid_b_lnk_f = None
            if lnk_idx_f.numel() > 0:
                lid_b_lnk_f = lid_b_f[lnk_idx_f]
                vel_f[lnk_idx_f] = (vel_f[lnk_idx_f]
                                    - (wv_star[lid_b_lnk_f]
                                       - wv_t[lid_b_lnk_f]) / dt)
            tan_vel_f = torch.einsum('kij,kmj->kmi', Proj_f, vel_f)
            u_xyz_f, omega_stack_f, _, _ = self._unified_u_to_xyz_omega(
                u_stack_f, n_hat_f)
            r_nx_f = torch.cross(n_hat_f.unsqueeze(1), va_curr_f, dim=2)
            rel_vel_f = (tan_vel_f - u_xyz_f.unsqueeze(1)
                         - omega_stack_f.view(K, 1, 1) * r_nx_f)

            va_h_fn_f = torch.cat([va_curr_f, ones_K1_f], 2)
            d_fn_f = -torch.einsum('kmi,ki->km', va_h_fn_f, pn_stack_f)
            _, bg_fn_f, _ = barrier_eval(d_fn_f, x0, self._barrier_d0_half)
            pn3_f = pn_stack_f[:, :3]
            f_vec_f = coef * bg_fn_f.unsqueeze(2) * pn3_f.unsqueeze(1)
            fn_sq_f = f_vec_f.pow(2).sum(dim=2)
            A_m_f = torch.sqrt(fn_sq_f + _eps_s) - self._sqrt_friction_eps_s

            s_norm_f = torch.sqrt(rel_vel_f.pow(2).sum(2) + _eps_s)
            inv_s_f = 1.0 / (s_norm_f + 1e-30)
            inv_s3_f = inv_s_f.pow(3)

            weight_f = self.friction / dt * A_m_f * vmask_a_f
            PMP_f = (Proj_f.unsqueeze(1) * inv_s_f.unsqueeze(2).unsqueeze(3)
                     - rel_vel_f.unsqueeze(-1) * rel_vel_f.unsqueeze(-2)
                       * inv_s3_f.unsqueeze(2).unsqueeze(3))
            H_fric_f = weight_f.unsqueeze(2).unsqueeze(3) * PMP_f

            dR_vel_A = R_star[lid_a_f] - R_t[lid_a_f]

            J_star_a_f = J_star[lid_a_f]
            contrib_fAA = torch.einsum(
                'kmci,kmcd,kdj->ikmj', J_star_a_f, H_fric_f, dR_vel_A)
            HThetaD.index_add_(1, lid_a_f, contrib_fAA)

            if lnk_idx_f.numel() > 0:
                dR_vel_B = R_star[lid_b_lnk_f] - R_t[lid_b_lnk_f]
                Hf_ll = H_fric_f[lnk_idx_f]
                dR_vel_A_ll = dR_vel_A[lnk_idx_f]
                Js_a_ll = J_star_a_f[lnk_idx_f]
                Js_b = J_star[lid_b_lnk_f]

                contrib_fBA = -torch.einsum(
                    'kmci,kmcd,kdj->ikmj', Js_b, Hf_ll, dR_vel_A_ll)
                HThetaD.index_add_(1, lid_a_f[lnk_idx_f], contrib_fBA)

                contrib_fAB = -torch.einsum(
                    'kmci,kmcd,kdj->ikmj', Js_a_ll, Hf_ll, dR_vel_B)
                HThetaD.index_add_(1, lid_b_lnk_f, contrib_fAB)

                contrib_fBB = torch.einsum(
                    'kmci,kmcd,kdj->ikmj', Js_b, Hf_ll, dR_vel_B)
                HThetaD.index_add_(1, lid_b_lnk_f, contrib_fBB)

        # ==================  D) FK correction: (∂J/∂d)^T @ g_v  ==================
        g_v = self._compute_gv_at_solution(
            wv_star, wv_t, wv_tm1, manifolds, kp, kd,
            theta_star, pd_target, theta_t,
            u_tangents=u_tangents)
        fk_corr = self._compute_fk_correction_d(J_star, g_v, theta_star, wv_star)
        HThetaD = HThetaD + fk_corr

        return HThetaD

    # g_v at converged solution (for FK correction in _compute_HThetaD)
    def _compute_gv_at_solution(self, wv_star, wv_t, wv_tm1,
                                manifolds, kp, kd,
                                theta_star, pd_target, theta_t,
                                u_tangents=None):
        dev, dty = self.device, self.dtype
        L, mM = self._L, self._max_M
        dt = self.dt
        coef = self.coef_barrier
        x0 = self.x0
        vmask = self._vert_mask_float
        K = len(manifolds)

        accel = wv_star - 2.0 * wv_t + wv_tm1
        g_v = ((1.0 / (dt * dt)) * (self._rho_2d * vmask).unsqueeze(2)
               * accel)
        g_v[:, :, 1] = g_v[:, :, 1] - self._rho_2d * self.gravity * vmask

        if K == 0:
            return g_v

        p_stack = torch.stack([m.p.detach() for m in manifolds])
        pn_stack = torch.stack([m.p.detach().clone() for m in manifolds])
        u_stack = (torch.stack(u_tangents) if u_tangents is not None
                   else torch.stack([m.u.detach() for m in manifolds]))
        lid_a = torch.tensor([m.link_a for m in manifolds],
                             device=dev, dtype=torch.long)
        lid_b = torch.tensor([m.link_b for m in manifolds],
                             device=dev, dtype=torch.long)
        is_ground = (lid_b < 0)
        lnk_idx = (~is_ground).nonzero(as_tuple=True)[0]
        p3 = p_stack[:, :3]
        ones_K1 = torch.ones(K, mM, 1, device=dev, dtype=dty)

        va_h = torch.cat([wv_star[lid_a], ones_K1], 2)
        d_a = -torch.einsum('kmi,ki->km', va_h, p_stack)
        _, bg_a, _ = barrier_eval(d_a, x0, self._barrier_d0_half)
        bg_m = bg_a * vmask[lid_a]
        g_v.index_add_(0, lid_a, -coef * bg_m.unsqueeze(2) * p3.unsqueeze(1))

        if lnk_idx.numel() > 0:
            lid_b_lnk = lid_b[lnk_idx]
            mask_b_lnk = vmask[lid_b_lnk]
            vb_next_lnk = wv_star[lid_b_lnk]
            ones_lnk = torch.ones(lnk_idx.numel(), mM, 1, device=dev, dtype=dty)
            vb_h_lnk = torch.cat([vb_next_lnk, ones_lnk], 2)
            d_b_lnk = torch.einsum('kmi,ki->km', vb_h_lnk, p_stack[lnk_idx])
            _, bg_b_lnk, _ = barrier_eval(d_b_lnk, x0, self._barrier_d0_half)
            bg_b_lnk_m = bg_b_lnk * mask_b_lnk
            p3_lnk = p3[lnk_idx]
            g_v.index_add_(0, lid_b_lnk,
                           coef * bg_b_lnk_m.unsqueeze(2) * p3_lnk.unsqueeze(1))

        if self._use_friction:
            _eps_n = self._friction_eps_n
            _eps_s = self._friction_eps_s
            n_vecs = pn_stack[:, :3]
            norm_n = torch.norm(n_vecs, dim=1, keepdim=True) + _eps_n
            n_hat = n_vecs / norm_n
            Proj = self._eye3.unsqueeze(0) - n_hat.unsqueeze(2) * n_hat.unsqueeze(1)
            va_next = wv_star[lid_a]
            va_curr = wv_t[lid_a]
            vel_a = (va_next - va_curr) / dt
            vel = vel_a.clone()
            if lnk_idx.numel() > 0:
                vb_next_f = wv_star[lid_b[lnk_idx]]
                vb_curr_f = wv_t[lid_b[lnk_idx]]
                vel[lnk_idx] = vel[lnk_idx] - (vb_next_f - vb_curr_f) / dt
            tan_vel = torch.einsum('kij,kmj->kmi', Proj, vel)
            u_xyz_gv, omega_stack_gv, _, _ = self._unified_u_to_xyz_omega(
                u_stack, n_hat)
            r_nx_gv = torch.cross(n_hat.unsqueeze(1), va_curr, dim=2)
            rel_vel = (tan_vel - u_xyz_gv.unsqueeze(1)
                       - omega_stack_gv.view(K, 1, 1) * r_nx_gv)

            va_h_fn = torch.cat([va_curr, ones_K1], 2)
            d_fn = -torch.einsum('kmi,ki->km', va_h_fn, pn_stack)
            _, bg_fn, _ = barrier_eval(d_fn, x0, self._barrier_d0_half)
            pn3_gv = pn_stack[:, :3]
            f_vec_gv = coef * bg_fn.unsqueeze(2) * pn3_gv.unsqueeze(1)
            fn_sq_gv = f_vec_gv.pow(2).sum(dim=2)
            A_m_gv = torch.sqrt(fn_sq_gv + _eps_s) - self._sqrt_friction_eps_s

            s_norm = torch.sqrt(rel_vel.pow(2).sum(2) + _eps_s)
            inv_s = 1.0 / (s_norm + 1e-30)
            w_gv = (A_m_gv * vmask[lid_a]).unsqueeze(2)
            rel_over_s = rel_vel * inv_s.unsqueeze(2)
            proj_rs = torch.einsum('kij,kmj->kmi', Proj, rel_over_s)
            g_fric_s = self.friction * w_gv * proj_rs
            g_v.index_add_(0, lid_a, g_fric_s)
            if lnk_idx.numel() > 0:
                g_v.index_add_(0, lid_b[lnk_idx], -g_fric_s[lnk_idx])

        return g_v

    # FK correction for vertex design derivatives: (∂J/∂d)^T @ g_v
    def _compute_fk_correction_d(self, J, g_v, theta, wv):
        dev, dty = self.device, self.dtype
        n = self.n_dof
        L, mM = self._L, self._max_M

        with torch.no_grad():
            transforms = self.robot.forward_kinematics(theta.detach())
            T_all = torch.stack(transforms)
            R_all = T_all[:, :3, :3]

        corr = torch.zeros(n, L, mM, 3, device=dev, dtype=dty)

        for ji, joint in enumerate(self.robot.joints):
            off = joint.dof_offset
            desc_links = self._descendants[ji]
            if not desc_links:
                continue
            desc = torch.tensor(desc_links, device=dev, dtype=torch.long)
            D = len(desc_links)
            parent = joint.parent_link
            R_par = (torch.eye(3, device=dev, dtype=dty)
                     if parent < 0 else T_all[parent, :3, :3])

            gv_desc = g_v[desc]
            R_desc = R_all[desc]

            if joint.jtype == 'hinge':
                a_norm = joint.axis / (joint.axis.norm() + 1e-12)
                a_w = R_par @ a_norm

                cross_gv_a = torch.linalg.cross(
                    gv_desc,
                    a_w.reshape(1, 1, 3).expand(D, mM, 3))
                fc = torch.einsum('dmc, dcj -> dmj', cross_gv_a, R_desc)
                corr[off, desc] = fc

            elif joint.jtype == 'free':
                roll  = theta[off + 3].detach()
                pitch = theta[off + 4].detach()
                yaw   = theta[off + 5].detach()
                cr, sr = torch.cos(roll),  torch.sin(roll)
                cp, sp = torch.cos(pitch), torch.sin(pitch)
                cy, sy = torch.cos(yaw),   torch.sin(yaw)
                z = torch.zeros_like(cr)

                dR_stack = torch.stack([
                    torch.stack([
                        torch.stack([z, cy*sp*cr+sy*sr, -cy*sp*sr+sy*cr]),
                        torch.stack([z, sy*sp*cr-cy*sr, -sy*sp*sr-cy*cr]),
                        torch.stack([z, cp*cr, -cp*sr])]),
                    torch.stack([
                        torch.stack([-cy*sp, cy*cp*sr, cy*cp*cr]),
                        torch.stack([-sy*sp, sy*cp*sr, sy*cp*cr]),
                        torch.stack([-cp, -sp*sr, -sp*cr])]),
                    torch.stack([
                        torch.stack([-sy*cp, -sy*sp*sr-cy*cr, -sy*sp*cr+cy*sr]),
                        torch.stack([cy*cp, cy*sp*sr-sy*cr, cy*sp*cr+sy*sr]),
                        torch.stack([z, z, z])])
                ])

                child = joint.child_link
                R_child_inv = T_all[child, :3, :3].T

                for r in range(3):
                    MR = R_par @ dR_stack[r] @ R_child_inv
                    MR_desc = torch.einsum('ij, djk -> dik', MR, R_desc)
                    fc = torch.einsum('dmc, dcj -> dmj', gv_desc, MR_desc)
                    corr[off + 3 + r, desc] = fc

        return corr

    # DEBUG_GRADIENT helper
    @staticmethod
    def _debug_gradient_cpp(name: str, a_val: float, err_val: float,
                            delta: float) -> None:
        th = math.sqrt(delta)
        red_on = "\033[1;31m"
        red_off = "\033[0m"
        if abs(err_val) > th:
            print(f"{red_on}{name}: {a_val} Err: {err_val}{red_off}")
        else:
            print(f"{name}: {a_val} Err: {err_val}")

    # DEBUG ENERGY — directional FD verification
    def debug_energy(
            self,
            scale: float = 0.1,
            custom_delta: Optional[float] = None,
            theta: Optional[torch.Tensor] = None,
            theta_t: Optional[torch.Tensor] = None,
            theta_tm1: Optional[torch.Tensor] = None,
            pd_target: Optional[torch.Tensor] = None,
            kp: float = 100.0,
            kd: float = 10.0,
            max_random_trials: int = 200,
            manifolds: Optional[List[ContactManifold]] = None,
            wv_curr_cache: Optional[torch.Tensor] = None,
            wv_last_cache: Optional[torch.Tensor] = None,
            compare_autograd: bool = True,
            normalize_dx: bool = True) -> None:
        """DE/DDE directional FD + optional autograd on _build_energy."""
        dev, dty = self.device, self.dtype
        n = self.n_dof
        L, mM = self._L, self._max_M
        delta = float(custom_delta) if custom_delta is not None else DEBUG_FD_EPS

        if (theta is None and theta_t is None and theta_tm1 is None
                and pd_target is None):
            ok = False
            for _ in range(max_random_trials):
                th0 = (2.0 * torch.rand(n, device=dev, dtype=dty) - 1.0) * scale
                tht = (2.0 * torch.rand(n, device=dev, dtype=dty) - 1.0) * scale
                thtm1 = (2.0 * torch.rand(n, device=dev, dtype=dty) - 1.0) * scale
                pd_ = torch.randn(n, device=dev, dtype=dty)
                if len(self._detect_contacts(th0)) == 0:
                    continue
                theta, theta_t, theta_tm1, pd_target = th0, tht, thtm1, pd_
                ok = True
                break
            if not ok:
                print("debug_energy: no contact in random trials; pass states.")
                return
        else:
            if None in (theta, theta_t, theta_tm1, pd_target):
                raise ValueError(
                    "debug_energy: pass all of theta, theta_t, theta_tm1, "
                    "pd_target, or none for random mode.")
            theta = theta.to(device=dev, dtype=dty)
            theta_t = theta_t.to(device=dev, dtype=dty)
            theta_tm1 = theta_tm1.to(device=dev, dtype=dty)
            pd_target = pd_target.to(device=dev, dtype=dty)

        dx = 2.0 * torch.rand(n, device=dev, dtype=dty) - 1.0
        if normalize_dx:
            nrm = dx.norm()
            if nrm < 1e-14:
                dx = torch.zeros(n, device=dev, dtype=dty)
                dx[0] = 1.0
            else:
                dx = dx / nrm

        if manifolds is not None and len(manifolds) > 0:
            mf_base = self._clone_manifolds(manifolds)
            for m in mf_base:
                key = (m.link_a, m.link_b) if m.link_b >= 0 else (m.link_a, -1)
                self._pair_manifolds[key] = (
                    m.p.detach().clone(), m.u.detach().clone())
        else:
            det = self._detect_contacts(theta)
            if len(det) == 0:
                print("debug_energy: no contacts at theta; abort.")
                return
            mf_base = self._clone_manifolds(det)

        pn0 = [m.p.detach().clone() for m in mf_base]
        un0 = [m.u.detach().clone() for m in mf_base]
        mf0 = self._clone_manifolds(mf_base)
        with torch.no_grad():
            if wv_curr_cache is not None and wv_last_cache is not None:
                wv_curr = wv_curr_cache
                wv_last = wv_last_cache
            else:
                # Must match JVP closures (they always FK(theta_t/tm1)); using
                # _last_step_wv_* broke DDE-L / DDE-XL vs autograd when states
                # differed from the last step().
                wv_curr = self._get_wv_stacked(theta_t)
                wv_last = self._get_wv_stacked(theta_tm1)

        # Friction planes frozen at θ_t (C++ manifoldsLast); contact uses m.p on mf0.
        fps_energy = self._friction_plane_snap_dict(self._detect_contacts(theta_t))
        pn_fric0 = self._friction_plane_list_from_snap(mf0, fps_energy)

        E0, g0, H0 = self._compute_energy(
            theta, theta_t, theta_tm1, pd_target, kp, kd,
            mf0, analytic_derivs=True, p_normals=pn0, u_tangents=un0,
            wv_curr_cache=wv_curr, wv_last_cache=wv_last,
            friction_plane_snap=fps_energy)

        # C++ debugEnergy: both energy() calls use the same _manifolds /
        # _manifoldsLast (frozen contact geometry). Re-detecting here makes
        # the FD probe a different energy than ∂E/∂θ and ∂²E/∂θ² assume.
        theta_p = theta + dx * delta
        mf_p = self._clone_manifolds(mf0)
        E_p, _, _ = self._compute_energy(
            theta_p, theta_t, theta_tm1, pd_target, kp, kd,
            mf_p, analytic_derivs=False, p_normals=pn0, u_tangents=un0,
            wv_curr_cache=wv_curr, wv_last_cache=wv_last,
            friction_plane_snap=fps_energy)
        ana_de = (g0 * dx).sum().item()
        fd_de = (E_p - E0) / delta
        self._debug_gradient_cpp("DE", ana_de, abs(ana_de - fd_de), delta)

        if compare_autograd:
            try:
                ag_t = self._directional_deriv_energy_autograd(
                    theta, theta_t, theta_tm1, pd_target, kp, kd,
                    mf0, pn_fric0, un0, dx,
                    wv_curr_cache=wv_curr, wv_last_cache=wv_last)
                ag_de = ag_t.item() if isinstance(ag_t, torch.Tensor) else float(ag_t)
                self._debug_gradient_cpp(
                    "DE-AD", ag_de, abs(ag_de - ana_de), delta)
                self._debug_gradient_cpp(
                    "DE-AD-FD", ag_de, abs(ag_de - fd_de), delta)
                print(
                    f"DE triple (dE·d̂, ‖d̂‖={'1' if normalize_dx else '‖dx‖'}):  "
                    f"ANA={ana_de:.12e}  FD={fd_de:.12e}  AD={ag_de:.12e}  "
                    f"|ANA-FD|={abs(ana_de - fd_de):.12e}  "
                    f"|ANA-AD|={abs(ana_de - ag_de):.12e}  "
                    f"|FD-AD|={abs(fd_de - ag_de):.12e}")
            except Exception as ex:
                print(f"DE-Autograd: skipped ({type(ex).__name__}: {ex})")

        theta_m = theta - dx * delta
        mf_m = self._clone_manifolds(mf_base)
        _, g_m, _ = self._compute_energy(
            theta_m, theta_t, theta_tm1, pd_target, kp, kd,
            mf_m, analytic_derivs=True, p_normals=pn0, u_tangents=un0,
            wv_curr_cache=wv_curr, wv_last_cache=wv_last,
            friction_plane_snap=fps_energy)
        mf_p_g = self._clone_manifolds(mf_base)
        _, g_p2, _ = self._compute_energy(
            theta_p, theta_t, theta_tm1, pd_target, kp, kd,
            mf_p_g, analytic_derivs=True, p_normals=pn0, u_tangents=un0,
            wv_curr_cache=wv_curr, wv_last_cache=wv_last,
            friction_plane_snap=fps_energy)
        Hdx = H0 @ dx
        fd_dg = (g_p2 - g_m) / (2.0 * delta)
        self._debug_gradient_cpp(
            "DDE", Hdx.norm().item(), (Hdx - fd_dg).norm().item(), delta)

        if compare_autograd:
            try:
                ag_h = self._hvp_energy_theta_dir_autograd(
                    theta, theta_t, theta_tm1, pd_target, kp, kd,
                    mf0, pn_fric0, un0, dx,
                    wv_curr_cache=wv_curr, wv_last_cache=wv_last)
                ah = ag_h.detach()
                self._debug_gradient_cpp(
                    "DDE-AD-ANA", ah.norm().item(), (Hdx - ah).norm().item(), delta)
                self._debug_gradient_cpp(
                    "DDE-AD-FD", ah.norm().item(), (fd_dg - ah).norm().item(), delta)
                print(
                    f"DDE triple (‖H·d̂‖):  ANA={Hdx.norm().item():.12e}  "
                    f"FD={fd_dg.norm().item():.12e}  AD={ah.norm().item():.12e}  "
                    f"|ANA-AD|={(Hdx - ah).norm().item():.12e}  "
                    f"|FD-AD|={(fd_dg - ah).norm().item():.12e}")
            except Exception as ex:
                print(f"DDE-Autograd: skipped ({type(ex).__name__}: {ex})")

        with torch.no_grad():
            wv_star, J_star, _, _ = self._compute_fk_jacobian_analytic(theta)
            _, J_t, _, _ = self._compute_fk_jacobian_analytic(theta_t)
            _, J_tm1, _, _ = self._compute_fk_jacobian_analytic(theta_tm1)
        dt = self.dt
        vmask = self._vert_mask_float
        w_rho = self._rho_2d * vmask
        H_L = ((-2.0 / (dt * dt)) * torch.einsum(
            'lmci,lm,lmcj->ij', J_star, w_rho, J_t)
               - kd * torch.diag(self.joint_mask ** 2))
        H_L = H_L + self._friction_cross_theta_t(
            wv_star, wv_curr, J_star, J_t, mf0, pn_fric0,
            u_tangents=un0)
        H_LL = (1.0 / (dt * dt)) * torch.einsum(
            'lmci,lm,lmcj->ij', J_star, w_rho, J_tm1)
        H_P = -kp * torch.diag(self.joint_mask ** 2)

        theta_t_p = theta_t + dx * delta
        theta_t_m = theta_t - dx * delta
        wv_ct_p = self._get_wv_stacked(theta_t_p)
        wv_ct_m = self._get_wv_stacked(theta_t_m)
        mf_Lp = self._clone_manifolds(mf_base)
        mf_Lm = self._clone_manifolds(mf_base)
        _, g_Lp, _ = self._compute_energy(
            theta, theta_t_p, theta_tm1, pd_target, kp, kd,
            mf_Lp, analytic_derivs=True, p_normals=pn0, u_tangents=un0,
            wv_curr_cache=wv_ct_p, wv_last_cache=wv_last,
            friction_plane_snap=fps_energy)
        _, g_Lm, _ = self._compute_energy(
            theta, theta_t_m, theta_tm1, pd_target, kp, kd,
            mf_Lm, analytic_derivs=True, p_normals=pn0, u_tangents=un0,
            wv_curr_cache=wv_ct_m, wv_last_cache=wv_last,
            friction_plane_snap=fps_energy)
        HLdx = H_L @ dx
        fd_dg_L = (g_Lp - g_Lm) / (2.0 * delta)
        self._debug_gradient_cpp(
            "DDE-L", HLdx.norm().item(),
            (HLdx - fd_dg_L).norm().item(), delta)

        if compare_autograd:
            try:
                ag_l = self._jvp_grad_theta_wrt_theta_t_autograd(
                    theta, theta_t, theta_tm1, pd_target, kp, kd,
                    mf0, pn_fric0, un0, dx, wv_last_cache=wv_last)
                al = ag_l.detach()
                self._debug_gradient_cpp(
                    "DDE-L-AD-ANA", al.norm().item(), (HLdx - al).norm().item(), delta)
                self._debug_gradient_cpp(
                    "DDE-L-AD-FD", al.norm().item(), (fd_dg_L - al).norm().item(), delta)
                print(
                    f"DDE-L triple (central FD):  ANA={HLdx.norm().item():.12e}  "
                    f"FD={fd_dg_L.norm().item():.12e}  AD={al.norm().item():.12e}  "
                    f"|ANA-AD|={(HLdx - al).norm().item():.12e}  "
                    f"|FD-AD|={(fd_dg_L - al).norm().item():.12e}")
            except Exception as ex:
                print(f"DDE-L-Autograd: skipped ({type(ex).__name__}: {ex})")

        theta_tm1_p = theta_tm1 + dx * delta
        theta_tm1_m = theta_tm1 - dx * delta
        wv_lp_p = self._get_wv_stacked(theta_tm1_p)
        wv_lp_m = self._get_wv_stacked(theta_tm1_m)
        mf_LL_p = self._clone_manifolds(mf_base)
        mf_LL_m = self._clone_manifolds(mf_base)
        _, g_LL_p, _ = self._compute_energy(
            theta, theta_t, theta_tm1_p, pd_target, kp, kd,
            mf_LL_p, analytic_derivs=True, p_normals=pn0, u_tangents=un0,
            wv_curr_cache=wv_curr, wv_last_cache=wv_lp_p,
            friction_plane_snap=fps_energy)
        _, g_LL_m, _ = self._compute_energy(
            theta, theta_t, theta_tm1_m, pd_target, kp, kd,
            mf_LL_m, analytic_derivs=True, p_normals=pn0, u_tangents=un0,
            wv_curr_cache=wv_curr, wv_last_cache=wv_lp_m,
            friction_plane_snap=fps_energy)
        HLLdx = H_LL @ dx
        fd_dg_LL = (g_LL_p - g_LL_m) / (2.0 * delta)
        self._debug_gradient_cpp(
            "DDE-LL", HLLdx.norm().item(),
            (HLLdx - fd_dg_LL).norm().item(), delta)

        if compare_autograd:
            try:
                ag_ll = self._jvp_grad_theta_wrt_theta_tm1_autograd(
                    theta, theta_t, theta_tm1, pd_target, kp, kd,
                    mf0, pn_fric0, un0, dx, wv_curr_cache=wv_curr)
                allv = ag_ll.detach()
                self._debug_gradient_cpp(
                    "DDE-LL-AD-ANA", allv.norm().item(),
                    (HLLdx - allv).norm().item(), delta)
                self._debug_gradient_cpp(
                    "DDE-LL-AD-FD", allv.norm().item(),
                    (fd_dg_LL - allv).norm().item(), delta)
                print(
                    f"DDE-LL triple (central FD):  ANA={HLLdx.norm().item():.12e}  "
                    f"FD={fd_dg_LL.norm().item():.12e}  AD={allv.norm().item():.12e}  "
                    f"|ANA-AD|={(HLLdx - allv).norm().item():.12e}  "
                    f"|FD-AD|={(fd_dg_LL - allv).norm().item():.12e}")
            except Exception as ex:
                print(f"DDE-LL-Autograd: skipped ({type(ex).__name__}: {ex})")

        pd_p = pd_target + dx * delta
        pd_m = pd_target - dx * delta
        mf_Pp = self._clone_manifolds(mf_base)
        mf_Pm = self._clone_manifolds(mf_base)
        _, g_Pp, _ = self._compute_energy(
            theta, theta_t, theta_tm1, pd_p, kp, kd,
            mf_Pp, analytic_derivs=True, p_normals=pn0, u_tangents=un0,
            wv_curr_cache=wv_curr, wv_last_cache=wv_last,
            friction_plane_snap=fps_energy)
        _, g_Pm, _ = self._compute_energy(
            theta, theta_t, theta_tm1, pd_m, kp, kd,
            mf_Pm, analytic_derivs=True, p_normals=pn0, u_tangents=un0,
            wv_curr_cache=wv_curr, wv_last_cache=wv_last,
            friction_plane_snap=fps_energy)
        HPdx = H_P @ dx
        fd_dg_P = (g_Pp - g_Pm) / (2.0 * delta)
        self._debug_gradient_cpp(
            "DDE-P", HPdx.norm().item(),
            (HPdx - fd_dg_P).norm().item(), delta)

        print("DDE-D: skipped (no separate D target in Python PD model).")
        print("DDE-Design: skipped (no setDesign counterpart).")

        V3 = L * mM * 3
        dc = 2.0 * torch.rand(V3, device=dev, dtype=dty) - 1.0
        dcn = dc.norm()
        if dcn < 1e-14:
            dc = torch.zeros(V3, device=dev, dtype=dty)
            dc[0] = 1.0
        else:
            dc = dc / dcn
        d0 = self._local_verts.clone()
        self._local_verts = (d0.reshape(-1) + dc * delta).reshape(L, mM, 3)
        mf_xp = self._clone_manifolds(mf0)
        with torch.no_grad():
            wv_curr_xp = self._get_wv_stacked(theta_t)
            wv_last_xp = self._get_wv_stacked(theta_tm1)
        _, g_Xp, _ = self._compute_energy(
            theta, theta_t, theta_tm1, pd_target, kp, kd,
            mf_xp, analytic_derivs=True, p_normals=pn0, u_tangents=un0,
            wv_curr_cache=wv_curr_xp, wv_last_cache=wv_last_xp,
            friction_plane_snap=fps_energy)
        self._local_verts = (d0.reshape(-1) - dc * delta).reshape(L, mM, 3)
        mf_xm = self._clone_manifolds(mf0)
        with torch.no_grad():
            wv_curr_xm = self._get_wv_stacked(theta_t)
            wv_last_xm = self._get_wv_stacked(theta_tm1)
        _, g_Xm, _ = self._compute_energy(
            theta, theta_t, theta_tm1, pd_target, kp, kd,
            mf_xm, analytic_derivs=True, p_normals=pn0, u_tangents=un0,
            wv_curr_cache=wv_curr_xm, wv_last_cache=wv_last_xm,
            friction_plane_snap=fps_energy)
        self._local_verts = d0
        H_D = self._compute_HThetaD(
            theta, theta_t, theta_tm1,
            wv_star, wv_curr, wv_last,
            J_star, pd_target, kp, kd, mf0,
            u_tangents=un0)
        HDdc = H_D.reshape(n, -1) @ dc
        fd_dg_X = (g_Xp - g_Xm) / (2.0 * delta)
        self._debug_gradient_cpp(
            "DDE-XL", HDdc.norm().item(),
            (HDdc - fd_dg_X).norm().item(), delta)

        if compare_autograd:
            try:
                ag_x = self._jvp_grad_theta_wrt_local_verts_autograd(
                    theta, theta_t, theta_tm1, pd_target, kp, kd,
                    mf0, pn_fric0, un0, dc.reshape(-1), d0,
                    wv_curr_cache=wv_curr, wv_last_cache=wv_last)
                ax = ag_x.detach()
                self._debug_gradient_cpp(
                    "DDE-XL-AD-ANA", ax.norm().item(),
                    (HDdc - ax).norm().item(), delta)
                self._debug_gradient_cpp(
                    "DDE-XL-AD-FD", ax.norm().item(),
                    (fd_dg_X - ax).norm().item(), delta)
                print(
                    f"DDE-XL triple (central FD, ‖dc‖=1):  ANA={HDdc.norm().item():.12e}  "
                    f"FD={fd_dg_X.norm().item():.12e}  AD={ax.norm().item():.12e}  "
                    f"|ANA-AD|={(HDdc - ax).norm().item():.12e}  "
                    f"|FD-AD|={(fd_dg_X - ax).norm().item():.12e}")
            except Exception as ex:
                print(f"DDE-XL-Autograd: skipped ({type(ex).__name__}: {ex})")

        print("BVH-E: skipped (no BVH vs brute-force split in Python).")
        print("debug_energy complete.")

    # DEBUG BACKWARD — IFT verification
    def debug_backward(
            self,
            scale: float = 0.1,
            custom_delta: Optional[float] = None,
            theta_t: Optional[torch.Tensor] = None,
            theta_tm1: Optional[torch.Tensor] = None,
            pd_target: Optional[torch.Tensor] = None,
            kp: float = 100.0,
            kd: float = 10.0,
            max_random_trials: int = 200,
            manifolds: Optional[List[ContactManifold]] = None,
            compare_autograd_backward: bool = True,
            normalize_dx: bool = True) -> None:
        """IFT backward blocks vs FD, mirroring C++ debugBackward."""
        dev, dty = self.device, self.dtype
        n = self.n_dof
        L, mM = self._L, self._max_M
        delta = float(custom_delta) if custom_delta is not None else DEBUG_FD_EPS

        # --- acquire (theta_t, theta_tm1, pd_target) ---
        if theta_t is None and theta_tm1 is None and pd_target is None:
            found = False
            for _ in range(max_random_trials):
                tt = (2.0 * torch.rand(n, device=dev, dtype=dty) - 1.0) * scale
                tm1 = (2.0 * torch.rand(n, device=dev, dtype=dty) - 1.0) * scale
                pd_ = torch.randn(n, device=dev, dtype=dty)
                mfs = self._detect_contacts(tt)
                if len(mfs) == 0:
                    continue
                theta_t, theta_tm1, pd_target = tt, tm1, pd_
                found = True
                break
            if not found:
                print("debug_backward: no contact in random trials; "
                      f"pass theta_t, theta_tm1, pd_target explicitly.")
                return
        else:
            if theta_t is None or theta_tm1 is None or pd_target is None:
                raise ValueError(
                    "debug_backward: provide all of theta_t, theta_tm1, "
                    "pd_target, or none for random mode.")
            theta_t = theta_t.to(device=dev, dtype=dty)
            theta_tm1 = theta_tm1.to(device=dev, dtype=dty)
            pd_target = pd_target.to(device=dev, dtype=dty)

        # Step-entry friction snap (same construction as :meth:`step`).
        if manifolds is not None and len(manifolds) > 0:
            mf_step_entry = self._clone_manifolds(manifolds)
        else:
            mf_step_entry = self._detect_contacts(theta_t)
        fps_dbg = self._friction_plane_snap_dict(mf_step_entry)

        snap_entry = self._snapshot_pair_manifolds()
        base_d = self._local_verts.clone()
        saved_out = self._output
        saved_alpha = self._alpha

        def restore_entry():
            self._local_verts = base_d.clone()
            self._restore_pair_manifolds(snap_entry)

        # Reference solve (+ backward matrices on converged state)
        restore_entry()
        self._output = False
        self._alpha = 1.0
        theta_star, gi, mf_ref = self.step(
            theta_t, theta_tm1, pd_target, kp, kd,
            initial_manifolds=manifolds)
        snap_conv = self._snapshot_pair_manifolds()
        self._alpha = saved_alpha
        self._output = saved_out

        dx = 2.0 * torch.rand(n, device=dev, dtype=dty) - 1.0
        if normalize_dx:
            nrm = dx.norm()
            if nrm < 1e-14:
                dx = torch.zeros(n, device=dev, dtype=dty)
                dx[0] = 1.0
            else:
                dx = dx / nrm

        def fd_solve(tt, tm1, pd_, *, theta_init):
            self._output = False
            sa = self._alpha
            self._alpha = 1.0
            self._local_verts = base_d.clone()
            self._restore_pair_manifolds(snap_conv)
            th, _, _ = self.step(
                tt, tm1, pd_, kp, kd,
                theta_init=theta_init)
            self._alpha = sa
            self._output = saved_out
            return th

        def check_block(tag: str, J: torch.Tensor, direction: torch.Tensor,
                        theta_pert: torch.Tensor):
            ana = J @ direction
            fdv = (theta_pert - theta_star) / delta
            a_n = ana.norm().item()
            e_n = (ana - fdv).norm().item()
            self._debug_gradient_cpp(tag, a_n, e_n, delta)

        # DTDL  —  d theta* / d theta_t  (C++ _HThetaL)
        # Warm-start from converged θ* (same as DTDLL/DTDP). Init = θ_t+δdx alone
        # is a poor FD probe for the IFT linearization and blows up FD vs analytic.
        tt_p = theta_t + dx * delta
        th_p_dt = fd_solve(tt_p, theta_tm1, pd_target, theta_init=theta_star)
        check_block("DTDL", gi.dtheta_dtheta_t, dx, th_p_dt)

        # DTDLL — d theta* / d theta_{t-1}  (C++ _HThetaLL)
        tm1_p = theta_tm1 + dx * delta
        th_p_dtm1 = fd_solve(theta_t, tm1_p, pd_target, theta_init=theta_star)
        check_block("DTDLL", gi.dtheta_dtheta_tm1, dx, th_p_dtm1)

        # DTDP — d theta* / d pd target P  (C++ _HThetaPTarget)
        pd_p = pd_target + dx * delta
        th_p_dpd = fd_solve(theta_t, theta_tm1, pd_p, theta_init=theta_star)
        check_block("DTDP", gi.dtheta_dpd, dx, th_p_dpd)

        # DDE-BACK-θ*: ‖∇²_θ E(θ*)·d̂‖ vs central FD & autograd (same style as debug_energy DDE).
        if len(mf_ref) > 0:
            mf_de = self._clone_manifolds(mf_ref)
            pn_de = [m.p.detach().clone() for m in mf_de]
            un_de = [m.u.detach().clone() for m in mf_de]
            wv_cu_d = self._get_wv_stacked(theta_t.detach())
            wv_la_d = self._get_wv_stacked(theta_tm1.detach())
            _, _, H0_de = self._compute_energy(
                theta_star, theta_t, theta_tm1, pd_target, kp, kd,
                mf_de, analytic_derivs=True, p_normals=pn_de, u_tangents=un_de,
                wv_curr_cache=wv_cu_d, wv_last_cache=wv_la_d,
                friction_plane_snap=fps_dbg)
            Hdx_de = H0_de @ dx
            th_p_de = theta_star + dx * delta
            th_m_de = theta_star - dx * delta
            mf_pp_de = self._clone_manifolds(mf_de)
            mf_mm_de = self._clone_manifolds(mf_de)
            _, g_pp_de, _ = self._compute_energy(
                th_p_de, theta_t, theta_tm1, pd_target, kp, kd,
                mf_pp_de, analytic_derivs=True, p_normals=pn_de, u_tangents=un_de,
                wv_curr_cache=wv_cu_d, wv_last_cache=wv_la_d,
                friction_plane_snap=fps_dbg)
            _, g_mm_de, _ = self._compute_energy(
                th_m_de, theta_t, theta_tm1, pd_target, kp, kd,
                mf_mm_de, analytic_derivs=True, p_normals=pn_de, u_tangents=un_de,
                wv_curr_cache=wv_cu_d, wv_last_cache=wv_la_d,
                friction_plane_snap=fps_dbg)
            fd_dde_b = (g_pp_de - g_mm_de) / (2.0 * delta)
            self._debug_gradient_cpp(
                "DDE-BACK-θ*", Hdx_de.norm().item(),
                (Hdx_de - fd_dde_b).norm().item(), delta)
            if compare_autograd_backward:
                try:
                    pn_fb = self._friction_plane_list_from_snap(mf_de, fps_dbg)
                    ah_de = self._hvp_energy_theta_dir_autograd(
                        theta_star, theta_t, theta_tm1, pd_target, kp, kd,
                        mf_de, pn_fb, un_de, dx,
                        wv_curr_cache=wv_cu_d, wv_last_cache=wv_la_d).detach()
                    self._debug_gradient_cpp(
                        "DDE-BACK-θ*-AD-ANA", ah_de.norm().item(),
                        (Hdx_de - ah_de).norm().item(), delta)
                    self._debug_gradient_cpp(
                        "DDE-BACK-θ*-AD-FD", ah_de.norm().item(),
                        (fd_dde_b - ah_de).norm().item(), delta)
                    print(
                        f"DDE-BACK-θ* triple (‖H·d̂‖):  "
                        f"ANA={Hdx_de.norm().item():.12e}  "
                        f"FD={fd_dde_b.norm().item():.12e}  "
                        f"AD={ah_de.norm().item():.12e}  "
                        f"|ANA-AD|={(Hdx_de - ah_de).norm().item():.12e}  "
                        f"|FD-AD|={(fd_dde_b - ah_de).norm().item():.12e}")
                except Exception as ex_h:
                    print(f"DDE-BACK-θ* Autograd: skipped ({type(ex_h).__name__}: {ex_h})")
            if gi.H_theta_O is not None:
                H_o_dx = gi.H_theta_O @ dx
                self._debug_gradient_cpp(
                    "DDE-BACK-vs-HθO",
                    Hdx_de.norm().item(), (Hdx_de - H_o_dx).norm().item(), delta)
                print(
                    f"DDE-BACK vs H_theta_O:  ‖H_E·d̂‖={Hdx_de.norm().item():.12e}  "
                    f"‖H_O·d̂‖={H_o_dx.norm().item():.12e}  "
                    f"‖diff‖={(Hdx_de - H_o_dx).norm().item():.12e}")

        # DTDD — separate D target: not in Python PD interface (see docstring).
        print("DTDD: skipped (Python model has no separate D target tensor; "
              "kd couples (theta* - theta_t) only).")

        # DTDDesign — C++ setDesign; no equivalent exposed here.
        print("DTDDesign: skipped (no ConvHullPBDSimulator::setDesign "
              "counterpart in this Python sim).")

        # DTDXL — vertex / convex-hull design  (C++ _HThetaD)
        V3 = L * mM * 3
        th_p_ddxl: Optional[torch.Tensor] = None
        if gi.dtheta_dd is not None:
            dc = 2.0 * torch.rand(V3, device=dev, dtype=dty) - 1.0
            _dcn = dc.norm()
            if _dcn < 1e-14:
                dc = torch.zeros(V3, device=dev, dtype=dty)
                dc[0] = 1.0
            else:
                dc = dc / _dcn
            d_flat = base_d.reshape(-1)
            d_pert = (d_flat + dc * delta).reshape(L, mM, 3)
            self._output = False
            sa = self._alpha
            self._alpha = 1.0
            self._local_verts = d_pert.clone()
            self._restore_pair_manifolds(snap_conv)
            th_p_ddxl, _, _ = self.step(
                theta_t, theta_tm1, pd_target, kp, kd,
                theta_init=theta_star)
            self._alpha = sa
            self._output = saved_out
            ana = gi.dtheta_dd @ dc
            fdv = (th_p_ddxl - theta_star) / delta
            self._debug_gradient_cpp(
                "DTDXL", ana.norm().item(), (ana - fdv).norm().item(), delta)
        else:
            print("DTDXL: skipped (dtheta_dd is None).")
            dc = None  # unused

        # DTDXL sets _local_verts to d_pert = base_d + dc*δ.  If we leave it there,
        # XL-RHS FD uses g(θ*) at d_pert vs g at d_pert again → ‖FD‖≈0, and
        # analytic/JVP use inconsistent hull geometry vs ``base_d``.
        self._local_verts = base_d.clone()

        if compare_autograd_backward:
            def _bb_ad_triple(tag: str, ana: torch.Tensor, fd_v: torch.Tensor,
                              ad: torch.Tensor) -> None:
                """Same layout as ``debug_energy`` (DDE-* triple lines)."""
                print(
                    f"{tag}:  ANA={ana.norm().item():.12e}  "
                    f"FD={fd_v.norm().item():.12e}  AD={ad.norm().item():.12e}  "
                    f"|ANA-AD|={(ana - ad).norm().item():.12e}  "
                    f"|FD-AD|={(fd_v - ad).norm().item():.12e}")

            try:
                mf_h = self._clone_manifolds(mf_ref)
                pn_h = [m.p.detach().clone() for m in mf_h]
                un_h = [m.u.detach().clone() for m in mf_h]
                pn_fric_h = self._friction_plane_list_from_snap(mf_h, fps_dbg)
                wv_cu = self._get_wv_stacked(theta_t.detach())
                wv_la = self._get_wv_stacked(theta_tm1.detach())
                K_h = len(mf_h)
                dim_h = n + K_h * 4 + K_h * 3

                _, g0_h, H0_ift = self._compute_energy(
                    theta_star, theta_t, theta_tm1, pd_target, kp, kd,
                    mf_h, analytic_derivs=True, p_normals=pn_h, u_tangents=un_h,
                    wv_curr_cache=wv_cu, wv_last_cache=wv_la,
                    friction_plane_snap=fps_dbg)
                g_z = torch.zeros(n, device=dev, dtype=dty)
                H_full, _ = self._assemble_full_system(
                    g_z, H0_ift, mf_h, 0.0)
                H_full = H_full + 1e-6 * torch.eye(
                    dim_h, device=dev, dtype=dty)
                H_full = 0.5 * (H_full + H_full.T)

                (dg_dt, dg_dtm1, dg_dpd, wv_s_ift, J_s_ift,
                 _wvt_ift, J_t_ift, _wvtm1_ift, _Jtm1_ift
                 ) = self._dg_cross_blocks_for_ift(
                    theta_star, theta_t, theta_tm1, mf_h,
                    wv_cu, pn_fric_h, un_h, kp, kd)
                dgu_dt = None
                if K_h > 0:
                    dgu_dt = self._compute_dgu_dtheta_t(
                        wv_s_ift, wv_cu, J_t_ift,
                        mf_h, pn_fric_h, u_tangents=un_h)

                # FD of ∂(∇_θ E)/∂(·) · d̂ — central difference (same as ``debug_energy`` DDE-L/LL/P)
                theta_t_p = theta_t + dx * delta
                theta_t_m = theta_t - dx * delta
                wv_ct_p = self._get_wv_stacked(theta_t_p.detach())
                wv_ct_m = self._get_wv_stacked(theta_t_m.detach())
                mf_t_p = self._clone_manifolds(mf_h)
                mf_t_m = self._clone_manifolds(mf_h)
                _, g_Ltp, _ = self._compute_energy(
                    theta_star, theta_t_p, theta_tm1, pd_target, kp, kd,
                    mf_t_p, analytic_derivs=True, p_normals=pn_h, u_tangents=un_h,
                    wv_curr_cache=wv_ct_p, wv_last_cache=wv_la,
                    friction_plane_snap=fps_dbg)
                _, g_Ltm, _ = self._compute_energy(
                    theta_star, theta_t_m, theta_tm1, pd_target, kp, kd,
                    mf_t_m, analytic_derivs=True, p_normals=pn_h, u_tangents=un_h,
                    wv_curr_cache=wv_ct_m, wv_last_cache=wv_la,
                    friction_plane_snap=fps_dbg)
                fd_rhs_dt = ((g_Ltp - g_Ltm) / (2.0 * delta)).detach()

                theta_tm1_p = theta_tm1 + dx * delta
                theta_tm1_m = theta_tm1 - dx * delta
                wv_lp_p = self._get_wv_stacked(theta_tm1_p.detach())
                wv_lp_m = self._get_wv_stacked(theta_tm1_m.detach())
                mf_tm1_p = self._clone_manifolds(mf_h)
                mf_tm1_m = self._clone_manifolds(mf_h)
                _, g_Lt1p, _ = self._compute_energy(
                    theta_star, theta_t, theta_tm1_p, pd_target, kp, kd,
                    mf_tm1_p, analytic_derivs=True, p_normals=pn_h, u_tangents=un_h,
                    wv_curr_cache=wv_cu, wv_last_cache=wv_lp_p,
                    friction_plane_snap=fps_dbg)
                _, g_Lt1m, _ = self._compute_energy(
                    theta_star, theta_t, theta_tm1_m, pd_target, kp, kd,
                    mf_tm1_m, analytic_derivs=True, p_normals=pn_h, u_tangents=un_h,
                    wv_curr_cache=wv_cu, wv_last_cache=wv_lp_m,
                    friction_plane_snap=fps_dbg)
                fd_rhs_dtm1 = ((g_Lt1p - g_Lt1m) / (2.0 * delta)).detach()

                pd_p_fd = pd_target + dx * delta
                pd_m_fd = pd_target - dx * delta
                mf_pd_p = self._clone_manifolds(mf_h)
                mf_pd_m = self._clone_manifolds(mf_h)
                _, g_Lpp, _ = self._compute_energy(
                    theta_star, theta_t, theta_tm1, pd_p_fd, kp, kd,
                    mf_pd_p, analytic_derivs=True, p_normals=pn_h, u_tangents=un_h,
                    wv_curr_cache=wv_cu, wv_last_cache=wv_la,
                    friction_plane_snap=fps_dbg)
                _, g_Lpm, _ = self._compute_energy(
                    theta_star, theta_t, theta_tm1, pd_m_fd, kp, kd,
                    mf_pd_m, analytic_derivs=True, p_normals=pn_h, u_tangents=un_h,
                    wv_curr_cache=wv_cu, wv_last_cache=wv_la,
                    friction_plane_snap=fps_dbg)
                fd_rhs_dpd = ((g_Lpp - g_Lpm) / (2.0 * delta)).detach()

                def _solve_r(Rv: torch.Tensor) -> torch.Tensor:
                    return -torch.linalg.solve(H_full, Rv)

                fd_ift_dt = ((th_p_dt - theta_star) / delta).detach()
                fd_ift_dtm1 = ((th_p_dtm1 - theta_star) / delta).detach()
                fd_ift_dpd = ((th_p_dpd - theta_star) / delta).detach()

                print(
                    "--- backward autograd (same print style as debug_energy) ---")

                # --- DTDL: RHS + IFT ---
                b_ag = self._jvp_grad_theta_wrt_theta_t_autograd(
                    theta_star, theta_t, theta_tm1, pd_target, kp, kd,
                    mf_h, pn_fric_h, un_h, dx, wv_last_cache=wv_la).detach()
                b_an = (dg_dt @ dx).detach()
                self._debug_gradient_cpp(
                    "DTDL-RHS-AD-ANA", b_ag.norm().item(),
                    (b_an - b_ag).norm().item(), delta)
                self._debug_gradient_cpp(
                    "DTDL-RHS-AD-FD", b_ag.norm().item(),
                    (fd_rhs_dt - b_ag).norm().item(), delta)
                _bb_ad_triple(
                    "DTDL-RHS triple (∂(∇E)/∂θ_t·d̂)",
                    b_an, fd_rhs_dt, b_ag)

                R_t = self._ift_rhs_vector_theta_t_u(
                    b_ag, dgu_dt, dx, n, K_h)
                ag_ift = _solve_r(R_t)[:n]
                cmp_t = (gi.dtheta_dtheta_t @ dx).detach()
                self._debug_gradient_cpp(
                    "DTDL-IFT-AD-ANA", ag_ift.norm().item(),
                    (cmp_t - ag_ift).norm().item(), delta)
                self._debug_gradient_cpp(
                    "DTDL-IFT-AD-FD", ag_ift.norm().item(),
                    (fd_ift_dt - ag_ift).norm().item(), delta)
                _bb_ad_triple(
                    "DTDL-IFT triple (dθ*/dθ_t·d̂)",
                    cmp_t, fd_ift_dt, ag_ift)

                # --- DTDLL ---
                b_ag_ll = self._jvp_grad_theta_wrt_theta_tm1_autograd(
                    theta_star, theta_t, theta_tm1, pd_target, kp, kd,
                    mf_h, pn_fric_h, un_h, dx, wv_curr_cache=wv_cu).detach()
                b_ll = (dg_dtm1 @ dx).detach()
                self._debug_gradient_cpp(
                    "DTDLL-RHS-AD-ANA", b_ag_ll.norm().item(),
                    (b_ll - b_ag_ll).norm().item(), delta)
                self._debug_gradient_cpp(
                    "DTDLL-RHS-AD-FD", b_ag_ll.norm().item(),
                    (fd_rhs_dtm1 - b_ag_ll).norm().item(), delta)
                _bb_ad_triple(
                    "DTDLL-RHS triple (∂(∇E)/∂θ_{t-1}·d̂)",
                    b_ll, fd_rhs_dtm1, b_ag_ll)

                R_ll = torch.zeros(dim_h, device=dev, dtype=dty)
                R_ll[:n] = b_ag_ll
                ag_ll = _solve_r(R_ll)[:n]
                cmp_ll = (gi.dtheta_dtheta_tm1 @ dx).detach()
                self._debug_gradient_cpp(
                    "DTDLL-IFT-AD-ANA", ag_ll.norm().item(),
                    (cmp_ll - ag_ll).norm().item(), delta)
                self._debug_gradient_cpp(
                    "DTDLL-IFT-AD-FD", ag_ll.norm().item(),
                    (fd_ift_dtm1 - ag_ll).norm().item(), delta)
                _bb_ad_triple(
                    "DTDLL-IFT triple (dθ*/dθ_{t-1}·d̂)",
                    cmp_ll, fd_ift_dtm1, ag_ll)

                # --- DTDP ---
                b_ag_p = self._jvp_grad_theta_wrt_pd_autograd(
                    theta_star, theta_t, theta_tm1, pd_target, kp, kd,
                    mf_h, pn_fric_h, un_h, dx,
                    wv_curr_cache=wv_cu, wv_last_cache=wv_la).detach()
                b_p = (dg_dpd @ dx).detach()
                self._debug_gradient_cpp(
                    "DTDP-RHS-AD-ANA", b_ag_p.norm().item(),
                    (b_p - b_ag_p).norm().item(), delta)
                self._debug_gradient_cpp(
                    "DTDP-RHS-AD-FD", b_ag_p.norm().item(),
                    (fd_rhs_dpd - b_ag_p).norm().item(), delta)
                _bb_ad_triple(
                    "DTDP-RHS triple (∂(∇E)/∂p_d·d̂)",
                    b_p, fd_rhs_dpd, b_ag_p)

                R_p = torch.zeros(dim_h, device=dev, dtype=dty)
                R_p[:n] = b_ag_p
                ag_p = _solve_r(R_p)[:n]
                cmp_p = (gi.dtheta_dpd @ dx).detach()
                self._debug_gradient_cpp(
                    "DTDP-IFT-AD-ANA", ag_p.norm().item(),
                    (cmp_p - ag_p).norm().item(), delta)
                self._debug_gradient_cpp(
                    "DTDP-IFT-AD-FD", ag_p.norm().item(),
                    (fd_ift_dpd - ag_p).norm().item(), delta)
                _bb_ad_triple(
                    "DTDP-IFT triple (dθ*/dp_d·d̂)",
                    cmp_p, fd_ift_dpd, ag_p)

                # --- DTDXL ---
                if gi.dtheta_dd is not None and dc is not None:
                    dg_dd_h = self._compute_HThetaD(
                        theta_star, theta_t, theta_tm1,
                        wv_s_ift, wv_cu, wv_la,
                        J_s_ift, pd_target, kp, kd, mf_h,
                        u_tangents=un_h).reshape(n, -1)
                    b_dd = (dg_dd_h @ dc).detach()
                    b_ag_x = self._jvp_grad_theta_wrt_local_verts_autograd(
                        theta_star, theta_t, theta_tm1, pd_target, kp, kd,
                        mf_h, pn_fric_h, un_h, dc.reshape(-1), base_d).detach()

                    saved_lv = self._local_verts.clone()
                    self._local_verts = (
                        (base_d.reshape(-1) + dc * delta)
                        .reshape(L, mM, 3)).clone()
                    mf_x_p = self._clone_manifolds(mf_h)
                    with torch.no_grad():
                        wv_cu_xp = self._get_wv_stacked(theta_t)
                        wv_la_xp = self._get_wv_stacked(theta_tm1)
                    _, g_Xfd_p, _ = self._compute_energy(
                        theta_star, theta_t, theta_tm1, pd_target, kp, kd,
                        mf_x_p, analytic_derivs=True, p_normals=pn_h,
                        u_tangents=un_h,
                        wv_curr_cache=wv_cu_xp, wv_last_cache=wv_la_xp,
                        friction_plane_snap=fps_dbg)
                    self._local_verts = (
                        (base_d.reshape(-1) - dc * delta)
                        .reshape(L, mM, 3)).clone()
                    mf_x_m = self._clone_manifolds(mf_h)
                    with torch.no_grad():
                        wv_cu_xm = self._get_wv_stacked(theta_t)
                        wv_la_xm = self._get_wv_stacked(theta_tm1)
                    _, g_Xfd_m, _ = self._compute_energy(
                        theta_star, theta_t, theta_tm1, pd_target, kp, kd,
                        mf_x_m, analytic_derivs=True, p_normals=pn_h,
                        u_tangents=un_h,
                        wv_curr_cache=wv_cu_xm, wv_last_cache=wv_la_xm,
                        friction_plane_snap=fps_dbg)
                    self._local_verts = saved_lv
                    fd_rhs_x = ((g_Xfd_p - g_Xfd_m) / (2.0 * delta)).detach()

                    self._debug_gradient_cpp(
                        "DTDXL-RHS-AD-ANA", b_ag_x.norm().item(),
                        (b_dd - b_ag_x).norm().item(), delta)
                    self._debug_gradient_cpp(
                        "DTDXL-RHS-AD-FD", b_ag_x.norm().item(),
                        (fd_rhs_x - b_ag_x).norm().item(), delta)
                    _bb_ad_triple(
                        "DTDXL-RHS triple (∂(∇E)/∂d·dc)",
                        b_dd, fd_rhs_x, b_ag_x)

                    R_x = torch.zeros(dim_h, device=dev, dtype=dty)
                    R_x[:n] = b_ag_x
                    ag_x = _solve_r(R_x)[:n]
                    cmp_x = (gi.dtheta_dd @ dc).detach()
                    assert th_p_ddxl is not None
                    fd_ift_x = ((th_p_ddxl - theta_star) / delta).detach()
                    self._debug_gradient_cpp(
                        "DTDXL-IFT-AD-ANA", ag_x.norm().item(),
                        (cmp_x - ag_x).norm().item(), delta)
                    self._debug_gradient_cpp(
                        "DTDXL-IFT-AD-FD", ag_x.norm().item(),
                        (fd_ift_x - ag_x).norm().item(), delta)
                    _bb_ad_triple(
                        "DTDXL-IFT triple (dθ*/dd·dc)",
                        cmp_x, fd_ift_x, ag_x)
            except Exception as ex:
                print(f"debug_backward autograd/IFT check: skipped "
                      f"({type(ex).__name__}: {ex})")

        restore_entry()
        self._alpha = saved_alpha

    # HELPERS
    def _clone_manifolds(self, manifolds):
        """Deep-copy contacts so trial ``_back_substitute`` sees ``g_*`` / ``H_*`` blocks.

        Cloning only ``p``/``u`` makes ``valid_p``/``valid_u`` empty and Schur
        back-substitution a no-op (would falsely show ``Δp=Δu=0``).
        """
        def _c(t):
            return None if t is None else t.detach().clone()

        return [
            ContactManifold(
                link_a=m.link_a, link_b=m.link_b,
                p=_c(m.p), u=_c(m.u),
                g_p=_c(m.g_p), H_pp=_c(m.H_pp), H_theta_p=_c(m.H_theta_p),
                g_u=_c(m.g_u), H_uu=_c(m.H_uu), H_theta_u=_c(m.H_theta_u))
            for m in manifolds
        ]


    # DEBUG: full θ gradient / Hessian vs FD (fixed p, u)
    def debug_verify_theta_derivatives_fd(
            self,
            theta: torch.Tensor,
            theta_t: torch.Tensor,
            theta_tm1: torch.Tensor,
            pd_target: torch.Tensor,
            kp: float = 100.0,
            kd: float = 10.0,
            manifolds: Optional[List[ContactManifold]] = None,
            eps: Optional[float] = None,
            eps_hess: Optional[float] = None,
            wv_curr_cache: Optional[torch.Tensor] = None,
            wv_last_cache: Optional[torch.Tensor] = None,
            max_n_full_hess: int = 24,
            n_hess_random: int = 8,
            compare_hvp_autograd: bool = True,
            compare_cross_hessian: bool = True,
            compare_cross_autograd: bool = True,
            verbose: bool = True,
            ) -> Dict[str, float]:
        """Full θ gradient/Hessian vs central FD (fixed p,u)."""
        dev, dty = self.device, self.dtype
        n = self.n_dof
        th = theta.detach().to(device=dev, dtype=dty)
        th_t = theta_t.detach().to(device=dev, dtype=dty)
        th_tm1 = theta_tm1.detach().to(device=dev, dtype=dty)
        pd = pd_target.detach().to(device=dev, dtype=dty)

        if eps is None:
            eps = DEBUG_FD_EPS
        if eps_hess is None:
            eps_hess = DEBUG_FD_EPS

        if manifolds is None:
            manifolds = self._detect_contacts(th)
        fixed = [
            ContactManifold(
                link_a=m.link_a, link_b=m.link_b,
                p=m.p.detach().clone(), u=m.u.detach().clone())
            for m in manifolds
        ]
        p_normals = [m.p.detach().clone() for m in fixed]
        u_tangents = [m.u.detach().clone() for m in fixed]
        fps = self._friction_plane_snap_dict(self._detect_contacts(th_t))
        pn_fric = self._friction_plane_list_from_snap(fixed, fps)

        E0, g_ana, H_ana = self._compute_energy(
            th, th_t, th_tm1, pd, kp, kd, fixed,
            analytic_derivs=True,
            p_normals=p_normals, u_tangents=u_tangents,
            wv_curr_cache=wv_curr_cache, wv_last_cache=wv_last_cache,
            friction_plane_snap=fps)
        if g_ana is None or H_ana is None:
            raise RuntimeError("analytic gradient/Hessian unavailable")

        # --- gradient FD ---
        g_fd = torch.empty(n, device=dev, dtype=dty)
        for i in range(n):
            ei = torch.zeros(n, device=dev, dtype=dty)
            ei[i] = 1.0
            Ep, _, _ = self._compute_energy(
                th + eps * ei, th_t, th_tm1, pd, kp, kd, fixed,
                analytic_derivs=False,
                p_normals=p_normals, u_tangents=u_tangents,
                wv_curr_cache=wv_curr_cache, wv_last_cache=wv_last_cache,
                friction_plane_snap=fps)
            Em, _, _ = self._compute_energy(
                th - eps * ei, th_t, th_tm1, pd, kp, kd, fixed,
                analytic_derivs=False,
                p_normals=p_normals, u_tangents=u_tangents,
                wv_curr_cache=wv_curr_cache, wv_last_cache=wv_last_cache,
                friction_plane_snap=fps)
            g_fd[i] = (Ep - Em) / (2.0 * eps)

        diff_g = (g_ana - g_fd).abs()
        max_abs_err_grad = float(diff_g.max().item())
        gn = g_fd.norm().item()
        rel_l2_err_grad = float((diff_g.norm() / (gn + 1e-30)).item())

        out: Dict[str, float] = {
            "E": float(E0),
            "eps": float(eps),
            "eps_hess": float(eps_hess),
            "n_contacts": float(len(fixed)),
            "max_abs_err_grad": max_abs_err_grad,
            "rel_l2_err_grad": rel_l2_err_grad,
            "max_abs_err_hess": float("nan"),
            "rel_frob_err_hess": float("nan"),
            "max_rel_err_hess_mv": float("nan"),
            "max_rel_err_ana_ad": float("nan"),
            "max_rel_err_cross_theta_t_mv": float("nan"),
            "max_rel_err_cross_theta_tm1_mv": float("nan"),
            "max_rel_err_cross_pd_mv": float("nan"),
            "max_rel_err_cross_hull_mv": float("nan"),
            "max_rel_err_cross_L_ana_ad": float("nan"),
            "max_rel_err_cross_LL_ana_ad": float("nan"),
            "max_rel_err_cross_P_ana_ad": float("nan"),
            "max_rel_err_cross_XL_ana_ad": float("nan"),
        }

        if verbose:
            print(
                f"[debug_verify_theta_derivatives_fd] n_dof={n}  "
                f"contacts={len(fixed)}  eps_grad={eps:.2e}  eps_hess={eps_hess:.2e}  "
                f"E={E0:.6e}")
            print(
                f"  gradient: max|g_ana-g_fd|={max_abs_err_grad:.3e}  "
                f"||diff||_2/||g_fd||_2={rel_l2_err_grad:.3e}")

        # --- Hessian: full matrix if n small ---
        if n <= max_n_full_hess:
            H_fd = torch.zeros(n, n, device=dev, dtype=dty)
            for j in range(n):
                ej = torch.zeros(n, device=dev, dtype=dty)
                ej[j] = 1.0
                _, gp, _ = self._compute_energy(
                    th + eps_hess * ej, th_t, th_tm1, pd, kp, kd, fixed,
                    analytic_derivs=True,
                    p_normals=p_normals, u_tangents=u_tangents,
                    wv_curr_cache=wv_curr_cache, wv_last_cache=wv_last_cache,
                    friction_plane_snap=fps)
                _, gm, _ = self._compute_energy(
                    th - eps_hess * ej, th_t, th_tm1, pd, kp, kd, fixed,
                    analytic_derivs=True,
                    p_normals=p_normals, u_tangents=u_tangents,
                    wv_curr_cache=wv_curr_cache, wv_last_cache=wv_last_cache,
                    friction_plane_snap=fps)
                H_fd[:, j] = (gp - gm) / (2.0 * eps_hess)

            H_sym = 0.5 * (H_ana + H_ana.T)
            diff_h = (H_sym - H_fd).abs()
            max_abs_err_hess = float(diff_h.max().item())
            hf = H_fd.norm().item()
            rel_frob_err_hess = float((diff_h.norm() / (hf + 1e-30)).item())
            out["max_abs_err_hess"] = max_abs_err_hess
            out["rel_frob_err_hess"] = rel_frob_err_hess
            if verbose:
                print(
                    f"  Hessian (full FD, n={n}, eps_hess): max|H_ana-H_fd|={max_abs_err_hess:.3e}  "
                    f"||diff||_F/||H_fd||_F={rel_frob_err_hess:.3e}")

        # --- random directions: Hessian-vector ---
        max_rel_mv = 0.0
        torch.manual_seed(0)
        for _ in range(max(1, n_hess_random)):
            v = torch.randn(n, device=dev, dtype=dty)
            v = v / (v.norm() + 1e-30)
            Hv = H_ana @ v
            _, gp, _ = self._compute_energy(
                th + eps_hess * v, th_t, th_tm1, pd, kp, kd, fixed,
                analytic_derivs=True,
                p_normals=p_normals, u_tangents=u_tangents,
                wv_curr_cache=wv_curr_cache, wv_last_cache=wv_last_cache,
                friction_plane_snap=fps)
            _, gm, _ = self._compute_energy(
                th - eps_hess * v, th_t, th_tm1, pd, kp, kd, fixed,
                analytic_derivs=True,
                p_normals=p_normals, u_tangents=u_tangents,
                wv_curr_cache=wv_curr_cache, wv_last_cache=wv_last_cache,
                friction_plane_snap=fps)
            hv_fd = (gp - gm) / (2.0 * eps_hess)
            num = (Hv - hv_fd).norm().item()
            den = hv_fd.norm().item() + 1e-30
            max_rel_mv = max(max_rel_mv, num / den)
        out["max_rel_err_hess_mv"] = float(max_rel_mv)
        if verbose:
            print(
                f"  Hessian-vector (random v, {n_hess_random} trials, eps_hess): "
                f"max ||Hv - fd||/||fd|| = {max_rel_mv:.3e}")

        # --- cross-Hessian ∂(∇E)/∂(θ_t, θ_tm1, p_d, hull) · direction (central FD) ---
        L, mM = self._L, self._max_M
        V3 = L * mM * 3
        max_cL = max_cLL = max_cP = max_cXL = 0.0
        max_adL = max_adLL = max_adP = max_adXL = 0.0
        adL_ok = adLL_ok = adP_ok = adXL_ok = True
        if compare_cross_hessian:
            if wv_curr_cache is None:
                wv_cu_b = self._get_wv_stacked(th_t)
            else:
                wv_cu_b = wv_curr_cache
            if wv_last_cache is None:
                wv_la_b = self._get_wv_stacked(th_tm1)
            else:
                wv_la_b = wv_last_cache
            dt = self.dt
            vmask = self._vert_mask_float
            w_rho = self._rho_2d * vmask
            with torch.no_grad():
                wv_star_c, J_star_c, _, _ = self._compute_fk_jacobian_analytic(th)
                _, J_t_c, _, _ = self._compute_fk_jacobian_analytic(th_t)
                _, J_tm1_c, _, _ = self._compute_fk_jacobian_analytic(th_tm1)
            H_L_c = ((-2.0 / (dt * dt)) * torch.einsum(
                'lmci,lm,lmcj->ij', J_star_c, w_rho, J_t_c)
                     - kd * torch.diag(self.joint_mask ** 2))
            H_L_c = H_L_c + self._friction_cross_theta_t(
                wv_star_c, wv_cu_b, J_star_c, J_t_c, fixed, pn_fric,
                u_tangents=u_tangents)
            H_LL_c = ((1.0 / (dt * dt)) * torch.einsum(
                'lmci,lm,lmcj->ij', J_star_c, w_rho, J_tm1_c))
            H_P_c = -kp * torch.diag(self.joint_mask ** 2)

            torch.manual_seed(42)
            for _ in range(max(1, n_hess_random)):
                v = torch.randn(n, device=dev, dtype=dty)
                v = v / (v.norm() + 1e-30)
                HLv = H_L_c @ v
                wv_tp = self._get_wv_stacked(th_t + eps_hess * v)
                wv_tm = self._get_wv_stacked(th_t - eps_hess * v)
                _, gp, _ = self._compute_energy(
                    th, th_t + eps_hess * v, th_tm1, pd, kp, kd, fixed,
                    analytic_derivs=True,
                    p_normals=p_normals, u_tangents=u_tangents,
                    wv_curr_cache=wv_tp, wv_last_cache=wv_la_b,
                    friction_plane_snap=fps)
                _, gm, _ = self._compute_energy(
                    th, th_t - eps_hess * v, th_tm1, pd, kp, kd, fixed,
                    analytic_derivs=True,
                    p_normals=p_normals, u_tangents=u_tangents,
                    wv_curr_cache=wv_tm, wv_last_cache=wv_la_b,
                    friction_plane_snap=fps)
                fdL = (gp - gm) / (2.0 * eps_hess)
                denL = fdL.norm().item() + 1e-30
                max_cL = max(max_cL, (HLv - fdL).norm().item() / denL)

                HLLv = H_LL_c @ v
                wv_lp = self._get_wv_stacked(th_tm1 + eps_hess * v)
                wv_lm = self._get_wv_stacked(th_tm1 - eps_hess * v)
                _, gp2, _ = self._compute_energy(
                    th, th_t, th_tm1 + eps_hess * v, pd, kp, kd, fixed,
                    analytic_derivs=True,
                    p_normals=p_normals, u_tangents=u_tangents,
                    wv_curr_cache=wv_cu_b, wv_last_cache=wv_lp,
                    friction_plane_snap=fps)
                _, gm2, _ = self._compute_energy(
                    th, th_t, th_tm1 - eps_hess * v, pd, kp, kd, fixed,
                    analytic_derivs=True,
                    p_normals=p_normals, u_tangents=u_tangents,
                    wv_curr_cache=wv_cu_b, wv_last_cache=wv_lm,
                    friction_plane_snap=fps)
                fdLL = (gp2 - gm2) / (2.0 * eps_hess)
                denLL = fdLL.norm().item() + 1e-30
                max_cLL = max(max_cLL, (HLLv - fdLL).norm().item() / denLL)

                HPv = H_P_c @ v
                _, gpp, _ = self._compute_energy(
                    th, th_t, th_tm1, pd + eps_hess * v, kp, kd, fixed,
                    analytic_derivs=True,
                    p_normals=p_normals, u_tangents=u_tangents,
                    wv_curr_cache=wv_cu_b, wv_last_cache=wv_la_b,
                    friction_plane_snap=fps)
                _, gpm, _ = self._compute_energy(
                    th, th_t, th_tm1, pd - eps_hess * v, kp, kd, fixed,
                    analytic_derivs=True,
                    p_normals=p_normals, u_tangents=u_tangents,
                    wv_curr_cache=wv_cu_b, wv_last_cache=wv_la_b,
                    friction_plane_snap=fps)
                fdP = (gpp - gpm) / (2.0 * eps_hess)
                denP = fdP.norm().item() + 1e-30
                max_cP = max(max_cP, (HPv - fdP).norm().item() / denP)

                if compare_cross_autograd:
                    try:
                        jvL = self._jvp_grad_theta_wrt_theta_t_autograd(
                            th, th_t, th_tm1, pd, kp, kd,
                            fixed, pn_fric, u_tangents, v,
                            wv_last_cache=wv_la_b)
                        den_j = jvL.norm().item() + 1e-30
                        max_adL = max(
                            max_adL, (HLv - jvL).norm().item() / den_j)
                    except Exception:
                        adL_ok = False
                    try:
                        jvLL = self._jvp_grad_theta_wrt_theta_tm1_autograd(
                            th, th_t, th_tm1, pd, kp, kd,
                            fixed, pn_fric, u_tangents, v,
                            wv_curr_cache=wv_cu_b)
                        den_j = jvLL.norm().item() + 1e-30
                        max_adLL = max(
                            max_adLL, (HLLv - jvLL).norm().item() / den_j)
                    except Exception:
                        adLL_ok = False
                    try:
                        jvP = self._jvp_grad_theta_wrt_pd_autograd(
                            th, th_t, th_tm1, pd, kp, kd,
                            fixed, pn_fric, u_tangents, v,
                            wv_curr_cache=wv_cu_b, wv_last_cache=wv_la_b)
                        den_j = jvP.norm().item() + 1e-30
                        max_adP = max(
                            max_adP, (HPv - jvP).norm().item() / den_j)
                    except Exception:
                        adP_ok = False

            torch.manual_seed(43)
            d0 = self._local_verts.clone()
            for _ in range(max(1, n_hess_random)):
                dc = torch.randn(V3, device=dev, dtype=dty)
                dc = dc / (dc.norm() + 1e-30)
                H_D = self._compute_HThetaD(
                    th, th_t, th_tm1,
                    wv_star_c, wv_cu_b, wv_la_b,
                    J_star_c, pd, kp, kd, fixed,
                    u_tangents=u_tangents).reshape(n, -1)
                HDdc = H_D @ dc
                saved = self._local_verts.clone()
                self._local_verts = (
                    (d0.reshape(-1) + eps_hess * dc).reshape(L, mM, 3))
                wvx_p = self._get_wv_stacked(th_t)
                wvl_p = self._get_wv_stacked(th_tm1)
                _, gxp, _ = self._compute_energy(
                    th, th_t, th_tm1, pd, kp, kd, fixed,
                    analytic_derivs=True,
                    p_normals=p_normals, u_tangents=u_tangents,
                    wv_curr_cache=wvx_p, wv_last_cache=wvl_p,
                    friction_plane_snap=fps)
                self._local_verts = (
                    (d0.reshape(-1) - eps_hess * dc).reshape(L, mM, 3))
                wvx_m = self._get_wv_stacked(th_t)
                wvl_m = self._get_wv_stacked(th_tm1)
                _, gxm, _ = self._compute_energy(
                    th, th_t, th_tm1, pd, kp, kd, fixed,
                    analytic_derivs=True,
                    p_normals=p_normals, u_tangents=u_tangents,
                    wv_curr_cache=wvx_m, wv_last_cache=wvl_m,
                    friction_plane_snap=fps)
                self._local_verts = saved
                fdXL = (gxp - gxm) / (2.0 * eps_hess)
                denXL = fdXL.norm().item() + 1e-30
                max_cXL = max(max_cXL, (HDdc - fdXL).norm().item() / denXL)
                if compare_cross_autograd:
                    try:
                        jvX = self._jvp_grad_theta_wrt_local_verts_autograd(
                            th, th_t, th_tm1, pd, kp, kd,
                            fixed, pn_fric, u_tangents, dc.reshape(-1), d0,
                            wv_curr_cache=wv_cu_b, wv_last_cache=wv_la_b)
                        den_j = jvX.norm().item() + 1e-30
                        max_adXL = max(
                            max_adXL, (HDdc - jvX).norm().item() / den_j)
                    except Exception:
                        adXL_ok = False

            out["max_rel_err_cross_theta_t_mv"] = float(max_cL)
            out["max_rel_err_cross_theta_tm1_mv"] = float(max_cLL)
            out["max_rel_err_cross_pd_mv"] = float(max_cP)
            out["max_rel_err_cross_hull_mv"] = float(max_cXL)
            out["max_rel_err_cross_L_ana_ad"] = (
                float(max_adL) if adL_ok and compare_cross_autograd else float("nan"))
            out["max_rel_err_cross_LL_ana_ad"] = (
                float(max_adLL) if adLL_ok and compare_cross_autograd else float("nan"))
            out["max_rel_err_cross_P_ana_ad"] = (
                float(max_adP) if adP_ok and compare_cross_autograd else float("nan"))
            out["max_rel_err_cross_XL_ana_ad"] = (
                float(max_adXL) if adXL_ok and compare_cross_autograd else float("nan"))
            if verbose:
                print(
                    f"  cross-Hessian (central FD, eps_hess, {n_hess_random} trials):  "
                    f"max ||·||/||fd||  θ_t={max_cL:.3e}  θ_tm1={max_cLL:.3e}  "
                    f"p_d={max_cP:.3e}  hull={max_cXL:.3e}")
                if compare_cross_autograd:
                    msg_ad = (
                        f"  cross vs autograd (max ||ana-ad||/||ad||):  "
                        f"θ_t={max_adL if adL_ok else float('nan'):.3e}  "
                        f"θ_tm1={max_adLL if adLL_ok else float('nan'):.3e}  "
                        f"p_d={max_adP if adP_ok else float('nan'):.3e}  "
                        f"hull={max_adXL if adXL_ok else float('nan'):.3e}")
                    print(msg_ad)

        # --- autograd HVP (same E as debug_energy / _build_energy) ---
        max_rel_ad = float("nan")
        if compare_hvp_autograd:
            max_rel_ad = 0.0
            ad_ok = True
            ad_err_msg = ""
            torch.manual_seed(0)
            for _ in range(max(1, n_hess_random)):
                v = torch.randn(n, device=dev, dtype=dty)
                v = v / (v.norm() + 1e-30)
                try:
                    h_ad = self._hvp_energy_theta_dir_autograd(
                        th, th_t, th_tm1, pd, kp, kd,
                        fixed, p_normals, u_tangents, v,
                        wv_curr_cache=wv_curr_cache,
                        wv_last_cache=wv_last_cache)
                except Exception as ex:
                    max_rel_ad = float("nan")
                    ad_ok = False
                    ad_err_msg = f"{type(ex).__name__}: {ex}"
                    break
                Hv = H_ana @ v
                den_ad = h_ad.norm().item() + 1e-30
                max_rel_ad = max(max_rel_ad, (Hv - h_ad).norm().item() / den_ad)
            out["max_rel_err_ana_ad"] = float(max_rel_ad)
            if verbose:
                if ad_ok:
                    print(
                        f"  Hessian-vector vs autograd (same v trials): "
                        f"max ||H_ana·v - H_ad·v||/||H_ad·v|| = {max_rel_ad:.3e}")
                else:
                    print(
                        f"  Hessian-vector vs autograd: skipped ({ad_err_msg})")

        return out
