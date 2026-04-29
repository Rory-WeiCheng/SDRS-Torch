"""Stage III quasi-static regularizer — energy + inner-solve + gradient-residual loss.

Implements the formulation in `formulation/stage_III_new_formulation.md`:
    L_phys(X, p) = || ∇_q E(q, ζ*; X, X^-) |_{q=q^∘(X)} ||²

with ζ = (ν, δ, u, ω) per-pair inner variables solved by `solve_zeta_star`,
and X^- = stopgrad(X). Auxiliary pose q^∘(X) = (0, c(X)) per body.

Public API (incremental — built up piece by piece):
    rodrigues(theta)                                         — R(θ) ∈ SO(3)
    compute_world_vertices(q, x_bar, link_to_body)           — X̃ = R·x̄ + t
    gravity_energy(wv, rho_2d, gravity, vmask)               — Ψ_g
    gravity_grad_q(q, x_bar, link_to_body, rho_2d, gravity, vmask)
                                                             — ∇_q Ψ_g  (analytic)
    contact_energy(wv, p_stack, lid_a, lid_b, ...)           — μ · Σ Ψ_hh'
    damping_energy(pn_stack, u_stack, wv_next, wv_curr, ...) — η · Σ D_hh'
    quasi_static_energy(q, ζ, X, X_minus, ...)              — full E       [TODO]
    solve_zeta_star(X, p, ...)                               — inner solve [TODO]
    physics_loss(X, p, ...)                                  — autograd.Fn [TODO]

This module reuses:
    - simulator.barrier_eval (with mode='log' for the global log barrier)
    - simulator.LM kernel + Schur primitives (for the inner solve)
    - Robot / forward kinematics from robot.py
"""

import torch

from simulator import barrier_eval


# ---------------------------------------------------------------------------
# Rotation + FK helpers (Rodrigues; specialised to free 6-DOF bodies)
# ---------------------------------------------------------------------------
#
# The new formulation parameterises rotation as ``R(θ) = exp([θ]_×)`` with
# ``θ ∈ ℝ³`` (axis-angle).  SDRS-Torch's ``Robot.forward_kinematics`` instead
# uses ZYX Euler for its 'free' joint type (robot.py:328-331); since our
# scene is just N free 6-DOF bodies (each with M_per_body[i] hulls connected
# by fixed joints), we bypass the Robot abstraction and compute world
# vertices directly via Rodrigues.  link_to_body[l] tells us which body a
# given hull belongs to.

# --- Rodrigues primitives (PyTorch3D-style: torch.sinc + denominator clamp) -
#
# Forward pattern matches pytorch3d.transforms.axis_angle_to_matrix:
# torch.sinc handles sin(t)/t at all t (including t=0); (1 − cos t)/t² and
# (t − sin t)/t³ use a `torch.where(t² == 0, 1, t²)` denominator clamp — at
# exactly t = 0 both numerators are also 0, so the masked value is 0/1 = 0,
# multiplied by terms that are themselves 0 at t = 0 → identity result.
# No explicit Taylor branch needed for typical use (fp64 precision).

def _skew_batched(v: torch.Tensor) -> torch.Tensor:
    """v [..., 3] → [v]_× [..., 3, 3]."""
    K = torch.zeros(*v.shape[:-1], 3, 3, device=v.device, dtype=v.dtype)
    vx, vy, vz = v[..., 0], v[..., 1], v[..., 2]
    K[..., 0, 1] = -vz; K[..., 0, 2] = vy
    K[..., 1, 0] = vz;  K[..., 1, 2] = -vx
    K[..., 2, 0] = -vy; K[..., 2, 1] = vx
    return K


def rodrigues(theta: torch.Tensor) -> torch.Tensor:
    """Axis-angle to rotation matrix, batched.   θ [..., 3] → R [..., 3, 3].

    R = I + cw · sin(t)/t + cw² · (1 − cos t)/t²    where cw = [θ]_×, t = ‖θ‖.
    """
    t = theta.norm(dim=-1, keepdim=True)                        # [..., 1]
    t2 = t * t
    t2_safe = torch.where(t2 == 0, torch.ones_like(t2), t2)
    K = _skew_batched(theta)
    sinc_t = torch.sinc(t / torch.pi).unsqueeze(-1)             # [..., 1, 1]
    one_m_cos_over_t2 = ((1.0 - torch.cos(t)) / t2_safe).unsqueeze(-1)
    eye = torch.eye(3, device=theta.device, dtype=theta.dtype).expand_as(K)
    return eye + sinc_t * K + one_m_cos_over_t2 * (K @ K)


def rodrigues_with_diffV(theta: torch.Tensor):
    """Rodrigues + per-axis derivative generators, batched.

    Returns:
        R     [..., 3, 3]
        diffV [..., 3, 3]   ``diffV[..., k, :]`` is the 3-vector v_k such that
                            ``∂R/∂θ_k = [v_k]_× · R``.

    Closed form (Utils/RotationUtils.h:236-280):
        diffV[k] = _tMst_t3 · θ_k · θ          (outer product term)
                 + _1Mct_t2 · (θ × e_k)         ( = -_1Mct_t2 · cw[..., k, :] )
                 + _st_t · e_k                  (identity term)
    """
    t = theta.norm(dim=-1, keepdim=True)                        # [..., 1]
    t2 = t * t
    t2_safe = torch.where(t2 == 0, torch.ones_like(t2), t2)
    t3_safe = t2_safe * torch.where(t == 0, torch.ones_like(t), t)
    cw = _skew_batched(theta)

    f_st_t    = torch.sinc(t / torch.pi)                        # sin t / t
    f_1Mct_t2 = (1.0 - torch.cos(t)) / t2_safe                  # (1 − cos t)/t²
    f_tMst_t3 = (t - torch.sin(t)) / t3_safe                    # (t − sin t)/t³

    eye = torch.eye(3, device=theta.device, dtype=theta.dtype).expand_as(cw)
    R = (eye
         + f_st_t.unsqueeze(-1) * cw
         + f_1Mct_t2.unsqueeze(-1) * (cw @ cw))

    term1 = (f_tMst_t3.unsqueeze(-1)
             * theta.unsqueeze(-1) * theta.unsqueeze(-2))       # θ_k · θ_j
    term2 = -f_1Mct_t2.unsqueeze(-1) * cw                         # θ × e_k
    term3 = f_st_t.unsqueeze(-1) * eye                            # e_k
    diffV = term1 + term2 + term3
    return R, diffV


def compute_world_vertices(q: torch.Tensor,
                           x_bar: torch.Tensor,
                           link_to_body: torch.Tensor) -> torch.Tensor:
    """X̃ = R(θ_i) · x̄_lk + t_i,  where i = ``link_to_body[l]``.

    Reference (analogue): ``simulator.py:478-484`` (``_get_wv_stacked``); we
    specialise for the free-body case and replace SDRS's Euler-based FK with
    Rodrigues.  ``q`` packs (θ, t) per body so that ``q[i, :3] = θ_i``,
    ``q[i, 3:] = t_i``; the per-body (R, t) are broadcast to per-link via
    ``link_to_body``.

    Args:
        q:            [N, 6]  per-body pose.
        x_bar:        [L, M, 3] body-frame canonical vertices.
        link_to_body: [L]  long, body index for each link.

    Returns:
        wv: [L, M, 3] world vertices.
    """
    theta = q[..., :3]
    t = q[..., 3:]
    R = rodrigues(theta)
    R_per_link = R[link_to_body]
    t_per_link = t[link_to_body]
    return torch.einsum('lij,lmj->lmi', R_per_link, x_bar) + t_per_link.unsqueeze(1)


# ---------------------------------------------------------------------------
# Gravity: energy + analytic ∇_q
# ---------------------------------------------------------------------------

def gravity_grad_q(q: torch.Tensor,
                   x_bar: torch.Tensor,
                   link_to_body: torch.Tensor,
                   rho_2d: torch.Tensor,
                   gravity: float,
                   vmask: torch.Tensor) -> torch.Tensor:
    """Analytic ``∇_q Ψ_g`` at general θ.   Returns ``[N, 6]``.

    Ψ_g = -gravity · Σ_lk ρ_lk · (R(θ_{b(l)}) x̄_lk + t_{b(l)})_y · vmask_lk

        ∂Ψ_g/∂t_{i, α} = (0, -gravity · m_i, 0)   ŷ-only

        ∂Ψ_g/∂θ_{i, k} = -gravity · Σ_{l ∈ i} Σ_v ρ_lv · vmask_lv ·
                          ([diffV[k]]_× · R · x̄_lv)_y
                       = -gravity · Σ ρ · vmask · (diffV[k] × (R · x̄_lv))_y
                          (scalar triple identity)

    Uses ``rodrigues_with_diffV`` so the formula is exact at any θ. At
    q^∘ = (0, c(X)) with x̄ centred on c(X): R = I, diffV = I, the formula
    reduces to ``Σ ρ · (e_k × x̄)_y`` which vanishes by centroid symmetry.
    """
    dev, dty = q.device, q.dtype
    N = q.shape[0]
    R, diffV = rodrigues_with_diffV(q[..., :3])                  # R, diffV [N, 3, 3]
    R_per_link = R[link_to_body]
    diffV_per_link = diffV[link_to_body]                          # [L, 3, 3]
    Rx = torch.einsum('lij,lmj->lmi', R_per_link, x_bar)         # [L, M, 3]

    w = rho_2d * vmask                                            # [L, M]

    # ∂/∂t : only ŷ-component
    grad_t = torch.zeros(N, 3, device=dev, dtype=dty)
    mass_per_link = w.sum(dim=1)                                  # [L]
    grad_t[..., 1] = -gravity * torch.zeros(N, device=dev, dtype=dty).scatter_add_(
        0, link_to_body, mass_per_link)

    # ∂/∂θ : for axis k, contribution is (diffV[k] × R·x̄)_y, weighted by ρ·vmask.
    # Compute (diffV[k] × R·x̄)_y for all (k, l, v) at once.
    # diffV [L, 3, 3]: [..., k, :] is the k-th generator vector.
    # cross with Rx: ((diffV[k]) × Rx) per-vertex; we want y-component.
    # einsum can pick out the y-row of the cross-product matrix-vector.
    # (a × b)_y = a_z·b_x - a_x·b_z, so:
    cross_y = (diffV_per_link[..., :, 2:3] * Rx[..., None, :, 0]            # diffV_z · Rx_x
               - diffV_per_link[..., :, 0:1] * Rx[..., None, :, 2])         # - diffV_x · Rx_z
    # cross_y shape: [L, 3, M, 1]; squeeze last → [L, 3, M].
    cross_y = cross_y.squeeze(-1)                                            # [L, 3, M]
    # Now weight by w (vertex mass-mask) and sum over v, then scatter to body.
    contrib_per_link = -gravity * (cross_y * w.unsqueeze(1)).sum(dim=-1)    # [L, 3]
    grad_theta = torch.zeros(N, 3, device=dev, dtype=dty).scatter_add_(
        0, link_to_body.unsqueeze(-1).expand(-1, 3), contrib_per_link)

    return torch.cat([grad_theta, grad_t], dim=-1)


# ---------------------------------------------------------------------------
# Energy terms (scalar values)
# ---------------------------------------------------------------------------

def gravity_energy(wv: torch.Tensor,
                   rho_2d: torch.Tensor,
                   gravity: float,
                   vmask: torch.Tensor) -> torch.Tensor:
    """Gravitational potential. Reference: ``simulator.py:594-595`` (``# 2. Gravity``).

    Direct port:
        E = -(rho_2d * gravity * wv[..., 1] * vmask).sum()

    World is Y-up (matches SDRS); ``gravity`` is a scalar magnitude with sign
    (e.g. ``-9.81``).

    Args:
        wv:      [..., L, M, 3] world vertices.
        rho_2d:  [L, M] mass per vertex (= link mass / vertex count).
        gravity: scalar.
        vmask:   [L, M] {0, 1} float vertex mask.

    Returns:
        Ψ_g: scalar (or [...] if batched).
    """
    return -(rho_2d * gravity * wv[..., 1] * vmask).sum(dim=(-2, -1))


def contact_energy(wv: torch.Tensor,
                   p_stack: torch.Tensor,
                   lid_a: torch.Tensor,
                   lid_b: torch.Tensor,
                   vmask: torch.Tensor,
                   ground_h: torch.Tensor,
                   coef: float,
                   x0: float,
                   d0_half: float) -> torch.Tensor:
    """Contact-barrier potential μ · Σ Ψ_hh' summed over all pairs.

    Reference: ``simulator.py:632-661`` (sub-blocks 5a, 5b, 5c).

    DIFFERENCE from SDRS: every ``barrier_eval(...)`` call uses ``mode='log'``
    (global ``-log(x)``) instead of the default ``mode='truncated'``. All other
    math, masking, and accumulation are identical to the SDRS reference.

    The ``x0`` argument is unused in ``mode='log'`` (no truncation) but kept
    in the signature for symmetry with ``mode='truncated'``; pass any value.

    Args:
        wv:       [L, M, 3] world vertices.
        p_stack:  [K, 4] (n, d) plane params for K contact pairs.
        lid_a:    [K] long, A-side link index per pair.
        lid_b:    [K] long, B-side link index per pair (-1 = ground).
        vmask:    [L, M] {0,1} float vertex mask.
        ground_h: [Mg, 4] homogeneous ground vertices.
        coef:     barrier scaling μ (= ``self.coef_barrier``).
        x0, d0_half: barrier knee + half-offset (passed through).

    Returns:
        E_contact: scalar.
    """
    dev, dty = wv.device, wv.dtype
    K, max_M = p_stack.shape[0], wv.shape[1]
    is_ground = lid_b < 0
    gnd_idx = is_ground.nonzero(as_tuple=True)[0]
    lnk_idx = (~is_ground).nonzero(as_tuple=True)[0]

    va_next = wv[lid_a]
    link_mask_a = vmask[lid_a]
    ones_KM1 = torch.ones(K, max_M, 1, device=dev, dtype=dty)

    # 5a. A-side d_a + normal-magnitude penalty   (simulator.py:632-639)
    va_h = torch.cat([va_next, ones_KM1], dim=2)
    d_a = -torch.einsum('kmi,ki->km', va_h, p_stack)
    val_a, _, _ = barrier_eval(d_a, x0, d0_half, mode='log')
    val_a = val_a * link_mask_a
    norm_p = torch.norm(p_stack[:, :3], dim=1)
    val_n, _, _ = barrier_eval(1.0 - norm_p, x0, mode='log')
    E = coef * (val_n.sum() + val_a.sum())

    # 5b. B-side ground   (simulator.py:644-648)
    if gnd_idx.numel() > 0:
        p_gnd = p_stack[gnd_idx]
        d_b_gnd = torch.einsum('gi,ki->kg', ground_h, p_gnd)
        val_b_gnd, _, _ = barrier_eval(d_b_gnd, x0, d0_half, mode='log')
        E = E + coef * val_b_gnd.sum()

    # 5c. B-side link-link   (simulator.py:651-661)
    if lnk_idx.numel() > 0:
        lid_b_lnk = lid_b[lnk_idx]
        p_lnk = p_stack[lnk_idx]
        vb_next_lnk = wv[lid_b_lnk]
        mask_b_lnk = vmask[lid_b_lnk]
        ones_lnk = torch.ones(lnk_idx.numel(), max_M, 1, device=dev, dtype=dty)
        vb_h_lnk = torch.cat([vb_next_lnk, ones_lnk], dim=2)
        d_b_lnk = torch.einsum('kmi,ki->km', vb_h_lnk, p_lnk)
        val_b_lnk, _, _ = barrier_eval(d_b_lnk, x0, d0_half, mode='log')
        val_b_lnk = val_b_lnk * mask_b_lnk
        E = E + coef * val_b_lnk.sum()

    return E


def _friction_tangent_basis(n_hat: torch.Tensor):
    """Orthonormal in-plane basis ``(t0, t1) ⟂ n_hat``. Reference: ``simulator.py:486-500``.

    Direct port (no differences).
    """
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


def _unified_u_to_xyz_omega(u_unified: torch.Tensor, n_hat: torch.Tensor):
    """``u_unified [K,3]`` → ``(u_xyz [K,3], ω [K], t0, t1)``. Reference: ``simulator.py:502-508``.

    Direct port (no differences).
    """
    t0, t1 = _friction_tangent_basis(n_hat)
    u_xyz = u_unified[:, 0:1] * t0 + u_unified[:, 1:2] * t1
    omega = u_unified[:, 2]
    return u_xyz, omega, t0, t1


def damping_energy(pn_stack: torch.Tensor,
                   u_stack: torch.Tensor,
                   wv_next: torch.Tensor,
                   wv_curr: torch.Tensor,
                   lid_a: torch.Tensor,
                   lid_b: torch.Tensor,
                   lnk_idx: torch.Tensor,
                   vmask: torch.Tensor,
                   friction: float,
                   coef: float,
                   x0: float,
                   d0_half: float,
                   dt: float,
                   eps_n: float = 1e-8,
                   eps_s: float = 1e-4) -> torch.Tensor:
    """Friction damping  η · Σ D_hh'.

    Reference: ``simulator.py:674-736`` (``_friction_energy_total_pn``).

    DIFFERENCES from SDRS:
      (1) ``barrier_eval`` for ``bg_fn`` uses ``mode='log'`` (consistent with
          ``contact_energy``). SDRS uses default ``'truncated'``.
      (2) ``eye3`` constructed inline; SDRS caches ``self._eye3``.
      (3) ``eps_n``, ``eps_s`` are explicit args (vs ``self._friction_eps_*``);
          ``sqrt_eps_s`` computed inline.
      (4) Helpers ``_friction_tangent_basis`` / ``_unified_u_to_xyz_omega``
          are this module's local ports of the SDRS methods.

    Note (formulation): for the quasi-static loss the caller passes
    ``wv_curr`` numerically equal to ``wv_next``. The role of ``X^-`` as a
    "frozen reference" is realised by the analytic gradient/Hessian routines
    (added in subsequent commits) which treat ``wv_curr`` as constant by
    construction — no tensor-level ``.detach()`` is required.

    Args:
        pn_stack: [K, 4]   per-pair (n, d) plane params.
        u_stack:  [K, 3]   per-pair u_unified (in (t0, t1, n) basis: α, β, ω).
        wv_next:  [L, M, 3] current world vertices X̃.
        wv_curr:  [L, M, 3] reference world vertices X^-.
        lid_a:    [K] long.
        lid_b:    [K] long (-1 = ground).
        lnk_idx:  [K_lnk] long, indices of link-link pairs.
        vmask:    [L, M] {0,1} float vertex mask.
        friction: η coefficient.
        coef:     barrier μ (used for normal-load magnitude).
        x0, d0_half: barrier knee + half-offset.
        dt:       time-step (= Δτ for quasi-static).
        eps_n, eps_s: smoothing constants.

    Returns:
        E_damping: scalar.
    """
    dev, dty = wv_next.device, wv_next.dtype
    K = pn_stack.shape[0]
    max_M = wv_next.shape[1]

    va_next = wv_next[lid_a]
    va_curr = wv_curr[lid_a]
    link_mask_a = vmask[lid_a]
    ones_KM1 = torch.ones(K, max_M, 1, device=dev, dtype=dty)

    n_vecs = pn_stack[:, :3]
    norm_n = torch.norm(n_vecs, dim=1, keepdim=True) + eps_n
    n_hat = n_vecs / norm_n
    eye3 = torch.eye(3, device=dev, dtype=dty)
    Proj = eye3.unsqueeze(0) - n_hat.unsqueeze(2) * n_hat.unsqueeze(1)
    u_xyz, omega_stack, _, _ = _unified_u_to_xyz_omega(u_stack, n_hat)

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
    _, bg_fn, _ = barrier_eval(d_fn, x0, d0_half, mode='log')
    pn3 = pn_stack[:, :3]
    f_vec = coef * bg_fn.unsqueeze(2) * pn3.unsqueeze(1)
    fn_sq = f_vec.pow(2).sum(dim=2)
    A_m = torch.sqrt(fn_sq + eps_s) - (eps_s ** 0.5)

    s_norm = torch.sqrt(rel_vel.pow(2).sum(dim=2) + eps_s)
    return (friction * dt * (A_m * s_norm * link_mask_a)).sum()


# ---------------------------------------------------------------------------
# Vertex → pose chain (used by every per-term q-gradient)
# ---------------------------------------------------------------------------

def vertex_grad_to_q(g_v: torch.Tensor,
                     q: torch.Tensor,
                     x_bar: torch.Tensor,
                     link_to_body: torch.Tensor) -> torch.Tensor:
    """``∂E/∂q[i] = Σ_{l ∈ i, k} J_lk^T · g_v_lk``  where ``J_lk`` is the FK Jacobian.

    For body i (link l = ``link_to_body^{-1}(i)``):
        ∂E/∂t_{i, α}    = Σ_v g_v_lkα                                 (translation)
        ∂E/∂θ_{i, α}    = Σ_v g_v_lk · (diffV[α] × R · x̄_lk)
                        = (Σ_v R·x̄_lk × g_v_lk) · diffV[α]            (scalar triple)

    Reference: simulator.py:1241-1243 — equivalent to ``J^T g_v`` with our
    direct-FK-for-free-bodies parameterisation.

    Args:
        g_v:          [L, M, 3] vertex-level gradient.
        q, x_bar, link_to_body: as in ``compute_world_vertices``.

    Returns:
        g_q: [N, 6] with ``g_q[i, :3] = ∂E/∂θ_i``, ``g_q[i, 3:] = ∂E/∂t_i``.
    """
    dev, dty = g_v.device, g_v.dtype
    N = q.shape[0]
    R, diffV = rodrigues_with_diffV(q[..., :3])                    # [N, 3, 3], [N, 3, 3]
    R_per_link = R[link_to_body]
    diffV_per_link = diffV[link_to_body]                           # [L, 3, 3]
    Rx = torch.einsum('lij,lmj->lmi', R_per_link, x_bar)           # [L, M, 3]

    # ∂/∂t : sum g_v over vertices, scatter to bodies.
    grad_t_per_link = g_v.sum(dim=1)                                # [L, 3]
    grad_t = torch.zeros(N, 3, device=dev, dtype=dty).scatter_add_(
        0, link_to_body.unsqueeze(-1).expand(-1, 3), grad_t_per_link)

    # ∂/∂θ : (Σ Rx × g_v) contracted with diffV[α] for each axis α.
    cross_sum_per_link = torch.cross(Rx, g_v, dim=-1).sum(dim=1)   # [L, 3]
    grad_theta_per_link = torch.einsum(
        'lj,laj->la', cross_sum_per_link, diffV_per_link)           # [L, 3]
    grad_theta = torch.zeros(N, 3, device=dev, dtype=dty).scatter_add_(
        0, link_to_body.unsqueeze(-1).expand(-1, 3), grad_theta_per_link)

    return torch.cat([grad_theta, grad_t], dim=-1)


# ---------------------------------------------------------------------------
# Contact: vertex-level + ζ-level gradients of  μ · Σ Ψ_hh'
# ---------------------------------------------------------------------------

def contact_vertex_grad(wv: torch.Tensor,
                        p_stack: torch.Tensor,
                        lid_a: torch.Tensor,
                        lid_b: torch.Tensor,
                        vmask: torch.Tensor,
                        coef: float,
                        x0: float,
                        d0_half: float) -> torch.Tensor:
    """``∂(coef · Σ Ψ_hh')/∂X̃_lk`` accumulated across all pairs.   Returns ``[L, M, 3]``.

    Reference: ``simulator.py:1181-1197``. Direct port with ``mode='log'``.
        A-side  (all pairs):  g_v[lid_a] += -coef · bg_a · vmask · ν
        B-side  (link-link):  g_v[lid_b] +=  coef · bg_b · vmask · ν
        B-side  (ground):     no contribution (ground vertices are static).
        Normal-magnitude term ``-log(1−‖ν‖)`` does not depend on X̃.
    """
    dev, dty = wv.device, wv.dtype
    K, max_M = p_stack.shape[0], wv.shape[1]
    L = wv.shape[0]
    is_ground = lid_b < 0
    lnk_idx = (~is_ground).nonzero(as_tuple=True)[0]

    g_v = torch.zeros(L, max_M, 3, device=dev, dtype=dty)
    if K == 0:
        return g_v

    va_next = wv[lid_a]                                             # [K, M, 3]
    link_mask_a = vmask[lid_a]                                      # [K, M]
    ones_KM1 = torch.ones(K, max_M, 1, device=dev, dtype=dty)
    p3 = p_stack[:, :3]                                             # [K, 3] = ν
    va_h = torch.cat([va_next, ones_KM1], dim=2)
    d_a = -torch.einsum('kmi,ki->km', va_h, p_stack)                # [K, M]
    _, bg_a, _ = barrier_eval(d_a, x0, d0_half, mode='log')
    bg_m = bg_a * link_mask_a                                       # [K, M]
    g_v.index_add_(0, lid_a, -coef * bg_m.unsqueeze(2) * p3.unsqueeze(1))

    if lnk_idx.numel() > 0:
        lid_b_lnk = lid_b[lnk_idx]
        mask_b_lnk = vmask[lid_b_lnk]
        vb_next_lnk = wv[lid_b_lnk]
        ones_lnk = torch.ones(lnk_idx.numel(), max_M, 1, device=dev, dtype=dty)
        vb_h_lnk = torch.cat([vb_next_lnk, ones_lnk], dim=2)
        d_b_lnk = torch.einsum('kmi,ki->km', vb_h_lnk, p_stack[lnk_idx])
        _, bg_b_lnk, _ = barrier_eval(d_b_lnk, x0, d0_half, mode='log')
        bg_b_lnk_m = bg_b_lnk * mask_b_lnk
        p3_lnk = p3[lnk_idx]
        g_v.index_add_(0, lid_b_lnk, coef * bg_b_lnk_m.unsqueeze(2) * p3_lnk.unsqueeze(1))

    return g_v


def contact_zeta_grad(wv: torch.Tensor,
                      p_stack: torch.Tensor,
                      lid_a: torch.Tensor,
                      lid_b: torch.Tensor,
                      vmask: torch.Tensor,
                      ground_h: torch.Tensor,
                      coef: float,
                      x0: float,
                      d0_half: float) -> torch.Tensor:
    """``∂(coef · Σ Ψ_hh')/∂(ν_p, δ_p)`` per pair.   Returns ``[K, 4]`` (ν: 0:3, δ: 3).

    Three contributions per pair (matching contact_energy's three sub-blocks):
        A-side  (5a):    -coef · Σ_k bg_a · vmask · (X̃_ak,  1)            [appears in ∂ν, ∂δ via ∂d_a]
        B-side ground:    coef · Σ_k bg_b · (X̃_ground_k, 1)
        B-side link-link: coef · Σ_k bg_b · vmask · (X̃_bk,  1)
        Normal magnitude: coef · bg_n · ((-ν / ‖ν‖), 0)                    [adds to ∂ν only]
    """
    dev, dty = wv.device, wv.dtype
    K, max_M = p_stack.shape[0], wv.shape[1]
    is_ground = lid_b < 0
    gnd_idx = is_ground.nonzero(as_tuple=True)[0]
    lnk_idx = (~is_ground).nonzero(as_tuple=True)[0]

    g_zeta = torch.zeros(K, 4, device=dev, dtype=dty)
    if K == 0:
        return g_zeta

    va_next = wv[lid_a]
    link_mask_a = vmask[lid_a]
    ones_KM1 = torch.ones(K, max_M, 1, device=dev, dtype=dty)
    va_h = torch.cat([va_next, ones_KM1], dim=2)                    # [K, M, 4]
    d_a = -torch.einsum('kmi,ki->km', va_h, p_stack)
    _, bg_a, _ = barrier_eval(d_a, x0, d0_half, mode='log')
    bg_m_a = (bg_a * link_mask_a).unsqueeze(2)                      # [K, M, 1]
    # ∂d_a/∂(ν, δ) = -(va_h). So contribution = -coef · bg · va_h.
    g_zeta = g_zeta + (-coef * bg_m_a * va_h).sum(dim=1)            # [K, 4]

    # Normal-magnitude term: -log(1 − ‖ν‖). Only affects ν (not δ).
    # Matches contact_energy:638 exactly (no eps_n) — SDRS uses bare norm
    # in the energy, eps_n only appears in the Hessian (intentional, see
    # contact_pp_hess docstring).
    p3 = p_stack[:, :3]
    norm_p = torch.norm(p3, dim=1, keepdim=True).clamp(min=1e-30)
    _, bg_n, _ = barrier_eval(1.0 - torch.norm(p3, dim=1), x0, mode='log')
    g_zeta[:, :3] = g_zeta[:, :3] + coef * bg_n.unsqueeze(1) * (-p3 / norm_p)

    if gnd_idx.numel() > 0:
        p_gnd = p_stack[gnd_idx]
        d_b_gnd = torch.einsum('gi,ki->kg', ground_h, p_gnd)        # [K_gnd, Mg]
        _, bg_b_gnd, _ = barrier_eval(d_b_gnd, x0, d0_half, mode='log')
        # ∂d_b_gnd/∂(ν, δ) = +ground_h.  contribution = +coef · bg · ground_h.
        # ground_h is shared across pairs; einsum to per-pair sum.
        contrib_gnd = coef * torch.einsum('kg,gi->ki', bg_b_gnd, ground_h)  # [K_gnd, 4]
        g_zeta[gnd_idx] = g_zeta[gnd_idx] + contrib_gnd

    if lnk_idx.numel() > 0:
        lid_b_lnk = lid_b[lnk_idx]
        mask_b_lnk = vmask[lid_b_lnk]
        vb_next_lnk = wv[lid_b_lnk]
        ones_lnk = torch.ones(lnk_idx.numel(), max_M, 1, device=dev, dtype=dty)
        vb_h_lnk = torch.cat([vb_next_lnk, ones_lnk], dim=2)        # [K_lnk, M, 4]
        d_b_lnk = torch.einsum('kmi,ki->km', vb_h_lnk, p_stack[lnk_idx])
        _, bg_b_lnk, _ = barrier_eval(d_b_lnk, x0, d0_half, mode='log')
        bg_m_b = (bg_b_lnk * mask_b_lnk).unsqueeze(2)               # [K_lnk, M, 1]
        contrib_lnk = (coef * bg_m_b * vb_h_lnk).sum(dim=1)         # [K_lnk, 4]
        g_zeta[lnk_idx] = g_zeta[lnk_idx] + contrib_lnk

    return g_zeta


# ---------------------------------------------------------------------------
# Damping (friction): vertex-level gradient of  η · Σ D_hh'
# ---------------------------------------------------------------------------

def damping_vertex_grad(pn_stack: torch.Tensor,
                        u_stack: torch.Tensor,
                        wv_next: torch.Tensor,
                        wv_curr: torch.Tensor,
                        lid_a: torch.Tensor,
                        lid_b: torch.Tensor,
                        lnk_idx: torch.Tensor,
                        vmask: torch.Tensor,
                        friction: float,
                        coef: float,
                        x0: float,
                        d0_half: float,
                        dt: float,
                        eps_n: float = 1e-8,
                        eps_s: float = 1e-4) -> torch.Tensor:
    """``∂(η · Σ D_hh')/∂X̃_lk`` via the slip-velocity term.   Returns ``[L, M, 3]``.

    Reference: ``simulator.py:1199-1238`` (friction part of ``_compute_energy``).
    Only the slip term ``s_norm = √(‖v^∥‖² + ε_s)`` depends on ``wv_next``;
    the normal-load weight ``A_m`` uses ``wv_curr`` (= X^-) and is treated as
    constant for this vertex gradient. dt cancels (analytic derivation):
        ∂(friction·dt·A_m·s_norm)/∂wv_next = friction · A_m · T·(rel_vel/s_norm).

    Same calling convention as ``damping_energy`` (caller chooses
    ``wv_curr = wv_next.detach()`` for quasi-static use).
    """
    dev, dty = wv_next.device, wv_next.dtype
    K = pn_stack.shape[0]
    L, max_M = wv_next.shape[0], wv_next.shape[1]

    g_v = torch.zeros(L, max_M, 3, device=dev, dtype=dty)
    if K == 0:
        return g_v

    va_next = wv_next[lid_a]
    va_curr = wv_curr[lid_a]
    link_mask_a = vmask[lid_a]
    ones_KM1 = torch.ones(K, max_M, 1, device=dev, dtype=dty)

    n_vecs = pn_stack[:, :3]
    norm_n = torch.norm(n_vecs, dim=1, keepdim=True) + eps_n
    n_hat = n_vecs / norm_n
    eye3 = torch.eye(3, device=dev, dtype=dty)
    Proj = eye3.unsqueeze(0) - n_hat.unsqueeze(2) * n_hat.unsqueeze(1)
    u_xyz, omega_stack, _, _ = _unified_u_to_xyz_omega(u_stack, n_hat)

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
    _, bg_fn, _ = barrier_eval(d_fn, x0, d0_half, mode='log')
    pn3 = pn_stack[:, :3]
    f_vec = coef * bg_fn.unsqueeze(2) * pn3.unsqueeze(1)
    fn_sq = f_vec.pow(2).sum(dim=2)
    A_m = torch.sqrt(fn_sq + eps_s) - (eps_s ** 0.5)

    s_norm = torch.sqrt(rel_vel.pow(2).sum(dim=2) + eps_s)
    inv_s = 1.0 / (s_norm + 1e-30)
    w_fric = (A_m * link_mask_a).unsqueeze(2)
    rel_over_s = rel_vel * inv_s.unsqueeze(2)
    proj_rs = torch.einsum('kij,kmj->kmi', Proj, rel_over_s)
    g_fric_s = friction * w_fric * proj_rs                          # [K, M, 3]

    g_v.index_add_(0, lid_a, g_fric_s)
    if lnk_idx.numel() > 0:
        lid_b_lnk = lid_b[lnk_idx]
        g_v.index_add_(0, lid_b_lnk, -g_fric_s[lnk_idx])

    return g_v


def damping_u_grad(pn_stack: torch.Tensor,
                   u_stack: torch.Tensor,
                   wv_next: torch.Tensor,
                   wv_curr: torch.Tensor,
                   lid_a: torch.Tensor,
                   lid_b: torch.Tensor,
                   lnk_idx: torch.Tensor,
                   vmask: torch.Tensor,
                   friction: float,
                   coef: float,
                   x0: float,
                   d0_half: float,
                   dt: float,
                   eps_n: float = 1e-8,
                   eps_s: float = 1e-4) -> torch.Tensor:
    """``∂(η · Σ D_hh')/∂u_unified`` per pair.   Returns ``[K, 3]``.

    Reference: ``simulator.py:1836-1841`` (``g_u3_mh``, ``g_om_mh``,
    ``_friction_reduce_g``).
        g_u3 = -Σ_v c · (rel_vel / s_norm)                       [K, 3] = ∂D/∂u_xyz
        g_om = -Σ_v c · (h / s_norm),  h = rel_vel · r_nx        [K]    = ∂D/∂ω
        g_u_unified = (g_u3 · t0,  g_u3 · t1,  g_om)
        where c = friction · dt · A_m · vmask.

    Note: matches SDRS's "friction-plane snap" pattern — the direct
    contributions of (ν, δ) to D are not included here (they're captured
    via the cross-Hessian / vertex-side gradient through wv).
    """
    dev, dty = wv_next.device, wv_next.dtype
    K = pn_stack.shape[0]
    if K == 0:
        return torch.zeros(0, 3, device=dev, dtype=dty)

    max_M = wv_next.shape[1]
    va_next = wv_next[lid_a]
    va_curr = wv_curr[lid_a]
    link_mask_a = vmask[lid_a]
    ones_KM1 = torch.ones(K, max_M, 1, device=dev, dtype=dty)

    n_vecs = pn_stack[:, :3]
    norm_n = torch.norm(n_vecs, dim=1, keepdim=True) + eps_n
    n_hat = n_vecs / norm_n
    eye3 = torch.eye(3, device=dev, dtype=dty)
    Proj = eye3.unsqueeze(0) - n_hat.unsqueeze(2) * n_hat.unsqueeze(1)
    u_xyz, omega_stack, t0, t1 = _unified_u_to_xyz_omega(u_stack, n_hat)

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
    _, bg_fn, _ = barrier_eval(d_fn, x0, d0_half, mode='log')
    pn3 = pn_stack[:, :3]
    f_vec = coef * bg_fn.unsqueeze(2) * pn3.unsqueeze(1)
    fn_sq = f_vec.pow(2).sum(dim=2)
    A_m = torch.sqrt(fn_sq + eps_s) - (eps_s ** 0.5)

    s_norm = torch.sqrt(rel_vel.pow(2).sum(dim=2) + eps_s)
    inv_s = 1.0 / (s_norm + 1e-30)

    c = friction * dt * A_m * link_mask_a                           # [K, M]
    rel_over_s = rel_vel * inv_s.unsqueeze(2)
    g_u3 = -(c.unsqueeze(2) * rel_over_s).sum(dim=1)                # [K, 3]
    h = (rel_vel * r_nx).sum(dim=2)                                  # [K, M]
    g_om = -(c * h * inv_s).sum(dim=1)                               # [K]

    g_alpha = (g_u3 * t0).sum(dim=1)
    g_beta = (g_u3 * t1).sum(dim=1)
    return torch.stack([g_alpha, g_beta, g_om], dim=1)              # [K, 3]


# ---------------------------------------------------------------------------
# Inner Hessians: H_pp (contact + normal-mag) and H_uu (friction)
# ---------------------------------------------------------------------------

def contact_pp_hess(wv: torch.Tensor,
                    p_stack: torch.Tensor,
                    lid_a: torch.Tensor,
                    lid_b: torch.Tensor,
                    vmask: torch.Tensor,
                    ground_h: torch.Tensor,
                    coef: float,
                    x0: float,
                    d0_half: float,
                    eps_n: float = 1e-8) -> torch.Tensor:
    """``∂²(coef · Σ Ψ_hh')/∂(ν, δ)²`` per pair.   Returns ``[K, 4, 4]`` (PSD).

    Reference: ``simulator.py:1742-1783``. Three contributions, summed:
        H_pp_a (A-side):   bh_a · vmask · va_h ⊗ va_h         [K, 4, 4]
        H_pp_b (B-side):   bh_b · vmask · vb_h ⊗ vb_h          [K, 4, 4] (ground or link-link)
        H_pp_n (normal):   [bh_n · n̂n̂ᵀ - bg_n · (I-n̂n̂ᵀ)/‖ν‖]  → [K, :3, :3] block only

    Final H_pp = coef · (H_pp_a + H_pp_b + H_pp_n), symmetrised.
    """
    dev, dty = wv.device, wv.dtype
    K, max_M = p_stack.shape[0], wv.shape[1]
    if K == 0:
        return torch.zeros(0, 4, 4, device=dev, dtype=dty)

    is_ground = lid_b < 0
    gnd_idx = is_ground.nonzero(as_tuple=True)[0]
    lnk_idx = (~is_ground).nonzero(as_tuple=True)[0]

    va = wv[lid_a]
    link_mask_a = vmask[lid_a]
    ones_KM1 = torch.ones(K, max_M, 1, device=dev, dtype=dty)
    va_h = torch.cat([va, ones_KM1], dim=2)
    d_a = -torch.einsum('kmi,ki->km', va_h, p_stack)
    _, _, bh_a = barrier_eval(d_a, x0, d0_half, mode='log')
    bh_m_a = bh_a * link_mask_a                                    # [K, M]
    H_pp_a = torch.einsum('km,kmi,kmj->kij', bh_m_a, va_h, va_h)   # [K, 4, 4]

    H_pp_b = torch.zeros(K, 4, 4, device=dev, dtype=dty)
    if gnd_idx.numel() > 0:
        p_gnd = p_stack[gnd_idx]
        d_b_gnd = torch.einsum('gi,ki->kg', ground_h, p_gnd)        # [K_gnd, Mg]
        _, _, bh_b_gnd = barrier_eval(d_b_gnd, x0, d0_half, mode='log')
        H_pp_b[gnd_idx] = torch.einsum(
            'kg,gi,gj->kij', bh_b_gnd, ground_h, ground_h)

    if lnk_idx.numel() > 0:
        lid_b_lnk = lid_b[lnk_idx]
        mask_b_lnk = vmask[lid_b_lnk]
        vb_next_lnk = wv[lid_b_lnk]
        ones_lnk = torch.ones(lnk_idx.numel(), max_M, 1, device=dev, dtype=dty)
        vb_h_lnk = torch.cat([vb_next_lnk, ones_lnk], dim=2)
        d_b_lnk = torch.einsum('kmi,ki->km', vb_h_lnk, p_stack[lnk_idx])
        _, _, bh_b_lnk = barrier_eval(d_b_lnk, x0, d0_half, mode='log')
        bh_b_m = bh_b_lnk * mask_b_lnk
        H_pp_b[lnk_idx] = torch.einsum(
            'km,kmi,kmj->kij', bh_b_m, vb_h_lnk, vb_h_lnk)

    # Normal-magnitude term: -log(1 - ‖ν‖). Affects only [:3, :3] sub-block.
    p3 = p_stack[:, :3]
    np3 = torch.norm(p3, dim=1, keepdim=True) + eps_n               # [K, 1]
    n_hat_p = p3 / np3                                              # [K, 3]
    s_p = 1.0 - np3.squeeze(1)
    _, bg_n, bh_n = barrier_eval(s_p, x0, mode='log')
    nn_p = n_hat_p.unsqueeze(2) * n_hat_p.unsqueeze(1)              # [K, 3, 3]
    eye3 = torch.eye(3, device=dev, dtype=dty)
    I_nn_p = eye3.unsqueeze(0) - nn_p
    H_pp_n33 = (bh_n.unsqueeze(1).unsqueeze(2) * nn_p
                - bg_n.unsqueeze(1).unsqueeze(2) * I_nn_p / np3.unsqueeze(2))
    H_pp_n = torch.zeros(K, 4, 4, device=dev, dtype=dty)
    H_pp_n[:, :3, :3] = H_pp_n33

    H_pp = coef * (H_pp_a + H_pp_b + H_pp_n)
    return 0.5 * (H_pp + H_pp.transpose(1, 2))


def damping_uu_hess(pn_stack: torch.Tensor,
                    u_stack: torch.Tensor,
                    wv_next: torch.Tensor,
                    wv_curr: torch.Tensor,
                    lid_a: torch.Tensor,
                    lid_b: torch.Tensor,
                    lnk_idx: torch.Tensor,
                    vmask: torch.Tensor,
                    friction: float,
                    coef: float,
                    x0: float,
                    d0_half: float,
                    dt: float,
                    eps_n: float = 1e-8,
                    eps_s: float = 1e-4) -> torch.Tensor:
    """``∂²(η · Σ D_hh')/∂u_unified²`` per pair.   Returns ``[K, 3, 3]`` (PSD).

    Reference: ``simulator.py:1820-1841`` + ``_friction_reduce_H``.

    Builds the 4×4 Hessian in ``(u_x, u_y, u_z, ω)`` basis from three blocks:
        H_uu_33 = α·I − VᵀV,     α = Σ_v c/s,   V_mr = √(c/s³)·rel_vel_mr
        col      = Σ_v c · [r_nx/s − h·rel_vel/s³]                (u↔ω coupling)
        H_ww     = Σ_v c · [‖r_nx‖²/s − h²/s³]                    (ω diagonal)
    where ``c = friction · dt · A_m · vmask``, ``h = rel_vel · r_nx``.
    Then reduces to 3×3 in (α, β, ω) basis via ``Jᵀ H_4 J``,
    ``J = [t0; t1; 0; 0; 0; 0; 1]`` (the same lift used by ``damping_u_grad``).
    """
    dev, dty = wv_next.device, wv_next.dtype
    K = pn_stack.shape[0]
    if K == 0:
        return torch.zeros(0, 3, 3, device=dev, dtype=dty)

    max_M = wv_next.shape[1]
    va_next = wv_next[lid_a]
    va_curr = wv_curr[lid_a]
    link_mask_a = vmask[lid_a]
    ones_KM1 = torch.ones(K, max_M, 1, device=dev, dtype=dty)

    n_vecs = pn_stack[:, :3]
    norm_n = torch.norm(n_vecs, dim=1, keepdim=True) + eps_n
    n_hat = n_vecs / norm_n
    eye3 = torch.eye(3, device=dev, dtype=dty)
    Proj = eye3.unsqueeze(0) - n_hat.unsqueeze(2) * n_hat.unsqueeze(1)
    u_xyz, omega_stack, t0, t1 = _unified_u_to_xyz_omega(u_stack, n_hat)

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
    _, bg_fn, _ = barrier_eval(d_fn, x0, d0_half, mode='log')
    pn3 = pn_stack[:, :3]
    f_vec = coef * bg_fn.unsqueeze(2) * pn3.unsqueeze(1)
    fn_sq = f_vec.pow(2).sum(dim=2)
    A_m = torch.sqrt(fn_sq + eps_s) - (eps_s ** 0.5)

    s_norm = torch.sqrt(rel_vel.pow(2).sum(dim=2) + eps_s)
    inv_s = 1.0 / (s_norm + 1e-30)
    inv_s3 = inv_s.pow(3)
    c = friction * dt * A_m * link_mask_a                           # [K, M]

    alpha_uu = (c * inv_s).sum(dim=1)                               # [K]
    v_w = rel_vel * (c * inv_s3).clamp(min=0).sqrt().unsqueeze(2)   # [K, M, 3]
    H_uu_33 = (alpha_uu.reshape(K, 1, 1) * eye3
               - torch.bmm(v_w.transpose(1, 2), v_w))               # [K, 3, 3]

    h = (rel_vel * r_nx).sum(dim=2)                                  # [K, M]
    t_mix = (r_nx / s_norm.unsqueeze(2)
             - (h / s_norm.pow(2)).unsqueeze(2)
             * rel_vel / s_norm.unsqueeze(2))                        # [K, M, 3]
    col = (c.unsqueeze(2) * t_mix).sum(dim=1)                       # [K, 3]
    rnx_sq = r_nx.pow(2).sum(dim=2)                                  # [K, M]
    H_ww = (c * (rnx_sq / s_norm - h.pow(2) / s_norm.pow(3))).sum(dim=1)  # [K]

    H_uu_4 = torch.zeros(K, 4, 4, device=dev, dtype=dty)
    H_uu_4[:, :3, :3] = 0.5 * (H_uu_33 + H_uu_33.transpose(1, 2))
    H_uu_4[:, :3, 3] = col
    H_uu_4[:, 3, :3] = col
    H_uu_4[:, 3, 3] = H_ww
    H_uu_4 = 0.5 * (H_uu_4 + H_uu_4.transpose(1, 2))

    # Lift J [K, 4, 3]: (α, β, ω) → (u_x, u_y, u_z, ω). Then H_3 = Jᵀ H_4 J.
    J = torch.zeros(K, 4, 3, device=dev, dtype=dty)
    J[:, :3, 0] = t0
    J[:, :3, 1] = t1
    J[:, 3, 2] = 1.0
    return torch.matmul(J.transpose(1, 2), torch.matmul(H_uu_4, J))
