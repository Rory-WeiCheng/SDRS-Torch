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
