"""Stage III physics-aware loss (formulation 3 — CVX friction).

Implements the formulation in `EXTENSION/formulation/stage3_latest.md`:
    L_phys(X, g) = ‖w(X, g) − w^∥(X, f*^∥(X, g))‖²

where w(X, g) is the frictionless residual wrench (gravity + implicit normal
contact, evaluated at q=0) and w^∥(X, f^∥) is the wrench of the explicit
tangential friction forces solved by an inner SOCP layer.

Public API — table of contents:

  Section 1  Kinematics (rotations + FK + vertex→pose chain)
      _skew_batched(v)                                    — utility
      rodrigues(theta)                                    — R(θ) ∈ SO(3)
      rodrigues_with_diffV(theta)                         — R + diffV
      compute_world_vertices(q, x_bar, link_to_body)      — X̃ = R·x̄ + t
      vertex_grad_to_q(g_v, q, x_bar, link_to_body)       — chain g_v → ∇_q

  Section 2  Gravity
      gravity_energy(wv, rho_2d, gravity, vmask)          — Ψ_g
      gravity_grad_q(...)                                 — ∇_q Ψ_g (analytic)

  Section 3  Contact primitives  (separating-plane barrier)
      contact_energy(wv, p_stack, ...)                    — μ · Σ Ψ_hh'
      _contact_energy_per_pair(...)                       — per-pair [K] (internal)
      contact_vertex_grad(...)                            — ∂Ψ_c/∂X̃   [L,M,3]
      contact_zeta_grad(...)                              — ∂Ψ_c/∂(ν,δ) [K,4]
      contact_pp_hess(...)                                — ∂²Ψ_c/∂(ν,δ)² [K,4,4]

  Section 4  Fused contact per-pair compute
      compute_contact_per_pair(p, wv, ...)                — single shared pass
        returns dict {E_per_pair, g_p, H_pp,
                      g_v_a, g_v_b_lnk, g_v_b_env, ...}

  Section 5  Normal-plane LM solve  (formulation 3, §H step 1)
      solve_normal_planes(p_init, wv, ...)                — (n*, d*) per pair

  Section 6  Wrenches  (§B and §D)
      frictionless_wrench(X, contact_data, g, ...)        — w ∈ ℝ^(N,6)
      friction_wrench(X, f^∥, lid_a, link_to_body, ...)   — w^∥ ∈ ℝ^(N,6)

  Section 7  CVX friction layer
      FrictionLayer(...)                                  — SOCP via CVXPYLayers
        forward: f*^∥ = argmin ‖w − w^∥(f)‖²
        backward: KKT-implicit differentiation

  Section 8  Physics-aware loss
      physics_loss(X, ..., g_vec)                         — L = ‖w − w^∥(f*)‖²
      physics_loss_aggregate(X, ..., g_vec_list)          — Σ_g L_phys(X, g)

This module reuses ``simulator.barrier_eval`` (mode='log') for the global log
barrier; everything else is self-contained.

Conventions:
  - q = (θ, t) per body, packed [N, 6] with q[i, :3] = θ_i, q[i, 3:] = t_i.
  - Wrench packs (torque, force) per body, matching q's (θ, t) order.
  - All physics-loss work in formulation 3 evaluates ∇_q at q = 0 (auxiliary
    differential pose; spec §A).  Some primitives are written for general θ
    (e.g. gravity_grad_q) for symmetry with SDRS, but called with q = 0 here.
"""

import torch

from simulator import barrier_eval


# =============================================================================
# Section 1: Kinematics (rotations + FK + vertex→pose chain)
# =============================================================================
#
# The new formulation parameterises rotation as ``R(θ) = exp([θ]_×)`` with
# ``θ ∈ ℝ³`` (axis-angle).  SDRS-Torch's ``Robot.forward_kinematics`` uses
# ZYX Euler for its 'free' joint type (robot.py:328-331); since our scene
# is just N free 6-DOF bodies (each with M_per_body[i] hulls connected by
# fixed joints), we bypass the Robot abstraction and compute world vertices
# directly via Rodrigues.  ``link_to_body[l]`` tells us which body a given
# hull belongs to.
#
# Forward Rodrigues uses the PyTorch3D-style closed form:
#   R = I + sin(t)/t · [θ]_× + (1 − cos t)/t² · [θ]²_×
# ``torch.sinc`` handles sin(t)/t at all t (including t = 0); the (1 − cos)/t²
# term uses a ``torch.where(t² == 0, 1, t²)`` denominator clamp — at t = 0
# the numerator is also 0, so 0/1 = 0 multiplied by [θ]²_× (also 0) → identity.
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


# =============================================================================
# Section 2: Gravity
# =============================================================================
#
# Ψ_g = -Σ_lk m_lk · g · X̃_lk      (Y-up scalar gravity in current API)
#
# In formulation 3 we evaluate at q = 0; the closed-form gravity wrench at
# q = 0 is built directly inside ``frictionless_wrench`` (§ section 6).  The
# more general ``gravity_grad_q`` (any θ) is kept here for symmetry with the
# SDRS reference and for possible reuse outside the q = 0 evaluation point.

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

    Uses ``rodrigues_with_diffV`` so the formula is exact at any θ.  In
    formulation 3 we evaluate at q = 0; the closed form there is preferred
    (see ``frictionless_wrench``), but this function works at general θ.
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
    # (a × b)_y = a_z·b_x − a_x·b_z, so:
    cross_y = (diffV_per_link[..., :, 2:3] * Rx[..., None, :, 0]
               - diffV_per_link[..., :, 0:1] * Rx[..., None, :, 2])
    cross_y = cross_y.squeeze(-1)                                            # [L, 3, M]
    contrib_per_link = -gravity * (cross_y * w.unsqueeze(1)).sum(dim=-1)    # [L, 3]
    grad_theta = torch.zeros(N, 3, device=dev, dtype=dty).scatter_add_(
        0, link_to_body.unsqueeze(-1).expand(-1, 3), contrib_per_link)

    return torch.cat([grad_theta, grad_t], dim=-1)


# =============================================================================
# Section 3: Contact primitives  (separating-plane barrier)
# =============================================================================
#
# Per pair (h, h'), the barrier potential is
#     Ψ_hh'(q, X, n, d) = -log(1 − ‖n‖)
#                         + Σ_k -log(n^T X̃_hk + d)
#                         + Σ_k' -log(-n^T X̃_h'k' − d).
# We expose three primitive views:
#     contact_energy           — scalar μ · Σ Ψ_hh'
#     contact_zeta_grad        — per-pair ∂Ψ_c/∂(ν, δ)   [K, 4]
#     contact_pp_hess          — per-pair ∂²Ψ_c/∂(ν, δ)² [K, 4, 4]
#     contact_vertex_grad      — per-link ∂Ψ_c/∂X̃        [L, M, 3]
# plus a per-pair-energy version ``_contact_energy_per_pair`` used by the LM.
# All match the SDRS reference (simulator.py:632-661 for energy, 1181-1197
# for vertex grad, 1742-1783 for the H_pp).  The fused per-pair compute in
# the next section avoids the redundant intermediate work these standalone
# primitives do; they are kept for tests and for occasional standalone use.

def contact_energy(wv: torch.Tensor,
                   p_stack: torch.Tensor,
                   lid_a: torch.Tensor,
                   lid_b: torch.Tensor,
                   vmask: torch.Tensor,
                   env_h: torch.Tensor,
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
        lid_b:    [K] long, B-side link index per pair (-1 = static env).
        vmask:    [L, M] {0,1} float vertex mask.
        env_h: [M_env, 4] homogeneous static-environment vertices.
        coef:     barrier scaling μ (= ``self.coef_barrier``).
        x0, d0_half: barrier knee + half-offset (passed through).

    Returns:
        E_contact: scalar.
    """
    dev, dty = wv.device, wv.dtype
    K, max_M = p_stack.shape[0], wv.shape[1]
    is_env = lid_b < 0
    env_idx = is_env.nonzero(as_tuple=True)[0]
    lnk_idx = (~is_env).nonzero(as_tuple=True)[0]

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

    # 5b. B-side static environment   (simulator.py:644-648)
    if env_idx.numel() > 0:
        p_env = p_stack[env_idx]
        d_b_env = torch.einsum('gi,ki->kg', env_h, p_env)
        val_b_env, _, _ = barrier_eval(d_b_env, x0, d0_half, mode='log')
        E = E + coef * val_b_env.sum()

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


def _contact_energy_per_pair(wv: torch.Tensor,
                             p_stack: torch.Tensor,
                             lid_a: torch.Tensor,
                             lid_b: torch.Tensor,
                             vmask: torch.Tensor,
                             env_h: torch.Tensor,
                             coef: float,
                             x0: float,
                             d0_half: float) -> torch.Tensor:
    """Per-pair contact-barrier potential ``coef · Ψ_hh'``.   Returns ``[K]``.

    Same math as ``contact_energy`` but exposes the per-pair contribution
    instead of summing.  Internal — used as a regression reference for the
    fused per-pair compute and (rarely) by stand-alone callers needing the
    per-pair breakdown.
    """
    dev, dty = wv.device, wv.dtype
    K, max_M = p_stack.shape[0], wv.shape[1]
    if K == 0:
        return torch.zeros(0, device=dev, dtype=dty)

    is_env = lid_b < 0
    env_idx = is_env.nonzero(as_tuple=True)[0]
    lnk_idx = (~is_env).nonzero(as_tuple=True)[0]

    va = wv[lid_a]
    link_mask_a = vmask[lid_a]
    ones_KM1 = torch.ones(K, max_M, 1, device=dev, dtype=dty)
    va_h = torch.cat([va, ones_KM1], dim=2)
    d_a = -torch.einsum('kmi,ki->km', va_h, p_stack)
    val_a, _, _ = barrier_eval(d_a, x0, d0_half, mode='log')
    val_a = val_a * link_mask_a
    norm_p = torch.norm(p_stack[:, :3], dim=1)
    val_n, _, _ = barrier_eval(1.0 - norm_p, x0, mode='log')
    E_k = val_n + val_a.sum(dim=1)                                  # [K]

    if env_idx.numel() > 0:
        p_env = p_stack[env_idx]
        d_b_env = torch.einsum('gi,ki->kg', env_h, p_env)
        val_b_env, _, _ = barrier_eval(d_b_env, x0, d0_half, mode='log')
        E_k = E_k.index_add(0, env_idx, val_b_env.sum(dim=1))

    if lnk_idx.numel() > 0:
        lid_b_lnk = lid_b[lnk_idx]
        p_lnk = p_stack[lnk_idx]
        vb = wv[lid_b_lnk]
        mask_b = vmask[lid_b_lnk]
        ones_lnk = torch.ones(lnk_idx.numel(), max_M, 1, device=dev, dtype=dty)
        vb_h = torch.cat([vb, ones_lnk], dim=2)
        d_b = torch.einsum('kmi,ki->km', vb_h, p_lnk)
        val_b, _, _ = barrier_eval(d_b, x0, d0_half, mode='log')
        val_b = val_b * mask_b
        E_k = E_k.index_add(0, lnk_idx, val_b.sum(dim=1))

    return coef * E_k


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
        B-side  (static env):     no contribution (env vertices are static).
        Normal-magnitude term ``-log(1−‖ν‖)`` does not depend on X̃.
    """
    dev, dty = wv.device, wv.dtype
    K, max_M = p_stack.shape[0], wv.shape[1]
    L = wv.shape[0]
    is_env = lid_b < 0
    lnk_idx = (~is_env).nonzero(as_tuple=True)[0]

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
                      env_h: torch.Tensor,
                      coef: float,
                      x0: float,
                      d0_half: float) -> torch.Tensor:
    """``∂(coef · Σ Ψ_hh')/∂(ν_p, δ_p)`` per pair.   Returns ``[K, 4]`` (ν: 0:3, δ: 3).

    Three contributions per pair (matching contact_energy's three sub-blocks):
        A-side  (5a):    -coef · Σ_k bg_a · vmask · (X̃_ak,  1)            [appears in ∂ν, ∂δ via ∂d_a]
        B-side static-env:    coef · Σ_k bg_b · (X̃_env_k, 1)
        B-side link-link: coef · Σ_k bg_b · vmask · (X̃_bk,  1)
        Normal magnitude: coef · bg_n · ((-ν / ‖ν‖), 0)                    [adds to ∂ν only]
    """
    dev, dty = wv.device, wv.dtype
    K, max_M = p_stack.shape[0], wv.shape[1]
    is_env = lid_b < 0
    env_idx = is_env.nonzero(as_tuple=True)[0]
    lnk_idx = (~is_env).nonzero(as_tuple=True)[0]

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

    if env_idx.numel() > 0:
        p_env = p_stack[env_idx]
        d_b_env = torch.einsum('gi,ki->kg', env_h, p_env)        # [K_env, M_env]
        _, bg_b_env, _ = barrier_eval(d_b_env, x0, d0_half, mode='log')
        # ∂d_b_env/∂(ν, δ) = +env_h.  contribution = +coef · bg · env_h.
        # env_h is shared across pairs; einsum to per-pair sum.
        contrib_env = coef * torch.einsum('kg,gi->ki', bg_b_env, env_h)  # [K_env, 4]
        g_zeta[env_idx] = g_zeta[env_idx] + contrib_env

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


def contact_pp_hess(wv: torch.Tensor,
                    p_stack: torch.Tensor,
                    lid_a: torch.Tensor,
                    lid_b: torch.Tensor,
                    vmask: torch.Tensor,
                    env_h: torch.Tensor,
                    coef: float,
                    x0: float,
                    d0_half: float,
                    eps_n: float = 1e-8) -> torch.Tensor:
    """``∂²(coef · Σ Ψ_hh')/∂(ν, δ)²`` per pair.   Returns ``[K, 4, 4]`` (PSD).

    Reference: ``simulator.py:1742-1783``. Three contributions, summed:
        H_pp_a (A-side):   bh_a · vmask · va_h ⊗ va_h         [K, 4, 4]
        H_pp_b (B-side):   bh_b · vmask · vb_h ⊗ vb_h          [K, 4, 4] (env or link-link)
        H_pp_n (normal):   [bh_n · n̂n̂ᵀ - bg_n · (I-n̂n̂ᵀ)/‖ν‖]  → [K, :3, :3] block only

    Final H_pp = coef · (H_pp_a + H_pp_b + H_pp_n), symmetrised.
    """
    dev, dty = wv.device, wv.dtype
    K, max_M = p_stack.shape[0], wv.shape[1]
    if K == 0:
        return torch.zeros(0, 4, 4, device=dev, dtype=dty)

    is_env = lid_b < 0
    env_idx = is_env.nonzero(as_tuple=True)[0]
    lnk_idx = (~is_env).nonzero(as_tuple=True)[0]

    va = wv[lid_a]
    link_mask_a = vmask[lid_a]
    ones_KM1 = torch.ones(K, max_M, 1, device=dev, dtype=dty)
    va_h = torch.cat([va, ones_KM1], dim=2)
    d_a = -torch.einsum('kmi,ki->km', va_h, p_stack)
    _, _, bh_a = barrier_eval(d_a, x0, d0_half, mode='log')
    bh_m_a = bh_a * link_mask_a                                    # [K, M]
    H_pp_a = torch.einsum('km,kmi,kmj->kij', bh_m_a, va_h, va_h)   # [K, 4, 4]

    H_pp_b = torch.zeros(K, 4, 4, device=dev, dtype=dty)
    if env_idx.numel() > 0:
        p_env = p_stack[env_idx]
        d_b_env = torch.einsum('gi,ki->kg', env_h, p_env)        # [K_env, M_env]
        _, _, bh_b_env = barrier_eval(d_b_env, x0, d0_half, mode='log')
        H_pp_b[env_idx] = torch.einsum(
            'kg,gi,gj->kij', bh_b_env, env_h, env_h)

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


# =============================================================================
# Section 4: Fused contact per-pair compute
# =============================================================================
#
# Mirrors SDRS' ``_compute_manifold_hessians`` (simulator.py:1702-1914) —
# contact half only — so that all per-pair contact quantities share the
# barrier intermediates ``va_h, d_a, bg_*, bh_*, n_hat_p, np3`` and avoid
# redundant ``barrier_eval`` calls.
#
# Used by:
#   solve_normal_planes      →  E_per_pair, g_p, H_pp
#   frictionless_wrench      →  g_v_a, g_v_b_lnk            (chain to body wrench)
#   solve_friction_forces    →  ‖g_v_*‖   = per-vertex      (Coulomb cone-bound RHS,
#                                normal-force magnitude       i.e. η · F_normal) [Sec 7]
#
# Physical interpretation of g_v_* (naming follows SDRS' ``g_v_contact_*``):
#   g_v_a [K, M, 3]      = ∂Ψ_hh' / ∂X̃_{hk}    (A-side dynamic vertex k of pair k_pair)
#   g_v_b_lnk [...]      = ∂Ψ_hh' / ∂X̃_{h'k'}  (B-side dynamic vertex, link-link only)
#   g_v_b_env [...]      = ∂Ψ_hh' / ∂X̃_{env_k'} (B-side static-env vertex, env pairs only)
# Each is a 3-vector aligned with the plane normal ν_k (since the contact
# barrier only feels the normal direction).  Its magnitude equals the
# per-vertex contact normal-force magnitude — the contact barrier acts as
# Coulomb's "normal force" generator, so ‖g_v‖ = F_normal at that vertex.
# Hence the friction cone bound  ‖f^∥‖ ≤ η · ‖g_v‖  is the standard
# Coulomb cone with the per-vertex normal force as the scale.

def compute_contact_per_pair(p_stack: torch.Tensor,
                             wv: torch.Tensor,
                             lid_a: torch.Tensor,
                             lid_b: torch.Tensor,
                             vmask: torch.Tensor,
                             env_h: torch.Tensor,
                             coef: float,
                             x0: float,
                             d0_half: float,
                             eps_n: float = 1e-8,
                             need_derivs: bool = True) -> dict:
    """Single fused pass over all contact pairs.

    Args:
        p_stack:  [K, 4]  separating-plane params (ν, δ).
        wv:       [L, M, 3]  world vertices at q = 0 (= X for quasi-static).
        lid_a, lid_b: [K] long pair link indices (lid_b = -1 → static env).
        vmask:    [L, M] {0, 1} float vertex mask.
        env_h: [M_env, 4] homogeneous static-environment vertices.
        coef:     barrier weight μ.
        x0, d0_half: barrier knee + half-offset.
        eps_n:    small protect for the I_nn_p / np3 division in H_pp.
        need_derivs: if False, skip g_p / H_pp / g_v assembly (only E_per_pair).

    Returns dict with always-present key ``E_per_pair`` [K] and (if
    ``need_derivs``):
        g_p             [K, 4]
        H_pp            [K, 4, 4]
        g_v_a           [K, M, 3]                 A-side per-pair vertex grad
        g_v_b_lnk       [K_lnk, M, 3]    | None   B-side, link-link pairs
        g_v_b_env       [K_env, M_env, 3]| None   B-side, static-env pairs
        lnk_idx_local   [K_lnk]                   indices of link-link pairs
        lid_b_lnk       [K_lnk]                   B-side link IDs (link-link)
        env_idx         [K_env]                   indices of static-env pairs

    Convention (matches existing primitives + SDRS energy@638 / Hessian@1772):
        E, g_p use bare ‖p3‖.   H_pp uses (‖p3‖ + eps_n) — eps_n protects
        only the I_nn_p / np3 division, not the barrier argument.  This
        requires two ``barrier_eval`` calls on the val_n term.
    """
    dev, dty = wv.device, wv.dtype
    K, max_M = p_stack.shape[0], wv.shape[1]
    if K == 0:
        out = {'E_per_pair': torch.zeros(0, device=dev, dtype=dty)}
        if need_derivs:
            out.update({
                'g_p':            torch.zeros(0, 4, device=dev, dtype=dty),
                'H_pp':           torch.zeros(0, 4, 4, device=dev, dtype=dty),
                'g_v_a':          torch.zeros(0, max_M, 3, device=dev, dtype=dty),
                'g_v_b_lnk':      None,
                'g_v_b_env':      None,
                'lnk_idx_local':  torch.zeros(0, dtype=torch.long, device=dev),
                'lid_b_lnk':      torch.zeros(0, dtype=torch.long, device=dev),
                'env_idx':        torch.zeros(0, dtype=torch.long, device=dev),
            })
        return out

    is_env = lid_b < 0
    env_idx = is_env.nonzero(as_tuple=True)[0]
    lnk_idx_local = (~is_env).nonzero(as_tuple=True)[0]
    eye3 = torch.eye(3, device=dev, dtype=dty)

    # ---- A-side shared intermediates ----
    va = wv[lid_a]                                                  # [K, M, 3]
    link_mask_a = vmask[lid_a]                                      # [K, M]
    ones_KM1 = torch.ones(K, max_M, 1, device=dev, dtype=dty)
    p3 = p_stack[:, :3]                                             # [K, 3]
    va_h = torch.cat([va, ones_KM1], dim=2)                         # [K, M, 4]
    d_a = -torch.einsum('kmi,ki->km', va_h, p_stack)                # [K, M]
    val_a, bg_a, bh_a = barrier_eval(d_a, x0, d0_half, mode='log')
    val_a = val_a * link_mask_a
    bg_m = bg_a * link_mask_a                                       # [K, M]
    bh_m = bh_a * link_mask_a                                       # [K, M]

    # Normal-magnitude term  -log(1 − ‖ν‖):  bare for E/g, eps for H.
    norm_p_bare = torch.norm(p3, dim=1, keepdim=True)               # [K, 1]
    np3 = norm_p_bare + eps_n                                       # [K, 1]
    n_hat_p = p3 / np3                                              # [K, 3]
    val_n, bg_n_bare, _ = barrier_eval(
        1.0 - norm_p_bare.squeeze(1), x0, mode='log')               # for E, g_p
    bg_n_eps = bh_n = None
    if need_derivs:
        _, bg_n_eps, bh_n = barrier_eval(
            1.0 - np3.squeeze(1), x0, mode='log')                   # for H_pp

    # ---- B-side static-env ----
    bg_b_env = bh_b_env = val_b_env = None
    if env_idx.numel() > 0:
        p_env = p_stack[env_idx]
        d_b_env = torch.einsum('gi,ki->kg', env_h, p_env)        # [K_env, M_env]
        val_b_env, bg_b_env, bh_b_env = barrier_eval(
            d_b_env, x0, d0_half, mode='log')

    # ---- B-side link-link ----
    vb_h_lnk = bg_b_lnk = bh_b_lnk = lid_b_lnk = val_b_lnk = None
    if lnk_idx_local.numel() > 0:
        lid_b_lnk = lid_b[lnk_idx_local]
        p_lnk = p_stack[lnk_idx_local]
        vb = wv[lid_b_lnk]                                          # [K_lnk, M, 3]
        mask_b = vmask[lid_b_lnk]
        ones_lnk = torch.ones(lnk_idx_local.numel(), max_M, 1,
                              device=dev, dtype=dty)
        vb_h_lnk = torch.cat([vb, ones_lnk], dim=2)                 # [K_lnk, M, 4]
        d_b_lnk = torch.einsum('kmi,ki->km', vb_h_lnk, p_lnk)
        val_b_lnk_raw, bg_b_lnk_raw, bh_b_lnk_raw = barrier_eval(
            d_b_lnk, x0, d0_half, mode='log')
        val_b_lnk = val_b_lnk_raw * mask_b
        bg_b_lnk = bg_b_lnk_raw * mask_b
        bh_b_lnk = bh_b_lnk_raw * mask_b

    # ---- E_per_pair ----
    E = coef * (val_n + val_a.sum(dim=1))
    if env_idx.numel() > 0:
        E = E.index_add(0, env_idx, coef * val_b_env.sum(dim=1))
    if lnk_idx_local.numel() > 0:
        E = E.index_add(0, lnk_idx_local, coef * val_b_lnk.sum(dim=1))

    out = {'E_per_pair': E}
    if not need_derivs:
        return out

    # ---- g_p ----
    g_p_a = -coef * torch.einsum('km,kmi->ki', bg_m, va_h)          # [K, 4]
    g_p_b = torch.zeros(K, 4, device=dev, dtype=dty)
    if env_idx.numel() > 0:
        g_p_b[env_idx] = coef * torch.einsum('kg,gi->ki', bg_b_env, env_h)
    if lnk_idx_local.numel() > 0:
        g_p_b[lnk_idx_local] = coef * torch.einsum(
            'km,kmi->ki', bg_b_lnk, vb_h_lnk)
    g_p_n = torch.zeros(K, 4, device=dev, dtype=dty)
    norm_p_safe = norm_p_bare.clamp(min=1e-30)
    g_p_n[:, :3] = coef * bg_n_bare.unsqueeze(1) * (-p3 / norm_p_safe)
    g_p = g_p_a + g_p_b + g_p_n

    # ---- H_pp ----
    H_pp_a = torch.einsum('km,kmi,kmj->kij', bh_m, va_h, va_h)
    H_pp_b = torch.zeros(K, 4, 4, device=dev, dtype=dty)
    if env_idx.numel() > 0:
        H_pp_b[env_idx] = torch.einsum(
            'kg,gi,gj->kij', bh_b_env, env_h, env_h)
    if lnk_idx_local.numel() > 0:
        H_pp_b[lnk_idx_local] = torch.einsum(
            'km,kmi,kmj->kij', bh_b_lnk, vb_h_lnk, vb_h_lnk)
    nn_p = n_hat_p.unsqueeze(2) * n_hat_p.unsqueeze(1)
    I_nn_p = eye3.unsqueeze(0) - nn_p
    H_pp_n33 = (bh_n.unsqueeze(1).unsqueeze(2) * nn_p
                - bg_n_eps.unsqueeze(1).unsqueeze(2)
                  * I_nn_p / np3.unsqueeze(2))
    H_pp_n = torch.zeros(K, 4, 4, device=dev, dtype=dty)
    H_pp_n[:, :3, :3] = H_pp_n33
    H_pp = coef * (H_pp_a + H_pp_b + H_pp_n)
    H_pp = 0.5 * (H_pp + H_pp.transpose(1, 2))

    # ---- Per-pair vertex gradients (= per-vertex contact normal-force vectors) ----
    # See the section-4 docstring above for the physical interpretation:
    # each g_v_* is a 3-vector along the plane normal ν_k whose magnitude
    # equals the contact normal-force magnitude at that vertex.  We use them
    # for the q-side wrench chain (where applicable) and as the Coulomb
    # cone-bound RHS  η·‖g_v‖  for the corresponding friction variables.
    #
    #   g_v_a     [K, M, 3]              A-side dynamic           — always present
    #   g_v_b_lnk [K_lnk, M, 3] | None   B-side dynamic           — link-link only
    #   g_v_b_env [K_env, M_env, 3]|None B-side static env        — env pairs only
    #
    # Naming follows SDRS' ``g_v_contact_*`` (simulator.py:1583, 1601).
    g_v_a = -coef * bg_m.unsqueeze(2) * p3.unsqueeze(1)             # [K, M, 3]
    g_v_b_lnk = None
    if lnk_idx_local.numel() > 0:
        p3_lnk = p3[lnk_idx_local]
        g_v_b_lnk = (coef * bg_b_lnk.unsqueeze(2)
                     * p3_lnk.unsqueeze(1))                          # [K_lnk, M, 3]
    g_v_b_env = None
    if env_idx.numel() > 0:
        p3_env = p3[env_idx]
        g_v_b_env = (coef * bg_b_env.unsqueeze(2)
                     * p3_env.unsqueeze(1))                          # [K_env, M_env, 3]

    out.update({
        'g_p':            g_p,
        'H_pp':           H_pp,
        'g_v_a':          g_v_a,
        'g_v_b_lnk':      g_v_b_lnk,
        'g_v_b_env':      g_v_b_env,
        'lnk_idx_local':  lnk_idx_local,
        'lid_b_lnk':      (lid_b_lnk if lid_b_lnk is not None
                           else torch.zeros(0, dtype=torch.long, device=dev)),
        'env_idx':        env_idx,
    })
    return out


# =============================================================================
# Section 5: Normal-plane LM solve  (formulation 3, §H step 1)
# =============================================================================
#
# Solve   (n*, d*) = argmin P^⊥(q=0, X, n, d)   per pair, block-diagonal
# across K contact pairs.  Each pair's p = (n, d) ∈ ℝ^4 is independent (no
# cross-pair coupling and no q-block since q is fixed at 0).  Per-pair
# Nielsen damped Newton — same accept/reject pattern as SDRS' _solve_lm
# (simulator.py:1997-2130), specialised to a 4-D inner variable per pair
# (no Schur step needed, no u block).

def solve_normal_planes(p_init: torch.Tensor,
                        wv: torch.Tensor,
                        lid_a: torch.Tensor,
                        lid_b: torch.Tensor,
                        vmask: torch.Tensor,
                        env_h: torch.Tensor,
                        coef: float,
                        x0: float,
                        d0_half: float,
                        max_iter: int = 50,
                        gtol: float = 1e-7,
                        alpha_init: float = 1e-3,
                        alpha_min: float = 1e-12,
                        alpha_max: float = 1e10,
                        verbose: bool = False):
    """Per-pair Nielsen LM for separating planes.   Returns ``(p_star, info)``.

    Args:
        p_init:   [K, 4]  initial (ν, δ).
        wv:       [L, M, 3]  world vertices at q = 0 (= X for quasi-static).
        lid_a, lid_b: [K] long pair link indices (lid_b = -1 → static env pair).
        vmask:    [L, M] {0, 1} float vertex mask.
        env_h: [M_env, 4] homogeneous static-environment vertices.
        coef:     barrier weight μ.
        x0, d0_half: barrier knee + half-offset (passed to ``barrier_eval``).
        max_iter, gtol, alpha_*, verbose: LM hyperparameters.

    Returns:
        p_star:   [K, 4]   converged separating-plane params.
        info:     dict     {'converged', 'iter', 'g_max'}.
    """
    dev, dty = wv.device, wv.dtype
    K = p_init.shape[0]
    if K == 0:
        return (p_init.clone(),
                {'converged': True, 'iter': 0, 'g_max': 0.0})

    p = p_init.clone()
    alpha = torch.full((K,), alpha_init, device=dev, dtype=dty)
    nu = torch.full((K,), 2.0, device=dev, dtype=dty)
    eye4 = torch.eye(4, device=dev, dtype=dty)

    g_max_final = float('inf')
    info_iter = 0
    for it in range(max_iter):
        # Fused per-pair compute: one shared pass for E, g_p, H_pp.
        data = compute_contact_per_pair(p, wv, lid_a, lid_b, vmask, env_h,
                                        coef, x0, d0_half)
        E    = data['E_per_pair']
        g_p  = data['g_p']
        H_pp = data['H_pp']

        g_max_final = g_p.abs().amax(dim=1).max().item()
        info_iter = it
        if verbose:
            print(f"  [LM {it:3d}] E={E.sum().item():.6e}  "
                  f"max|g_p|={g_max_final:.3e}  "
                  f"alpha=[{alpha.min().item():.2e}, "
                  f"{alpha.max().item():.2e}]")
        if g_max_final < gtol:
            break

        # Damped Newton step.
        H_pp_reg = H_pp + alpha.view(K, 1, 1) * eye4
        try:
            dp = -torch.linalg.solve(H_pp_reg, g_p.unsqueeze(2)).squeeze(2)
        except RuntimeError:
            dp = -1e-2 * g_p

        p_trial = p + dp
        # Trial energy: skip g/H to save compute.
        E_trial = compute_contact_per_pair(
            p_trial, wv, lid_a, lid_b, vmask, env_h,
            coef, x0, d0_half, need_derivs=False)['E_per_pair']

        # Trust-region predicted reduction (per pair).
        Hpp_dp = torch.bmm(H_pp, dp.unsqueeze(2)).squeeze(2)
        pred = (dp * (g_p + 0.5 * Hpp_dp)).sum(dim=1)
        finite_pred = pred.abs() > 1e-30
        finite_E = torch.isfinite(E_trial)
        rho = torch.where(
            finite_pred & finite_E,
            (E_trial - E) / pred.where(finite_pred, torch.ones_like(pred)),
            torch.zeros_like(pred))
        accept = (E_trial < E) & (rho > 0) & finite_pred & finite_E

        # Commit; rejected pairs grow α (Nielsen).
        p = torch.where(accept.unsqueeze(1), p_trial, p)
        rho_clamped = (2.0 * rho - 1.0).clamp(min=-100.0, max=100.0)
        accept_factor = torch.maximum(torch.full_like(rho, 1.0 / 3.0),
                                       1.0 - rho_clamped.pow(3))
        alpha = torch.where(accept, alpha * accept_factor, alpha * nu)
        alpha = alpha.clamp(min=alpha_min, max=alpha_max)
        nu = torch.where(accept, torch.full_like(nu, 2.0), nu * 2.0)
        info_iter = it + 1

    return p, {'converged': g_max_final < gtol,
               'iter': info_iter,
               'g_max': g_max_final}


# =============================================================================
# Section 6: Wrenches  (formulation 3, §B and §D)
# =============================================================================
#
# w(X, g)  = -∇_q [P^g + P^⊥(·, p*)] |_{q=0}      (frictionless residual)
# w^∥(X, f^∥) = Σ_{h: i(h)=i} Σ_{h'} Σ_k G_hk · f^∥_{hk, h'}
#            with G_hk = [[X_hk]_×; I]            (friction wrench map)
#
# Both produce ∈ ℝ^(N, 6) packed as (torque, force) per body, matching the
# q = (θ, t) convention (q[i, :3] = θ_i, q[i, 3:] = t_i).
#
# Closed forms at q = 0 (R = I, diffV = I) — no FK chain through rotation:
#   ∂Ψ/∂t_i = Σ_{l∈i, k} g_v_lk
#   ∂Ψ/∂θ_i = Σ_{l∈i, k} (X_lk × g_v_lk)            [via triple product]
# So a per-vertex "force-side" gradient g_v gives a wrench (torque, force) =
# (-Σ X×g_v, -Σ g_v) per body.

def frictionless_wrench(X: torch.Tensor,
                        contact_data: dict,
                        g: torch.Tensor,
                        link_to_body: torch.Tensor,
                        rho_2d: torch.Tensor,
                        vmask: torch.Tensor,
                        lid_a: torch.Tensor) -> torch.Tensor:
    """w(X, g) = -∇_q [P^g + P^⊥(·, p*)] at q=0.   Returns ``[N, 6]``.

    Args:
        X:            [L, M, 3]  world vertices at q = 0.
        contact_data: dict from ``compute_contact_per_pair(p*, X, ...)``.
                      Reads ``g_v_a, g_v_b_lnk, lid_b_lnk``.
        g:            [3]  gravity vector (e.g., ``[0, -9.81, 0]``).
        link_to_body: [L]  long.
        rho_2d:       [L, M]  mass per vertex.
        vmask:        [L, M] {0, 1} float vertex mask.
        lid_a:        [K]  long, A-side link index per pair.

    Returns:
        w: [N, 6] = (torque, force) per body.
    """
    dev, dty = X.device, X.dtype
    L, max_M, _ = X.shape
    N = int(link_to_body.max().item()) + 1

    # ---- Gravity wrench (closed form at q = 0) ----
    # w_g_force_i = M_i · g,    w_g_torque_i = Σ_{l∈i, k} m·(X × g)
    m = rho_2d * vmask                                              # [L, M]
    body_mass = torch.zeros(N, device=dev, dtype=dty).scatter_add_(
        0, link_to_body, m.sum(dim=1))                              # [N]
    w_force = body_mass.unsqueeze(1) * g.unsqueeze(0)               # [N, 3]
    cross_X_g = torch.cross(X, g.expand_as(X), dim=-1)              # [L, M, 3]
    w_torque_per_link = (m.unsqueeze(2) * cross_X_g).sum(dim=1)     # [L, 3]
    w_torque = torch.zeros(N, 3, device=dev, dtype=dty).scatter_add_(
        0, link_to_body.unsqueeze(-1).expand(-1, 3),
        w_torque_per_link)                                          # [N, 3]

    # ---- Contact wrench: chain per-pair g_v through scatter + (X × g_v) ----
    # Reconstruct per-link g_v by index-add of fused per-pair fragments.
    g_v = torch.zeros(L, max_M, 3, device=dev, dtype=dty)
    g_v.index_add_(0, lid_a, contact_data['g_v_a'])
    if contact_data.get('g_v_b_lnk') is not None:
        g_v.index_add_(0, contact_data['lid_b_lnk'], contact_data['g_v_b_lnk'])
    grad_t_per_link = g_v.sum(dim=1)                                # [L, 3]
    cross_X_gv = torch.cross(X, g_v, dim=-1)                        # [L, M, 3]
    grad_th_per_link = cross_X_gv.sum(dim=1)                        # [L, 3]
    body_idx = link_to_body.unsqueeze(-1).expand(-1, 3)
    grad_t = torch.zeros(N, 3, device=dev, dtype=dty).scatter_add_(
        0, body_idx, grad_t_per_link)
    grad_th = torch.zeros(N, 3, device=dev, dtype=dty).scatter_add_(
        0, body_idx, grad_th_per_link)
    # Wrench = -gradient.
    w_force = w_force - grad_t
    w_torque = w_torque - grad_th

    return torch.cat([w_torque, w_force], dim=-1)                   # [N, 6]


def friction_wrench(X: torch.Tensor,
                    f_par_a: torch.Tensor,
                    lid_a: torch.Tensor,
                    link_to_body: torch.Tensor,
                    f_par_b_lnk: torch.Tensor = None,
                    lid_b_lnk: torch.Tensor = None) -> torch.Tensor:
    """w^∥(X, f^∥) = Σ G_hk · f^∥   per body.   Returns ``[N, 6]``.

    Linear in the per-vertex tangential forces.  G_hk(X) = [[X_hk]_×; I],
    so each vertex contributes (X_hk × f, f) to its body's (torque, force).

    Args:
        X:            [L, M, 3]  world vertices.
        f_par_a:      [K, M, 3]  tangential forces on A-side vertices.
        lid_a:        [K] long.
        link_to_body: [L] long.
        f_par_b_lnk:  [K_lnk, M, 3] or None.  B-side forces (link-link only).
        lid_b_lnk:    [K_lnk] long or None.

    Returns:
        w_par: [N, 6] = (torque, force) per body.
    """
    dev, dty = X.device, X.dtype
    N = int(link_to_body.max().item()) + 1

    # A-side: per-pair sum, scatter to body.
    X_a = X[lid_a]                                                  # [K, M, 3]
    torque_per_pair_a = torch.cross(X_a, f_par_a, dim=-1).sum(dim=1) # [K, 3]
    force_per_pair_a = f_par_a.sum(dim=1)                            # [K, 3]
    body_a_idx = link_to_body[lid_a].unsqueeze(-1).expand(-1, 3)
    w_torque = torch.zeros(N, 3, device=dev, dtype=dty).scatter_add_(
        0, body_a_idx, torque_per_pair_a)
    w_force = torch.zeros(N, 3, device=dev, dtype=dty).scatter_add_(
        0, body_a_idx, force_per_pair_a)

    # B-side (link-link only).
    if f_par_b_lnk is not None and f_par_b_lnk.shape[0] > 0:
        X_b = X[lid_b_lnk]                                          # [K_lnk, M, 3]
        torque_per_pair_b = torch.cross(X_b, f_par_b_lnk, dim=-1).sum(dim=1)
        force_per_pair_b = f_par_b_lnk.sum(dim=1)
        body_b_idx = link_to_body[lid_b_lnk].unsqueeze(-1).expand(-1, 3)
        w_torque.scatter_add_(0, body_b_idx, torque_per_pair_b)
        w_force.scatter_add_(0, body_b_idx, force_per_pair_b)

    return torch.cat([w_torque, w_force], dim=-1)                   # [N, 6]


# =============================================================================
# Section 7: CVX friction layer  (formulation 3, §C-E)
# =============================================================================
#
# Solve, given the frictionless residual wrench ``w(X, g)`` and the contact
# state from ``compute_contact_per_pair``:
#
#     f*^∥(X, g) = argmin_{f}  ‖w(X, g) − w^∥(X, f)‖²
#                  s.t.  tangentiality      n^T f_v   = 0          per vertex
#                        cone (Coulomb)     ‖f_v‖     ≤ η · ‖g_v_v‖  per vertex
#                        force balance      Σ f_a + Σ f_b = 0      per pair
#                        torque balance     n · Σ (X × f) = 0      per pair
#
# Variables (one 3-vec per "friction vertex"):
#   f_a         [K, M, 3]            A-side dynamic vertices       (always)
#   f_b_lnk     [K_lnk, M, 3]        B-side dynamic vertices       (link-link only)
#   f_b_env     [K_env, M_env, 3]    B-side static-env vertices    (env pairs only)
# Stacked into a single CVXPY variable ``f`` of shape ``(F_total, 3)``.
#
# Structure exploited (see discussion in code-review notes):
#   - Constraints decouple per pair (separable feasible set).
#   - Objective couples pairs that share a body (per-body wrench sum).
# Per-component decomposition for scaling is left for a future wrapper;
# this layer is the building block that wrapper would call.
#
# Implementation: single global SOCP via CVXPYLayers.  Topology (K, K_lnk,
# K_env, M, M_env, link_to_body, lid_a, lid_b) is fixed at construction; the
# parametric problem accepts X, n_hat, cone bounds, and w as parameters.

import cvxpy as cp
from cvxpylayers.torch import CvxpyLayer


class FrictionLayer:
    """Differentiable friction-force SOCP.

    Construct once per scene topology, call repeatedly per X-update:

        layer = FrictionLayer(link_to_body, lid_a, lid_b, M=8, M_env=4)
        f_a, f_b_lnk, f_b_env = layer(X, contact_data, w_target, eta=0.3,
                                      env_h=env_h, p_star=p_star)

    The forward pass returns the optimal friction forces; backward propagates
    gradients via CVXPYLayers' KKT-implicit differentiation.
    """

    def __init__(self,
                 link_to_body: torch.Tensor,
                 lid_a: torch.Tensor,
                 lid_b: torch.Tensor,
                 M: int,
                 M_env: int):
        """
        Args:
            link_to_body, lid_a, lid_b, M, M_env: scene topology.

        Forward solve uses Clarabel (Newton-based interior-point) routed via
        cvxpylayers' DIFFCP backward path (KKT-implicit differentiation).
        """
        N = int(link_to_body.max().item()) + 1
        K = lid_a.shape[0]
        is_env = lid_b < 0
        env_idx = is_env.nonzero(as_tuple=True)[0]
        lnk_idx = (~is_env).nonzero(as_tuple=True)[0]
        K_env = env_idx.shape[0]
        K_lnk = lnk_idx.shape[0]

        self.N, self.K, self.M = N, K, M
        self.K_lnk, self.K_env, self.M_env = K_lnk, K_env, M_env
        self.link_to_body = link_to_body
        self.lid_a = lid_a
        self.lid_b = lid_b
        self.env_idx = env_idx
        self.lnk_idx = lnk_idx
        self.lid_b_lnk = lid_b[lnk_idx]
        # Vertex layout:
        #   rows [0,            K*M)              = A-side dynamic, pair k row k*M + m
        #   rows [K*M,         K*M + K_lnk*M)     = link-link B-side, k_lnk row K*M + k_lnk*M + m
        #   rows [K*M + K_lnk*M, F_total)         = env B-side, k_env row K*M + K_lnk*M + k_env*M_env + m'
        self.F_a = K * M
        self.F_b_lnk = K_lnk * M
        self.F_b_env = K_env * M_env
        self.F_total = self.F_a + self.F_b_lnk + self.F_b_env

        # Per-vertex pair index (which pair in [0, K) does this vertex belong to).
        # Lets us scatter per-vertex data → per-pair sums (for force/torque balance).
        # Built once at init, used at solve time and inside the F_balance constant.
        self._vertex_pair = torch.zeros(self.F_total, dtype=torch.long)
        for k in range(K):
            self._vertex_pair[k * M:(k + 1) * M] = k                       # A-side
        for kk in range(K_lnk):
            base = self.F_a + kk * M
            self._vertex_pair[base:base + M] = lnk_idx[kk]                 # link-link B
        for kk in range(K_env):
            base = self.F_a + self.F_b_lnk + kk * M_env
            self._vertex_pair[base:base + M_env] = env_idx[kk]             # env B

        # F_balance is a CONSTANT (depends only on topology, not on parameters).
        # Shape: [3*K, F_total*3].  Per-pair force balance:  Σ_v_in_pair f[v, c] = 0
        # for each c=0..2.  In Fortran-order vec(f), f[v, c] sits at index c*F_total + v.
        F_b_const = torch.zeros(3 * K, self.F_total * 3)
        for k in range(K):
            in_pair = (self._vertex_pair == k)                              # [F_total]
            v_idx = in_pair.nonzero(as_tuple=True)[0]
            for c in range(3):
                F_b_const[3 * k + c, c * self.F_total + v_idx] = 1.0
        self._F_balance_const = F_b_const

        # ---- Build CVXPY problem ----
        f = cp.Variable((self.F_total, 3))

        W           = cp.Parameter((6 * N, self.F_total * 3))
        n_hat_per_v = cp.Parameter((self.F_total, 3))
        cone_bounds = cp.Parameter(self.F_total, nonneg=True)
        T_balance   = cp.Parameter((K, self.F_total * 3))
        w_target    = cp.Parameter(6 * N)

        # Explicit Fortran order: f_vec[c * F_total + v] = f[v, c].  Required by
        # CVXPY 1.7+ to suppress deprecation warning and lock in the convention.
        f_vec = cp.vec(f, order='F')

        constraints = [
            cp.norm(f, 2, axis=1) <= cone_bounds,               # per-vertex SOC
            cp.sum(cp.multiply(n_hat_per_v, f), axis=1) == 0,   # tangentiality
            self._F_balance_const @ f_vec == 0,                  # per-pair force balance
            T_balance @ f_vec == 0,                              # per-pair torque balance
        ]
        objective = cp.Minimize(cp.sum_squares(w_target - W @ f_vec))
        problem = cp.Problem(objective, constraints)
        assert problem.is_dpp(), "friction-layer problem is not DPP-compliant"

        # cvxpylayers >=1.1 default backward path is DIFFCP; the forward cone
        # solver (Clarabel here) is selected per-call via
        # ``solver_args={'solve_method': 'Clarabel', ...}`` in __call__.
        self.layer = CvxpyLayer(
            problem,
            parameters=[W, n_hat_per_v, cone_bounds, T_balance, w_target],
            variables=[f],
        )

    # ---- Build per-call parameter values from PyTorch state ----
    def _build_W(self, X: torch.Tensor, env_h: torch.Tensor) -> torch.Tensor:
        """Wrench-map matrix W [6N, F_total*3] s.t. w^∥ = W @ vec(f).

        env vertices contribute NOTHING to W (env body has no DOF / wrench).
        Fortran-order vec(f): column j = c * F_total + v ↔ f[v, c].
        For each dynamic friction vertex v with body b(v) and world pos X_v:
            Force  part: W[6b + 3 + c,  c·F_total + v] = 1
            Torque part: W[6b + r,      c·F_total + v] = [X_v]_× [r, c]
        """
        dev, dty = X.device, X.dtype
        n_v_dyn = self.F_a + self.F_b_lnk    # only A-side + link-link B (env skipped)

        # Per-dynamic-friction-vertex: body and world-position lookup.
        bodies = torch.empty(n_v_dyn, dtype=torch.long, device=dev)
        X_per_v = torch.empty(n_v_dyn, 3, dtype=dty, device=dev)
        bodies[:self.F_a] = self.link_to_body[self.lid_a].repeat_interleave(self.M)
        X_per_v[:self.F_a] = X[self.lid_a].reshape(-1, 3)
        if self.K_lnk > 0:
            bodies[self.F_a:] = (
                self.link_to_body[self.lid_b_lnk].repeat_interleave(self.M))
            X_per_v[self.F_a:] = X[self.lid_b_lnk].reshape(-1, 3)

        skew_X = _skew_batched(X_per_v)              # [n_v_dyn, 3, 3]
        cs = torch.arange(3, device=dev)
        v_arr = torch.arange(n_v_dyn, device=dev)
        F_total = self.F_total

        W = torch.zeros(6 * self.N, F_total * 3, device=dev, dtype=dty)

        # Force part (constant 1.0):  W[6b + 3 + c, c·F_total + v] = 1.
        rows_f = (6 * bodies)[None, :] + 3 + cs[:, None]              # [3, n_v_dyn]
        cols_f = cs[:, None] * F_total + v_arr[None, :]               # [3, n_v_dyn]
        W[rows_f.flatten(), cols_f.flatten()] = 1.0

        # Torque part (parameter, depends on X_v):
        #   W[6b + r, c·F_total + v] = skew_X[v, r, c]
        # Vectorize over (r, c, v).
        rows_t = (6 * bodies)[None, None, :] + cs[:, None, None]      # [3 (r), 1, n_v_dyn]
        rows_t = rows_t.expand(3, 3, n_v_dyn)                         # [3 (r), 3 (c), n_v_dyn]
        cols_t = (cs[None, :, None] * F_total
                  + v_arr[None, None, :]).expand(3, 3, n_v_dyn)
        # values[r, c, v] = skew_X[v, r, c]  →  permute(1, 2, 0).
        values_t = skew_X.permute(1, 2, 0).contiguous()               # [3, 3, n_v_dyn]
        W[rows_t.flatten(), cols_t.flatten()] = values_t.flatten()
        return W

    def _build_n_hat_per_v(self, n_hat_per_pair: torch.Tensor) -> torch.Tensor:
        """Broadcast n_hat[k] to all vertices in pair k. Shape [F_total, 3]."""
        return n_hat_per_pair[self._vertex_pair]

    def _build_T_balance(self, n_hat_per_pair: torch.Tensor,
                         X: torch.Tensor, env_h: torch.Tensor) -> torch.Tensor:
        """Per-pair torque balance: row k = (n_hat_k × X_v)_c · f[v, c] coeffs.

        Constraint per pair k:   n · Σ_v (X_v × f_v)  = 0
                              =  Σ_v (n × X_v) · f_v          (cyclic identity)
        So row k has coefficients (n_k × X_v)[c] for every (v ∈ pair k, c).
        """
        dev, dty = X.device, X.dtype
        F_total = self.F_total

        # Per-friction-vertex world position X_per_v [F_total, 3] (incl. env).
        X_per_v = torch.empty(F_total, 3, dtype=dty, device=dev)
        X_per_v[:self.F_a] = X[self.lid_a].reshape(-1, 3)
        if self.K_lnk > 0:
            X_per_v[self.F_a:self.F_a + self.F_b_lnk] = X[self.lid_b_lnk].reshape(-1, 3)
        if self.K_env > 0:
            env_3 = env_h[:, :3]                                     # [M_env, 3]
            X_per_v[self.F_a + self.F_b_lnk:] = env_3.repeat(self.K_env, 1)

        # n × X per vertex (per-pair n_hat broadcast via _vertex_pair).
        n_per_v = n_hat_per_pair[self._vertex_pair]                  # [F_total, 3]
        cross_per_v = torch.cross(n_per_v, X_per_v, dim=-1)          # [F_total, 3]

        # T_b[pair(v), c·F_total + v] = cross_per_v[v, c]
        T_b = torch.zeros(self.K, F_total * 3, device=dev, dtype=dty)
        cs = torch.arange(3, device=dev)
        v_arr = torch.arange(F_total, device=dev)
        rows = self._vertex_pair[None, :].expand(3, F_total)         # [3, F_total]
        cols = cs[:, None] * F_total + v_arr[None, :]                # [3, F_total]
        T_b[rows.flatten(), cols.flatten()] = cross_per_v.T.contiguous().flatten()
        return T_b

    def _build_cone_bounds(self, eta: float, contact_data: dict) -> torch.Tensor:
        """Pack η · ‖g_v_*‖ for every friction vertex into [F_total]."""
        cone = torch.zeros(self.F_total,
                           device=contact_data['g_v_a'].device,
                           dtype=contact_data['g_v_a'].dtype)
        cone[:self.F_a] = eta * contact_data['g_v_a'].norm(dim=-1).flatten()
        if self.K_lnk > 0 and contact_data['g_v_b_lnk'] is not None:
            base = self.F_a
            cone[base:base + self.F_b_lnk] = (
                eta * contact_data['g_v_b_lnk'].norm(dim=-1).flatten())
        if self.K_env > 0 and contact_data['g_v_b_env'] is not None:
            base = self.F_a + self.F_b_lnk
            cone[base:base + self.F_b_env] = (
                eta * contact_data['g_v_b_env'].norm(dim=-1).flatten())
        return cone

    def __call__(self,
                 X: torch.Tensor,
                 env_h: torch.Tensor,
                 contact_data: dict,
                 w_target: torch.Tensor,
                 eta: float,
                 p_star: torch.Tensor,
                 solver_args: dict = None):
        """Forward solve.   Returns ``(f_a, f_b_lnk, f_b_env)``.

        Args:
            X:            [L, M, 3]  world vertices.
            env_h:        [M_env, 4] homogeneous env vertices.
            contact_data: dict from ``compute_contact_per_pair`` (with
                          ``need_derivs=True``).
            w_target:     [N, 6] frictionless residual wrench.
            eta:          friction coefficient (Coulomb cone slope).
            p_star:       [K, 4] converged separating-plane params (used to
                          extract per-pair n_hat).
        """
        # n_hat per pair: normalize the first three components of p_star.
        p3 = p_star[:, :3]
        n_hat_per_pair = p3 / p3.norm(dim=-1, keepdim=True).clamp(min=1e-30)

        # Build all parameters.
        W           = self._build_W(X, env_h)
        n_hat_per_v = self._build_n_hat_per_v(n_hat_per_pair)
        cone_bounds = self._build_cone_bounds(eta, contact_data)
        T_balance   = self._build_T_balance(n_hat_per_pair, X, env_h)
        w_flat      = w_target.flatten()

        # Solve via Clarabel (Newton-based interior-point) routed through
        # diffcp.  ``tol_*`` are Clarabel's per-residual tolerances.
        sa = {'solve_method': 'Clarabel',
              'tol_gap_abs': 1e-9, 'tol_gap_rel': 1e-9, 'tol_feas': 1e-9,
              'max_iter': 200, 'verbose': False}
        if solver_args:
            sa.update(solver_args)
        (f,) = self.layer(W, n_hat_per_v, cone_bounds, T_balance, w_flat,
                          solver_args=sa)
        # Reshape into the three friction-variable groups.
        f_a     = f[:self.F_a].reshape(self.K, self.M, 3)
        f_b_lnk = (f[self.F_a:self.F_a + self.F_b_lnk]
                   .reshape(self.K_lnk, self.M, 3)) if self.K_lnk > 0 else None
        f_b_env = (f[self.F_a + self.F_b_lnk:]
                   .reshape(self.K_env, self.M_env, 3)) if self.K_env > 0 else None
        return f_a, f_b_lnk, f_b_env


# =============================================================================
# Section 8: Physics-aware loss  (formulation 3, §F)
# =============================================================================
#
#     L_phys(X, g) = ‖ w(X, g) − w^∥(X, f*^∥(X, g)) ‖²
#
# Wires the upstream pieces into a single end-to-end scalar loss:
#   1. p*           = solve_normal_planes(p_init, X, ...)
#   2. contact_data = compute_contact_per_pair(p*, X, ...)
#   3. w            = frictionless_wrench(X, contact_data, g, ...)
#   4. f*           = friction_layer(X, env_h, contact_data, w, η, p*)
#   5. w^∥          = friction_wrench(X, f*, ...)
#   6. L            = ‖w − w^∥‖²
#
# All steps are differentiable in X via PyTorch autograd:
#   - normal-plane LM: autograd through ~6 iterations of damped Newton
#     (torch.linalg.solve, torch.where are all autograd-supported);
#   - contact_data, wrenches: pure analytic primitives;
#   - friction layer: KKT-implicit differentiation built into CVXPYLayers.
# So ``loss.backward()`` yields ∂L/∂X correctly out of the box; the
# Implicit-Function-Theorem optimisation of the inner solves is left as a
# future speed-up, not a correctness requirement.

def physics_loss(X: torch.Tensor,
                 env_h: torch.Tensor,
                 lid_a: torch.Tensor,
                 lid_b: torch.Tensor,
                 link_to_body: torch.Tensor,
                 vmask: torch.Tensor,
                 rho_2d: torch.Tensor,
                 g_vec: torch.Tensor,
                 eta: float,
                 friction_layer: 'FrictionLayer',
                 coef: float,
                 x0: float,
                 d0_half: float,
                 p_init: torch.Tensor,
                 normal_solve_kwargs: dict = None,
                 friction_solver_args: dict = None):
    """End-to-end physics-aware loss for a single gravity vector.

    Args:
        X:               [L, M, 3]  world vertices.
        env_h:           [M_env, 4] homogeneous static-environment vertices.
        lid_a, lid_b:    [K]        per-pair link indices (lid_b = -1 → env).
        link_to_body:    [L]        link → body index.
        vmask, rho_2d:   [L, M]     vertex mask + per-vertex mass.
        g_vec:           [3]        gravity vector.
        eta:             float      Coulomb friction coefficient.
        friction_layer:  FrictionLayer matching the topology.
        coef, x0, d0_half: barrier params.
        p_init:          [K, 4]     warm-start for the normal-plane LM.
        normal_solve_kwargs: dict, forwarded to ``solve_normal_planes``.

    Returns:
        loss: scalar torch.Tensor, autograd-connected to X.
        info: dict with intermediate states (``p_star``, ``w``, ``w_par``,
              ``residual``, ``f_a``, ``f_b_lnk``, ``f_b_env``,
              ``normal_solve_info``) — useful for diagnostics + tests.
    """
    kw = dict(normal_solve_kwargs or {})

    # 1. Normal-plane inner solve.
    p_star, ns_info = solve_normal_planes(
        p_init, X, lid_a, lid_b, vmask, env_h, coef, x0, d0_half, **kw)

    # 2. Per-pair contact data at the converged p*.
    contact_data = compute_contact_per_pair(
        p_star, X, lid_a, lid_b, vmask, env_h, coef, x0, d0_half)

    # 3. Frictionless residual wrench  w(X, g).
    w = frictionless_wrench(X, contact_data, g_vec,
                            link_to_body, rho_2d, vmask, lid_a)

    # 4. Friction SOCP  →  f*^∥(X, g).
    f_a, f_b_lnk, f_b_env = friction_layer(
        X, env_h, contact_data, w, eta, p_star,
        solver_args=friction_solver_args)

    # 5. Friction wrench  w^∥(X, f*).
    w_par = friction_wrench(
        X, f_a, lid_a, link_to_body,
        f_par_b_lnk=f_b_lnk, lid_b_lnk=friction_layer.lid_b_lnk)

    # 6. Loss  =  ‖w − w^∥‖².
    residual = w - w_par
    loss = (residual * residual).sum()

    info = {
        'p_star': p_star,
        'normal_solve_info': ns_info,
        'contact_data': contact_data,
        'w': w,
        'w_par': w_par,
        'residual': residual,
        'f_a': f_a,
        'f_b_lnk': f_b_lnk,
        'f_b_env': f_b_env,
    }
    return loss, info


def physics_loss_aggregate(X: torch.Tensor,
                           env_h: torch.Tensor,
                           lid_a: torch.Tensor,
                           lid_b: torch.Tensor,
                           link_to_body: torch.Tensor,
                           vmask: torch.Tensor,
                           rho_2d: torch.Tensor,
                           g_vec_list,
                           eta: float,
                           friction_layer: 'FrictionLayer',
                           coef: float,
                           x0: float,
                           d0_half: float,
                           p_init: torch.Tensor,
                           normal_solve_kwargs: dict = None):
    """Sum ``physics_loss`` over a finite set of gravity directions  (spec §F).

    L_phys(X) = Σ_{g ∈ G} L_phys(X, g)
    """
    total = X.new_zeros(())
    infos = []
    for g_vec in g_vec_list:
        loss_g, info_g = physics_loss(
            X, env_h, lid_a, lid_b, link_to_body, vmask, rho_2d,
            g_vec, eta, friction_layer, coef, x0, d0_half, p_init,
            normal_solve_kwargs=normal_solve_kwargs)
        total = total + loss_g
        infos.append(info_g)
    return total, infos
