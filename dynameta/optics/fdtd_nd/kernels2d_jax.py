"""2D-TE normal-incidence JAX kernel (lax.scan; differentiable).

Split from the former monolithic fdtd_nd.py; see the package __init__ docstring
for conventions. Bodies are verbatim from the original module.
"""
from __future__ import annotations

from dynameta.constants import EPS0, MU0



def _run_2d_te_jax(eps_inf, wp, gam, chi3, dx, dz, dt, nsteps, k_src, k_pL, k_pR, src, cpml, lor=None,
                   chi2=None, raman=None, gain=None):
    """JAX (XLA) backend -- the SAME 2D-TE physics as _run_2d_te, expressed as a single traced, compiled
    lax.scan time loop. Two payoffs: (1) it is DIFFERENTIABLE end-to-end, so a downstream jax.grad gives
    d(R,T)/d(geometry/material) for gradient-based inverse design; (2) XLA fuses the whole step (no
    per-op Python overhead) on CPU and, on a JAX-GPU build (WSL2 on Windows), on the device. Functional
    (immutable .at[]) updates replace the in-place ones; float64 is forced so it matches the reference.
    Returns the four probe x-lines as JAX arrays (the dispatcher converts to NumPy for the FFT/R-T
    extraction; staying in JAX lets a caller jax.grad a scalar objective straight through the time loop,
    the inverse-design path -- see validation/fdtd_2d_autodiff.py). cpml from _cpml_z. `lor`=(C1,C2,C3)
    per-cell Lorentz ADE coefficients (a second polarization PL in the carry) or None (no pole).
    chi2/raman/gain (R15/R20) mirror the numpy kernel; their states extend the scan carry only when
    active (None -> identical carry/trace to the pre-R15 path), and remain DIFFERENTIABLE -- jax.grad
    flows through the SHG/Raman/gain polarizations like every other carry component."""
    import jax
    jax.config.update("jax_enable_x64", True)               # FDTD needs float64 to match the reference
    import jax.numpy as jnp
    from jax import lax
    (ke, be, ce), (kh, bh, ch) = cpml
    ke, be, ce = jnp.asarray(ke), jnp.asarray(be), jnp.asarray(ce)
    kh, bh, ch = jnp.asarray(kh), jnp.asarray(bh), jnp.asarray(ch)
    eps_inf = jnp.asarray(eps_inf); chi3 = jnp.asarray(chi3)
    gam = jnp.asarray(gam); wp = jnp.asarray(wp)
    aJ = (1.0 - gam * dt / 2.0) / (1.0 + gam * dt / 2.0)
    bJ = (EPS0 * wp ** 2 * dt / 2.0) / (1.0 + gam * dt / 2.0)
    nx, nz = eps_inf.shape
    cmu = dt / MU0
    do_lor = lor is not None
    if do_lor:
        C1, C2, C3 = jnp.asarray(lor[0]), jnp.asarray(lor[1]), jnp.asarray(lor[2])
    do_chi2 = chi2 is not None
    if do_chi2:
        chi2 = jnp.asarray(chi2)
    do_raman = raman is not None
    if do_raman:
        R1, R2, R3, chi3R = (jnp.asarray(raman[0]), jnp.asarray(raman[1]),
                             jnp.asarray(raman[2]), jnp.asarray(raman[3]))
    do_gain = gain is not None
    if do_gain:
        G1, G2, G3 = jnp.asarray(gain[0]), jnp.asarray(gain[1]), jnp.asarray(gain[2])

    def step(carry, src_n):
        Ey, Hx, Hz, Jy, psi_h, psi_e, PL, PLp = carry[:8]
        extra = list(carry[8:])
        if do_gain:
            PG, PGp = extra.pop(0), extra.pop(0)
        if do_chi2:
            P2 = extra.pop(0)
        if do_raman:
            Q, Qp, PR = extra.pop(0), extra.pop(0), extra.pop(0)
        dEy_dz = (Ey[:, 1:] - Ey[:, :-1]) / dz
        psi_h = psi_h.at[:, :-1].set(bh[:-1] * psi_h[:, :-1] + ch[:-1] * dEy_dz)
        Hx = Hx.at[:, :-1].add(cmu * (dEy_dz / kh[:-1] + psi_h[:, :-1]))
        Hz = Hz - cmu * (jnp.roll(Ey, -1, axis=0) - Ey) / dx
        dHx_dz = (Hx[:, 1:] - Hx[:, :-1]) / dz
        psi_e = psi_e.at[:, 1:].set(be[1:] * psi_e[:, 1:] + ce[1:] * dHx_dz)
        curl = jnp.zeros((nx, nz))
        curl = curl.at[:, 1:].add(dHx_dz / ke[1:] + psi_e[:, 1:])
        curl = curl - (Hz - jnp.roll(Hz, 1, axis=0)) / dx
        if do_lor:                                          # Lorentz ADE: dPL/dt enters the E-update
            PLnew = C1 * PL + C2 * PLp + C3 * Ey
            curl = curl - (PLnew - PL) / dt
            PLp, PL = PL, PLnew
        if do_gain:                                         # R20 clamped-inversion gain line
            PGnew = G1 * PG + G2 * PGp + G3 * Ey
            curl = curl - (PGnew - PG) / dt
            PGp, PG = PG, PGnew
        if do_chi2:                                         # R15 chi2 SHG polarization
            P2new = EPS0 * chi2 * Ey ** 2
            curl = curl - (P2new - P2) / dt
            P2 = P2new
        if do_raman:                                        # R15 Raman: ADE on E^2 + P_R = eps0 chiR E Q
            Qnew = R1 * Q + R2 * Qp + R3 * Ey ** 2
            PRnew = EPS0 * chi3R * Ey * Qnew
            curl = curl - (PRnew - PR) / dt
            Qp, Q, PR = Q, Qnew, PRnew
        eps_eff = eps_inf + chi3 * Ey ** 2
        denom = EPS0 * eps_eff / dt + bJ / 2.0
        Eyn = (EPS0 * eps_eff / dt * Ey + curl - 0.5 * (1.0 + aJ) * Jy - 0.5 * bJ * Ey) / denom
        Jy = aJ * Jy + bJ * (Eyn + Ey)
        Eyn = Eyn.at[:, k_src].add(src_n)                   # soft plane source
        Eyn = Eyn.at[:, 0].set(0.0).at[:, nz - 1].set(0.0)  # PEC backing the CPML
        out = (Eyn[:, k_pL], 0.5 * (Hx[:, k_pL] + Hx[:, k_pL - 1]),
               Eyn[:, k_pR], 0.5 * (Hx[:, k_pR] + Hx[:, k_pR - 1]))
        new_extra = []
        if do_gain:
            new_extra += [PG, PGp]
        if do_chi2:
            new_extra += [P2]
        if do_raman:
            new_extra += [Q, Qp, PR]
        return (Eyn, Hx, Hz, Jy, psi_h, psi_e, PL, PLp) + tuple(new_extra), out

    z0 = jnp.zeros((nx, nz))
    n_extra = (2 if do_gain else 0) + (1 if do_chi2 else 0) + (3 if do_raman else 0)
    _, (eyL, hxL, eyR, hxR) = lax.scan(step, tuple(z0 for _ in range(8 + n_extra)),
                                       jnp.asarray(src))
    return eyL, hxL, eyR, hxR                               # JAX arrays (differentiable); dispatcher -> NumPy
