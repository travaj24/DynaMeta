"""3D JAX kernels (normal incidence and oblique envelope).

Split from the former monolithic fdtd_nd.py; see the package __init__ docstring
for conventions. Bodies are verbatim from the original module.
"""
from __future__ import annotations

from dynameta.constants import EPS0, MU0
from dynameta.optics.fdtd_nd.backends import check_scan_length

def run_3d_jax(eps_inf, wp, gam, chi3, dx, dy, dz, dt, nsteps, k_src, k_pL, k_pR, src, cpml, lor=None,
               chi2=None, raman=None, gain=None):
    """JAX (XLA lax.scan) 3D backend -- the SAME six-component physics as run_3d, but DIFFERENTIABLE end
    to end: a downstream jax.grad gives d(R,T)/d(geometry/material) for 3D inverse design. Functional
    (immutable .at[]) updates, float64 forced to match the reference. Returns the eight probe planes as JAX
    arrays (the dispatcher converts to NumPy; staying in JAX lets a caller grad straight through the loop).

    `lor` / `chi2` / `raman` / `gain` (audit S2-12, wave-7 2026-11-02) are run_3d's per-component
    Lorentz ADE, diagonal chi2 SHG polarization, ONE isotropic Raman coordinate Q on |E|^2 with
    P_R,i = eps0 chi3R E_i Q, and clamped-inversion gain line -- ported term for term from run_3d
    (the 2-D twin run_2d_te_jax already carried all four; this kernel used to raise). Each extends
    the scan carry only when active, so with all four None the carry and the traced step are the
    pre-S2-12 ones (byte-identical probes, gated), and each stays differentiable. Parity vs the numpy
    run_3d: ~1e-13 relative on every probe plane, every term (the 2-D twins' level; gate
    tests/test_fdtd_backend_contracts.py::test_s2_12_jax_3d_carries_lorentz_chi2_raman_gain)."""
    check_scan_length("run_3d_jax", src, nsteps)                  # audit X-16
    import jax
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    from jax import lax
    (ke, be, ce), (kh, bh, ch) = cpml
    nx, ny, nz = eps_inf.shape
    rs = (lambda a: jnp.asarray(a).reshape(1, 1, nz))       # z-profile -> broadcast
    ke, be, ce = rs(ke), rs(be), rs(ce)
    kh, bh, ch = rs(kh), rs(bh), rs(ch)
    eps_inf = jnp.asarray(eps_inf); chi3 = jnp.asarray(chi3)
    gam = jnp.asarray(gam); wp = jnp.asarray(wp)
    aJ = (1.0 - gam * dt / 2.0) / (1.0 + gam * dt / 2.0)
    bJ = (EPS0 * wp ** 2 * dt / 2.0) / (1.0 + gam * dt / 2.0)
    cmu = dt / MU0
    zz = jnp.zeros((nx, ny, nz))
    do_lor = lor is not None
    if do_lor:
        C1, C2, C3 = jnp.asarray(lor[0]), jnp.asarray(lor[1]), jnp.asarray(lor[2])
    do_gain = gain is not None
    if do_gain:
        G1, G2, G3 = jnp.asarray(gain[0]), jnp.asarray(gain[1]), jnp.asarray(gain[2])
    do_chi2 = chi2 is not None
    if do_chi2:
        chi2 = jnp.asarray(chi2)
    do_raman = raman is not None
    if do_raman:
        R1, R2, R3, chi3R = (jnp.asarray(raman[0]), jnp.asarray(raman[1]),
                             jnp.asarray(raman[2]), jnp.asarray(raman[3]))
    # carry layout: the 13 base fields, then (only when active, in this order) the Lorentz
    # PL{x,y,z} + PLp{x,y,z}, the gain PG{x,y,z} + PGp{x,y,z}, the chi2 P2{x,y,z}, the Raman Q, Qp,
    # PR{x,y,z} -- exactly run_3d's state, so a None term adds nothing to the trace.
    n_extra = 6 * do_lor + 6 * do_gain + 3 * do_chi2 + 5 * do_raman

    def _pol_terms(curl, E, st, c):
        """run_3d's per-component polarization currents, in run_3d's order (Lorentz, gain, chi2,
        Raman), for component index c; `st` is the dict of extra states (updated in place)."""
        if do_lor:
            PLn = C1 * st["PL"][c] + C2 * st["PLp"][c] + C3 * E
            curl = curl - (PLn - st["PL"][c]) / dt
            st["PLp"][c], st["PL"][c] = st["PL"][c], PLn
        if do_gain:
            PGn = G1 * st["PG"][c] + G2 * st["PGp"][c] + G3 * E
            curl = curl - (PGn - st["PG"][c]) / dt
            st["PGp"][c], st["PG"][c] = st["PG"][c], PGn
        if do_chi2:
            P2n = EPS0 * chi2 * E ** 2
            curl = curl - (P2n - st["P2"][c]) / dt
            st["P2"][c] = P2n
        if do_raman:
            PRn = EPS0 * chi3R * E * st["Q"]
            curl = curl - (PRn - st["PR"][c]) / dt
            st["PR"][c] = PRn
        return curl

    def step(carry, src_n):
        Ex, Ey, Ez, Hx, Hy, Hz, Jx, Jy, Jz, psi_Hx, psi_Hy, psi_Ex, psi_Ey = carry[:13]
        ex = list(carry[13:])
        st = {}
        if do_lor:
            st["PL"], st["PLp"] = ex[0:3], ex[3:6]; ex = ex[6:]
        if do_gain:
            st["PG"], st["PGp"] = ex[0:3], ex[3:6]; ex = ex[6:]
        if do_chi2:
            st["P2"] = ex[0:3]; ex = ex[3:]
        if do_raman:
            st["Q"], st["Qp"], st["PR"] = ex[0], ex[1], ex[2:5]
        # ---- H update ----
        dEy_dz = (Ey[:, :, 1:] - Ey[:, :, :-1]) / dz
        psi_Hx = psi_Hx.at[:, :, :-1].set(bh[:, :, :-1] * psi_Hx[:, :, :-1] + ch[:, :, :-1] * dEy_dz)
        sEy = zz.at[:, :, :-1].set(dEy_dz / kh[:, :, :-1] + psi_Hx[:, :, :-1])
        Hx = Hx - cmu * ((jnp.roll(Ez, -1, axis=1) - Ez) / dy - sEy)
        dEx_dz = (Ex[:, :, 1:] - Ex[:, :, :-1]) / dz
        psi_Hy = psi_Hy.at[:, :, :-1].set(bh[:, :, :-1] * psi_Hy[:, :, :-1] + ch[:, :, :-1] * dEx_dz)
        sEx = zz.at[:, :, :-1].set(dEx_dz / kh[:, :, :-1] + psi_Hy[:, :, :-1])
        Hy = Hy - cmu * (sEx - (jnp.roll(Ez, -1, axis=0) - Ez) / dx)
        Hz = Hz - cmu * ((jnp.roll(Ey, -1, axis=0) - Ey) / dx - (jnp.roll(Ex, -1, axis=1) - Ex) / dy)
        # ---- E update (per-component Drude ADE + Kerr) ----
        if do_raman:                                        # ONE isotropic Q on |E^n|^2, once per step
            Qn = R1 * st["Q"] + R2 * st["Qp"] + R3 * (Ex ** 2 + Ey ** 2 + Ez ** 2)
            st["Qp"], st["Q"] = st["Q"], Qn
        eps_eff = eps_inf + 3.0 * chi3 * (Ex ** 2 + Ey ** 2 + Ez ** 2)  # standard chi3 (C3-2)
        ce_dt = EPS0 * eps_eff / dt
        denom = ce_dt + bJ / 2.0
        coef = 0.5 * (1.0 + aJ)
        dHy_dz = (Hy[:, :, 1:] - Hy[:, :, :-1]) / dz
        psi_Ex = psi_Ex.at[:, :, 1:].set(be[:, :, 1:] * psi_Ex[:, :, 1:] + ce[:, :, 1:] * dHy_dz)
        sHy = zz.at[:, :, 1:].set(dHy_dz / ke[:, :, 1:] + psi_Ex[:, :, 1:])
        curlx = _pol_terms((Hz - jnp.roll(Hz, 1, axis=1)) / dy - sHy, Ex, st, 0)
        Exn = (ce_dt * Ex + curlx - coef * Jx - 0.5 * bJ * Ex) / denom
        Jx = aJ * Jx + bJ * (Exn + Ex)
        dHx_dz = (Hx[:, :, 1:] - Hx[:, :, :-1]) / dz
        psi_Ey = psi_Ey.at[:, :, 1:].set(be[:, :, 1:] * psi_Ey[:, :, 1:] + ce[:, :, 1:] * dHx_dz)
        sHx = zz.at[:, :, 1:].set(dHx_dz / ke[:, :, 1:] + psi_Ey[:, :, 1:])
        curly = _pol_terms(sHx - (Hz - jnp.roll(Hz, 1, axis=0)) / dx, Ey, st, 1)
        Eyn = (ce_dt * Ey + curly - coef * Jy - 0.5 * bJ * Ey) / denom
        Jy = aJ * Jy + bJ * (Eyn + Ey)
        curlz = _pol_terms((Hy - jnp.roll(Hy, 1, axis=0)) / dx - (Hx - jnp.roll(Hx, 1, axis=1)) / dy,
                           Ez, st, 2)
        Ezn = (ce_dt * Ez + curlz - coef * Jz - 0.5 * bJ * Ez) / denom
        Jz = aJ * Jz + bJ * (Ezn + Ez)
        Eyn = Eyn.at[:, :, k_src].add(src_n)                # soft y-pol source
        Exn = Exn.at[:, :, 0].set(0.0).at[:, :, nz - 1].set(0.0)   # PEC: tangential Ex,Ey only
        Eyn = Eyn.at[:, :, 0].set(0.0).at[:, :, nz - 1].set(0.0)
        Ex, Ey, Ez = Exn, Eyn, Ezn
        out = (Ex[:, :, k_pL], Ey[:, :, k_pL],
               0.5 * (Hx[:, :, k_pL] + Hx[:, :, k_pL - 1]), 0.5 * (Hy[:, :, k_pL] + Hy[:, :, k_pL - 1]),
               Ex[:, :, k_pR], Ey[:, :, k_pR],
               0.5 * (Hx[:, :, k_pR] + Hx[:, :, k_pR - 1]), 0.5 * (Hy[:, :, k_pR] + Hy[:, :, k_pR - 1]))
        new_ex = []
        if do_lor:
            new_ex += list(st["PL"]) + list(st["PLp"])
        if do_gain:
            new_ex += list(st["PG"]) + list(st["PGp"])
        if do_chi2:
            new_ex += list(st["P2"])
        if do_raman:
            new_ex += [st["Q"], st["Qp"]] + list(st["PR"])
        return (Ex, Ey, Ez, Hx, Hy, Hz, Jx, Jy, Jz, psi_Hx, psi_Hy, psi_Ex, psi_Ey) + tuple(new_ex), out

    carry0 = tuple(zz for _ in range(13 + n_extra))
    _, outs = lax.scan(step, carry0, jnp.asarray(src))
    return outs                                             # 8-tuple of (nsteps,nx,ny) JAX arrays


def run_3d_oblique_jax(eps_inf, wp, gam, dx, dy, dz, dt, nsteps, k_src, k_pL, k_pR, src, cpml,
                       kx, ky, sx, sy):
    """JAX (XLA lax.scan) twin of _run_3d_oblique -- the full-vector COMPLEX-ENVELOPE oblique 3D engine
    (2D transverse Bloch envelope d/dx->d/dx+i kx, d/dy->d/dy+i ky; semi-implicit Drude ADE per
    E-component; CFS-CPML + PEC in z; s-pol plane source on (sx,sy)) as a single traced/compiled
    DIFFERENTIABLE complex128 time loop -- so jax.grad flows d(R,T)/d(geometry/material) through the
    oblique 3D scan. (kx,ky)=0 reduces to the real normal-incidence response. Byte-equal to the NumPy
    reference to ~1e-9. Drude only. Returns the complex Ex,Ey probe planes at the left/right z-planes."""
    check_scan_length("run_3d_oblique_jax", src, nsteps)                  # audit X-16
    import jax
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    from jax import lax
    (ke, be, ce), (kh, bh, ch) = cpml
    nx, ny, nz = eps_inf.shape
    rs = (lambda a: jnp.asarray(a).reshape(1, 1, nz))
    ke, be, ce = rs(ke), rs(be), rs(ce)
    kh, bh, ch = rs(kh), rs(bh), rs(ch)
    eps_inf = jnp.asarray(eps_inf); gam = jnp.asarray(gam); wp = jnp.asarray(wp)
    aJ = (1.0 - gam * dt / 2.0) / (1.0 + gam * dt / 2.0)
    bJ = (EPS0 * wp ** 2 * dt / 2.0) / (1.0 + gam * dt / 2.0)
    cmu = dt / MU0; e0dt = EPS0 / dt
    ce_dt = e0dt * eps_inf; denom = ce_dt + bJ / 2.0; coef = 0.5 * (1.0 + aJ)
    ikx, iky = 1j * kx, 1j * ky
    zz = jnp.zeros((nx, ny, nz), dtype=jnp.complex128)

    def dxf(F):
        return (jnp.roll(F, -1, axis=0) - F) / dx + ikx * F

    def dyf(F):
        return (jnp.roll(F, -1, axis=1) - F) / dy + iky * F

    def dxb(F):
        return (F - jnp.roll(F, 1, axis=0)) / dx + ikx * F

    def dyb(F):
        return (F - jnp.roll(F, 1, axis=1)) / dy + iky * F

    def step(carry, src_n):
        Ex, Ey, Ez, Hx, Hy, Hz, Jx, Jy, Jz, psi_Hx, psi_Hy, psi_Ex, psi_Ey = carry
        # ---- H update (reads old E; forward envelope derivs) ----
        dEy_dz = (Ey[:, :, 1:] - Ey[:, :, :-1]) / dz
        psi_Hx = psi_Hx.at[:, :, :-1].set(bh[:, :, :-1] * psi_Hx[:, :, :-1] + ch[:, :, :-1] * dEy_dz)
        sEy = zz.at[:, :, :-1].set(dEy_dz / kh[:, :, :-1] + psi_Hx[:, :, :-1])
        Hx = Hx - cmu * (dyf(Ez) - sEy)
        dEx_dz = (Ex[:, :, 1:] - Ex[:, :, :-1]) / dz
        psi_Hy = psi_Hy.at[:, :, :-1].set(bh[:, :, :-1] * psi_Hy[:, :, :-1] + ch[:, :, :-1] * dEx_dz)
        sEx = zz.at[:, :, :-1].set(dEx_dz / kh[:, :, :-1] + psi_Hy[:, :, :-1])
        Hy = Hy - cmu * (sEx - dxf(Ez))
        Hz = Hz - cmu * (dxf(Ey) - dyf(Ex))
        # ---- E update (reads new H, old E/J; backward envelope derivs) ----
        dHy_dz = (Hy[:, :, 1:] - Hy[:, :, :-1]) / dz
        psi_Ex = psi_Ex.at[:, :, 1:].set(be[:, :, 1:] * psi_Ex[:, :, 1:] + ce[:, :, 1:] * dHy_dz)
        sHy = zz.at[:, :, 1:].set(dHy_dz / ke[:, :, 1:] + psi_Ex[:, :, 1:])
        Exn = (ce_dt * Ex + (dyb(Hz) - sHy) - coef * Jx - 0.5 * bJ * Ex) / denom
        Jx = aJ * Jx + bJ * (Exn + Ex)
        dHx_dz = (Hx[:, :, 1:] - Hx[:, :, :-1]) / dz
        psi_Ey = psi_Ey.at[:, :, 1:].set(be[:, :, 1:] * psi_Ey[:, :, 1:] + ce[:, :, 1:] * dHx_dz)
        sHx = zz.at[:, :, 1:].set(dHx_dz / ke[:, :, 1:] + psi_Ey[:, :, 1:])
        Eyn = (ce_dt * Ey + (sHx - dxb(Hz)) - coef * Jy - 0.5 * bJ * Ey) / denom
        Jy = aJ * Jy + bJ * (Eyn + Ey)
        curlz = dxb(Hy) - dyb(Hx)
        Ezn = (ce_dt * Ez + curlz - coef * Jz - 0.5 * bJ * Ez) / denom
        Jz = aJ * Jz + bJ * (Ezn + Ez)
        Exn = Exn.at[:, :, k_src].add(sx * src_n)
        Eyn = Eyn.at[:, :, k_src].add(sy * src_n)
        Exn = Exn.at[:, :, 0].set(0.0 + 0.0j).at[:, :, nz - 1].set(0.0 + 0.0j)
        Eyn = Eyn.at[:, :, 0].set(0.0 + 0.0j).at[:, :, nz - 1].set(0.0 + 0.0j)
        out = (Exn[:, :, k_pL], Eyn[:, :, k_pL], Exn[:, :, k_pR], Eyn[:, :, k_pR])
        return (Exn, Eyn, Ezn, Hx, Hy, Hz, Jx, Jy, Jz, psi_Hx, psi_Hy, psi_Ex, psi_Ey), out

    carry0 = tuple(zz for _ in range(13))
    _, (exL, eyL, exR, eyR) = lax.scan(step, carry0, jnp.asarray(src, dtype=jnp.complex128))
    return exL, eyL, exR, eyR
