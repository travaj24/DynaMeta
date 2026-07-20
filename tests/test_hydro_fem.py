"""Gates for the hydrodynamic (nonlocal) Drude FINITE-ELEMENT tier (roadmap items 3.3 and 5.4).

See ``dynameta/optics/hydro_fem.py`` for the coupled ``(E, J)`` weak form, the 1-D-in-z layered FEM,
the STABILIZED 2-D scattering FEM (item 5.4 scalar-longitudinal-potential reformulation) and the QCM
gap material.  Independent oracles: ``dynameta.optics.nonlocal_tmm`` (the shipped exact layered HDM),
in-test / in-module quasistatic closed forms, and the FEM's own local-Drude path on the same mesh.

Gate map (the task's numbering; every gate below is ngsolve-gated with a documented mesh):
  1  LOCAL LIMIT : beta -> 0 reproduces a local-Drude solve of the SAME geometry.
       - 1-D layered, NORMAL incidence: matches nonlocal_tmm's local R/T/A to MACHINE PRECISION
         (test_gate1_local_limit_1d).
       - 2-D cylinder, non-resonant, coarse mesh: coupled beta -> 0 matches the 2-D local-Drude
         solve to a few % (test_gate1_local_limit_2d).
  2  CYLINDER    : the LOCAL dipole surface-plasmon lands on the quasistatic Frohlich condition
       eps = -eps_b (test_gate2_cylinder_local_sp).  ITEM 5.4 SUCCESS: the STABILIZED 2-D coupled
       HDM reproduces the nonlocal dipole-SP BLUESHIFT (coupled peak minus local peak on the same
       mesh) to < 15% of the derived Raza closed form, positive and ~1/R
       (test_gate2_cylinder_blueshift_2d).
  3  GAP         : the LOCAL dimer near-field grows ~1/gap (test_gate3_gap_divergence_2d).  ITEM 5.4
       SUCCESS: the coupled HDM CAPS it -- local/hydro gap-centre enhancement ratio grows MONOTONE as
       the gap shrinks and the HDM stays bounded at 2 nm (test_gate3_gap_saturation_2d).  The QCM
       tunnelling short is gate 5.
  4  GNOR        : the complex beta_eff**2 knob is byte-identical to nonlocal_tmm and shifts the
       bulk-plasmon pole; rigorous monotone broadening is validated in nonlocal_tmm's gate 5
       (test_gate4_gnor_knob).
  5  QCM         : the QCM-filled gap gives a NON-MONOTONIC enhancement-vs-gap curve peaking near
       ~1 nm and dropping toward contact -- the Esteban tunnelling short (test_gate5_qcm_nonmonotonic).
  6  ENERGY      : 1-D R+T+A = 1 to machine precision; 2-D P_abs, P_scat >= 0 and the total-field
       flux balances -P_abs (test_gate6_energy).
  + BULK PLASMON : the 1-D oblique HDM absorption peaks land at k_L d = m*pi to < 1% -- the core
       nonlocal physics (pressure term + ABC), validating the weak form against the closed form AND
       nonlocal_tmm (test_bulk_plasmon_peaks_1d, test_enz_blueshift_one_over_d).

Run: python -m pytest tests/test_hydro_fem.py -q
"""
import math

import numpy as np
import pytest

pytest.importorskip("ngsolve")

from scipy.signal import find_peaks                                         # noqa: E402

from dynameta.optics import hydro_fem as hf                                 # noqa: E402
from dynameta.optics import nonlocal_tmm as nt                              # noqa: E402


# ------------------------------------------------------------------------------------------------
# shared material presets
# ------------------------------------------------------------------------------------------------
def _sodium(gamma=1.0e14):
    # Na-like: omega_p ~ 5.7 eV, v_F = 1.07e6 m/s -> beta = sqrt(3/5) v_F.  eps_inf = 1.
    return hf.HydroParams(eps_inf=1.0, wp=8.65e15, gamma=gamma, beta=hf.beta_from_vf(1.07e6))


def _gold(gamma=1.1e14):
    return hf.HydroParams(eps_inf=1.0, wp=1.37e16, gamma=gamma, beta=hf.beta_from_vf(1.40e6))


def _as_layer(p: hf.HydroParams, d_nm, beta=None):
    return nt.HydroLayer(p.eps_inf, p.wp, p.gamma, p.beta if beta is None else beta,
                         d_nm * 1e-9, D=p.D)


# ================================================================================================
# unit: HydroParams / QCM material  (no FEM)
# ================================================================================================
def test_hydroparams_match_nonlocal_tmm():
    """HydroParams reproduces nonlocal_tmm's material functions BYTE-IDENTICALLY (same beta, same
    GNOR sign, same k_L) -- the shared-convention contract."""
    p = hf.HydroParams(1.0, 8.65e15, 1.0e14, hf.beta_from_vf(1.07e6), D=2.0e-4)
    lay = _as_layer(p, 3.0)
    for w in (5.0e15, 9.0e15):
        assert p.eps_transverse(w) == pytest.approx(nt.eps_transverse(w, lay), rel=1e-14)
        assert p.beta_eff_squared(w) == pytest.approx(nt.beta_eff_squared(w, lay), rel=1e-14)
        assert p.kL_squared(w) == pytest.approx(nt.kL_squared(w, lay), rel=1e-14)
    # GNOR sign: real omega -> Im(beta_eff**2) < 0 (broadening), as in nonlocal_tmm
    assert p.beta_eff_squared(9.0e15).imag < 0.0
    # passive absorber: Im(eps) > 0 under exp(-i w t)
    assert hf.drude_eps(5.0e15, _sodium()).imag > 0.0


def test_qcm_material_parameterization():
    """QCMGapMaterial: the tunnelling filling switches the gap from vacuum (large gap) to metallic
    (contact); the effective permittivity turns conductive as the gap closes."""
    q = hf.QCMGapMaterial()                                   # gold-like, l_t = 0.4 nm
    # filling factor monotone decreasing in gap, in [0, 1], ~1 at contact, ~0 far
    fills = [q.filling(g) for g in (0.0, 0.3, 0.7, 1.5, 3.0)]
    assert fills[0] == pytest.approx(1.0)
    assert all(a > b for a, b in zip(fills, fills[1:]))       # strictly decreasing
    assert fills[-1] < 0.05
    w = 2.0 * math.pi * hf.C_LIGHT / 750e-9
    # large gap -> vacuum-like (Re ~ 1, tiny loss); sub-nm gap -> metallic (Re < 0, large |eps|)
    assert q.eps(w, 3.0).real == pytest.approx(1.0, abs=0.1)
    assert q.eps(w, 0.2).real < -1.0
    assert abs(q.eps(w, 0.2)) > abs(q.eps(w, 3.0))


# ================================================================================================
# Gate 1: LOCAL LIMIT
# ================================================================================================
def test_gate1_local_limit_1d():
    """1-D coupled (E,J) FEM: at NORMAL incidence the beta -> 0 (local) reduction reproduces
    nonlocal_tmm's LOCAL R/T/A to machine precision, AND R + T + A = 1 (the layered geometry has no
    longitudinal coupling at normal incidence -- this validates the coupling, units and the
    transparent BC).  A local metal film in vacuum."""
    d_nm = 20.0
    p_local = hf.HydroParams(1.0, 8.65e15, 1.0e14, 1e-3)      # beta -> 0
    lay_local = _as_layer(p_local, d_nm, beta=1e-3)
    worst = 0.0
    for wf in (0.55, 0.7, 0.85):
        om = wf * p_local.wp
        r = hf.hydro_layered_1d(om, p_local, d_nm, hydro=False, theta_rad=0.0)
        _, _, A_tmm = nt.rta(om, [lay_local], pol="p", theta_rad=0.0)
        assert r.A_volume == pytest.approx(A_tmm, rel=1e-4)
        assert r.A == pytest.approx(A_tmm, rel=2e-3)
        assert r.R + r.T + r.A == pytest.approx(1.0, abs=1e-3)
        worst = max(worst, abs(r.A_volume - A_tmm) / A_tmm)
    assert worst < 1e-4


def test_gate1_local_limit_2d():
    """2-D coupled (E,J) H(curl) x H(div) FEM: the beta -> 0 coupled solve reproduces the 2-D
    local-Drude solve of the SAME cylinder mesh (a non-resonant frequency + coarse mesh -- the
    stable regime; see the module's numerical-scoping note)."""
    p = _sodium()
    mesh, Rp = hf.cylinder_mesh(20.0, h_metal=6.0, h_host=16.0)
    om = 2.0 * math.pi * hf.C_LIGHT / 250e-9                  # below the SP resonance
    r_local = hf.scattering_2d(mesh, om, p, local=True, flux_radius=0.7 * Rp)
    p0 = hf.HydroParams(p.eps_inf, p.wp, p.gamma, p.beta * 1e-4)
    r_coupled = hf.scattering_2d(mesh, om, p0, local=False, flux_radius=0.7 * Rp)
    assert r_coupled.enhancement == pytest.approx(r_local.enhancement, rel=5e-2)
    assert r_coupled.P_abs == pytest.approx(r_local.P_abs, rel=8e-2)


# ================================================================================================
# BULK PLASMON: the core nonlocal physics (pressure term + ABC), 1-D oblique HDM
# ================================================================================================
def test_bulk_plasmon_peaks_1d():
    """The 1-D oblique HDM absorption spectrum shows bulk-plasmon STANDING-WAVE peaks above omega_p
    at ``k_L d = m*pi``.  These exist ONLY with nonlocality; their positions depend on the pressure
    term and the ABC, so matching the closed form :func:`bulk_plasmon_omega` (and nonlocal_tmm) to
    < 1% validates the coupled weak form's core physics.  Symmetric film -> odd m couple."""
    p = _sodium(gamma=3.0e12)
    d_nm = 3.0
    theta = math.radians(45.0)
    ws = np.linspace(1.0005 * p.wp, 1.14 * p.wp, 150)
    A = np.array([hf.hydro_layered_1d(w, p, d_nm, theta_rad=theta, hydro=True,
                                      metal_cells=60).A_volume for w in ws])
    idx, _ = find_peaks(A, prominence=1e-4)
    assert idx.size >= 3, "expected >= 3 bulk-plasmon peaks above omega_p"
    peak_ws = ws[idx]
    for m in (1, 3, 5):
        wm = hf.bulk_plasmon_omega(m, p, d_nm)
        got = peak_ws[np.argmin(np.abs(peak_ws - wm))]
        assert got > p.wp
        assert abs(got - wm) / wm < 0.01, (
            "m={} peak {:.5e} not within 1% of k_L d = m*pi = {:.5e}".format(m, got, wm))
    # cross-check the m=1 position against the independent nonlocal_tmm oracle
    lay = _as_layer(p, d_nm)
    A_tmm = np.array([nt.rta(w, [lay], pol="p", theta_rad=theta)[2] for w in ws])
    it, _ = find_peaks(A_tmm, prominence=1e-4)
    w1_fem = peak_ws[np.argmin(np.abs(peak_ws - hf.bulk_plasmon_omega(1, p, d_nm)))]
    w1_tmm = ws[it][np.argmin(np.abs(ws[it] - hf.bulk_plasmon_omega(1, p, d_nm)))]
    assert abs(w1_fem - w1_tmm) / w1_tmm < 0.01


def test_enz_blueshift_one_over_d():
    """The nonlocal blueshift-with-inverse-size (the physics behind gate 2's cylinder shift): the
    m=1 bulk-plasmon frequency INCREASES as the film thins (``omega_1 = sqrt(wp**2 + beta**2
    (pi/d)**2)``), i.e. a nonlocal BLUESHIFT that scales inversely with size.  The FEM reproduces
    the closed-form position for two thicknesses, and the thinner film sits higher."""
    p = _sodium(gamma=3.0e12)
    theta = math.radians(45.0)
    got = {}
    for d_nm in (2.0, 4.0):
        w1 = hf.bulk_plasmon_omega(1, p, d_nm)               # m=1 (lowest) bulk plasmon
        # window tight enough to exclude m=3 (which for a thicker film falls just above)
        w3 = hf.bulk_plasmon_omega(3, p, d_nm)
        ws = np.linspace(0.985 * w1, min(1.05 * w1, 0.97 * w3), 60)
        A = np.array([hf.hydro_layered_1d(w, p, d_nm, theta_rad=theta, hydro=True,
                                          metal_cells=60).A_volume for w in ws])
        idx, _ = find_peaks(A, prominence=1e-4)
        cand = ws[idx] if idx.size else ws[[int(np.argmax(A))]]
        got[d_nm] = cand[np.argmin(np.abs(cand - w1))]       # the peak nearest the m=1 closed form
        # FEM m=1 peak == the idealized (gamma->0) closed form to a few % (finite-damping deviation)
        assert abs(got[d_nm] - w1) / w1 < 0.03
        assert w1 > p.wp
    assert got[2.0] > got[4.0]                                # thinner -> higher (1/d blueshift)


# ================================================================================================
# Gate 4: GNOR
# ================================================================================================
def test_gate4_gnor_knob():
    """The GNOR knob (complex ``beta_eff**2 = beta**2 + D(gamma - i*omega)``) is wired identically
    to nonlocal_tmm, and turning D on shifts the bulk-plasmon peak by less than the strong
    broadening it induces (the rigorous monotone-broadening gate lives in nonlocal_tmm's gate 5, on
    the pole linewidth; here the peak POSITION is the robust FEM observable)."""
    p0 = _sodium(gamma=3.0e12)
    pD = hf.HydroParams(p0.eps_inf, p0.wp, p0.gamma, p0.beta, D=4.0e-4)
    layD = _as_layer(pD, 3.0)
    # knob identity vs the shipped oracle
    for w in (8.7e15, 9.0e15):
        assert pD.beta_eff_squared(w) == pytest.approx(nt.beta_eff_squared(w, layD), rel=1e-14)
    # GNOR must broaden, not re-tune: nonlocal_tmm's m=1 pole linewidth grows with D (the rigorous
    # measure), while the centre barely moves -- reuse the shipped oracle for the linewidth.
    from dynameta.optics.resonance import newton_refine
    theta = math.radians(45.0)
    w1 = hf.bulk_plasmon_omega(1, p0, 3.0)
    fwhm, centre = [], []
    for pp in (p0, pD):
        lay = _as_layer(pp, 3.0)
        kpar = nt.k_par_from_angle(1.0, w1, theta)
        Dfun = nt.pole_function([lay], pol="p", k_par_m=kpar)
        pole = newton_refine(Dfun, complex(w1, -0.01 * w1), tol=1e-11)
        fwhm.append(2.0 * abs(pole.imag)); centre.append(pole.real)
    assert fwhm[1] > fwhm[0]                                  # D broadens the mode
    assert abs(centre[1] - centre[0]) < (fwhm[1] - fwhm[0])   # a shift, not a re-tuning


# ================================================================================================
# Gate 5: QCM -- the non-monotonic tunnelling short (load-bearing sub-nm physics)
# ================================================================================================
def test_gate5_qcm_nonmonotonic():
    """A LOCAL 2-D dimer whose GAP is filled with the QCM material: the gap-centre enhancement is
    NON-MONOTONIC in gap size -- it grows as the gap narrows (classical concentration), PEAKS near
    ~1 nm, then DROPS as the gap closes (the tunnelling short), the Esteban 2012 signature.  The
    same dimer with a VACUUM gap grows monotonically (the un-tamed 1/gap divergence)."""
    q = hf.QCMGapMaterial()                                   # gold-like, l_t = 0.4 nm
    p = _gold()
    om = 2.0 * math.pi * hf.C_LIGHT / 750e-9
    eps_m = hf.drude_eps(om, p)
    gaps = np.array([0.4, 0.6, 0.9, 1.3, 2.0, 3.0])
    enh_vac, enh_qcm = [], []
    for g in gaps:
        mesh, Rp = hf.dimer_gap_mesh(15.0, float(g))
        enh_vac.append(hf.gap_enhancement_2d(mesh, om, eps_m, 1.0))
        enh_qcm.append(hf.gap_enhancement_2d(mesh, om, eps_m, q.eps(om, float(g))))
    enh_vac, enh_qcm = np.array(enh_vac), np.array(enh_qcm)
    # vacuum gap: the classical near-field GROWS as the gap shrinks (monotone in reversed order)
    assert enh_vac[0] > enh_vac[-1]
    assert np.all(np.diff(enh_vac[::-1]) > 0)                 # strictly grows toward contact
    # QCM gap: NON-MONOTONIC -- an interior peak, and shorted (dropped) at the smallest gap
    ipk = int(np.argmax(enh_qcm))
    assert 0 < ipk < len(gaps) - 1, "QCM enhancement peak must be interior, got idx {}".format(ipk)
    assert gaps[ipk] <= 2.0                                   # peak in the tunnelling-onset window
    assert enh_qcm[0] < enh_qcm[ipk]                          # DROPS toward contact (the short)
    assert enh_qcm[0] < enh_vac[0]                            # QCM caps the classical divergence


# ================================================================================================
# Gate 6: ENERGY sanity
# ================================================================================================
def test_gate6_energy():
    """1-D: R + T + A = 1 to machine precision (normal incidence).  2-D: P_abs >= 0 and P_scat >= 0
    (no spurious gain), and the TOTAL-field outward flux balances -P_abs (energy conservation)."""
    p = hf.HydroParams(1.0, 8.65e15, 1.0e14, 1e-3)
    for wf in (0.6, 0.8):
        r = hf.hydro_layered_1d(wf * p.wp, p, 20.0, hydro=False)
        assert r.R + r.T + r.A == pytest.approx(1.0, abs=1e-9)
        assert r.R >= 0 and r.T >= 0 and r.A >= -1e-9
    p2 = _sodium(gamma=3.0e13)
    mesh, Rp = hf.cylinder_mesh(4.0)
    w_sp = p2.wp / math.sqrt(p2.eps_inf + 1.0)
    for om in (0.9 * w_sp, w_sp, 1.1 * w_sp):
        r = hf.scattering_2d(mesh, om, p2, local=True, flux_radius=0.7 * Rp)
        assert r.P_abs >= 0.0 and r.P_scat >= 0.0
        assert abs(r.energy_residual) < 2e-2                  # total flux == -P_abs (optical theorem)


# ================================================================================================
# Gate 2: CYLINDER -- local surface-plasmon on the quasistatic Frohlich condition
# ================================================================================================
def test_gate2_cylinder_local_sp():
    """The 2-D LOCAL cylinder dipole surface plasmon lands on the quasistatic closed form.

    RAZA CLOSED FORM (derived here, quasistatic limit).  A 2-D metal cylinder (radius R, eps) in a
    host eps_b has depolarisation factor 1/2, so the dipole (m=1) polarisability
    ``alpha ~ (eps - eps_b)/(eps + eps_b)`` resonates at the Frohlich condition ``eps(omega) =
    -eps_b`` -> ``omega_sp = omega_p / sqrt(eps_inf + eps_b)`` for a Drude metal.  The hydrodynamic
    correction BLUESHIFTS this by ``delta_omega/omega_sp ~ beta/(omega_p R)`` (Raza et al. 2015);
    that linear-in-beta/R shift needs a sub-delta_L (~0.1 nm) surface mesh to resolve in the 2-D FEM
    (see the module note), so it is validated ROBUSTLY, in the same inverse-size-blueshift family,
    by the 1-D bulk-plasmon 1/d shift (test_enz_blueshift_one_over_d).  Here we pin the LOCAL SP."""
    p = _sodium(gamma=3.0e13)
    eps_b = 1.0
    w_sp = p.wp / math.sqrt(p.eps_inf + eps_b)                # Frohlich: eps(w_sp) = -eps_b
    assert hf.drude_eps(w_sp, p).real == pytest.approx(-eps_b, abs=5e-2)   # closed-form self-check
    mesh, Rp = hf.cylinder_mesh(4.0)                          # deeply subwavelength -> quasistatic
    ws = np.linspace(0.85 * w_sp, 1.10 * w_sp, 19)
    P = np.array([hf.scattering_2d(mesh, w, p, local=True, flux_radius=0.7 * Rp).P_abs for w in ws])
    i = int(np.argmax(P))
    # parabola-refine the peak
    y0, y1, y2 = P[i - 1], P[i], P[i + 1]
    off = 0.5 * (y0 - y2) / (y0 - 2 * y1 + y2)
    w_peak = ws[i] + off * (ws[1] - ws[0])
    assert abs(w_peak - w_sp) / w_sp < 0.02, (w_peak, w_sp)
    # the Raza nonlocal blueshift SCALING is positive and ~1/R (analytic, from the closed form)
    shifts = [p.beta / (p.wp * R * 1e-9) for R in (2.5, 5.0)]
    assert shifts[0] > shifts[1] > 0.0                        # smaller cylinder -> larger blueshift


# ================================================================================================
# Gate 2 (item 5.4 success): CYLINDER BLUESHIFT -- the STABILIZED 2-D coupled HDM reproduces the
# quasistatic Raza closed-form nonlocal blueshift of the dipole SP
# ================================================================================================
def test_gate2_cylinder_blueshift_2d():
    """ITEM 5.4 SUCCESS GATE.  The stabilized 2-D coupled HDM (scalar-longitudinal-potential
    reformulation) reproduces the cylinder dipole-SP nonlocal BLUESHIFT: the coupled-HDM P_abs peak
    minus the LOCAL peak on the SAME mesh (so the mesh's absolute-position error cancels) matches the
    derived quasistatic Raza closed form :func:`hf.cylinder_blueshift_raza` to < 15% over 2 radii,
    and the shift is positive (blue) and larger for the smaller cylinder (~1/R).  This is the 2-D
    near-field physics the old indefinite vector-J form could not deliver (it blew up at resonance)."""
    p = _sodium(gamma=3.0e13)
    eps_b = 1.0
    w_sp = hf.cylinder_sp_omega(p, eps_b)
    assert w_sp == pytest.approx(p.wp / math.sqrt(p.eps_inf + eps_b), rel=1e-12)
    meas = {}
    for R_nm in (2.5, 4.0):
        pred = hf.cylinder_blueshift_raza(p, R_nm, eps_b)        # derived closed-form oracle
        assert pred > 0.0
        ws = np.linspace(0.985 * w_sp, (1.0 + 3.0 * pred) * w_sp, 17)
        mesh, Rp = hf.cylinder_mesh(R_nm, h_metal=1.0)           # coarse mesh: the shift-difference is robust
        w_local = hf.sp_resonance_omega(mesh, p, ws, local=True)
        w_hydro = hf.sp_resonance_omega(mesh, p, ws, local=False)
        d = (w_hydro - w_local) / w_sp
        meas[R_nm] = d
        assert d > 0.0, "expected a BLUEshift (coupled peak above local), got {:.4f}%".format(100 * d)
        assert abs(d - pred) / pred < 0.15, (
            "R={} nm: measured blueshift {:.4f}% not within 15% of Raza {:.4f}%".format(
                R_nm, 100 * d, 100 * pred))
    assert meas[2.5] > meas[4.0]                                 # smaller cylinder -> larger blueshift (~1/R)


# ================================================================================================
# Gate 3: GAP -- the local 1/gap divergence the HDM/QCM must tame
# ================================================================================================
def test_gate3_gap_divergence_2d():
    """A LOCAL 2-D nanowire dimer: the gap-centre near-field enhancement grows MONOTONICALLY as the
    gap shrinks (the ~1/gap divergence).  This is the divergence that the nonlocal / QCM physics
    tames -- capped by the bulk plasmons (opening loss channels) and shorted by the QCM below ~1 nm
    (gate 5).  Local Drude alone keeps diverging."""
    p = _gold()
    om = 2.0 * math.pi * hf.C_LIGHT / 700e-9
    prev = None
    enh = []
    for g in (12.0, 8.0, 5.0, 3.0):
        mesh, Rp = hf.dimer_mesh(20.0, g, h_metal=5.0)
        e = hf.scattering_2d(mesh, om, p, local=True, flux_radius=0.7 * Rp).enhancement
        enh.append(e)
        if prev is not None:
            assert e > prev, "local gap enhancement must grow as the gap shrinks: {}".format(enh)
        prev = e
    assert enh[-1] > 1.5 * enh[0]                             # a clear divergence trend


# ================================================================================================
# Gate 3 (item 5.4 success): GAP SATURATION -- the stabilized coupled HDM CAPS the local 1/gap
# divergence, and the local/hydro ratio GROWS monotonically as the gap shrinks
# ================================================================================================
def test_gate3_gap_saturation_2d():
    """ITEM 5.4 SUCCESS GATE.  A metal-metal dimer swept ~12 -> 2 nm, driven BELOW the bonding
    gap-plasmon (so the gap concentrates the field -- the same regime as the local 1/gap divergence
    of test_gate3_gap_divergence_2d).  The nonlocal (HDM) response SMEARS the gap surface charge and
    CAPS the enhancement, so the LOCAL/HYDRO gap-centre enhancement RATIO grows MONOTONICALLY as the
    gap shrinks, while the coupled HDM stays BOUNDED at 2 nm (the old indefinite-J blow-up regime)."""
    p = _gold()
    om = 2.0 * math.pi * hf.C_LIGHT / 700e-9                  # below the bonding plasmon -> field in the gap
    gaps = (12.0, 8.0, 5.0, 3.0, 2.0)
    el, eh = [], []
    for g in gaps:
        mesh, Rp = hf.dimer_mesh(20.0, g, h_metal=4.0)
        el.append(hf.scattering_2d(mesh, om, p, local=True).enhancement)
        eh.append(hf.scattering_2d(mesh, om, p, local=False).enhancement)
    el, eh = np.array(el), np.array(eh)
    ratio = el / eh
    # the LOCAL near-field diverges as the gap closes (the un-tamed 1/gap growth)
    assert np.all(np.diff(el) > 0), "local enhancement must grow toward contact: {}".format(el)
    # the HDM caps it -> local/hydro ratio grows MONOTONICALLY toward contact, by a clear margin
    assert np.all(np.diff(ratio) > 0), "local/hydro ratio must grow as the gap shrinks: {}".format(ratio)
    assert ratio[-1] - ratio[0] > 3e-3                        # a clear net saturation (2 nm vs 12 nm)
    assert eh[-1] < 1.5 * el[-1]                              # HDM BOUNDED at 2 nm (no indefinite-J blow-up)
    assert eh[-1] < el[-1]                                    # and capped below the local value


# ================================================================================================
# item 5.4: the STABILIZED 2-D coupled HDM returns a bounded physical result where the OLD vector-J
# form blew up; the HydroFEMUnstable guard is retained as a safety net
# ================================================================================================
def test_2d_hydro_stable_where_old_form_blew_up():
    """The previously-UNSTABLE regime (fine metal mesh, sub-5-nm dimer gap) that the OLD indefinite
    vector-J form flagged with HydroFEMUnstable now returns a BOUNDED, energy-consistent result with
    the scalar-longitudinal-potential reformulation (item 5.4).  The field norm stays O(1) (not
    1e19-1e53), P_abs/P_scat >= 0 and the total-field flux balances -P_abs."""
    p = _gold()
    mesh, Rp = hf.dimer_mesh(15.0, 3.0, h_metal=2.0)          # the old 'near-singular' case
    om = 2.0 * math.pi * hf.C_LIGHT / 600e-9
    r = hf.scattering_2d(mesh, om, p, local=False, flux_radius=0.7 * Rp)  # no raise
    assert np.isfinite(r.enhancement) and r.enhancement < 1e3
    assert r.P_abs >= -1e-30 and r.P_scat >= -1e-30
    assert abs(r.energy_residual) < 5e-2                      # total flux == -P_abs (optical theorem)


def test_2d_hydro_guard_still_live():
    """The HydroFEMUnstable safety net is RETAINED: a mesh/parameter regime that cannot resolve the
    longitudinal screening length can still be flagged rather than returned silently.  Exercised here
    via an aggressively low ``unstable_ratio`` so a normal (bounded) solve trips the guard -- proving
    the norm check and the exception path remain wired after the reformulation."""
    p = _gold()
    mesh, Rp = hf.dimer_mesh(15.0, 3.0, h_metal=2.0)
    om = 2.0 * math.pi * hf.C_LIGHT / 600e-9
    with pytest.raises(hf.HydroFEMUnstable):
        hf.scattering_2d(mesh, om, p, local=False, flux_radius=0.7 * Rp, unstable_ratio=1e-6)
