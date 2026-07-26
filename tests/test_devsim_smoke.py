"""Skip-gated DEVSIM smoke tests (audit 1.2/6.3): the DEVSIM solver modules previously had
ZERO pytest presence -- an API/install break surfaced only tens of minutes into a manual
`make validate`, unlike the NGSolve drivers which got test_fem_drivers.py. These smokes
build the smallest proven device (the uniform bar of validation/contact_current_drivers)
and run one equilibrium + one unipolar-DD solve end to end; they self-skip without devsim
(matching the CI legs, where the heavy stack is absent) and take seconds where present.
Correctness pins live in the validation gates -- these only guard 'builds, solves,
returns finite fields'."""
import numpy as np
import pytest

devsim = pytest.importorskip("devsim")

P = 300e-9
T_SI = 40e-9
ND = 1.0e24


def _const(v):
    return lambda n: np.full_like(np.asarray(n, dtype=float), v)


def _bar_design(physics):
    from dynameta.constants import M_E
    from dynameta.geometry import Design, Electrode, Layer, Stack, UnitCell
    from dynameta.geometry.specs import Mesh2DSpec
    from dynameta.materials import ConstantOptical, Material, MaterialRegistry, TransportModel
    reg = MaterialRegistry()
    reg.add(Material("air", ConstantOptical(1.0 + 0j)))
    reg.add(Material("cmetal", ConstantOptical(-50.0 + 5.0j), is_metal=True))
    reg.add(Material("Si_bar", ConstantOptical(12.0 + 0j),
                     transport=TransportModel(n_bg_m3=ND, eps_static=11.7,
                                              dos_mass_kg_of_n_m3=_const(1.08 * M_E),
                                              band_gap_eV=1.12, chi_eV=4.05, physics=physics,
                                              mobility_m2Vs_of_n_m3=_const(0.135))))
    stack = Stack(layers=[Layer("si", T_SI, "Si_bar")], superstrate_material="air",
                  substrate_material="air")
    electrodes = [Electrode("anode", "si", "x_lo", role="biased"),
                  Electrode("cathode", "si", "x_hi", role="ground", fixed_voltage_V=0.0)]
    return Design(name="smoke_" + physics, unit_cell=UnitCell.square(P), stack=stack,
                  electrodes=electrodes, materials=reg, mesh_2d=Mesh2DSpec())


def _solve(physics, bias_V, tag):
    from dynameta.carriers import eq_registry as _R
    from dynameta.carriers.devsim_layered import LayeredDevsimBuilder
    from dynameta.sweep import BiasPoint
    b = LayeredDevsimBuilder(_bar_design(physics), mesh_name="smk_m_" + tag,
                             device_name="smk_d_" + tag)
    try:
        return b.solve(BiasPoint({"anode": bias_V}, tag))
    finally:
        try:
            devsim.delete_device(device=b.device)
            devsim.delete_mesh(mesh=b.mesh_name)
            _R.clear(b.device)
        except Exception:
            pass


def _finite_fields(cf):
    assert "si" in cf.regions
    fields = cf.regions["si"].grid_fields
    assert fields, "no grid fields extracted"
    for key, arr in fields.items():
        a = np.asarray(arr, dtype=float)
        assert np.all(np.isfinite(a)), "non-finite values in field {!r}".format(key)
    return fields


def test_devsim_equilibrium_smoke():
    fields = _finite_fields(_solve("equilibrium", 0.0, "eq"))
    if "n_m3" in fields:            # a flat neutral bar sits at its background density
        n = np.asarray(fields["n_m3"], dtype=float)
        assert 0.5 * ND < float(np.median(n)) < 2.0 * ND


def test_devsim_dd_smoke():
    fields = _finite_fields(_solve("drift_diffusion", 0.01, "dd"))
    if "n_m3" in fields:
        assert float(np.max(np.asarray(fields["n_m3"], dtype=float))) > 1e20


# =====================================================================================================
# audit C-10 (wave-5): the bipolar OHMIC CONTACT pinned the built-in potential with BOLTZMANN
# mass-action while the interior transport ran the Fermi-Dirac g-factor (fd_enhancement=True, which
# both shipped builders hardcode) -- 0.17 V of silent error per degenerate contact at eta = 10.
# =====================================================================================================

V_T_SI = 0.025851999786187648        # KB*T_REF/Q_E, the module's own thermal voltage


def test_c10_fd_potential_shift_matches_the_exact_fermi_dirac_offset():
    """The correction Delta(u) = int_0^u (g-1) dx/x is the built-in offset the FD-enhanced
    Scharfetter-Gummel interior implies (zero current => dpsi = g V_t dln n). Its oracle is the
    EXACT Fermi-Dirac offset eta - ln(F_1/2(eta)) -- an independent route through the
    Aymerich-Humet F_1/2 and its inversion, sharing no code with the rational g fit."""
    from dynameta.carriers.physics_bipolar_dd import fd_potential_shift
    from dynameta.carriers.physics_equilibrium import F12_aymerich_humet
    assert fd_potential_shift(0.0) == 0.0                    # Boltzmann off-switch, EXACT
    assert fd_potential_shift(1e-30) < 1e-25                 # and continuous into it
    # The AH fit is the FRAME this assertion works in, so pin the frame itself against EXACT
    # quadrature of F_1/2 (audit C-10 residual: the previous line asserted invert_F12(F12(eta)) ==
    # eta, which is true of ANY invertible pair and validated nothing about the physics).
    quad = pytest.importorskip("scipy.integrate").quad
    import math

    def F12_exact(eta):
        f = lambda x: math.sqrt(x) / (1.0 + math.exp(min(700.0, x - eta)))
        pts = [max(eta, 0.0)] if eta > 0 else None
        a, _ = quad(f, 0.0, max(eta, 0.0) + 60.0, limit=400, points=pts)
        b, _ = quad(f, max(eta, 0.0) + 60.0, max(eta, 0.0) + 260.0, limit=200)
        return (a + b) / math.gamma(1.5)

    for eta in (0.0, 10.0, 30.0):                            # three eta, the AH paper's own claim
        assert F12_aymerich_humet(eta) == pytest.approx(F12_exact(eta), rel=5e-3)
    worst = 0.0
    for eta in (0.0, 2.0, 5.0, 10.0, 15.0, 20.0, 30.0):
        u = F12_aymerich_humet(eta)
        exact = eta - np.log(u)                              # = (E_F - E_c)/V_t - ln(n/N_c)
        got = fd_potential_shift(u)
        worst = max(worst, abs(got - exact) / exact)
        if eta >= 10.0:
            assert abs(got - exact) / exact < 0.01           # <1 % where degeneracy matters
    assert worst < 0.05
    # ... and the same claim in the EXACT frame, which is where the docstring's "<0.5 % for
    # eta >= 15" does NOT hold (0.60 % at 15, 0.50 % at 25, 0.40 % at the eta=32 ceiling).
    e15 = abs(fd_potential_shift(F12_exact(15.0)) / (15.0 - math.log(F12_exact(15.0))) - 1.0)
    assert 0.005 < e15 < 0.01
    for eta in (10.0, 20.0, 32.0):
        ue = F12_exact(eta)
        rel = fd_potential_shift(ue) / (eta - math.log(ue)) - 1.0
        assert 0.0 < rel < 0.01                              # OVERSHOOTS, by under 1 %, at the ceiling
    # the headline number the audit measured: 0.176 V at eta = 10, and it is a RISE in |V_bi|
    d10 = fd_potential_shift(F12_aymerich_humet(10.0))
    assert d10 * V_T_SI == pytest.approx(0.176, rel=0.02)
    assert np.all(np.diff(fd_potential_shift(np.array([0.0, 0.1, 1.0, 10.0, 100.0]))) > 0.0)


def _bipolar_1d_device(dev, fd_enhancement, nd_m3, n_dos_m3, n_i_m3=1.0e16):
    """Smallest 1-D bipolar device: uniform n-type bar with one ohmic contact."""
    from dynameta.carriers import eq_registry as _R
    from dynameta.carriers import physics_bipolar_dd as BP
    mesh = dev + "_mesh"
    for _d in list(devsim.get_device_list()):
        if _d == dev:
            devsim.delete_device(device=_d)
    for _m in list(devsim.get_mesh_list()):
        if _m == mesh:
            devsim.delete_mesh(mesh=_m)
    _R.clear(dev)
    devsim.create_1d_mesh(mesh=mesh)
    devsim.add_1d_mesh_line(mesh=mesh, pos=0.0, ps=2e-9, tag="lo")
    devsim.add_1d_mesh_line(mesh=mesh, pos=100e-9, ps=2e-9, tag="hi")
    devsim.add_1d_contact(mesh=mesh, name="lo", tag="lo", material="metal")
    devsim.add_1d_contact(mesh=mesh, name="hi", tag="hi", material="metal")
    devsim.add_1d_region(mesh=mesh, material="Si", region="bulk", tag1="lo", tag2="hi")
    devsim.finalize_mesh(mesh=mesh)
    devsim.create_device(mesh=mesh, device=dev)
    devsim.node_model(device=dev, region="bulk", name="NetDoping",
                      equation="{:.10e}".format(nd_m3))
    BP.setup_bipolar_region(dev, "bulk", eps_static=11.7, n_dos_m3=n_dos_m3, n_i_m3=n_i_m3,
                            mobility_n_m2Vs=0.135, mobility_p_m2Vs=0.048, tau_n_s=1e-7,
                            tau_p_s=1e-7, fd_enhancement=fd_enhancement)
    BP.setup_equilibrium_seed_models(dev, "bulk")
    BP.setup_contact_ohmic_bipolar(dev, "lo")
    return mesh


def _cleanup(dev, mesh):
    from dynameta.carriers import eq_registry as _R
    try:
        devsim.delete_device(device=dev)
        devsim.delete_mesh(mesh=mesh)
        _R.clear(dev)
    except Exception:
        pass


def _pinned_offset_V(dev):
    """The built-in offset the ohmic contact actually pins [V]: the contact node model evaluates to
    Potential - bias + offset, and both Potential and the bias are 0 before any solve, so its value
    AT THE CONTACT NODE is the offset itself."""
    v = np.asarray(devsim.get_node_model_values(device=dev, region="bulk",
                                                name="lo_potential_dirichlet"), dtype=float)
    return float(v[0])


def test_c10_degenerate_contact_carries_the_fermi_dirac_built_in_offset():
    """A DEGENERATE ohmic contact (n0/N_c = 24.1, i.e. eta = 10 -- the ITO-class regime the module's
    own docstring invites) must pin the built-in potential with the FD offset its FD interior
    implies. Measured on the SHIPPED contact model, against the same device built with
    fd_enhancement=False: the difference is the 0.178 V the audit predicted, and it agrees with the
    EXACT Fermi-Dirac offset V_t (eta - ln u) to 1 %."""
    from dynameta.carriers.physics_bipolar_dd import fd_potential_shift
    from dynameta.carriers.physics_equilibrium import invert_F12
    N_c, n0, n_i = 1.0e25, 2.41022e26, 1.0e16                 # u = n0/N_c = 24.1022 -> eta = 10
    u = n0 / N_c                                              # the n_i^2 / +1 terms are negligible
    dev, dev_b = "c10_fd", "c10_fd_bz"
    mesh = _bipolar_1d_device(dev, True, n0, N_c, n_i)
    try:
        off_fd = _pinned_offset_V(dev)
        assert "FDContactShift" in devsim.get_node_model_list(device=dev, region="bulk")
        vals = np.asarray(devsim.get_node_model_values(device=dev, region="bulk",
                                                       name="FDContactShift"), dtype=float)
        assert float(np.max(np.abs(vals - fd_potential_shift(u)))) < 1e-12
    finally:
        _cleanup(dev, mesh)
    mesh_b = _bipolar_1d_device(dev_b, False, n0, N_c, n_i)   # SAME device, Boltzmann statistics
    try:
        off_bz = _pinned_offset_V(dev_b)
    finally:
        _cleanup(dev_b, mesh_b)
    assert off_bz == pytest.approx(-V_T_SI * np.log(u * N_c / n_i), rel=1e-9)   # the legacy value
    d_fd = off_fd - off_bz
    assert d_fd == pytest.approx(-V_T_SI * fd_potential_shift(u), rel=1e-9)
    assert abs(d_fd) == pytest.approx(0.178, rel=0.02)        # the audit's ~0.17 V per contact
    exact = -V_T_SI * (invert_F12(u) - np.log(u))             # exact FD, independent route
    assert d_fd == pytest.approx(exact, rel=0.01)


def test_c10_boltzmann_region_and_nondegenerate_contact_are_unchanged():
    """fd_enhancement=False keeps the pure-Boltzmann contact expression -- no node model, no
    reference -- and a NON-degenerate FD region (the validated Si diode regime, n/N_c = 4e-2)
    carries a correction of only ~0.3 mV, so the shipped diode gates cannot move."""
    from dynameta.carriers.physics_bipolar_dd import fd_potential_shift
    dev = "c10_bz"
    mesh = _bipolar_1d_device(dev, False, 1.0e24, 2.8e25)
    try:
        assert "FDContactShift" not in devsim.get_node_model_list(device=dev, region="bulk")
        off_bz = _pinned_offset_V(dev)
        assert off_bz == pytest.approx(-V_T_SI * np.log(1.0e24 / 1.0e16), rel=1e-9)
    finally:
        _cleanup(dev, mesh)
    dev2 = "c10_si"
    mesh2 = _bipolar_1d_device(dev2, True, 1.0e24, 2.8e25)     # the shipped Si diode doping
    try:
        vals = np.asarray(devsim.get_node_model_values(device=dev2, region="bulk",
                                                       name="FDContactShift"), dtype=float)
        assert abs(_pinned_offset_V(dev2) - off_bz) < 5.0e-4   # sub-mV: the Si diode cannot move
        assert float(np.max(vals)) * V_T_SI < 5.0e-4
        assert float(np.max(vals)) == pytest.approx(fd_potential_shift(1.0e24 / 2.8e25), rel=1e-9)
    finally:
        _cleanup(dev2, mesh2)


# ----------------------------------------------------------------------------------------------
# audit C-10, wave-5 RESIDUALS: the SEED potential's sign on a p-type body, the silent material
# lookup fallback, and the quadrature's peak memory.
# ----------------------------------------------------------------------------------------------

def test_c10_equilibrium_seed_matches_the_pinned_contact_on_BOTH_polarities():
    """The staged bipolar solvers seed Potential from the charge-neutral equilibrium and go STRAIGHT
    into the coupled Newton, so the seed must sit in the frame the ohmic contact PINS. Adding a bare
    +FDContactShift was right on the n side and wrong by 2 V_t Delta on the p side (IntrinsicElectrons
    is n_i^2/p0 there, so the log term already flipped sign and the correction did not): measured
    +0.355 V of seed error at eta = 10, exactly twice the 0.178 V correction, on a p-type body.

    This drives the SHIPPED expression builder -- the one devsim_3d, devsim_layered and
    validation/bipolar_diode all call -- on a real DEVSIM device of each polarity."""
    from dynameta.carriers import physics_bipolar_dd as BP
    N_c, n_i = 2.8e25, 1.0e16                                 # N_dos_p defaults to N_dos (== N_c)
    u = 24.0847                                               # eta = 10 on either side
    for tag, nd in (("n", +u * N_c), ("p", -u * N_c)):
        dev = "c10_seed_" + tag
        mesh = _bipolar_1d_device(dev, True, nd, N_c, n_i)
        try:
            shift = BP.fd_shift_model(dev, "bulk")
            assert shift == "FDContactShift"                  # the FD frame is live on both sides
            devsim.node_model(device=dev, region="bulk", name="_seed_psi",
                              equation=BP.equilibrium_seed_psi_expr(shift))
            seed = float(np.asarray(devsim.get_node_model_values(
                device=dev, region="bulk", name="_seed_psi"), dtype=float)[0])
            pinned = -_pinned_offset_V(dev)                   # contact pins Potential = bias - offset
            assert seed == pytest.approx(pinned, abs=1e-12), tag
            # and it agrees with the single-source scalar the 3-D gate reference uses
            assert seed == pytest.approx(
                BP.ohmic_built_in_offset_V(nd, n_i_m3=n_i, n_dos_m3=N_c, n_dos_p_m3=N_c), abs=1e-12)
            # the pre-fix expression is reproduced as the BUG on the p side (+2 V_t Delta) and is
            # exact on the n side -- which is why it survived
            devsim.node_model(device=dev, region="bulk", name="_seed_stale",
                              equation="V_t*(log(IntrinsicElectrons/n_i) + FDContactShift)")
            stale = float(np.asarray(devsim.get_node_model_values(
                device=dev, region="bulk", name="_seed_stale"), dtype=float)[0])
            err = stale - pinned
            if tag == "n":
                assert err == pytest.approx(0.0, abs=1e-12)
            else:
                assert err == pytest.approx(0.355, rel=0.02)
                assert err == pytest.approx(2.0 * V_T_SI * BP.fd_potential_shift(u), rel=1e-6)
        finally:
            _cleanup(dev, mesh)
    # fd_enhancement=False keeps the legacy string CHARACTER-FOR-CHARACTER
    assert BP.equilibrium_seed_psi_expr(None) == "V_t*log(IntrinsicElectrons/n_i)"


def test_c10_failed_material_lookup_warns_instead_of_reverting_to_boltzmann_silently():
    """`_fd_contact_shift` returns None on any exception. For a region with no `fd_enhancement`
    parameter that is correct and quiet -- it is not a bipolar region. But a region that DECLARES
    fd_enhancement > 0 and then fails an n_i / N_dos / N_dos_p lookup silently reverted its contact
    to Boltzmann while the interior kept the FD g-factor, i.e. re-opened C-10 with no diagnostic."""
    import warnings as _w
    from dynameta.carriers import physics_bipolar_dd as BP
    dev = "c10_nolookup"
    mesh = _bipolar_1d_device(dev, True, 24.0847 * 2.8e25, 2.8e25)
    try:
        real = devsim.get_parameter

        def fake(**kw):
            if kw.get("name") == "N_dos_p":
                raise RuntimeError("N_dos_p not found")
            return real(**kw)

        BP.ds.get_parameter = fake
        try:
            with _w.catch_warnings(record=True) as caught:
                _w.simplefilter("always")
                got = BP._fd_contact_shift(dev, "hi")
        finally:
            BP.ds.get_parameter = real
        assert got is None                                     # it still falls back ...
        msgs = [str(x.message) for x in caught]
        assert any("N_dos_p" in m and "C-10" in m for m in msgs), msgs   # ... but says so
        assert any("BOLTZMANN" in m for m in msgs), msgs
    finally:
        _cleanup(dev, mesh)


def test_c10_quadrature_is_chunked_and_bit_identical():
    """fd_potential_shift evaluated an (n_unique, 96) block in one allocation: a graded doping
    profile with 1e6 distinct u values took 5.5 GB. It is now chunked; rows are independent, so the
    result must be BIT-identical at any chunk size."""
    from dynameta.carriers import physics_bipolar_dd as BP
    rng = np.random.default_rng(0)
    u = rng.uniform(0.0, 130.0, size=5000)
    ref = np.array(BP.fd_potential_shift(u), dtype=float)
    old_chunk = BP._FD_CHUNK
    try:
        for chunk in (1, 7, 512, 10 ** 9):                     # 1e9 == the unchunked original
            BP._FD_CHUNK = chunk
            got = np.array(BP.fd_potential_shift(u), dtype=float)
            assert got.tobytes() == ref.tobytes(), chunk
    finally:
        BP._FD_CHUNK = old_chunk
    # the budget itself: one block is ~12 MB, not ~770 MB for a 1e6-value profile
    assert BP._FD_CHUNK * 8 * BP._FD_GL_N <= 16 * 1024 * 1024
    assert BP._FD_CHUNK >= 1024                                # ... and not so small it crawls
    # shapes/scalars survive the flatten-reshape
    assert isinstance(BP.fd_potential_shift(1.0), float)
    assert np.asarray(BP.fd_potential_shift(np.zeros((3, 4)) + 2.0)).shape == (3, 4)
