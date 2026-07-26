"""Fast (no FEM) tests for the TMM reference helper -- validates the n_list/d_list/units
wiring against analytic Fresnel + energy conservation, and the Design extractor. Uses the
`tmm` dependency (already required); runs in CI."""
import numpy as np
import pytest

from dynameta.optics.tmm_reference import stack_rta, design_layer_stack


# ------------------------------------------------------------------------------------------------
# Independent Abeles (characteristic-matrix) TMM oracle -- hand-written, NOT the `tmm` package that
# stack_rta calls, so the layer-ORDER gate below is not self-referential (audit V-1).
# Convention exp(-i omega t): forward wave ~ exp(+i kz z), Im(eps) >= 0 = loss, so the per-layer
# characteristic matrix mapping (E_tan, H_tan) from the layer's TOP face to its BOTTOM face is
#   m = [[cos q, -i sin q / Y], [-i Y sin q, cos q]],  q = kz d,  Y = kz (s-pol) or eps/kz (p-pol).
# ------------------------------------------------------------------------------------------------
def _abeles_rta(n_super, layers_super_first, n_sub, lambda_m, *, theta_deg=0.0, pol="s"):
    """(R, T, A) for super | layers | sub, layers ordered SUPERSTRATE-side first."""
    k0 = 2.0 * np.pi / float(lambda_m)
    eps_s, eps_b = complex(n_super) ** 2, complex(n_sub) ** 2
    k_par = complex(n_super) * k0 * np.sin(np.radians(float(theta_deg)))

    def kz(eps):
        return np.sqrt(complex(eps) * k0 * k0 - k_par * k_par + 0j)

    def adm(eps):
        return kz(eps) if pol == "s" else complex(eps) / kz(eps)

    M = np.eye(2, dtype=np.complex128)
    for n_j, d_j in layers_super_first:
        eps_j = complex(n_j) ** 2
        q = kz(eps_j) * float(d_j)
        Y = adm(eps_j)
        M = M @ np.array([[np.cos(q), -1j * np.sin(q) / Y],
                          [-1j * Y * np.sin(q), np.cos(q)]], dtype=np.complex128)
    Ys, Yb = adm(eps_s), adm(eps_b)
    B = M[0, 0] + M[0, 1] * Yb                       # [B; C] = M @ [1; Y_sub]
    C = M[1, 0] + M[1, 1] * Yb
    r = (Ys * B - C) / (Ys * B + C)
    t = 2.0 * Ys / (Ys * B + C)
    R = float(abs(r) ** 2)
    T = float((Yb.real / Ys.real) * abs(t) ** 2)
    return R, T, 1.0 - R - T


def test_single_interface_fresnel_normal():
    # bare n=1 | n=1.5 interface, normal incidence: R = ((1-1.5)/(1+1.5))^2 = 0.04
    R, T, A = stack_rta(1.0, [], 1.5, 1300e-9, theta_deg=0.0, pol="s")
    assert abs(R - 0.04) < 1e-6
    assert abs(A) < 1e-9                       # lossless -> no absorption
    assert abs(R + T + A - 1.0) < 1e-9
    # T carries the index factor: T = 1 - R for this interface (tmm convention)
    assert abs(T - 0.96) < 1e-6


def test_lossless_slab_energy_conserves():
    R, T, A = stack_rta(1.0, [(2.0, 250e-9)], 1.0, 1300e-9, theta_deg=20.0, pol="p")
    assert abs(A) < 1e-9
    assert abs(R + T + A - 1.0) < 1e-9
    assert 0.0 < R < 1.0


def test_lossy_slab_absorbs():
    R, T, A = stack_rta(1.0, [(2.0 + 0.1j, 250e-9)], 1.0, 1300e-9, pol="s")
    assert A > 0.0                              # a lossy slab absorbs
    assert abs(R + T + A - 1.0) < 1e-9


def test_s_p_differ_at_angle():
    rs = stack_rta(1.0, [(2.0, 250e-9)], 1.5, 1300e-9, theta_deg=45.0, pol="s")
    rp = stack_rta(1.0, [(2.0, 250e-9)], 1.5, 1300e-9, theta_deg=45.0, pol="p")
    assert abs(rs[0] - rp[0]) > 1e-3            # s and p reflectance differ at oblique


@pytest.mark.parametrize("theta,pol", [(0.0, "s"), (35.0, "s"), (35.0, "p")])
def test_abeles_oracle_matches_stack_rta(theta, pol):
    """Pin the hand-written oracle against `tmm` on an ASYMMETRIC lossy stack (both orderings are
    distinct here), so the ordering gate below judges with a trustworthy independent TMM."""
    layers = [(2.0 + 0.0j, 95e-9), (np.sqrt(6.0 + 0.6j), 210e-9)]
    for lam in (1300e-9, 1550e-9):
        got = stack_rta(1.0, layers, 1.5, lam, theta_deg=theta, pol=pol)
        ref = _abeles_rta(1.0, layers, 1.5, lam, theta_deg=theta, pol=pol)
        assert max(abs(a - b) for a, b in zip(got, ref)) < 1e-12


def test_design_layer_stack_is_superstrate_first():
    """AUDIT V-1: design_layer_stack must return layers SUPERSTRATE-first (the order stack_rta
    documents and every sibling Design->stack builder produces). `Stack.layers` is bottom-to-top,
    so an unreversed extractor vertically flips the stack -- and the energy budget closes in BOTH
    orderings (|R+T+A-1| = 0), so only an independent TMM catches it. Gated against the
    hand-written Abeles oracle AND against the repo's own correct extractor
    (layered_stack_from_design + layered_rta)."""
    from dynameta.materials import Material, MaterialRegistry, ConstantOptical
    from dynameta.geometry import UnitCell, Stack, Layer, Design
    from dynameta.optics.tmm_reference import layered_rta, layered_stack_from_design
    reg = MaterialRegistry()
    reg.add(Material("air", ConstantOptical(1.0 + 0j)))
    reg.add(Material("glass", ConstantOptical(2.25 + 0j)))       # n = 1.5 substrate
    reg.add(Material("mid", ConstantOptical(4.0 + 0j)))          # n = 2, lossless
    reg.add(Material("abs", ConstantOptical(6.0 + 0.6j)))        # lossy, thick
    # bottom-to-top = [abs, mid]  ->  superstrate-first = [mid, abs]; air | ... | glass
    d = Design(name="asym", unit_cell=UnitCell.square(300e-9),
               stack=Stack(layers=[Layer("bot", 210e-9, "abs"), Layer("top", 95e-9, "mid")],
                           superstrate_material="air", substrate_material="glass"),
               electrodes=[], materials=reg)
    lam = 1300e-9
    n_sup, layers, n_sub = design_layer_stack(d, lam)
    assert abs(complex(n_sup) - 1.0) < 1e-12 and abs(complex(n_sub) - 1.5) < 1e-12
    # the extractor's own ordering: top layer ('mid', n=2, 95 nm) first
    assert abs(complex(layers[0][0]) - 2.0) < 1e-12 and abs(layers[0][1] - 95e-9) < 1e-18
    for theta, pol in [(0.0, "s"), (35.0, "s"), (35.0, "p")]:
        got = stack_rta(n_sup, layers, n_sub, lam, theta_deg=theta, pol=pol)
        ref = _abeles_rta(n_sup, layers, n_sub, lam, theta_deg=theta, pol=pol)
        assert max(abs(a - b) for a, b in zip(got, ref)) < 1e-12
        # the FLIPPED (bottom-first) ordering is a different, wrong answer -- the regression this
        # gate exists for; energy still closes there, which is why nothing downstream caught it.
        flip = _abeles_rta(n_sup, list(reversed(layers)), n_sub, lam, theta_deg=theta, pol=pol)
        assert abs(flip[0] + flip[1] + flip[2] - 1.0) < 1e-12    # energy closes in BOTH orderings
        assert abs(flip[0] - ref[0]) > 0.1 * ref[0]              # ... but R differs grossly
        # cross-gate: the repo's own reversed extractor must agree exactly
        stack = layered_stack_from_design(d, lam)
        assert max(abs(a - b) for a, b in
                   zip(layered_rta(stack, lam, theta_deg=theta, pol=pol), ref)) < 1e-12


def test_design_layer_stack_extract_and_reject_inclusions():
    from dynameta.materials import Material, MaterialRegistry, ConstantOptical
    from dynameta.geometry import UnitCell, Stack, Layer, Inclusion, Design, centered_square
    reg = MaterialRegistry()
    reg.add(Material("air", ConstantOptical(1.0 + 0j)))
    reg.add(Material("hi", ConstantOptical(4.0 + 0j)))   # n=2
    cell = UnitCell.square(300e-9)
    # uniform stack -> extracts and matches a manual stack_rta
    d_uniform = Design(name="u", unit_cell=cell,
                       stack=Stack(layers=[Layer("slab", 250e-9, "hi")],
                                    superstrate_material="air", substrate_material="air"),
                       electrodes=[], materials=reg)
    n_sup, layers, n_sub = design_layer_stack(d_uniform, 1300e-9)
    assert abs(complex(layers[0][0]) - 2.0) < 1e-9 and abs(layers[0][1] - 250e-9) < 1e-18
    R1, _, _ = stack_rta(n_sup, layers, n_sub, 1300e-9)
    R2, _, _ = stack_rta(1.0, [(2.0, 250e-9)], 1.0, 1300e-9)
    assert abs(R1 - R2) < 1e-9
    # a layer with an inclusion is laterally structured -> TMM must refuse
    d_struct = Design(name="s", unit_cell=cell,
                      stack=Stack(layers=[Layer("p", 250e-9, "air",
                                   inclusions=[Inclusion(centered_square(cell, 120e-9), "hi")])],
                                   superstrate_material="air", substrate_material="air"),
                      electrodes=[], materials=reg)
    with pytest.raises(ValueError):
        design_layer_stack(d_struct, 1300e-9)
