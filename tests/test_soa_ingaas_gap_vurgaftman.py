"""Audit 2026-11-06 (v0.14 row 3.2-14): the Ga0.47In0.53As (InGaAs lattice-matched to InP) band gap
of optics.soa against Vurgaftman, Meyer & Ram-Mohan, J. Appl. Phys. 89, 5815 (2001).

THE DEFECT. temperature.VARSHNI_PARAMS["InGaAs_LM"] and the BulkGainParams default (qw_gain) both
carried a Varshni triple (0.816 eV, 2.9e-4 eV/K, 193 K) "from Vurgaftman". Vurgaftman tabulates
Varshni parameters for BINARIES only; an alloy gap is Eq. (4.1) over the two binary laws with a
bowing parameter, which for GaInAs the paper takes temperature-independent (Sec. IV.A.2). The old
triple matches at 0 K (816 meV vs the alloy rule's 816.12) but is 3.97 meV too wide at 77 K,
25.80 meV at 300 K and 36.40 meV at 400 K, and its 300 K slope (-0.2456 meV/K) is 30 % below the alloy rule's (-0.3505 meV/K).

THE ORACLE is the paper's tables RE-TYPED below and Eq. (4.1) evaluated here, independently of the
module (no dynameta call in _oracle_eg):
  GaAs  Table I    Eg(Gamma) = 1.519 eV, alpha = 0.5405 meV/K, beta = 204 K
  InAs  Table III  Eg(Gamma) = 0.417 eV, alpha = 0.276  meV/K, beta = 93 K
  InP   Table VI   Eg(Gamma) = 1.4236 eV, alpha = 0.363 meV/K, beta = 162 K
  GaInAs Table XIII  Gamma-gap bowing C = 0.477 eV;  Ga(1-x)In(x)As at x = 0.53 (Sec. IV.A.2)
plus the two numbers the paper itself QUOTES: Eg(T=0) = 816 meV for Ga0.47In0.53As (Sec. IV.A.2,
"composite average" of 810-821 meV measurements), and the 300 K end points of the InP-lattice-matched
GaInAsP law, 1.353 eV (InP) and 0.737 eV (Ga0.47In0.53As) (Sec. V.B.1). The paper quotes no measured
room-temperature value for this alloy (it says only that a temperature-dependent C would put the
room-temperature gap "higher than nearly all experimental values").

ERROR BUDGETS. Oracle vs kernel: the same closed form in a different operation order, <= ~20
double-precision operations on O(1 eV) terms -> <= 20 * 2.2e-16 * 1.5 eV ~ 7e-15 eV; tolerance
1e-13 eV (14x). Quoted paper values: rounded to 1 meV in the last digit -> +-0.5 meV.
"""
import numpy as np
import pytest
from dataclasses import replace

from dynameta.constants import C_LIGHT, H_PLANCK, Q_E
from dynameta.optics.soa import qw_gain as qw
from dynameta.optics.soa import temperature as tm
from dynameta.optics.soa.qw_gain import BulkGainParams

# --- Vurgaftman 2001, re-typed (eV, eV/K, K) ---------------------------------------------------
GAAS = (1.519, 0.5405e-3, 204.0)      # Table I
INAS = (0.417, 0.276e-3, 93.0)        # Table III
INP = (1.4236, 0.363e-3, 162.0)       # Table VI
C_GAINAS = 0.477                      # Table XIII, Gamma-gap bowing (eV), T-independent
X_IN = 0.53                           # Ga0.47In0.53As lattice-matched to InP

# the pre-v0.14 shipped triple (the defect)
OLD_TRIPLE = (0.816, 2.9e-4, 193.0)

TOL_EV = 1.0e-13                      # module docstring budget
TOL_QUOTED_EV = 0.5e-3                # paper values rounded to 1 meV


def _varshni(T, e0, a, b):
    return e0 - a * T ** 2 / (T + b)


def _oracle_eg(T, x=X_IN, C=C_GAINAS):
    """Vurgaftman Eq. (4.1): Eg(A(1-x)B(x)) = (1-x) Eg(A) + x Eg(B) - x (1-x) C, A = GaAs, B = InAs."""
    return (1.0 - x) * _varshni(T, *GAAS) + x * _varshni(T, *INAS) - x * (1.0 - x) * C


@pytest.mark.parametrize("T", [0.0, 77.0, 300.0, 400.0])
def test_ingaas_lm_gap_matches_retyped_vurgaftman_alloy_rule(T):
    """Fails on the old triple by -0.12 / 3.97 / 25.80 / 36.40 meV at 0 / 77 / 300 / 400 K (MEASURED:
    the old law's value minus _oracle_eg, asserted below as the discrimination check)."""
    ref = _oracle_eg(T)
    assert abs(tm.varshni_eg_ev(T, "InGaAs_LM") - ref) < TOL_EV
    # the gate RESOLVES the defect: the old triple misses this oracle by the documented amount
    miss_old = _varshni(T, *OLD_TRIPLE) - ref
    expected_miss = {0.0: -0.12e-3, 77.0: 3.97e-3, 300.0: 25.80e-3, 400.0: 36.40e-3}[T]
    assert abs(miss_old - expected_miss) < 0.01e-3, (T, miss_old)
    if T > 0.0:
        assert miss_old > 1.0e3 * TOL_EV


def test_ingaas_lm_gap_matches_the_values_the_paper_quotes():
    assert abs(tm.varshni_eg_ev(0.0, "InGaAs_LM") - 0.816) < TOL_QUOTED_EV        # Sec. IV.A.2
    assert abs(tm.varshni_eg_ev(300.0, "InGaAs_LM") - 0.737) < TOL_QUOTED_EV      # Sec. V.B.1
    assert abs(tm.varshni_eg_ev(300.0, "InP") - 1.353) < TOL_QUOTED_EV            # Sec. V.B.1
    # the old triple fails the paper's own 300 K end point by 26 meV
    assert abs(_varshni(300.0, *OLD_TRIPLE) - 0.737) > 50 * TOL_QUOTED_EV


def test_binary_rows_match_retyped_tables():
    for name, row in (("GaAs", GAAS), ("InAs", INAS), ("InP", INP)):
        assert tm.VARSHNI_PARAMS[name] == pytest.approx(row, rel=1e-15)
        for T in (0.0, 77.0, 300.0, 400.0):
            assert abs(tm.varshni_eg_ev(T, name) - _varshni(T, *row)) < TOL_EV
    assert "InGaAs_LM" not in tm.VARSHNI_PARAMS          # an alloy has no Varshni row
    assert tm.ALLOY_GAP_PARAMS["InGaAs_LM"] == ("GaAs", "InAs", X_IN, C_GAINAS)


def test_ingaas_lm_slope_is_the_alloy_weighted_binary_slope():
    """dEg/dT against a central difference of the INDEPENDENT oracle (h = 1e-3 K: truncation
    ~h^2 Eg''' ~ 1e-15 eV/K, rounding ~eps * Eg / h ~ 2e-13 eV/K -> rel 1e-8 on 3.5e-4 eV/K is a
    > 1000x margin). Old triple at 300 K: -2.456e-4 eV/K, 30 % below the alloy rule (MEASURED)."""
    h = 1.0e-3
    for T in (77.0, 300.0, 400.0):
        fd = (_oracle_eg(T + h) - _oracle_eg(T - h)) / (2.0 * h)
        assert tm.d_eg_dT_ev_per_K(T, "InGaAs_LM") == pytest.approx(fd, rel=1e-8)
    old = -OLD_TRIPLE[1] * 300.0 * (300.0 + 2 * OLD_TRIPLE[2]) / (300.0 + OLD_TRIPLE[2]) ** 2
    assert abs(old / tm.d_eg_dT_ev_per_K(300.0, "InGaAs_LM") - 1.0) > 0.29


def test_adversarial_composition_and_bowing_perturbations_break_the_gate():
    """A swapped In/Ga composition (x = 0.47) or the Paul et al. bowing (C = 0.475 eV, quoted in
    Sec. IV.A.2) through the kernel's custom-ternary path must MISS the oracle by far more than
    TOL_EV -- the gate pins the composition and the bowing, not just "something near 0.74 eV"."""
    ref = _oracle_eg(300.0)
    swapped = tm.varshni_eg_ev(300.0, ("GaAs", "InAs", 0.47, C_GAINAS))
    paul = tm.varshni_eg_ev(300.0, ("GaAs", "InAs", X_IN, 0.475))
    assert abs(swapped - ref) > 0.05                    # ~ 62 meV
    assert abs(paul - ref) > 4.0e-4                     # x(1-x) * 2 meV = 0.50 meV
    # and the custom path itself reproduces the preset exactly
    assert tm.varshni_eg_ev(300.0, ("GaAs", "InAs", X_IN, C_GAINAS)) == \
        tm.varshni_eg_ev(300.0, "InGaAs_LM")
    with pytest.raises(ValueError, match="not a binary"):
        tm.varshni_eg_ev(300.0, ("InGaAs_LM", "InAs", 0.5, 0.0))
    with pytest.raises(ValueError, match="unknown material"):
        tm.varshni_eg_ev(300.0, "InGaAsP")


# --- the SOA consumers -------------------------------------------------------------------------

def test_bulk_gain_default_gap_is_the_vurgaftman_alloy_law():
    p = BulkGainParams()
    assert p.gap_law == "InGaAs_LM"
    for T in (77.0, 298.0, 300.0, 400.0):
        eg = qw.band_edge_Eg_J(0.0, replace(p, T_K=T)) / Q_E
        assert abs(eg - _oracle_eg(T)) < TOL_EV, T
    # the explicit device-effective override is all-or-nothing
    with pytest.raises(ValueError, match="ALL THREE"):
        BulkGainParams(Eg0_eV=0.80)
    cband = BulkGainParams(Eg0_eV=OLD_TRIPLE[0], varshni_alpha_eV_K=OLD_TRIPLE[1],
                           varshni_beta_K=OLD_TRIPLE[2])
    assert abs(qw.band_edge_Eg_J(0.0, cband) / Q_E - _varshni(298.0, *OLD_TRIPLE)) < TOL_EV


def test_gain_spectrum_moves_by_exactly_the_gap_change():
    """Rule-7 cross-check that the default-device change is ONLY the gap: in the Connelly form the
    gap enters through sqrt(nu - Eg/h) and the 1/nu^2 prefactor alone (the quasi-Fermi levels depend
    on N, T and the masses only), so at the same N and T, g(Eg + x) (Eg + x)^2 is the SAME function
    of the excess photon energy x for any gap (re-derived from the qw_gain module formula; budget
    ~eps * Eg / x ~ 1e-14 relative -> rel 1e-12). Then the MEASURED peak moves by dEg up to the
    1/nu^2 tilt of the peak (|2 dEg / E^2| over the ~1/(30 meV)^2 curvature of ln g: < 0.15 meV)
    and the peak-search grid (0.25 eV / 4000 = 62.5 ueV per device): budget 0.3 meV."""
    new = BulkGainParams()
    old = BulkGainParams(Eg0_eV=OLD_TRIPLE[0], varshni_alpha_eV_K=OLD_TRIPLE[1],
                         varshni_beta_K=OLD_TRIPLE[2])
    N = 1.6950415e24                                   # the 130 mA operating point (P = 0)
    x_J = np.linspace(1.0e-3, 0.12, 60) * Q_E
    eg_new, eg_old = qw.band_edge_Eg_J(N, new), qw.band_edge_Eg_J(N, old)
    g_new = qw.material_gain_per_m((eg_new + x_J) / H_PLANCK, N, new) * (eg_new + x_J) ** 2
    g_old = qw.material_gain_per_m((eg_old + x_J) / H_PLANCK, N, old) * (eg_old + x_J) ** 2
    np.testing.assert_allclose(g_new, g_old, rtol=1e-12, atol=0.0)
    d_eg_eV = (eg_new - eg_old) / Q_E                  # -25.59 meV at 298 K (MEASURED)
    assert d_eg_eV == pytest.approx(_oracle_eg(298.0) - _varshni(298.0, *OLD_TRIPLE), abs=TOL_EV)
    nu_new, _ = qw.gain_peak(N, new)
    nu_old, _ = qw.gain_peak(N, old)
    d_pk_eV = H_PLANCK * (nu_new - nu_old) / Q_E      # -25.66 meV: 1560.4 -> 1612.5 nm (MEASURED)
    assert abs(d_pk_eV - d_eg_eV) < 0.3e-3, (d_pk_eV, d_eg_eV)
    print("\n[ingaas] peak {:.1f} -> {:.1f} nm  dEg={:.2f} meV  dpeak={:.2f} meV".format(
        C_LIGHT / nu_old * 1e9, C_LIGHT / nu_new * 1e9, d_eg_eV * 1e3, d_pk_eV * 1e3))
