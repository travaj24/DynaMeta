"""DEVSIM gates of the v0.15 III-V recombination presets (consolidated plan item 4.2.2, P2 item 3: R3, the
DEVSIM half of R4 and R5; user decision 30). Pure gates: tests/test_iii_v_presets.py. Audit note:
docs/audit/2026-11-06-v0.15-extended-precision-design-iii-v.md.

The fixture is the R26 uniform-generation slab of tests/test_bipolar_recombination_bgn_ionization.py
(minority-blocking majority contact, graded mesh, far end local), REUSED, with GaAs-scale inputs:
n_i = 2.1e6 cm^-3 (the commonly tabulated GaAs value [secondary]; inferred cross-check from Vurgaftman,
Meyer & Ram-Mohan, J. Appl. Phys. 89, 5815 (2001) Table I -- E_g(300 K) = 1.519 eV - 0.5405 meV/K
T^2/(T + 204 K) = 1.4225 eV, m_e 0.067, Luttinger gamma1 6.98 / gamma2 2.06 -> m_hh 0.350, m_lh 0.090,
DOS hole mass 0.380: n_i = 1.8e6 cm^-3, same decade), n0 = 1e17 cm^-3. n_i^2 = 4.4e24 m^-6 must be
representable (it is: 4.41e24). ORACLE: the scipy brentq root of G = B_eff((n0+d)(p0+d) - n_i^2) + U_SRH
written in the R26 module (_U_oracle), with B_eff = 3.3e-10 x (1 - 0.76) cm^3/s RE-TYPED here from Niemeyer
et al. 2019 (never read from the preset). Tolerances are R26-d's (node-wise 1e-12, root 1e-9; measured
there 0 / 2.2e-16).

FAILS ON OLD CODE (40b1ede): semiconductor_presets has no III-V presets (ImportError at collection) and
RadiativeModel takes no kind / photon_recycling_factor (TypeError). Runtime: ~3 s (4 tests).
"""
import math
import warnings

import numpy as np
import pytest

devsim = pytest.importorskip("devsim")
from scipy.optimize import brentq                                          # noqa: E402

import test_bipolar_recombination_bgn_ionization as R26                    # noqa: E402  (the R26 fixture)
from dynameta.carriers import bipolar_options as BO                       # noqa: E402
from dynameta.materials.semiconductor_presets import RADIATIVE_GAAS_NIEMEYER_2019_INTERNAL   # noqa: E402
from dynameta.warning_types import AdvisoryWarning                        # noqa: E402

CM3, CM3S = 1e6, 1e-6
NI_GAAS = 2.1e6 * CM3                          # m^-3
NC_GAAS, NV_GAAS = 4.7e17 * CM3, 9.0e18 * CM3  # inputs only (Boltzmann transport: not used by the oracle)
N0 = 1e17 * CM3
B_INT = 3.3e-10 * CM3S                         # Niemeyer 2019, abstract / Sec. VI (re-typed)
P_R = 0.76                                     # Niemeyer 2019, abstract: 2 um DH
B_EFF = B_INT * (1.0 - P_R)                    # Eq. (1)
TAU, G, LENGTH = 1e-4, 1e28, 500e-6            # SRH 1/tau = 1e4 /s vs B_eff n0 = 7.9e6 /s (99.9 % radiative);
#                                                G -> dp ~ 1.3 % of n0; L_p = 12.5 um, the slab is 40 L_p


def _gaas_slab(dev, radiative):
    lines = [(0.0, 5e-9, "lo"), (2e-6, 100e-9, None), (LENGTH, LENGTH / 40.0, "hi")]
    R26._device(dev, lines, "{:.10e}".format(N0),
                options=BO.BipolarOptions(radiative=radiative, generation_m3s=0.0), tau=TAU, ohmic=(),
                majority_contact=("lo",), n_i=NI_GAAS, n_c=NC_GAAS, n_v=NV_GAAS)
    R26._solve()


def _b_rad(dev):
    return float(devsim.get_parameter(device=dev, region="bulk", name="B_rad"))


def test_r3_recycling_factor_reaches_the_kernel_gaas_slab():
    """GATE R3. GaAs-scale radiative-dominated slab with RADIATIVE_GAAS_NIEMEYER_2019_INTERNAL.
    with_photon_recycling(0.76): (i) the B_rad parameter IS B_int (1 - P_r) (re-typed) bit for bit, (ii) the
    URad node model equals B_eff (n p - n_i^2) re-evaluated from the SOLVED n, p at every node to 1e-12, (iii)
    the far-end excess density is the brentq root of G = B_eff(...) + U_SRH to 1e-9, (iv) the far-end
    lifetime d/G is 1/(B_eff n0) to within the SRH share (1/tau / (B_eff n0) = 1.3e-3) plus the injection
    term (d/n0 = 1.25e-2) -- budget 2e-2. MEASURED: (ii) max rel 0, (iii) 2.2e-16, (iv) tau B_eff n0 =
    0.9865. ADVERSARIAL: the same slab with the factor dropped (the internal B used as the net loss) has a
    4.12x smaller excess density (root ratio asserted > 4; 1/(1 - P_r) = 4.17 less the SRH/injection terms),
    so a factor that never reached the kernel misses (iii) by 3e9 x its tolerance."""
    dev = "iiiv_r3"
    model = RADIATIVE_GAAS_NIEMEYER_2019_INTERNAL.with_photon_recycling(P_R)
    try:
        _gaas_slab(dev, model)
        R26._ramp_G(dev, G)
        b_rad = _b_rad(dev)
        n, p, urad = R26._v(dev, "Electrons"), R26._v(dev, "Holes"), R26._v(dev, "URad")
    finally:
        R26._drop(dev)
    assert b_rad == B_EFF                                                    # (i) one factor, bitwise
    assert NI_GAAS * NI_GAAS == pytest.approx(4.41e24, rel=1e-12, abs=0.0)  # representable
    assert np.allclose(urad, B_EFF * (n * p - NI_GAAS * NI_GAAS), rtol=1e-12, atol=0.0)   # (ii)
    n0 = 0.5 * N0 + math.sqrt(0.25 * N0 * N0 + NI_GAAS * NI_GAAS)
    p0 = NI_GAAS * NI_GAAS / n0

    def root(b):
        return brentq(lambda d: R26._U_oracle(n0 + d, p0 + d, tau=TAU, b=b, ni=NI_GAAS) - G, 0.0, 1e26,
                      xtol=1e-300, rtol=1e-15)
    d_eff = root(B_EFF)
    assert p[-1] - p0 == pytest.approx(d_eff, rel=1e-9, abs=0.0)             # (iii)
    assert (d_eff / G) * B_EFF * n0 == pytest.approx(1.0, abs=2e-2)          # (iv) tau = 1/(B_eff n0)
    assert d_eff / root(B_INT) > 4.0                                         # adversarial: 4.12x


def test_r4_internal_without_factor_warns_exactly_once_and_uses_B_int():
    """GATE R4 (DEVSIM half). An INTERNAL preset with no photon_recycling_factor builds with exactly ONE
    AdvisoryWarning for the region (recycling ignored -> radiative loss overestimated) and writes B_int
    itself (the advisory changes nothing numerically). An effective B and a legacy B build silently."""
    dev = "iiiv_r4"
    try:
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter("always")
            _gaas_slab(dev, RADIATIVE_GAAS_NIEMEYER_2019_INTERNAL)
        hits = [w for w in rec if issubclass(w.category, AdvisoryWarning)
                and "photon recycling is ignored" in str(w.message)]
        assert len(hits) == 1, [str(w.message)[:80] for w in rec]
        assert "Niemeyer" in str(hits[0].message) and "'bulk'" in str(hits[0].message)
        assert _b_rad(dev) == B_INT
    finally:
        R26._drop(dev)
    for silent in (BO.RadiativeModel(B_m3s=B_EFF, kind="effective"), BO.RadiativeModel(B_m3s=B_EFF)):
        dev = "iiiv_r4s"
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("error", AdvisoryWarning)
                _gaas_slab(dev, silent)
            assert _b_rad(dev) == B_EFF
        finally:
            R26._drop(dev)


@pytest.mark.parametrize("model", [BO.RADIATIVE_SI_TRUPKE_2003, BO.RadiativeModel(B_m3s=0.1e-15 + 0.2e-15)],
                         ids=["trupke", "unlabelled"])
def test_r5_legacy_radiative_writes_the_same_B_rad_bits(model):
    """GATE R5 (DEVSIM half). A legacy (unlabelled) RadiativeModel writes B_rad = its B_m3s bit for bit (the
    region-parameter readback is compared with ==, the float the user typed). The call-log digest of the
    two option-free fixtures (test_bipolar_recombination_bgn_ionization::test_defaults_off_*) is the other
    half: unchanged, not re-recorded."""
    dev = "iiiv_r5"
    try:
        _gaas_slab(dev, model)
        assert _b_rad(dev) == model.B_m3s
    finally:
        R26._drop(dev)
