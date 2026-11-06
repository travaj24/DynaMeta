"""v0.14.0 corrections to the shipped bipolar material presets and the incomplete-ionization model
(consolidated plan rows 3.2-4, 3.2-5, 3.2-11; audit note docs/audit/2026-11-06-v0.14-materials-
corrections.md). Pure numpy -- no DEVSIM -- so these run on every `test` leg; the DEVSIM-solved twins
(relaxed slabs, the per-region UnreliableResultWarnings) live in tests/test_bipolar_recombination_bgn_ionization.py.

EVERY ORACLE IS INDEPENDENT OF THE CODE UNDER TEST: literature constants are re-typed below in the
units and with the equation/table numbers of the primary sources; freeze-out roots are solved with
scipy brentq on a hand-written residual; k/q come from scipy.constants.

EVERY GATE FAILS ON THE PRE-FIX CODE (189c648), measured by running this file against it:
  * del Alamo basis: the p-type 1e19 cm^-3 node gave dEg = 49.73 meV against the oracle's 0, the
    compensated (1e19 / 1e18) node 51.51 meV against 49.73 (+1.78 meV);
  * Sb: 39 meV against the table's 42.74 meV (-3.74 meV);
  * units guard: alpha_D_eVm = 3.1e-8 (an eV*cm value) was accepted silently; pearson_bardeen,
    ionization_energies_eV, IonizationUnitsError, bgn_basis_advisory and mott_advisory did not exist.
"""
import math

import numpy as np
import pytest
from scipy.constants import e as QE, k as KB
from scipy.optimize import brentq

CM3 = 1e6                                         # cm^-3 -> m^-3


# =====================================================================================================
# 3.2-4: del Alamo BGN is an n-type fit in N_D
# =====================================================================================================

def _dEg_del_alamo_ted1987(ND_cm3):
    """del Alamo & Swanson, IEEE TED ED-34, 1580 (1987), Eq. (11) (= IEDM 1985 p. 290, Eq. (7)),
    re-typed: dE_g^app = 18.7 meV ln(N_D / 7e17 cm^-3) for N_D > 7e17, else 0. N_D ONLY."""
    return 18.7e-3 * math.log(ND_cm3 / 7e17) if ND_cm3 > 7e17 else 0.0


# (N_D, N_A) cm^-3: pure n, pure p (three levels), compensated n, compensated p, light compensation
_NODES = [(1e19, 0.0), (0.0, 1e18), (0.0, 1e19), (0.0, 1e20), (1e19, 1e18), (1e18, 1e19), (5e17, 5e17)]


def test_del_alamo_is_evaluated_in_donors_only():
    """GATE 3.2-4a. equilibrium_fields with the del Alamo preset gives, at every node, the TED 1987
    Eq. (11) value in N_D -- including 0 on p-type nodes -- to 1e-12 relative (+1e-15 eV absolute for the
    exact zeros): only round-off of one log separates them. Pre-fix (N_D + N_A basis) the p-type 1e19
    node was 49.73 meV off and the compensated 1e19/1e18 node +1.78 meV off.
    ADVERSARIAL: the same fit on the TOTAL basis (the pre-fix behaviour, doping_basis='total') must miss
    the oracle by > 1 meV at the compensated node and by the full 49.73 meV at the p-type one."""
    from dynameta.carriers.bipolar_options import (BGN_SI_DEL_ALAMO_1985, BGNModel, BipolarOptions,
                                                   equilibrium_fields)
    ND = np.array([n[0] for n in _NODES]) * CM3
    NA = np.array([n[1] for n in _NODES]) * CM3
    oracle = np.array([_dEg_del_alamo_ted1987(n[0]) for n in _NODES])

    def dEg(bgn):
        f = equilibrium_fields(ND - NA, n_i_m3=1e16, N_c_m3=2.8e25, N_v_m3=1.04e25,
                               options=BipolarOptions(bgn=bgn), fd_enhancement=False,
                               donors_m3=ND, acceptors_m3=NA)
        return np.asarray(f["dEg_eV"])
    got = dEg(BGN_SI_DEL_ALAMO_1985)
    assert np.all(np.abs(got - oracle) <= 1e-12 * oracle + 1e-15), (got, oracle)
    assert BGN_SI_DEL_ALAMO_1985.doping_basis == "donors"
    # a user-built del_alamo model takes the kind's own basis by default
    assert BGNModel(kind="del_alamo", E_ref_eV=18.7e-3, N_ref_m3=7e23).doping_basis == "donors"
    # the citation names where the fit is (TED 1987 Eq. 11 / IEDM 1985 Eq. 7), not SSE 28, 47
    src = BGN_SI_DEL_ALAMO_1985.source
    assert "1580" in src and "Eq. (11)" in src and "Eq. (7)" in src and "28, 47" not in src
    # adversarial: the total-basis evaluation is far off the oracle where the bases differ
    total = dEg(BGNModel(kind="del_alamo", E_ref_eV=18.7e-3, N_ref_m3=7e23, doping_basis="total"))
    assert abs(total[4] - oracle[4]) > 1e-3 and abs(total[2] - oracle[2]) > 0.049


def test_slotboom_total_basis_is_unchanged():
    """The default 'total' basis of the Slotboom kind is the pre-fix arithmetic N_D + N_A, bitwise (no
    gate moves): re-typed Slotboom & de Graaff 1977 (9 mV, 1e17 cm^-3, C = 0.5) at N_D + N_A, 1e-12."""
    from dynameta.carriers.bipolar_options import BGN_SI_SLOTBOOM_1977, BipolarOptions, equilibrium_fields
    ND = np.array([n[0] for n in _NODES]) * CM3
    NA = np.array([n[1] for n in _NODES]) * CM3
    f = equilibrium_fields(ND - NA, n_i_m3=1e16, N_c_m3=2.8e25, N_v_m3=1.04e25,
                           options=BipolarOptions(bgn=BGN_SI_SLOTBOOM_1977), fd_enhancement=False,
                           donors_m3=ND, acceptors_m3=NA)
    assert BGN_SI_SLOTBOOM_1977.doping_basis == "total"
    for k, (nd, na) in enumerate(_NODES):
        x = math.log((nd + na) / 1e17)
        assert f["dEg_eV"][k] == pytest.approx(9e-3 * (x + math.sqrt(x * x + 0.5)), rel=1e-12)
    assert np.array_equal(BGN_SI_SLOTBOOM_1977.basis_density_m3(ND, NA), ND + NA)


def test_del_alamo_basis_advisory_names_the_unsupported_nodes():
    """GATE 3.2-4b (pure half; the DEVSIM half checks the region warns exactly once). The advisory
    text exists exactly when some node has N_A above the fit's onset N_ref = 7e17 cm^-3: a pure n+
    region and an n+/p diode whose base is 1e17 (the R27-c fixture) are silent; a p+ 1e19 node and a
    compensated 1e19/1e18 node are named. Slotboom (total basis) is never advised."""
    from dynameta.carriers.bipolar_options import (BGN_SI_DEL_ALAMO_1985, BGN_SI_SLOTBOOM_1977,
                                                   bgn_basis_advisory)
    a = np.array
    assert bgn_basis_advisory("r", BGN_SI_DEL_ALAMO_1985, a([1e25, 1e25]), a([0.0, 0.0])) is None
    assert bgn_basis_advisory("r", BGN_SI_DEL_ALAMO_1985, a([1e25, 0.0]), a([0.0, 1e23])) is None
    assert bgn_basis_advisory("r", BGN_SI_DEL_ALAMO_1985, a([0.0, 0.0]), a([7e23, 7e23])) is None
    msg = bgn_basis_advisory("r", BGN_SI_DEL_ALAMO_1985, a([1e25, 0.0, 1e25]), a([0.0, 1e25, 1e24]))
    assert msg is not None and "'r'" in msg and "2 of 3 nodes" in msg and "N_D only" in msg
    assert "1e+19 cm^-3" in msg                                     # the worst N_A, in cm^-3
    assert bgn_basis_advisory("r", BGN_SI_SLOTBOOM_1977, a([0.0]), a([1e26])) is None


# =====================================================================================================
# 3.2-5: Sb ionization energy
# =====================================================================================================

# Ramdas & Rodriguez, Rep. Prog. Phys. 44, 1297 (1981), Table 4 ("Binding energies of the energy levels
# of donors in silicon (meV)"), row 1s(A1), re-typed: P 45.59, As 53.76, Sb 42.74. Quoted to 0.01 meV.
_RR_1S_A1_MEV = {"P": 45.59, "As": 53.76, "Sb": 42.74}


def test_sb_ionization_energy_is_the_spectroscopic_value():
    """GATE 3.2-5. SI_DONOR_IONIZATION_EV['Sb'] equals the 1s(A1) binding energy of Table 4 within half
    its last quoted digit (0.005 meV). Pre-fix: 39 meV, 3.74 meV (= 0.14 kT at 300 K) too shallow.
    ADVERSARIAL: the rounded 42.7 meV misses by 0.04 meV and must fail the same check. P and As keep
    their Sze & Ng values (45, 54 meV), within 1 meV of the table (0.59 / 0.24 meV) -- the plan's
    'optional, < 1 meV' rows, deliberately not moved so the R27-d freeze-out gates stay put."""
    from dynameta.carriers.bipolar_options import SI_DONOR_IONIZATION_EV
    half_digit = 0.005e-3
    assert abs(SI_DONOR_IONIZATION_EV["Sb"] - _RR_1S_A1_MEV["Sb"] * 1e-3) <= half_digit
    assert abs(42.7e-3 - _RR_1S_A1_MEV["Sb"] * 1e-3) > half_digit            # adversarial
    for sp in ("P", "As"):
        assert abs(SI_DONOR_IONIZATION_EV[sp] - _RR_1S_A1_MEV[sp] * 1e-3) < 1e-3


def test_sb_freeze_out_root_uses_the_corrected_level():
    """The corrected level reaches the charge-neutral root: Si:Sb 1e16 cm^-3 at 77 K (N_c = 2.8e19
    (T/300)^1.5 cm^-3, g_D = 2, N_A = 0) -- ionized fraction from equilibrium_fields equals the brentq
    root of n = N_D/(1 + 2 n/n1), n1 = N_c exp(-42.74 meV/kT) (p << n at 77 K: n_i^2/n ~ 1e-25 of n)
    to 1e-11 (the R27-d tolerance). Moved (measured): 0.5033 (39 meV) -> 0.4128 (42.74 meV)."""
    from dynameta.carriers.bipolar_options import IonizationModel, SI_DONOR_IONIZATION_EV
    T, ND = 77.0, 1e22
    nc = 2.8e25 * (T / 300.0) ** 1.5
    kT = KB * T / QE
    ion = IonizationModel(dE_D_eV=SI_DONOR_IONIZATION_EV["Sb"], dE_A_eV=0.045)
    f = equilibrium_fields_at(T, ND, nc, ion)
    n1 = nc * math.exp(-42.74e-3 / kT)
    n = math.exp(brentq(lambda ln_n: math.exp(ln_n) - ND / (1.0 + 2.0 * math.exp(ln_n) / n1),
                        math.log(1e10), math.log(2 * ND), xtol=1e-15, rtol=1e-15, maxiter=500))
    assert f["n0"][0] == pytest.approx(n, rel=1e-11)
    assert 0.40 < n / ND < 0.42 and n / ND < 0.5                    # 39 meV gave 0.503


def equilibrium_fields_at(T, ND, nc, ion):
    from dynameta.carriers.bipolar_options import BipolarOptions, equilibrium_fields
    nv = 1.04e25 * (T / 300.0) ** 1.5
    eg = 1.17 - 4.73e-4 * T * T / (T + 636.0)
    ni = math.sqrt(nc * nv) * math.exp(-eg * QE / (2.0 * KB * T))
    return equilibrium_fields(np.array([ND]), n_i_m3=ni, N_c_m3=nc, N_v_m3=nv,
                              options=BipolarOptions(ionization=ion, T_K=T), fd_enhancement=False)


# =====================================================================================================
# 3.2-11: Pearson-Bardeen units guard, converting constructor, Mott advisory
# =====================================================================================================

# Silvaco ATLAS User's Manual (2004), Eqs. (3-53)/(3-54), re-typed as EXAMPLE values (a code default,
# not a measurement): E_D = 0.044 - 3.6e-8 N_D^(1/3) eV, E_A = 0.0438 - 3.037e-8 N_A^(1/3) eV, N in cm^-3.
_ATLAS = dict(dE_D=0.044, dE_A=0.0438, aD_eVcm=3.6e-8, aA_eVcm=3.037e-8)


def test_pearson_bardeen_constructor_converts_ev_cm():
    """GATE 3.2-11a (P2 I1/I2). pearson_bardeen(..., alpha_*_eVcm) gives the ATLAS energies at
    N = 1e18 cm^-3 evaluated in CGS (0.044 - 3.6e-8 * 1e6 = 8.0 meV; 0.0438 - 3.037e-8 * 1e6 =
    13.43 meV) to 1e-15 eV (round-off of one cube root and one subtraction, ~1e-17); 3.1e-8 eV cm becomes
    exactly the eV m literal 3.1e-10. ADVERSARIAL: the same number passed unconverted into the eV m field
    is refused (next gate) -- and, were the guard absent, would clamp the energy to 0 at 1e18."""
    from dynameta.carriers.bipolar_options import IonizationModel
    m = IonizationModel.pearson_bardeen(_ATLAS["dE_D"], _ATLAS["dE_A"], alpha_D_eVcm=_ATLAS["aD_eVcm"],
                                        alpha_A_eVcm=_ATLAS["aA_eVcm"])
    dED, dEA = m.ionization_energies_eV(np.array([1e18 * CM3]), np.array([1e18 * CM3]))
    assert float(dED[0]) == pytest.approx(0.044 - 3.6e-8 * 1e6, abs=1e-15)
    assert float(dEA[0]) == pytest.approx(0.0438 - 3.037e-8 * 1e6, abs=1e-15)
    assert IonizationModel.pearson_bardeen(0.045, 0.045, alpha_D_eVcm=3.1e-8).alpha_D_eVm == 3.1e-10
    assert IonizationModel.pearson_bardeen(0.045, 0.045, alpha_A_eVcm=3.1e-8).alpha_A_eVm == 3.1e-10


def test_unconverted_alpha_is_refused():
    """GATE 3.2-11b (P2 I2). The units guard refuses an alpha whose zero crossing (dE0/alpha)^3 lies
    below 1e17 cm^-3, with a named error (a ValueError) that says eV*cm vs eV*m. Oracle threshold
    alpha_c = dE0 / (1e23 m^-3)^(1/3) computed here: 1.01 alpha_c raises, 0.99 alpha_c passes, for both
    species. The literature value 3.1e-8 (eV cm) passed into the eV m field raises (it would zero a
    45 meV level at 3.06e12 cm^-3); 3.1e-10 and the ATLAS example (zero at 1.8e18) pass. Pre-fix:
    nothing raised. alpha with dE0 = 0 is inert (no crossing to guard) and is accepted."""
    from dynameta.carriers.bipolar_options import IonizationModel, IonizationUnitsError
    assert issubclass(IonizationUnitsError, ValueError)
    with pytest.raises(IonizationUnitsError, match=r"eV\*cm.*eV\*m"):
        IonizationModel(dE_D_eV=0.045, dE_A_eV=0.045, alpha_D_eVm=3.1e-8)
    with pytest.raises(IonizationUnitsError, match="alpha_A_eVm"):
        IonizationModel(dE_D_eV=0.045, dE_A_eV=0.045, alpha_A_eVm=3.1e-8)
    IonizationModel(dE_D_eV=0.045, dE_A_eV=0.045, alpha_D_eVm=3.1e-10, alpha_A_eVm=3.1e-10)
    for dE0 in (0.045, 0.0438):
        alpha_c = dE0 / (1e17 * CM3) ** (1.0 / 3.0)
        IonizationModel(dE_D_eV=dE0, dE_A_eV=dE0, alpha_D_eVm=0.99 * alpha_c, alpha_A_eVm=0.99 * alpha_c)
        with pytest.raises(IonizationUnitsError):
            IonizationModel(dE_D_eV=dE0, dE_A_eV=dE0, alpha_D_eVm=1.01 * alpha_c)
        with pytest.raises(IonizationUnitsError):
            IonizationModel(dE_D_eV=dE0, dE_A_eV=dE0, alpha_A_eVm=1.01 * alpha_c)
    IonizationModel(dE_D_eV=0.0, dE_A_eV=0.045, alpha_D_eVm=3.1e-8)
    IonizationModel(dE_D_eV=0.045, dE_A_eV=0.045, alpha_D_eVm=1e-300)            # no overflow
    with pytest.raises(ValueError):
        IonizationModel(dE_D_eV=0.045, dE_A_eV=0.045, alpha_D_eVm=math.inf)


def _sip_non_ionized_300K(ND):
    """Si:P (45 meV, g_D = 2) at 300 K, N_c = 2.8e19 cm^-3, uncompensated: brentq of n = N_D+(n);
    returns the non-ionized fraction 1 - n/N_D (holes negligible: n_i^2/n < 1e-12 of n here)."""
    n1 = 2.8e25 * math.exp(-0.045 / (KB * 300.0 / QE))
    n = math.exp(brentq(lambda ln_n: math.exp(ln_n) - ND / (1.0 + 2.0 * math.exp(ln_n) / n1),
                        math.log(1e10), math.log(2 * ND), xtol=1e-15, rtol=1e-15, maxiter=500))
    return 1.0 - n / ND


def test_mott_advisory_fires_above_the_mott_density_only():
    """GATE 3.2-11c (P2 I3, pure half; the DEVSIM half checks one warning per region). Si:P with the
    shipped IONIZATION_SI_P_B: the advisory text exists at 1e19 cm^-3 (> N_crit = 3.74e18) and not at
    1e18; it quotes the model's non-ionized fraction, which must be the brentq root's 61 % (the
    'legacy model 39 % ionized at 1e19' of the plan). Acceptors use the 4.5e18 B proxy (5e18 advised,
    4e18 not). Silent when the Pearson-Bardeen lowering has already zeroed the energy there (ATLAS
    alpha: dE_D = 0 above 1.83e18), and with N_mott_D_m3 = inf. Pre-fix: no advisory existed."""
    from dynameta.carriers.bipolar_options import (IONIZATION_SI_P_B, BipolarOptions, IonizationModel,
                                                   equilibrium_fields, mott_advisory)

    def msg_for(ion, ND, NA=0.0):
        net = np.array([ND - NA])
        f = equilibrium_fields(net, n_i_m3=1e16, N_c_m3=2.8e25, N_v_m3=1.04e25,
                               options=BipolarOptions(ionization=ion, T_K=300.0), fd_enhancement=False,
                               donors_m3=np.array([ND]), acceptors_m3=np.array([NA]))
        return mott_advisory("r", ion, f["N_D"], f["N_A"], f["n0"], f["p0"], f["n1D"], f["p1A"])
    m19 = msg_for(IONIZATION_SI_P_B, 1e25)
    assert m19 is not None and "'r'" in m19 and "Mott" in m19 and "N_D up to 1e+19" in m19
    assert "{:.0f} % NON-ionized".format(100.0 * _sip_non_ionized_300K(1e25)) in m19
    assert round(100.0 * _sip_non_ionized_300K(1e25)) == 61
    assert msg_for(IONIZATION_SI_P_B, 1e24) is None
    assert msg_for(IONIZATION_SI_P_B, 0.0, 5e24) is not None
    assert msg_for(IONIZATION_SI_P_B, 0.0, 4e24) is None
    atlas = IonizationModel.pearson_bardeen(_ATLAS["dE_D"], _ATLAS["dE_A"],
                                            alpha_D_eVcm=_ATLAS["aD_eVcm"], alpha_A_eVcm=_ATLAS["aA_eVcm"])
    assert msg_for(atlas, 1e25) is None
    quiet = IonizationModel(dE_D_eV=0.045, dE_A_eV=0.045, N_mott_D_m3=math.inf)
    assert msg_for(quiet, 1e25) is None
