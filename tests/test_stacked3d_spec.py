"""Fast solver-free unit tests for the 3D gated-cap Stacked3DSpec: the bipolar-param validation, the
multi-semiconductor heterostack stack assembly, and the equilibrium-only guard. No DEVSIM solve -- only
the dataclass + layer_stack_nm logic (the rigorous solver oracles live in validation/carriers_3d_*)."""
import pytest

pytest.importorskip("devsim")  # devsim_3d imports the physics modules (devsim) at module load

from dynameta.carriers.devsim_3d import Stacked3DSpec
from dynameta.carriers.physics_equilibrium import M_E

BASE = dict(semi_material="ITO", oxide_material="HfO2", lateral_m=12e-9, semi_thk_m=12e-9,
            oxide_thk_m=8e-9, n_bg_m3=4e26, eps_semi=9.5, eps_oxide=18.0, dos_mass_kg=0.35 * M_E)


def test_physics_value_validated():
    with pytest.raises(ValueError):
        Stacked3DSpec(physics="nonsense", **BASE)


def test_bipolar_requires_n_i_and_lifetimes():
    with pytest.raises(ValueError):
        Stacked3DSpec(physics="bipolar_dd", **BASE)                       # missing n_i_m3
    with pytest.raises(ValueError):
        Stacked3DSpec(physics="bipolar_dd", n_i_m3=1e16, tau_srh_s=0.0,
                      mobility_p_m2Vs=0.02, **BASE)                        # tau <= 0
    # a complete bipolar spec constructs
    s = Stacked3DSpec(physics="bipolar_dd", n_i_m3=1e16, tau_srh_s=1e-7, mobility_p_m2Vs=0.02, **BASE)
    assert s.physics == "bipolar_dd"


def test_extra_semiconductors_unique_names():
    with pytest.raises(ValueError):
        Stacked3DSpec(extra_semiconductors=[("semi", "ITO", 5e-9, 9.5, 4e26)], **BASE)  # clashes with primary
    with pytest.raises(ValueError):
        Stacked3DSpec(extra_semiconductors=[("xs", "ITO", 5e-9, 9.5, 4e26),
                                            ("xs", "ITO", 5e-9, 9.5, 4e26)], **BASE)     # duplicate


def test_multisemi_equilibrium_only_guard():
    # multi-semiconductor is supported only for equilibrium (semi-semi carrier continuity for DD/bipolar
    # is not yet wired) -- the spec must reject it loudly.
    for phys, extra in (("drift_diffusion", {}),
                        ("bipolar_dd", dict(n_i_m3=1e16, tau_srh_s=1e-7, mobility_p_m2Vs=0.02))):
        with pytest.raises(ValueError):
            Stacked3DSpec(physics=phys, extra_semiconductors=[("xs", "ITO", 5e-9, 9.5, 4e26)],
                          **extra, **BASE)


def test_heterostack_layer_stack_and_field_region():
    s = Stacked3DSpec(semi_thk_m=6e-9,
                      extra_semiconductors=[("xs", "ITO", 5e-9, 9.5, 2e26)], **{k: v for k, v in BASE.items()
                                                                                if k != "semi_thk_m"})
    stack = s.layer_stack_nm()
    names = [r[0] for r in stack]
    roles = [r[2] for r in stack]
    assert names == ["semi", "xs", "oxide"]                              # primary | extra | oxide
    assert roles == ["semiconductor", "semiconductor", "dielectric"]
    # contiguous z-ranges: semi [0,6], xs [6,11], oxide [11,19] (nm)
    assert abs(stack[0][4] - 6.0) < 1e-9 and abs(stack[1][3] - 6.0) < 1e-9
    assert abs(stack[1][4] - 11.0) < 1e-9 and abs(stack[2][3] - 11.0) < 1e-9
    assert s.field_devsim_region() == "xs"                               # gate-adjacent emitted
    assert abs(s.semiconductor_nbg("xs") - 2e26) < 1e18                  # per-region n_bg
    assert abs(s.semiconductor_nbg("semi") - 4e26) < 1e18


def test_single_semiconductor_unchanged():
    s = Stacked3DSpec(**BASE)
    assert s.field_devsim_region() == "semi"
    assert [r[0] for r in s.layer_stack_nm()] == ["semi", "oxide"]
