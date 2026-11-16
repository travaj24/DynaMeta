"""v0.16: drude_eps_jax refuses DrudeOptical options it cannot trace. Needs lumenairy (core)."""
import pytest


def test_drude_eps_jax_refuses_options_it_cannot_trace():
    """Before the fix drude_eps_jax returned the plain single-pole eps for a DrudeOptical with a
    background model (0.5756+0.2618j at 8e26 m^-3, 1200 nm, while eps() gives -0.3323+0.2618j) and for one with
    boundary scattering (the bulk value): a gradient-design run optimized a different material. Now any option
    beyond the single pole is refused (the census is the dataclass fields, so a later option is covered too);
    the plain model is unchanged (the existing byte-identity pin in test_lumenairy_bridge.py)."""
    from dynameta.constants import M_E
    from dynameta.materials import BoundaryScattering, DrudeOptical, TabulatedBackground
    from dynameta.optics.lumenairy_bridge import drude_eps_jax
    plain = DrudeOptical(4.0, 0.30 * M_E, 1.2e14)
    assert plain.options_beyond_single_pole() == ()
    bs = DrudeOptical(4.0, 0.30 * M_E, 1.2e14, boundary=BoundaryScattering(perp_coefficient=0.75, thickness_m=6e-9))
    bg = DrudeOptical(4.0, 0.30 * M_E, 1.2e14, background=TabulatedBackground(
        n_m3=(1e25, 1e26, 1e27), eps_inf=(4.05, 3.95, 3.0)))
    for model, name in ((bs, "boundary"), (bg, "background")):
        assert model.options_beyond_single_pole() == (name,)
        with pytest.raises(NotImplementedError, match=name):
            drude_eps_jax(model)
    assert abs(complex(drude_eps_jax(plain)(8e26, 1200e-9)) - complex(plain.eps(1200e-9, n_m3=8e26))) < 1e-14
