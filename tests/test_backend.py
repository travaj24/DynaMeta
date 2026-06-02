"""Unit coverage for the array-backend seam (core.backend) and the xp-agnostic constitutive
EffectModels. The NumPy-path tests always run; the JAX tests are skipped if jax is not installed.
Run: python -m pytest tests/test_backend.py -q
"""
import subprocess
import sys

import numpy as np
import pytest

from dynameta.core.backend import (array_namespace, backend_name, to_numpy,
                                    is_numpy_array, JAX_AVAILABLE)


def test_numpy_default_namespace_and_predicates():
    assert array_namespace(np.zeros(3)) is np
    assert array_namespace(1.0, [1, 2], None) is np          # scalars / lists / None -> NumPy
    assert array_namespace() is np
    assert backend_name(np) == "numpy"
    assert is_numpy_array(np.zeros(3)) and not is_numpy_array([1, 2])
    assert np.array_equal(to_numpy(np.arange(3)), np.arange(3))


def test_numpy_path_does_not_import_jax_or_cupy():
    # REGRESSION GUARD: a constitutive map on NumPy input must NOT eagerly import jax/cupy (the
    # is_*_array predicates import them; array_namespace checks NumPy first to avoid it). Run in a
    # fresh process so the result is independent of any jax-using test that ran earlier in-suite.
    code = (
        "import sys, numpy as np;"
        "from dynameta.core.effects import ThermoOpticModel, PockelsEffect;"
        "ThermoOpticModel(eps_ref=complex(12.1,0.0), dn_dT=1.8e-4).eps({'T':350.0}, 1.3e-6);"
        "PockelsEffect(eps_bg=np.eye(3)*4+0j, r_voigt=np.zeros((6,3)))"
        ".eps({'E':np.array([0.,0.,1e7])}, 1.3e-6);"
        "assert 'jax' not in sys.modules, 'jax imported on the numpy path';"
        "assert 'cupy' not in sys.modules, 'cupy imported on the numpy path';"
        "print('clean')")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert out.returncode == 0 and "clean" in out.stdout, out.stderr


@pytest.mark.skipif(not JAX_AVAILABLE, reason="jax not installed")
def test_mixed_backend_raises():
    import jax.numpy as jnp
    with pytest.raises(TypeError):
        array_namespace(np.zeros(3), jnp.zeros(3))           # mixing backends in one call


@pytest.mark.skipif(not JAX_AVAILABLE, reason="jax not installed")
def test_jax_constitutive_matches_numpy_in_float64():
    import jax
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    from dynameta.core.effects import (PockelsEffect, KerrEffect, FranzKeldyshEffect,
                                       ThermoOpticModel, PCMModel, LiquidCrystalModel)
    from dynameta.core.graphene import graphene_sigma
    from dynameta.constants import Q_E as Q
    no, ne, r13, r33 = 2.21, 2.14, 9.6e-12, 30.9e-12
    eps_bg = np.diag([no ** 2, no ** 2, ne ** 2]).astype(complex)
    r = np.zeros((6, 3)); r[0, 2] = r13; r[1, 2] = r13; r[2, 2] = r33
    Ez = 1.0e7
    pairs = [
        (PockelsEffect(eps_bg=eps_bg, r_voigt=r), {"E": np.array([0., 0., Ez])}, {"E": jnp.array([0., 0., Ez])}),
        (KerrEffect(eps_bg=(2.0 ** 2) * np.eye(3) + 0j, s_kerr=1e-18), {"E": np.array([0., 0., 1e8])}, {"E": jnp.array([0., 0., 1e8])}),
        (FranzKeldyshEffect(eps_bg=complex(12.0, 0.1), beta=1e-8), {"E": np.array([0., 0., 1e6])}, {"E": jnp.array([0., 0., 1e6])}),
        (ThermoOpticModel(eps_ref=complex(3.48 ** 2, 0.0), dn_dT=1.8e-4), {"T": 350.0}, {"T": jnp.asarray(350.0)}),
        (PCMModel(complex(16.0, 0.5), complex(36.0, 6.0)), {"crystalline_fraction": 0.4}, {"crystalline_fraction": jnp.asarray(0.4)}),
        (LiquidCrystalModel(1.53, 1.71), {"director_angle_rad": 0.4}, {"director_angle_rad": jnp.asarray(0.4)}),
    ]
    for model, f_np, f_jx in pairs:
        e_np = np.asarray(model.eps(f_np, 1300e-9))
        e_jx = model.eps(f_jx, 1300e-9)
        assert "complex128" in str(e_jx.dtype) or "float64" in str(e_jx.dtype)   # x64 enforced
        assert np.allclose(np.asarray(e_jx), e_np, atol=1e-12)
    s_np = complex(graphene_sigma(0.3 * Q, 1.55e-6))
    s_jx = complex(graphene_sigma(jnp.asarray(0.3 * Q), 1.55e-6))
    assert np.allclose(s_jx, s_np, atol=1e-12)


@pytest.mark.skipif(not JAX_AVAILABLE, reason="jax not installed")
def test_jax_autodiff_pockels_matches_analytic_slope():
    # the POINT of the seam: a constitutive map is differentiable. d(n_x)/d(E_z) of the Pockels
    # response must equal the analytic linear-EO slope -0.5 n_o^3 r13 (to the linearization error).
    import jax
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    from dynameta.core.effects import PockelsEffect
    no, ne, r13 = 2.21, 2.14, 9.6e-12
    eps_bg = np.diag([no ** 2, no ** 2, ne ** 2]).astype(complex)
    r = np.zeros((6, 3)); r[0, 2] = r13; r[1, 2] = r13
    pk = PockelsEffect(eps_bg=eps_bg, r_voigt=r)

    def nx(Ez):
        e = pk.eps({"E": jnp.stack([0.0 * Ez, 0.0 * Ez, Ez])}, 1300e-9)
        return jnp.sqrt(jnp.real(e[0, 0]))

    g = float(jax.grad(nx)(jnp.asarray(1.0e7)))
    assert abs(g - (-0.5 * no ** 3 * r13)) < 1e-3 * abs(0.5 * no ** 3 * r13)
