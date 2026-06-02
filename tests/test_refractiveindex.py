"""Unit coverage for RefractiveIndexInfoOptical (materials.optical_model) -- the refractiveindex.info
loader ported from the sibling Lumenairy library. Skipped when the optional `refractiveindex`
package is not installed. Run: python -m pytest tests/test_refractiveindex.py -q
"""
import numpy as np
import pytest

from dynameta.materials.optical_model import (RefractiveIndexInfoOptical, TabulatedOptical,
                                              _REFRACTIVEINDEX_AVAILABLE)

pytestmark = pytest.mark.skipif(not _REFRACTIVEINDEX_AVAILABLE,
                                reason="optional `refractiveindex` package not installed")


def test_lossless_dielectric_sio2():
    eps = complex(RefractiveIndexInfoOptical("main", "SiO2", "Malitson").eps(1550e-9))
    assert np.sqrt(eps.real) == pytest.approx(1.444, abs=3e-3)   # fused-silica n at 1550 nm
    assert abs(eps.imag) < 1e-9                                  # n-only (lossless) entry -> k = 0


def test_lossy_metal_au_passive_convention():
    eps = complex(RefractiveIndexInfoOptical("main", "Au", "Johnson").eps(1550e-9))
    assert eps.imag > 0.0                                        # Im(eps) = 2 n k > 0 (exp(-iwt) passive)
    assert eps.real < 0.0                                        # metal below the plasma frequency
    assert 0.3 < np.sqrt(eps).real < 0.8                         # n ~ 0.52 (Johnson-Christy Au @1550)


def test_density_broadcast_and_out_of_range_raises():
    m = RefractiveIndexInfoOptical("main", "SiO2", "Malitson")
    out = m.eps(1550e-9, n_m3=np.zeros((2, 3)))                  # density-independent -> broadcast
    assert out.shape == (2, 3) and np.allclose(out, complex(m.eps(1550e-9)))
    with pytest.raises(ValueError):
        m.eps(1.0e-3)                                           # 1 mm: far outside the entry's range


def test_to_tabulated_snapshot_matches_on_the_fly():
    m = RefractiveIndexInfoOptical("main", "Si", "Aspnes")       # covers ~207-830 nm
    lams = np.array([400e-9, 500e-9, 600e-9, 700e-9])
    tab = m.to_tabulated(lams)
    assert isinstance(tab, TabulatedOptical)
    for L in lams:                                              # exact at the sampled nodes
        assert complex(tab.eps(float(L))) == pytest.approx(complex(m.eps(float(L))), rel=1e-9)
