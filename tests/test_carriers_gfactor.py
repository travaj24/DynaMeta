"""Fast unit coverage for the generalized-Einstein g-factor fit (carriers/einstein.py) -- the
load-bearing replacement for the old degenerate asymptote (audit F1 / DD-3). Pure numpy+scipy,
no devsim (the DD modules' DEVSIM-string g_expr cannot be unit-tested directly, so this pins
the SHARED source the string is built from). Checks: the Boltzmann VALUE limit g(0)=1; the fit
tracks the exact F_1/2(eta)/F_-1/2(eta) to the documented ~1.1% peak over eta in [-4,32] and
<0.5% across ITO's regime (eta>=10); and the shipped coefficients are the audited values.
Run: python -m pytest tests/test_carriers_gfactor.py -q
"""
from math import gamma

import numpy as np
import pytest
from scipy.integrate import quad

from dynameta.carriers.einstein import g_einstein, GA, GB, GC, GD


def _fd(j, eta):
    """Complete Fermi-Dirac integral F_j(eta) = 1/Gamma(j+1) int_0^inf t^j/(1+exp(t-eta)) dt."""
    # clamp the exponent so the deep-tail integrand underflows cleanly (no overflow warning)
    val, _ = quad(lambda t: t ** j / (1.0 + np.exp(np.minimum(t - eta, 700.0))), 0.0, np.inf,
                  limit=200)
    return val / gamma(j + 1.0)


def test_g_boltzmann_limit_is_exact():
    assert g_einstein(0.0) == pytest.approx(1.0, abs=1e-12)
    assert g_einstein(1e-6) == pytest.approx(1.0, abs=1e-4)


def test_g_coefficients_are_the_audited_values():
    # pins the SHARED source the DEVSIM g_expr strings build from -- a coefficient typo here
    # would otherwise pass every test and only surface in a minutes-long DEVSIM validation.
    assert (GA, GB, GC, GD) == (0.33717, 0.13356, 0.14143, 0.20570)


def test_g_tracks_exact_fermi_ratio_within_documented_bounds():
    etas = np.linspace(-4.0, 32.0, 73)
    err = np.array([abs(g_einstein(_fd(0.5, e)) - _fd(0.5, e) / _fd(-0.5, e))
                    / (_fd(0.5, e) / _fd(-0.5, e)) for e in etas])
    assert err.max() < 0.013                       # ~1.1% peak (documented)
    assert err[etas >= 10.0].max() < 0.005         # <0.5% across ITO's regime
    # at ITO's operating point (eta~22) the fit is sub-0.5%, far better than the old
    # degenerate asymptote's 6.6% (the F1 improvement this fit delivers)
    e22 = _fd(0.5, 22.0)
    assert abs(g_einstein(e22) - _fd(0.5, 22.0) / _fd(-0.5, 22.0)) / (
        _fd(0.5, 22.0) / _fd(-0.5, 22.0)) < 0.005
