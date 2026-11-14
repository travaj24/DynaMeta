"""Gates for the 2026-07-25 v0.9.0 audit's carriers-package hygiene items closed in wave 6
(2026-11-02):

  V-20   `dynameta.carriers.__all__` named 8 submodules that were NOT attributes until something
         imported them (`dynameta.carriers.llg` raised AttributeError in a fresh process although
         __all__ and dir() advertised it). Every submodule now resolves lazily via __getattr__.
  X-21   `carriers.__dir__` returned sorted(__all__), hiding exactly the pure-numpy submodules the
         package docstring advertises as the CI-testable path.
  X-15   the effective-DOS N_c formula had FOUR implementations; it now has one
         (carriers.physics_equilibrium.effective_dos_m3), pinned by a source census, and every
         former copy's caller is bit-identical to the old inline value.

Run: python -m pytest tests/test_audit_2026_07_25_carriers_hygiene.py -q
"""
import pathlib
import re
import subprocess
import sys

import numpy as np

_PKG = pathlib.Path(__file__).resolve().parents[1] / "dynameta"


def test_v20_x21_every_advertised_carriers_name_resolves_in_a_fresh_process():
    """Fresh interpreter, so no earlier test has bound the submodules onto the package. Importing
    the package must stay devsim-free; dir() must list the pure-numpy submodules; every name in
    __all__ and dir() that does not need devsim must resolve as an attribute."""
    code = "\n".join([
        "import sys, dynameta.carriers as c",
        "assert 'devsim' not in sys.modules, 'import dynameta.carriers pulled devsim'",
        "d = dir(c)",
        "assert {'schrodinger_poisson', 'sp_carrier', 'einstein', 'lc_director'} <= set(d), d",
        "assert set(c.__all__) <= set(d)",
        "pure = ['llg', 'lc_director', 'lc_director_2d', 'lc_dynamics', 'carrier_heating',"
        " 'schrodinger_poisson', 'sp_carrier', 'einstein']",
        "mods = [getattr(c, n) for n in pure]",
        "assert all(m.__name__ == 'dynameta.carriers.' + n for m, n in zip(mods, pure))",
        "assert 'devsim' not in sys.modules, 'a pure-numpy submodule pulled devsim'",
        "missing = []",
        "for n in sorted(set(c.__all__) | set(d)):",
        "    try:",
        "        getattr(c, n)",
        "    except AttributeError:",
        "        missing.append(n)",
        "    except ImportError:",              # a leg without the optional devsim
        "        pass",
        "assert not missing, missing",
        "print('clean')"])
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                         cwd=str(_PKG.parent))
    assert out.returncode == 0 and "clean" in out.stdout, out.stderr
    import dynameta.carriers as c
    try:
        c.definitely_not_a_submodule
    except AttributeError:
        pass
    else:
        raise AssertionError("a non-existent name must still raise AttributeError")


def test_resolving_every_carriers_name_does_not_load_ngsolve():
    """v0.14 task 8 (user decision 2026-11-04): the FEM carrier modules (electrostatics_fem,
    electrothermal, fem_mesh, thermal_fem.*) import ngsolve LAZILY (dynameta.lazy_import.LazyModule).
    ngsolve's own OpenBLAS commits ~4 GB per process at load; MEASURED (box A) for this child: private
    commit 6170 MB on ee0ad04 (ngsolve + netgen loaded) vs 1652 MB now, = `import dynameta.carriers`
    alone (1649 MB). Fails on ee0ad04 ('ngsolve' in sys.modules). The FEM tests (test_fem_drivers,
    test_electrothermal, ...) are the gate that the lazy proxy still solves."""
    code = "\n".join([
        "import sys, dynameta.carriers as c",
        "for n in sorted(set(c.__all__) | set(dir(c))):",
        "    try:",
        "        getattr(c, n)",
        "    except ImportError:",
        "        pass",
        "import dynameta.carriers.thermal_fem, dynameta.carriers.thermal_fem.kirchhoff",
        "import dynameta.carriers.thermal_fem.twotemp, dynameta.carriers.electrothermal",
        "loaded = sorted(m for m in ('ngsolve', 'netgen') if m in sys.modules)",
        "assert not loaded, loaded",
        "print('lazy')"])
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                         cwd=str(_PKG.parent))
    assert out.returncode == 0 and "lazy" in out.stdout, out.stderr[-3000:]


def test_lazy_module_proxy_contract():
    """LazyModule forwards attributes, imports once, and a MISSING package raises the same
    ModuleNotFoundError (with .name) the eager `import` did -- at first use."""
    from dynameta.lazy_import import LazyModule
    j = LazyModule("json")
    assert not j.is_loaded
    import json
    assert j.dumps is json.dumps and j.is_loaded and "loaded" in repr(j)
    missing = LazyModule("dynameta_no_such_pkg_xyz")
    try:
        missing.anything
    except ModuleNotFoundError as e:
        assert e.name == "dynameta_no_such_pkg_xyz"
    else:
        raise AssertionError("a missing package must raise ModuleNotFoundError at first use")


_NC_RE = re.compile(r"HBAR\s*\*\*\s*2\s*\)\s*\)\s*\*\*\s*1\.5")


def test_x15_one_effective_dos_implementation_and_callers_bit_identical():
    hits = []
    for p in _PKG.rglob("*.py"):
        for i, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1):
            if _NC_RE.search(line):
                hits.append("{}:{}".format(p.relative_to(_PKG).as_posix(), i))
    assert len(hits) == 1 and hits[0].startswith("carriers/physics_equilibrium.py:"), hits
    from dynameta.carriers.physics_equilibrium import effective_dos_m3
    from dynameta.constants import HBAR, KB, M_E, T_REF
    from dynameta.optics.soa.qw_gain import _band_dos
    for m_rel in (0.067, 0.35, 0.5):
        m = m_rel * M_E
        old_np = 2.0 * (m * KB * 300.0 / (2.0 * np.pi * HBAR ** 2)) ** 1.5      # devsim_* copies
        old_pe = 2.0 * (m * KB * T_REF / (2.0 * np.pi * HBAR ** 2)) ** 1.5      # setup_phi_c0 inline
        assert effective_dos_m3(m) == old_np == old_pe
        for T in (77.0, 298.0, 400.0):
            old_soa = 2.0 * (m_rel * M_E * KB * T / (2.0 * np.pi * HBAR ** 2)) ** 1.5
            assert _band_dos(m_rel, T) == old_soa
