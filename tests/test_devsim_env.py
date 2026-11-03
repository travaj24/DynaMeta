"""D0 (DEVSIM 2.11.2 core groundwork, 2026-11-03): DEVSIM's BLAS is pinned to a pip-installed library
by dynameta's ONE devsim import home (dynameta.carriers.devsim_env) before the first `import devsim`.

Why (measured, module docstring of devsim_env): with DEVSIM_MATH_LIBS unset, devsim 2.11.2 on
Windows resolves "libopenblas.dll;liblapack.dll;libblas.dll" through PATH, and on the dev box that
loaded an Elmer 8.4 install's OpenBLAS 0.3.7; 24 concurrent copies of the fresh-process carriers
hygiene test then crashed 9/72 times with 0xC0000005 right after that load.

Gates:
  * test_decoy_on_the_search_path_does_not_capture_devsim -- a directory of decoy libopenblas /
    liblapack / libblas files is put FIRST on the loader search path (PATH on Windows,
    LD_LIBRARY_PATH elsewhere) of a fresh child that imports the package's real devsim entry point
    (dynameta.carriers.devsim_layered). The child must import, report the pinned library in
    devsim's info['math_libraries'], and map it (psutil memory_maps) -- and nothing from the decoy
    directory. On the pre-D0 code the child dies in devsim's import ("Error loading math
    libraries" -> RuntimeError: Issues initializing DEVSIM), because the bare names hit the decoys.
  * test_user_math_libs_is_passed_through_verbatim -- a user DEVSIM_MATH_LIBS always wins: the same
    child with the user pointing at a decoy fails INSIDE devsim on exactly that file.
  * the decision unit tests run devsim-free (every CI leg).
  * test_no_devsim_import_outside_the_home -- AST census over dynameta/ and validation/.
"""
import ast
import importlib.util
import json
import os
import pathlib
import subprocess
import sys

import pytest

import dynameta
from dynameta.carriers import devsim_env as E

PKG = pathlib.Path(dynameta.__file__).resolve().parent
REPO = PKG.parent
HAVE_DEVSIM = importlib.util.find_spec("devsim") is not None
_DEVSIM_ENV = ("DEVSIM_MATH_LIBS", "DEVSIM_BLAS_PREFIX", "DEVSIM_BLAS_SUFFIX", "DEVSIM_NEW_SYMBOLIC",
               E.MODE_ENV)

_CHILD = "\n".join([
    "import json, os",
    "import dynameta.carriers.devsim_layered",          # the package's real first devsim import
    "import devsim, psutil",
    "info = devsim.get_parameter(name='info')",
    "maps = sorted({os.path.normcase(m.path) for m in psutil.Process().memory_maps()})",
    "print('RESULT ' + json.dumps({'libs': list(info['math_libraries']), 'maps': maps,",
    "                              'solver': info.get('direct_solver')}))"])


def _expected_pin():
    """The library the default resolution picks in THIS environment (None: nothing pip-installed)."""
    return E._mkl_rt() or E._scipy_openblas()


def _decoys(tmp_path):
    ext = {"nt": ".dll"}.get(os.name, ".so")
    d = tmp_path / "decoy_blas"
    d.mkdir()
    for stem in ("libopenblas", "liblapack", "libblas"):
        (d / (stem + ext)).write_bytes(b"not a shared library -- D0 decoy\n" * 64)
    return d


def _child_env(decoy_dir, **extra):
    env = {k: v for k, v in os.environ.items() if k not in _DEVSIM_ENV}   # conftest exported a pin
    if os.name == "nt":
        env["PATH"] = str(decoy_dir) + os.pathsep + env.get("PATH", "")
    else:
        env["LD_LIBRARY_PATH"] = str(decoy_dir) + os.pathsep + env.get("LD_LIBRARY_PATH", "")
        env[E.MODE_ENV] = "pin"                # the default leaves POSIX to the dynamic loader
    env.update(extra)
    return env


def _run_child(env):
    return subprocess.run([sys.executable, "-c", _CHILD], capture_output=True, text=True,
                          cwd=str(REPO), env=env, timeout=300)


@pytest.mark.skipif(not HAVE_DEVSIM, reason="needs devsim")
def test_decoy_on_the_search_path_does_not_capture_devsim(tmp_path):
    pytest.importorskip("psutil")
    want = _expected_pin()
    if want is None:
        pytest.skip("no pip-installed BLAS to pin to (no pip mkl, no scipy-bundled OpenBLAS)")
    decoy = _decoys(tmp_path)
    out = _run_child(_child_env(decoy))
    assert out.returncode == 0, "devsim import failed under a decoy-first search path:\n{}\n{}".format(
        out.stdout[-1500:], out.stderr[-1500:])
    res = json.loads(out.stdout.split("RESULT ", 1)[1])
    assert res["libs"] == [want], res["libs"]
    nc = os.path.normcase
    assert nc(want) in res["maps"], "pinned library not mapped in the child"
    assert not [m for m in res["maps"] if m.startswith(nc(str(decoy)))], "a decoy was mapped"


@pytest.mark.skipif(not HAVE_DEVSIM, reason="needs devsim")
def test_user_math_libs_is_passed_through_verbatim(tmp_path):
    decoy = _decoys(tmp_path)
    user_lib = str(next(decoy.iterdir()))
    out = _run_child(_child_env(decoy, DEVSIM_MATH_LIBS=user_lib))
    assert out.returncode != 0                           # devsim tried the user's (decoy) file ...
    text = out.stdout + out.stderr
    assert 'Searching DEVSIM_MATH_LIBS="{}"'.format(user_lib) in text, text[-1500:]
    assert "RESULT" not in out.stdout                    # ... and nothing else was substituted


# ------------------------------------------- decision unit tests (devsim-free) ------------------

@pytest.fixture
def fresh(monkeypatch):
    for k in _DEVSIM_ENV:
        # setenv first: delenv of an ABSENT key records nothing, so a value configure_math_libs()
        # writes later would survive the test and leak into the next one's children.
        monkeypatch.setenv(k, "unset-by-fixture")
        monkeypatch.delenv(k)
    monkeypatch.setattr(E, "_DECISION", None)
    monkeypatch.delitem(sys.modules, "devsim", raising=False)
    return monkeypatch


def test_user_value_wins_and_nothing_else_is_written(fresh):
    fresh.setenv("DEVSIM_MATH_LIBS", "my_blas.dll")
    fresh.setattr(E, "_mkl_rt", lambda: pytest.fail("resolution must not run under a user value"))
    d = E.configure_math_libs()
    assert d["source"] == "user" and d["math_libs"] == "my_blas.dll"
    assert os.environ["DEVSIM_MATH_LIBS"] == "my_blas.dll"
    assert "DEVSIM_BLAS_PREFIX" not in os.environ and "DEVSIM_BLAS_SUFFIX" not in os.environ


def test_resolution_order_mkl_then_scipy_then_devsim_default(fresh):
    fresh.setenv(E.MODE_ENV, "pin")
    fresh.setattr(E, "_mkl_rt", lambda: "/x/mkl_rt.3.dll")
    fresh.setattr(E, "_scipy_openblas", lambda: "/x/libscipy_openblas-ab12.dll")
    fresh.setattr(E, "_devsim_honours_blas_prefix", lambda: True)
    assert E.configure_math_libs()["source"] == "mkl"
    assert os.environ["DEVSIM_MATH_LIBS"] == "/x/mkl_rt.3.dll"
    assert "DEVSIM_BLAS_PREFIX" not in os.environ          # MKL uses devsim's standard names
    assert os.environ["DEVSIM_NEW_SYMBOLIC"] == "1"         # no PARDISO symbolic reuse (docstring)
    assert E.configure_math_libs()["source"] == "mkl"      # idempotent: the first call decides

    fresh.setattr(E, "_DECISION", None)
    fresh.delenv("DEVSIM_MATH_LIBS")
    fresh.setattr(E, "_mkl_rt", lambda: None)
    d = E.configure_math_libs()
    assert (d["source"], d["prefix"], d["suffix"]) == ("scipy-openblas", "scipy_", "_")
    assert (os.environ["DEVSIM_MATH_LIBS"], os.environ["DEVSIM_BLAS_PREFIX"],
            os.environ["DEVSIM_BLAS_SUFFIX"]) == ("/x/libscipy_openblas-ab12.dll", "scipy_", "_")

    for k in ("DEVSIM_MATH_LIBS", "DEVSIM_BLAS_PREFIX", "DEVSIM_BLAS_SUFFIX"):
        fresh.delenv(k)
    fresh.setattr(E, "_DECISION", None)
    fresh.setattr(E, "_devsim_honours_blas_prefix", lambda: False)   # devsim < 2.11.1: no prefix
    assert E.configure_math_libs()["source"] == "devsim-default"
    assert "DEVSIM_MATH_LIBS" not in os.environ

    fresh.setattr(E, "_DECISION", None)
    fresh.setattr(E, "_devsim_honours_blas_prefix", lambda: True)
    fresh.setattr(E, "_scipy_openblas", lambda: None)
    assert E.configure_math_libs()["source"] == "devsim-default"
    assert "DEVSIM_MATH_LIBS" not in os.environ


def test_mkl_pin_keeps_a_user_new_symbolic_setting(fresh):
    fresh.setenv(E.MODE_ENV, "pin")
    fresh.setenv("DEVSIM_NEW_SYMBOLIC", "0")
    fresh.setattr(E, "_mkl_rt", lambda: "/x/mkl_rt.3.dll")
    assert E.configure_math_libs()["source"] == "mkl"
    assert os.environ["DEVSIM_NEW_SYMBOLIC"] == "0"


def test_mode_off_posix_default_and_already_imported(fresh):
    fresh.setattr(E, "_mkl_rt", lambda: "/x/mkl_rt.3.dll")
    fresh.setenv(E.MODE_ENV, "off")
    assert E.configure_math_libs()["source"] == "off" and "DEVSIM_MATH_LIBS" not in os.environ
    fresh.setattr(E, "_DECISION", None)
    fresh.setenv(E.MODE_ENV, "auto")
    fresh.setattr(E, "_is_windows", lambda: False)
    assert E.configure_math_libs()["source"] == "posix-default"
    assert "DEVSIM_MATH_LIBS" not in os.environ
    fresh.setattr(E, "_DECISION", None)
    fresh.setenv(E.MODE_ENV, "pin")
    fresh.setitem(sys.modules, "devsim", object())       # too late: devsim loaded its libraries
    assert E.configure_math_libs()["source"] == "already-imported"
    fresh.setattr(E, "_DECISION", None)
    fresh.setenv(E.MODE_ENV, "bogus")
    with pytest.raises(ValueError, match="expected 'auto', 'pin' or 'off'"):
        E.configure_math_libs()


@pytest.mark.parametrize("name,ok", [
    ("libscipy_openblas-64eda39e79589aedb16f58e5547eb599.dll", True),   # scipy 1.17.1 win_amd64
    ("libscipy_openblas-6cdc3b4a.so", True),                             # scipy 1.17.1 manylinux
    ("libscipy_openblas.dylib", True),                                   # macOS scipy/.dylibs
    ("libscipy_openblas64_-63c857e738469261263c764a36be9436.dll", False),  # numpy's ILP64 copy
    ("libopenblas.dll", False),
])
def test_scipy_openblas_name_pattern(name, ok):
    assert bool(E._SCIPY_OB_RE.match(name)) is ok


@pytest.mark.parametrize("name,ok", [
    ("mkl_rt.3.dll", True), ("mkl_rt.2.dll", True), ("mkl_rt.dll", True),
    ("libmkl_rt.so.2", True), ("libmkl_rt.so", True), ("libmkl_rt.2.dylib", True),
    ("mkl_core.3.dll", False), ("mkl_rt.3.pdb", False),
])
def test_mkl_rt_name_pattern(name, ok):
    assert bool(E._MKL_RE.match(name)) is ok


def test_no_devsim_import_outside_the_home():
    """One import home: no module in dynameta/ or validation/ imports devsim except devsim_env
    (tests/ go through conftest's configure_math_libs())."""
    bad = []
    for root in (PKG, REPO / "validation"):
        for p in sorted(root.rglob("*.py")):
            if p == PKG / "carriers" / "devsim_env.py":
                continue
            for n in ast.walk(ast.parse(p.read_text(encoding="utf-8"))):
                hit = (isinstance(n, ast.Import) and any(a.name.split(".")[0] == "devsim" for a in n.names)) or \
                      (isinstance(n, ast.ImportFrom) and (n.module or "").split(".")[0] == "devsim")
                if hit:
                    bad.append("{}:{}".format(p.relative_to(REPO).as_posix(), n.lineno))
    assert not bad, "devsim imported outside dynameta.carriers.devsim_env: {}".format(bad)


# ------------------------------------- devsim >= 2.11.2 contract: reset_devsim() ---------------
# validation/ac_capacitance.py clears state between geometries with ds.reset_devsim(). On devsim
# 2.10.0 the reset wiped the "direct_solver"/"solver_callback" parameters umfshim set at import and
# the next solve died ('Unrecognized "direct_solver" parameter value "unknown"', MEASURED in a 2.10.0
# venv, D0 2026-11-03); 2.11.2 (issue #169) re-registers UMFPACK inside reset_devsim().
_RESET_CHILD = "\n".join([
    "from dynameta.carriers.devsim_env import import_devsim",
    "ds = import_devsim()",
    "def build():",
    "    ds.create_1d_mesh(mesh='m')",
    "    ds.add_1d_mesh_line(mesh='m', pos=0.0, ps=2e-9, tag='bot')",
    "    ds.add_1d_mesh_line(mesh='m', pos=1e-7, ps=2e-9, tag='top')",
    "    for c in ('bot', 'top'):",
    "        ds.add_1d_contact(mesh='m', name=c, tag=c, material='metal')",
    "    ds.add_1d_region(mesh='m', material='ox', region='r', tag1='bot', tag2='top')",
    "    ds.finalize_mesh(mesh='m'); ds.create_device(mesh='m', device='d')",
    "    ds.set_parameter(device='d', region='r', name='Permittivity', value=3.45e-11)",
    "    ds.node_solution(device='d', region='r', name='Potential')",
    "    ds.edge_from_node_model(device='d', region='r', node_model='Potential')",
    "    ds.edge_model(device='d', region='r', name='F',",
    "                  equation='Permittivity*(Potential@n0-Potential@n1)*EdgeInverseLength')",
    "    ds.edge_model(device='d', region='r', name='F:Potential@n0', equation='Permittivity*EdgeInverseLength')",
    "    ds.edge_model(device='d', region='r', name='F:Potential@n1', equation='-Permittivity*EdgeInverseLength')",
    "    ds.equation(device='d', region='r', name='PE', variable_name='Potential', edge_model='F')",
    "    for c, v in (('bot', 0.0), ('top', 1.0)):",
    "        ds.set_parameter(device='d', name=c + '_bias', value=v)",
    "        ds.contact_node_model(device='d', contact=c, name=c + 'nm', equation='Potential - %s_bias' % c)",
    "        ds.contact_node_model(device='d', contact=c, name=c + 'nm:Potential', equation='1')",
    "        ds.contact_equation(device='d', contact=c, name='PE', node_model=c + 'nm', edge_charge_model='F')",
    "    ds.solve(type='dc', absolute_error=1e-12, relative_error=1e-12, maximum_iterations=10)",
    "    return list(ds.get_node_model_values(device='d', region='r', name='Potential'))",
    "s0 = ds.get_parameter(name='direct_solver'); v0 = build()",
    "ds.reset_devsim()",
    "s1 = ds.get_parameter(name='direct_solver'); v1 = build()",
    "print('RESULT', s0, s1, v0 == v1, min(v0), max(v0))"])


@pytest.mark.skipif(not HAVE_DEVSIM, reason="needs devsim")
def test_reset_devsim_keeps_the_direct_solver_and_the_answer():
    out = subprocess.run([sys.executable, "-c", _RESET_CHILD], capture_output=True, text=True,
                         cwd=str(REPO), timeout=300)
    assert out.returncode == 0, (out.stdout[-1500:], out.stderr[-1500:])
    s0, s1, same, vmin, vmax = out.stdout.split("RESULT", 1)[1].split()
    assert s0 == s1 and s0 in ("custom", "mkl_pardiso"), (s0, s1)   # UMFPACK shim or MKL-PARDISO
    assert same == "True"                                            # bit-identical rebuild
    assert abs(float(vmin)) < 1e-12 and abs(float(vmax) - 1.0) < 1e-12
