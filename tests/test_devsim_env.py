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
    return E._mkl_rt() or (E._scipy_openblas() if E._devsim_honours_blas_prefix() else None)


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


# ------------------------------- v0.15.1: UMFPACK 5.1's obsolete 2-by-2 strategy ----------------
# devsim_env.umfpack_2by2_guard (docs/audit/2026-11-10-v0.15.1-dg-iteration-cost.md, sections 4 and 12).
# Three small matrices on which UMFPACK 5.1's AUTO picks each of its strategies (MEASURED,
# Info[UMFPACK_STRATEGY_USED] 3 / 2 / 1 = symmetric / 2-by-2 / unsymmetric). The 2-by-2 one does NOT fill in
# (its symbolic nz(L+U) bound equals the symmetric one's), so the v2 guard must KEEP all three bit for bit as
# devsim's own call makes them; with the bound-ratio threshold forced below 1 the same matrix must be redone
# as symmetric (the replacement branch; the real threshold on real matrices is gated by
# test_dg_closure_jacobian_factorizes_without_fill_in and test_umfpack_2by2_guard_keeps_2by2_on_a_2d_diode).
# On the v1 guard (4cd3af7, always replaces) the 2-by-2 case comes back symmetric; before v0.15.1 the wrapped
# method does not exist (AttributeError).

def _umf_test_matrix(m, small, extra):
    import scipy.sparse as sp
    n = 2 * m
    A = sp.lil_matrix((n, n))
    for k in range(m):
        u, c = 2 * k, 2 * k + 1
        A[u, u] = 2.0
        if k > 0:
            A[u, u - 2] = -1.0
        if k < m - 1:
            A[u, u + 2] = -1.0
        A[u, c] = 0.5
        A[c, c] = small
        A[c, u] = 1.0
        for e in extra:
            if 0 <= u + e < n:
                A[c, u + e] = -0.3
    return A.tocsc()


def _umf_factor_solve(L, symbolic, A):
    """devsim's own UMFPACK path (umf_control -> symbolic -> numeric -> solve) with `symbolic` as the
    symbolic step; returns (Info[STRATEGY_USED] after the symbolic step, nz(L) + nz(U), x)."""
    import array

    import numpy as np
    uc = L.umf_control(L.global_data_instance, "real")
    M = L.matrix(uc=uc, Ap=array.array("i", A.indptr.astype(np.int32)),
                 Ai=array.array("i", A.indices.astype(np.int32)), Ax=array.array("d", A.data))
    S = symbolic(uc, M)
    strategy = int(uc.Info[18])
    N = uc.numeric(matrix=M, Symbolic=S)
    b = array.array("d", (np.arange(A.shape[0]) % 7 + 1.0).tolist())
    x = array.array("d", b)
    uc.solve(matrix=M, Numeric=N, b=b, transpose=False, x=x)
    return strategy, uc.Info[43] + uc.Info[44], np.asarray(x)


@pytest.mark.skipif(not HAVE_DEVSIM, reason="needs devsim")
def test_umfpack_2by2_guard_keeps_what_does_not_fill_in(monkeypatch):
    import numpy as np
    ds = E.import_devsim()
    g = E.umfpack_2by2_guard()
    if g["mode"] != "symmetric" or "devsim.umfpack.umfshim" not in sys.modules:
        pytest.skip("guard not active here: {}".format(g["reason"]))
    import devsim.umfpack.umfpack_loader as L
    if not L.global_data_instance.is_initialized:
        info = ds.get_parameter(name="info")
        L.global_data_instance.initialize_umfpack(info["math_libraries"], info["blas_symbol_prefix"],
                                                  info["blas_symbol_suffix"])
    assert g["installed"], g
    guarded = L.umf_control.symbolic
    original = guarded.dynameta_original
    cases = {"symmetric": (_umf_test_matrix(5, 1e-20, ()), 3),
             "2-by-2": (_umf_test_matrix(6, 1e-20, (-2, 2)), 2),
             "unsymmetric": (_umf_test_matrix(6, 1e-20, (2, 4, 6)), 1)}
    for name, (A, auto) in cases.items():
        s0, f0, x0 = _umf_factor_solve(L, lambda uc, M: original(uc, M), A)
        before = dict(g)
        s1, f1, x1 = _umf_factor_solve(L, lambda uc, M: guarded(uc, M), A)
        assert (s0, s1) == (auto, auto), name                   # AUTO's choice stands
        assert g.get("auto_2by2", 0) - before.get("auto_2by2", 0) == (1 if auto == 2 else 0), name
        assert g.get("kept", 0) - before.get("kept", 0) == (1 if auto == 2 else 0), name
        assert g["replaced"] == before["replaced"], name
        assert f1 == f0 and np.array_equal(x0, x1), name        # untouched: bit for bit
        r = A @ x1 - (np.arange(A.shape[0]) % 7 + 1.0)
        assert np.max(np.abs(r)) <= 1e-12 * (abs(A) @ np.abs(x1)).max(), name
    # the replacement branch: same 2-by-2 matrix, threshold forced below any bound ratio
    monkeypatch.setattr(E, "_UMF_2BY2_BOUND_RATIO", 0.5)
    A = cases["2-by-2"][0]
    before = dict(g)
    s2, _, x2 = _umf_factor_solve(L, lambda uc, M: guarded(uc, M), A)
    assert s2 == 3 and g["replaced"] - before["replaced"] == 1 and g["kept"] == before["kept"]
    r = A @ x2 - (np.arange(A.shape[0]) % 7 + 1.0)
    assert np.max(np.abs(r)) <= 1e-12 * (abs(A) @ np.abs(x2)).max()
    assert g["auto_2by2"] == g["replaced"] + g["kept"]


_GUARD_CHILD = "\n".join([
    "import json",
    "from dynameta.carriers import devsim_env as E",
    "ds = E.import_devsim()",
    "import devsim.umfpack.umfpack_loader as L",
    "def wrapped():",
    "    return hasattr(L.umf_control.symbolic, 'dynameta_original')",
    "g = dict(E.umfpack_2by2_guard()); w0 = wrapped()",
    "ds.reset_devsim()",
    "print('RESULT ' + json.dumps({'guard': g, 'before': w0, 'after_reset': wrapped(),",
    "                              'solver': ds.get_parameter(name='direct_solver')}))"])


@pytest.mark.skipif(not HAVE_DEVSIM, reason="needs devsim")
@pytest.mark.parametrize("mode", [None, "keep"])
def test_umfpack_2by2_guard_default_on_keep_off_and_survives_reset(mode):
    env = {k: v for k, v in os.environ.items() if k != E.UMFPACK_2BY2_ENV}
    if mode:
        env[E.UMFPACK_2BY2_ENV] = mode
    out = subprocess.run([sys.executable, "-c", _GUARD_CHILD], capture_output=True, text=True,
                         cwd=str(REPO), env=env, timeout=300)
    assert out.returncode == 0, (out.stdout[-1500:], out.stderr[-1500:])
    r = json.loads(out.stdout.split("RESULT", 1)[1])
    if r["solver"] != "custom":
        pytest.skip("devsim's direct solver is {!r}, not the UMFPACK shim".format(r["solver"]))
    on = mode is None
    assert r["guard"]["installed"] is on and r["guard"]["mode"] == (mode or "symmetric"), r
    assert r["before"] is on and r["after_reset"] is on, r


def test_umfpack_2by2_mode_is_validated(monkeypatch):
    monkeypatch.setattr(E, "_GUARD", None)
    monkeypatch.setenv(E.UMFPACK_2BY2_ENV, "bogus")
    with pytest.raises(ValueError, match="expected 'symmetric' or 'keep'"):
        E.umfpack_2by2_guard()


# A device without the density gradient on which UMFPACK 5.1's AUTO picks 2-by-2 and 2-by-2 does NOT fill in:
# validation/bipolar_diode_2d's lateral p-n diode (bipolar DD + SRH, no DG, no closure) on a coarse mesh
# (1081 nodes, 2970 unknowns; one forward bias, ~3 s). MEASURED (Windows, unguarded + a symmetric twin of every
# factorization): AUTO picks 2-by-2 for all 10 fresh symbolic steps (51 numeric), its symbolic nz(L+U) bound
# equals the symmetric one's (ratio 1.000) and so does the actual fill. The v2 guard must keep every one of them:
# the guarded run is compared with an unguarded child (DYNAMETA_UMFPACK_2BY2=keep), factorization by
# factorization (strategy used, nz(L) + nz(U)) and in the converged state bit for bit. On the v1 guard
# (4cd3af7), which redid every 2-by-2 choice as symmetric, the strategies read 3 and the states differ.
_DIODE_CHILD = "\n".join([
    "import dataclasses, json",
    "import validation.bipolar_diode_2d as D",
    "from dynameta.carriers import devsim_env as E",
    "from dynameta.carriers.devsim_layered import LayeredDevsimBuilder",
    "from dynameta.geometry.specs import Mesh2DSpec",
    "from dynameta.sweep import BiasPoint",
    "ds = D.ds",
    "solver = ds.get_parameter(name='direct_solver')",
    "seen = []",
    "if solver == 'custom':",
    "    import devsim.umfpack.umfpack_loader as L",
    "    numeric = L.umf_control.numeric",
    "    def spy(self, matrix, Symbolic):",
    "        out = numeric(self, matrix, Symbolic)",
    "        seen.append([int(self.Info[18]), self.Info[43] + self.Info[44], len(matrix.Ax)])",
    "        return out",
    "    L.umf_control.numeric = spy",
    "m2 = Mesh2DSpec(x_spacing_feature_mid_m=10e-9, x_spacing_feature_edge_m=5e-9, coarse_ps_m=2e-9,",
    "                interface_ps_m=5e-10)",
    "b = LayeredDevsimBuilder(dataclasses.replace(D.build_diode_design(), mesh_2d=m2), mesh_name='g2_m',",
    "                         device_name='g2_d')",
    "b.solve(BiasPoint({'anode': 0.4}, 'fwd'))",
    "sol = {}",
    "for r in ds.get_region_list(device=b.device):",
    "    for eq in ds.get_equation_list(device=b.device, region=r):",
    "        v = ds.get_equation_command(device=b.device, region=r, name=eq)['variable_name']",
    "        sol[r + '/' + v] = [x.hex() for x in ds.get_node_model_values(device=b.device, region=r, name=v)]",
    "print('RESULT ' + json.dumps({'guard': dict(E.umfpack_2by2_guard()), 'seen': seen, 'sol': sol,",
    "                              'I': D._terminal_current(b.device, 'anode').hex(), 'solver': solver}))"])


@pytest.mark.skipif(not HAVE_DEVSIM, reason="needs devsim")
def test_umfpack_2by2_guard_keeps_2by2_on_a_2d_diode():
    runs = {}
    for mode in (None, "keep"):
        env = {k: v for k, v in os.environ.items() if k != E.UMFPACK_2BY2_ENV}
        if mode:
            env[E.UMFPACK_2BY2_ENV] = mode
        out = subprocess.run([sys.executable, "-c", _DIODE_CHILD], capture_output=True, text=True,
                             cwd=str(REPO), env=env, timeout=600)
        assert out.returncode == 0, (mode, out.stdout[-1500:], out.stderr[-1500:])
        runs[mode] = json.loads(out.stdout.split("RESULT", 1)[1])
    guarded, plain = runs[None], runs["keep"]
    if guarded["solver"] != "custom":
        pytest.skip("devsim's direct solver is {!r}, not the UMFPACK shim".format(guarded["solver"]))
    assert not plain["guard"]["installed"] and guarded["guard"]["installed"], runs
    n2 = sum(1 for s, _, _ in plain["seen"] if s == 2)
    assert n2 >= 10, plain["seen"]                       # AUTO picks 2-by-2 on this device (51 of 51)
    assert guarded["seen"] == plain["seen"]              # same strategy (2) and fill, step by step
    assert guarded["sol"] == plain["sol"] and guarded["I"] == plain["I"]   # bit for bit
    g = guarded["guard"]
    assert g.get("replaced") == 0 and g.get("kept") == g.get("auto_2by2") >= 10, g
