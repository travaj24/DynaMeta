"""Fast, dependency-light tests for the DielectricDB (no JARVIS/MP/network)."""
import importlib.util

import pytest

from dynameta.materials import DielectricDB, DielectricRecord, normalize_formula


def test_formula_normalization():
    nf = normalize_formula
    assert nf("HfO2") == nf("Hf1O2") == nf("O2Hf") == nf("Hf2O4")
    assert nf("HfO2") != nf("Al2O3")


def test_override_precedence(tmp_path):
    db = DielectricDB(
        overrides={"HfO2": 18.0,
                    "Al2O3": DielectricRecord.measured("Al2O3", 9.0, "in-house C-V")},
        cache_path=str(tmp_path / "c.json"), verbose=False)
    r = db.eps_static("HfO2")
    assert r.eps_static == 18.0 and r.source == "override"
    r2 = db.eps_static("Al2O3")
    assert r2.eps_static == 9.0 and r2.source == "measured" and r2.identifier == "in-house C-V"
    assert db.eps_for("Hf1O2") == 18.0          # normalization-based override match


def test_disk_cache_round_trip(tmp_path):
    cp = str(tmp_path / "c.json")
    db = DielectricDB(backend="jarvis", cache_path=cp, verbose=False)
    db._query = lambda f: DielectricRecord(formula=f, eps_static=22.5,
                                            source="jarvis-dft", identifier="JVASP-X",
                                            kind="dft-dfpt")
    assert db.eps_static("ZrO2").eps_static == 22.5     # queries stub -> writes cache

    db2 = DielectricDB(backend="jarvis", cache_path=cp, verbose=False)
    def _boom(_f):
        raise AssertionError("should have hit the disk cache, not the backend")
    db2._query = _boom
    assert db2.eps_static("ZrO2").eps_static == 22.5    # served from disk cache


def test_missing_backend_raises():
    # Skip if the backend dep happens to be installed (then it would try a real query).
    if importlib.util.find_spec("mp_api") is not None:
        pytest.skip("mp-api is installed; cannot test the missing-backend path")
    db = DielectricDB(backend="mp", verbose=False)
    with pytest.raises(RuntimeError, match="mp-api"):
        db.eps_static("TiO2")
