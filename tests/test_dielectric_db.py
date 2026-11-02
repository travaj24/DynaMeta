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


# ---- audit Q-25 (2026-11-02): no `or` on a float that can legitimately be 0.0 --------------------
def test_q25_mp_picks_the_ground_state_polymorph_with_zero_hull_energy(monkeypatch, tmp_path):
    """`energy_above_hull or 1e9` turned the GROUND STATE's 0.0 into 1e9 and sorted it LAST, so a
    metastable polymorph was returned. Fake MPRester, three polymorphs (0.05, 0.0, None eV)."""
    import sys
    import types
    docs = [types.SimpleNamespace(material_id="mp-meta", e_total=30.0, e_ionic=None,
                                  e_electronic=None, symmetry=None, energy_above_hull=0.05),
            types.SimpleNamespace(material_id="mp-ground", e_total=20.0, e_ionic=None,
                                  e_electronic=None, symmetry=None, energy_above_hull=0.0),
            types.SimpleNamespace(material_id="mp-unknown", e_total=10.0, e_ionic=None,
                                  e_electronic=None, symmetry=None, energy_above_hull=None)]

    class _Summary:
        def search(self, formula, fields):
            return list(docs)

    class _MPRester:
        def __init__(self, api_key):
            self.materials = types.SimpleNamespace(summary=_Summary())

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    client = types.ModuleType("mp_api.client")
    client.MPRester = _MPRester
    monkeypatch.setitem(sys.modules, "mp_api", types.ModuleType("mp_api"))
    monkeypatch.setitem(sys.modules, "mp_api.client", client)
    db = DielectricDB(backend="mp", mp_api_key="x", cache_path=str(tmp_path / "c.json"),
                      verbose=False)
    rec = db._query_mp("TiO2")
    assert rec.identifier == "mp-ground" and rec.eps_static == 20.0


def test_q25_jarvis_electronic_eps_zero_sentinel_falls_back_explicitly(monkeypatch, tmp_path):
    """The JARVIS electronic permittivity: 0.0 is unphysical (eps_inf >= 1) and means "not
    computed", so it falls back to the epsx/y/z average -- what the old `or` did, now explicit;
    a genuine value is used as is."""
    import sys
    import types
    jar = types.ModuleType("jarvis.db.figshare")
    jar.data = lambda name: []
    for mod in ("jarvis", "jarvis.db"):
        monkeypatch.setitem(sys.modules, mod, types.ModuleType(mod))
    monkeypatch.setitem(sys.modules, "jarvis.db.figshare", jar)
    db = DielectricDB(backend="jarvis", cache_path=str(tmp_path / "c.json"), verbose=False)
    base = {"formula": "HfO2", "jid": "JVASP-1", "formation_energy_peratom": -3.0,
            "dfpt_piezo_max_dielectric": 25.0, "epsx": 4.0, "epsy": 5.0, "epsz": 6.0}
    db._jarvis_dataset = [dict(base, dfpt_piezo_max_dielectric_electronic=0.0)]
    assert db._query_jarvis("HfO2").eps_electronic == 5.0
    db._jarvis_dataset = [dict(base, dfpt_piezo_max_dielectric_electronic=4.4)]
    assert db._query_jarvis("HfO2").eps_electronic == 4.4
