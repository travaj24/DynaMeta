"""
Static (DC..GHz) dielectric constants from open materials databases, with
provenance and a measured-value override path -- so Stage-1 gate permittivities
come from an auditable source instead of a hardcoded literature guess.

Physics: from DC up through GHz (well below the THz lattice phonons) the
permittivity of a good insulator is flat at the STATIC value
    eps(0) = eps_electronic + eps_ionic
The optical model in this library is eps_electronic only (e.g. HfO2 ~4); the
ionic/lattice part is what carries it up to the DC value (HfO2 ~18-25). That
ionic part is exactly what these DFPT databases provide.

Backends (both opt-in -- imported lazily, not hard dependencies):
  "jarvis" (default): NIST JARVIS-DFT via `jarvis-tools`. No API key. The first
      lookup downloads + caches the dft_3d dataset (~1.6 GB, one time); per-
      formula results are then cached locally by this module.
  "mp"            : Materials Project via `mp-api`. Needs a free API key
      (https://materialsproject.org). Lightweight per-query REST.

CAVEATS (read these):
  * These are DFT/DFPT *computed* values, not measured -- ~10-30% error on
    high-k oxides, mostly in the ionic part. A principled, provenance-bearing
    default, NOT ground truth.
  * Polymorph dependence is large for exactly these materials. HfO2 is
    monoclinic (~16-18), tetragonal/cubic (~25-70), or amorphous ALD film
    (~16-25). The database returns a value for ONE crystal structure (the most
    stable by default). Your ALD thin film is amorphous + processing-dependent,
    so the device-relevant number is ultimately a C-V measurement -> use the
    `overrides` path for those. Measured overrides ALWAYS win.

Usage:
    from dynameta.materials.db import DielectricDB, DielectricRecord
    db = DielectricDB(overrides={
        "Al2O3": DielectricRecord.measured("Al2O3", 9.0, "in-house C-V 2026"),
    })
    rec = db.eps_static("HfO2")          # -> DielectricRecord (queries JARVIS)
    print(rec.eps_static, rec.source, rec.identifier)
    eps = db.eps_for("HfO2")             # -> float only
    db.apply(material, formula="HfO2")   # sets material.eps_static_dc + provenance

See docs/dielectrics.md for the measured-vs-DFPT comparison of the Park-stack
oxides (HfO2/Al2O3/In2O3) and guidance on choosing and overriding these values.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple, Union


# ---------------------------------------------------------------- record ----
@dataclass(frozen=True)
class DielectricRecord:
    """A resolved static permittivity with provenance."""
    formula:        str
    eps_static:     float                       # total static (elec + ionic)
    source:         str                         # measured|override|jarvis-dft|materials-project
    identifier:     str = ""                    # jid / mp-id / citation
    kind:           str = ""                    # measured | dft-dfpt
    eps_electronic: Optional[float] = None
    eps_ionic:      Optional[float] = None
    spacegroup:     Optional[str] = None
    note:           str = ""

    @classmethod
    def measured(cls, formula: str, eps_static: float, citation: str) -> "DielectricRecord":
        """A measured value (e.g. C-V of your own film). Highest priority."""
        return cls(formula=formula, eps_static=float(eps_static),
                    source="measured", identifier=citation, kind="measured")

    def __str__(self) -> str:
        parts = ["{}: eps_static={:.3f} [{}".format(self.formula, self.eps_static, self.source)]
        if self.identifier:
            parts.append(" {}".format(self.identifier))
        parts.append("]")
        if self.eps_electronic is not None and self.eps_ionic is not None:
            parts.append(" (elec {:.2f} + ionic {:.2f})".format(
                self.eps_electronic, self.eps_ionic))
        if self.spacegroup:
            parts.append(" spg={}".format(self.spacegroup))
        return "".join(parts)


# ------------------------------------------------------------ formula util --
_TOKEN = re.compile(r"([A-Z][a-z]?)(\d*\.?\d*)")


def normalize_formula(formula: str) -> Tuple[Tuple[str, float], ...]:
    """Parse 'HfO2'/'Hf1O2'/'O2Hf' -> a canonical sorted element-count tuple, so
    differently written formulas for the same compound compare equal. Counts are
    normalized by their GCD-like smallest so 'Hf2O4' == 'HfO2'."""
    counts: Dict[str, float] = {}
    for el, num in _TOKEN.findall(formula.replace(" ", "")):
        counts[el] = counts.get(el, 0.0) + (float(num) if num else 1.0)
    if not counts:
        return tuple()
    smallest = min(v for v in counts.values() if v > 0)
    return tuple(sorted((el, round(v / smallest, 4)) for el, v in counts.items()))


# ----------------------------------------------------------------- the DB ---
def _default_cache_path() -> str:
    return os.path.join(os.path.expanduser("~"), ".dynameta",
                         "dielectric_cache.json")


class DielectricDB:
    def __init__(self, *, overrides: Optional[Dict[str, Union[float, DielectricRecord]]] = None,
                  backend: str = "jarvis", cache_path: Optional[str] = None,
                  mp_api_key: Optional[str] = None, verbose: bool = True) -> None:
        if backend not in ("jarvis", "mp"):
            raise ValueError("backend must be 'jarvis' or 'mp', got {!r}".format(backend))
        self.backend = backend
        self.mp_api_key = mp_api_key or os.environ.get("MP_API_KEY")
        self.cache_path = cache_path or _default_cache_path()
        self.verbose = verbose
        self.overrides: Dict[Tuple, DielectricRecord] = {}
        for f, v in (overrides or {}).items():
            rec = v if isinstance(v, DielectricRecord) else DielectricRecord(
                formula=f, eps_static=float(v), source="override", kind="manual")
            self.overrides[normalize_formula(f)] = rec
        self._cache: Dict[str, dict] = self._load_cache()
        self._jarvis_dataset = None     # lazily loaded bulk dataset

    # ---- public ----
    def eps_static(self, formula: str, *, refresh: bool = False) -> DielectricRecord:
        """Resolve the static permittivity. Order: measured/override -> cache ->
        backend query. Raises if the backend is selected but unavailable."""
        key = normalize_formula(formula)
        if key in self.overrides:
            return self.overrides[key]
        ckey = "{}:{}".format(self.backend, _key_str(key))
        if (not refresh) and ckey in self._cache:
            return DielectricRecord(**self._cache[ckey])
        rec = self._query(formula)
        self._cache[ckey] = asdict(rec)
        self._save_cache()
        if self.verbose:
            print("[dielectric.db] {}".format(rec), flush=True)
        return rec

    def eps_for(self, formula: str, *, refresh: bool = False) -> float:
        return self.eps_static(formula, refresh=refresh).eps_static

    def apply(self, material, formula: Optional[str] = None, *, refresh: bool = False):
        """Set material.eps_static_dc from the DB and stash the record on the
        material as `._eps_static_dc_record` for provenance. Returns the record."""
        rec = self.eps_static(formula or material.name, refresh=refresh)
        material.eps_static_dc = rec.eps_static
        try:
            object.__setattr__(material, "_eps_static_dc_record", rec)
        except Exception:
            pass
        return rec

    # ---- backend dispatch ----
    def _query(self, formula: str) -> DielectricRecord:
        return self._query_jarvis(formula) if self.backend == "jarvis" \
            else self._query_mp(formula)

    # ---- JARVIS-DFT ----
    # DFPT total (electronic+ionic) is the DC value we want; epsx/y/z are
    # electronic-only. Probe candidate keys defensively (names vary by release)
    # and record which was used; if none are present, raise listing the actual
    # keys so the truth comes from the live record, not an assumption here.
    _J_TOTAL = ("dfpt_piezo_max_dielectric",)
    _J_ELEC = ("dfpt_piezo_max_dielectric_electronic",)
    _J_IONIC = ("dfpt_piezo_max_dielectric_ionic",)
    _J_ELEC_OPT = ("epsx", "epsy", "epsz")

    def _query_jarvis(self, formula: str) -> DielectricRecord:
        try:
            from jarvis.db.figshare import data as _jdata
        except ImportError:
            raise RuntimeError(
                "JARVIS backend needs 'jarvis-tools' (pip install jarvis-tools). "
                "The first call downloads the dft_3d dataset (~1.6 GB, cached). "
                "Or use backend='mp', or supply a measured override.")
        if self._jarvis_dataset is None:
            if self.verbose:
                print("[dielectric.db] loading JARVIS dft_3d (first call downloads "
                      "~1.6 GB, then cached by jarvis-tools)...", flush=True)
            self._jarvis_dataset = _jdata("dft_3d")
        target = normalize_formula(formula)
        cands = [r for r in self._jarvis_dataset
                 if normalize_formula(str(r.get("formula", ""))) == target]
        if not cands:
            raise LookupError("JARVIS-DFT has no entry matching formula {!r}".format(formula))
        # The most stable polymorph that actually HAS a numeric DFPT total
        # dielectric. NB many entries carry the dielectric KEYS but a non-numeric
        # 'na' VALUE (DFPT not run for that material) -- _first_num rejects those,
        # so a key-present-but-na material lands here, not in with_diel.
        with_diel = [r for r in cands if _first_num(r, self._J_TOTAL) is not None]
        if not with_diel:
            raise LookupError(
                "JARVIS-DFT has {} polymorph(s) of {} but none has a computed DFPT "
                "dielectric (the dfpt_piezo_max_dielectric value is 'na'). Use "
                "backend='mp' or supply a measured/literature override.".format(
                    len(cands), formula))
        with_diel.sort(key=lambda r: _num_or(r.get("formation_energy_peratom"), 1e9))
        rec = with_diel[0]
        total = _first_num(rec, self._J_TOTAL)
        elec = _first_num(rec, self._J_ELEC) or _avg_num(rec, self._J_ELEC_OPT)
        ionic = _first_num(rec, self._J_IONIC)
        return DielectricRecord(
            formula=formula, eps_static=float(total), source="jarvis-dft",
            identifier=str(rec.get("jid", "")), kind="dft-dfpt",
            eps_electronic=elec, eps_ionic=ionic,
            spacegroup=str(rec.get("spg_symbol") or rec.get("spg_number") or "") or None)

    # ---- Materials Project ----
    def _query_mp(self, formula: str) -> DielectricRecord:
        try:
            from mp_api.client import MPRester
        except ImportError:
            raise RuntimeError(
                "MP backend needs 'mp-api' (pip install mp-api) and an API key "
                "(MP_API_KEY env var or mp_api_key=...). Or use backend='jarvis', "
                "or supply a measured override.")
        if not self.mp_api_key:
            raise RuntimeError("Materials Project needs an API key: set MP_API_KEY "
                                "or pass mp_api_key=... (free at materialsproject.org).")
        fields = ["material_id", "e_total", "e_ionic", "e_electronic",
                  "symmetry", "energy_above_hull"]
        with MPRester(api_key=self.mp_api_key) as mpr:
            docs = mpr.materials.summary.search(formula=formula, fields=fields)
        docs = [d for d in docs if getattr(d, "e_total", None) is not None]
        if not docs:
            raise LookupError("Materials Project has no dielectric entry for {!r}".format(formula))
        docs.sort(key=lambda d: (getattr(d, "energy_above_hull", None) or 1e9))
        d = docs[0]
        sym = getattr(d, "symmetry", None)
        spg = getattr(sym, "symbol", None) if sym is not None else None
        return DielectricRecord(
            formula=formula, eps_static=float(d.e_total), source="materials-project",
            identifier=str(getattr(d, "material_id", "")), kind="dft-dfpt",
            eps_electronic=_num_or(getattr(d, "e_electronic", None), None),
            eps_ionic=_num_or(getattr(d, "e_ionic", None), None),
            spacegroup=str(spg) if spg else None)

    # ---- cache io ----
    def _load_cache(self) -> Dict[str, dict]:
        try:
            with open(self.cache_path, "r", encoding="utf-8") as fh:
                return json.load(fh)
        except (FileNotFoundError, ValueError):
            return {}

    def _save_cache(self) -> None:
        os.makedirs(os.path.dirname(self.cache_path), exist_ok=True)
        with open(self.cache_path, "w", encoding="utf-8") as fh:
            json.dump(self._cache, fh, indent=2, sort_keys=True)


# ------------------------------------------------------------- numeric util -
def _num_or(v, default):
    try:
        f = float(v)
        return f if f == f else default      # reject NaN
    except (TypeError, ValueError):
        return default


def _first_num(rec: dict, keys) -> Optional[float]:
    for k in keys:
        v = _num_or(rec.get(k), None)
        if v is not None:
            return v
    return None


def _avg_num(rec: dict, keys) -> Optional[float]:
    vals = [v for v in (_num_or(rec.get(k), None) for k in keys) if v is not None]
    return sum(vals) / len(vals) if vals else None


def _key_str(norm: Tuple) -> str:
    return "".join("{}{}".format(el, n) for el, n in norm)


# --------------------------------------------------------------- demo cli ---
if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Look up a static dielectric constant.")
    p.add_argument("formula", nargs="+", help="chemical formula(e), e.g. HfO2 Al2O3")
    p.add_argument("--backend", default="jarvis", choices=["jarvis", "mp"])
    p.add_argument("--mp-key", default=None)
    args = p.parse_args()
    db = DielectricDB(backend=args.backend, mp_api_key=args.mp_key)
    for f in args.formula:
        try:
            print(db.eps_static(f))
        except Exception as e:
            print("{}: ERROR {}".format(f, e))
