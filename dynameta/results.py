"""SweepResults -- a gridded, serializable view of a run_pipeline sweep (the List[SweepRow] it returns).

run_pipeline yields a flat list of SweepRow(bias_label, lambda_nm, OpticalResult). SweepResults.from_rows
pivots that into (n_bias, n_wl) arrays (R/T/A/R_flux/T_flux/phase/solve_time + complex r/t), so the whole
sweep is one tidy object you can index, save/load (HDF5 or Zarr, via dynameta.io.store), and hand to the
plotting helpers (dynameta.viz). Missing fields (T/A/... None before transmission) become NaN. Complex r/t
are stored split as <name>_real/<name>_imag on disk and recombined on load."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from dynameta.io.store import load_arrays, save_arrays

_SCHEMA = 1
# the per-(bias,wavelength) scalar fields lifted off OpticalResult into (n_bias, n_wl) real arrays
_REAL_FIELDS = ("R", "T", "A", "A_independent", "R_flux", "T_flux", "phase_deg", "solve_time_s")
_CPLX_FIELDS = ("r", "t")


@dataclass
class SweepResults:
    bias_labels: List[str]
    wavelengths_nm: np.ndarray                              # (n_wl,) sorted ascending
    fields: dict = field(default_factory=dict)             # name -> (n_bias, n_wl) array (real or complex)

    @property
    def n_bias(self) -> int:
        return len(self.bias_labels)

    @property
    def n_wl(self) -> int:
        return int(self.wavelengths_nm.size)

    def __getattr__(self, name):                            # sr.R, sr.T, ... -> the gridded field
        fields = self.__dict__.get("fields", {})
        if name in fields:
            return fields[name]
        raise AttributeError(name)

    # ---- construction ---------------------------------------------------------------------------
    @classmethod
    def from_rows(cls, rows: List["object"]) -> "SweepResults":
        """Pivot a List[SweepRow] (bias_label, lambda_nm, result) into the gridded container. Bias order =
        first appearance; wavelengths are sorted ascending."""
        labels = list(dict.fromkeys(r.bias_label for r in rows))          # unique, order-preserving
        wl = np.array(sorted({float(r.lambda_nm) for r in rows}), dtype=float)
        bi = {lab: i for i, lab in enumerate(labels)}
        wj = {round(float(w), 6): j for j, w in enumerate(wl)}
        if len(wj) != wl.size:               # two wavelengths within 1e-6 nm would silently OVERWRITE
            raise ValueError("from_rows: distinct wavelengths collide at 1e-6 nm rounding resolution "
                             "({} unique keys for {} wavelengths) -- a degenerate sweep would silently "
                             "overwrite rows.".format(len(wj), wl.size))
        nb, nw = len(labels), wl.size
        flds = {k: np.full((nb, nw), np.nan) for k in _REAL_FIELDS}
        flds.update({k: np.full((nb, nw), np.nan + 0j) for k in _CPLX_FIELDS})
        for row in rows:
            i = bi[row.bias_label]; j = wj[round(float(row.lambda_nm), 6)]
            res = row.result
            for k in _REAL_FIELDS:
                v = getattr(res, k, None)
                if v is not None:
                    flds[k][i, j] = float(v)
            for k in _CPLX_FIELDS:
                v = getattr(res, k, None)
                if v is not None:
                    flds[k][i, j] = complex(v)
        return cls(bias_labels=labels, wavelengths_nm=wl, fields=flds)

    # ---- analysis convenience -------------------------------------------------------------------
    def spectrum(self, bias_label: str, metric: str = "R") -> np.ndarray:
        """The (n_wl,) metric spectrum at one bias."""
        return self.fields[metric][self.bias_labels.index(bias_label)]

    def contrast(self, metric: str = "R", ref: Optional[str] = None) -> np.ndarray:
        """|metric(bias) - metric(ref)| per (bias, wavelength) -- the modulation against a reference bias
        (default: the first bias). The OFF/ON contrast spectrum of the modulator."""
        a = self.fields[metric]
        r = a[self.bias_labels.index(ref)] if ref is not None else a[0]
        # A reference bias with no data (all-NaN, e.g. a None->NaN field) would make the WHOLE
        # contrast array NaN -- and max_contrast then reads 'no modulation' when the truth is 'missing
        # reference data'. Fail loud (like from_rows does for a wavelength collision) instead.
        if np.all(np.isnan(r)):
            raise ValueError(
                "contrast: reference bias {!r} has no {!r} data (all-NaN) -- the contrast would be "
                "silently all-NaN. Choose a reference bias with data, or check the solve populated "
                "{!r}.".format(ref if ref is not None else self.bias_labels[0], metric, metric))
        return np.abs(a - r[None, :])

    def max_contrast(self, metric: str = "R", ref: Optional[str] = None) -> float:
        """The largest |delta metric| over all biases and wavelengths (the peak modulation).

        contrast() already guards the all-NaN REFERENCE; this also guards the symmetric case: if EVERY
        non-reference (comparison) bias is all-NaN, np.nanmax silently returns 0.0 (the reference row's
        self-contrast) -- reading 'no modulation' when the truth is 'no comparison data'. Raise instead.
        A PARTIALLY-NaN comparison bias (a solve that failed at some wavelengths) is legitimate and still
        reduces over its present cells; a single-bias sweep (no comparison row) correctly returns 0.0."""
        c = self.contrast(metric, ref)                              # raises if the reference is all-NaN
        ref_idx = self.bias_labels.index(ref) if ref is not None else 0
        comp = np.delete(c, ref_idx, axis=0)                        # the non-reference modulation rows
        if comp.size and bool(np.all(np.isnan(comp))):
            raise ValueError(
                "max_contrast: every non-reference bias is all-NaN for {!r} -- the peak modulation "
                "would read 0.0 silently though there is NO comparison data. Check the solve populated "
                "{!r} for the modulated bias(es).".format(metric, metric))
        return float(np.nanmax(c))

    # ---- serialization (HDF5 / Zarr) ------------------------------------------------------------
    def save(self, path: str, *, fmt: str = "auto") -> str:
        """Persist to HDF5 (.h5/.hdf5) or Zarr (.zarr) -- chosen by extension or `fmt`. Complex r/t are
        split into <name>_real/<name>_imag; bias labels + schema go in the metadata."""
        arrays = {"wavelengths_nm": self.wavelengths_nm}
        for k in _REAL_FIELDS:
            arrays[k] = np.asarray(self.fields[k], dtype=float)
        for k in _CPLX_FIELDS:
            c = np.asarray(self.fields[k], dtype=complex)
            arrays[k + "_real"] = c.real; arrays[k + "_imag"] = c.imag
        attrs = {"schema": _SCHEMA, "bias_labels": list(self.bias_labels),
                 "real_fields": list(_REAL_FIELDS), "cplx_fields": list(_CPLX_FIELDS)}
        return save_arrays(path, arrays, attrs, fmt=fmt)

    @classmethod
    def load(cls, path: str, *, fmt: str = "auto") -> "SweepResults":
        arrays, attrs = load_arrays(path, fmt=fmt)
        labels = list(attrs.get("bias_labels", []))
        wl = np.asarray(arrays["wavelengths_nm"], dtype=float)
        flds = {}
        for k in attrs.get("real_fields", _REAL_FIELDS):
            flds[k] = np.asarray(arrays[k], dtype=float)
        for k in attrs.get("cplx_fields", _CPLX_FIELDS):
            flds[k] = np.asarray(arrays[k + "_real"], dtype=float) + 1j * np.asarray(arrays[k + "_imag"], dtype=float)
        return cls(bias_labels=labels, wavelengths_nm=wl, fields=flds)
