"""Sweep + BiasPoint: the (bias, wavelength) grid a pipeline run covers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class BiasPoint:
    voltages: Dict[str, float]      # {electrode_name: volts} for biased electrodes
    label:    str

    def __post_init__(self) -> None:
        if not self.label:
            raise ValueError("BiasPoint requires a label")


@dataclass
class Sweep:
    bias_points:    List[BiasPoint]
    wavelengths_nm: List[float] = field(default_factory=list)

    def __post_init__(self) -> None:
        # audit C6-3: duplicate bias labels silently COLLAPSED -- the label-keyed
        # carrier-field dict kept only the LAST duplicate's field for all its rows and
        # SweepResults.from_rows silently overwrote the grid row, while the library
        # fail-louds on every sibling degeneracy (wavelength collision, duplicate gate
        # biases, empty sweep).
        labels = [bp.label for bp in self.bias_points]
        dupes = sorted({l for l in labels if labels.count(l) > 1})
        if dupes:
            raise ValueError("Sweep: duplicate bias-point labels {} -- labels key the "
                             "carrier-field and results grids, so duplicates silently "
                             "collapse to the last point".format(dupes))
        # audit R-4: the comment above named "wavelength collision" as a sibling hazard the
        # library already fail-louds on. It did NOT: duplicate and non-positive wavelengths
        # were both accepted, and SweepResults.from_rows keys the grid off a SET of
        # wavelengths, so a duplicate silently overwrote its own row with the LAST solve --
        # after paying for the extra solve. Same fail-loud policy as the labels.
        wl = [float(w) for w in (self.wavelengths_nm or [])]
        bad = [w for w in wl if not (w > 0.0) or w != w or w in (float("inf"),)]
        if bad:
            raise ValueError(
                "Sweep: wavelengths_nm must all be finite and > 0 (got {}); a non-positive or "
                "non-finite wavelength has no solve and would surface far downstream as a "
                "division or a NaN result row.".format(bad))
        wdupes = sorted({w for w in wl if wl.count(w) > 1})
        if wdupes:
            raise ValueError(
                "Sweep: duplicate wavelengths_nm {} -- SweepResults.from_rows keys the results "
                "grid off the SET of wavelengths, so a duplicate silently overwrites its own "
                "row with the last solve while doubling the solve cost invisibly.".format(
                    wdupes))
