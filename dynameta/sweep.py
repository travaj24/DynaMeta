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
