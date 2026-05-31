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
    voltage_step_V: float = 0.25    # bias-ramp increment for Newton stability
