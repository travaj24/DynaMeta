"""
BiasPoint + Sweep: declarative specification of the voltage / wavelength
sweeps a Design will be simulated under.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class BiasPoint:
    """One DC bias setting of the device.

    `voltages` keys are electrode names (matching Electrode.name in the
    Design). Electrodes NOT listed get their Electrode.fixed_voltage_V
    (typically 0 V for grounds).

    `label` is a human-readable identifier used in filenames + plots.

    Example:
        BiasPoint(voltages={"top_contact": +2.0, "bot_contact": 0.0},
                  label="patch+2V")
    """
    voltages:  Dict[str, float]
    label:     str

    def __post_init__(self) -> None:
        if not self.label:
            raise ValueError("BiasPoint requires a non-empty label")


@dataclass
class Sweep:
    """A list of bias points + a list of wavelengths to simulate at.

    The total cost is len(bias_points) * len(wavelengths_nm) FEM solves.
    """
    bias_points:        List[BiasPoint]
    wavelengths_nm:     List[float] = field(default_factory=list)
    # Per-stage controls
    voltage_step_m:     float = 0.1     # Newton-ramp dV at each ramp step
    rel_tol:            float = 1e-5    # DC solve rel tolerance
    max_iter_dc:        int   = 100     # Newton iterations cap

    def __post_init__(self) -> None:
        if not self.bias_points:
            raise ValueError("Sweep requires at least 1 BiasPoint")
        labels = [bp.label for bp in self.bias_points]
        if len(set(labels)) != len(labels):
            raise ValueError("Duplicate BiasPoint labels: {}".format(labels))
