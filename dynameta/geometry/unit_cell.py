"""
UnitCell: the lateral (x, y) periodicity of the device. Rectangular in
general; square is the special case period_x_m == period_y_m. The cell
origin is at (0, 0); the cell spans [0, period_x] x [0, period_y].
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class UnitCell:
    period_x_m: float
    period_y_m: float

    def __post_init__(self) -> None:
        if self.period_x_m <= 0 or self.period_y_m <= 0:
            raise ValueError("UnitCell periods must be positive")

    @property
    def is_square(self) -> bool:
        return abs(self.period_x_m - self.period_y_m) < 1e-15 * self.period_x_m

    @property
    def center_m(self) -> "tuple[float, float]":
        return (0.5 * self.period_x_m, 0.5 * self.period_y_m)

    def lattice_symmetry(self) -> str:
        """Coarse lattice point-group tag: 'c4v' for a square cell, 'c2v'
        for a rectangular one. The DEVICE symmetry (used to gate the
        carrier-field lift) is the intersection of this with the inclusion
        shapes' symmetry -- computed at the Design level."""
        return "c4v" if self.is_square else "c2v"

    @classmethod
    def square(cls, period_m: float) -> "UnitCell":
        return cls(period_x_m=period_m, period_y_m=period_m)
