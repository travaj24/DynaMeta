"""Stage 1: DC drift-diffusion + Poisson via 2D DEVSIM."""

from dynameta.stage1_carriers.solver import run_stage1
from dynameta.stage1_carriers.io    import dump_carrier_field

__all__ = ["run_stage1", "dump_carrier_field"]
