"""
Single source of the SI physical constants used across the library (CODATA 2018).

The carrier (DEVSIM) and optical (NGSolve/Drude) sides previously re-declared these
independently under inconsistent names (Q_E / Q / _Q_E for the elementary charge, plus a
function-local EPS0 in physics_drift_diffusion), which risked a silent drift between the two
halves of the bridge. They now live here.

This is a top-level LEAF module with NO imports (not under a package whose __init__ pulls in
materials/optics), so importing it can never trigger a circular import -- both the core spine
and either solver side can import it freely.
"""

Q_E     = 1.602176634e-19     # elementary charge, C
EPS0    = 8.8541878128e-12    # vacuum permittivity, F/m
KB      = 1.380649e-23        # Boltzmann constant, J/K
HBAR    = 1.054571817e-34     # reduced Planck constant, J s
M_E     = 9.1093837015e-31    # electron rest mass, kg
C_LIGHT = 2.99792458e8        # speed of light in vacuum, m/s
T_REF   = 300.0               # reference temperature, K
V_T     = KB * T_REF / Q_E    # thermal voltage at T_REF, V
