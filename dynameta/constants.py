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
HBAR    = 1.054571817e-34     # reduced Planck constant, J s (= h/2pi, CODATA-rounded literal)
H_PLANCK = 6.62607015e-34     # Planck constant, J s (exact SI definition; several soa modules
                              # used to re-derive it as 2 pi HBAR or re-type the literal)
M_E     = 9.1093837015e-31    # electron rest mass, kg
C_LIGHT = 2.99792458e8        # speed of light in vacuum, m/s
MU0     = 1.0 / (EPS0 * C_LIGHT ** 2)   # vacuum permeability, H/m (= 1/(eps0 c^2); the FDTD modules used to re-derive this locally)
T_REF   = 300.0               # reference temperature, K
V_T     = KB * T_REF / Q_E    # thermal voltage at T_REF, V
KB_EV_K = 8.617333262e-5      # Boltzmann constant, eV/K (CODATA-rounded literal, NOT computed as
                              # KB/Q_E -- the quotient differs at ~1.5e-11 rel and downstream
                              # Arrhenius pins are tighter; was re-declared in 8 reliability modules)

# Library-wide field time convention (Drude Im(eps) sign, NGSolve PML, FDTD ADEs). A string,
# not physics -- lives here because this is the only no-import leaf every layer can reach
# (was re-typed in core/bridge.py, core/carrier_field.py x2, core/eps_field.py).
SOLVER_TIME_CONVENTION = "exp(-iwt)"
GAMMA_E_RAD_ST = 1.760859630e11   # electron gyromagnetic ratio [rad/(s T)] (CODATA; audit S1-12)
