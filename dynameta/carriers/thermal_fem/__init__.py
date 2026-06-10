"""
FEM heat-equation thermal driver + electro-thermal Joule coupling -- the volumetric/lateral
generalization of carriers.thermal (which is the exact series-thermal-resistance profile for a 1D
flux-driven stack). Solves the steady heat equation div(k grad T) = -Q on a layered box with the
bottom face at the sink temperature (Dirichlet), a heat flux into the top face (Neumann), and an
optional volumetric Joule source Q [W/m^3] (e.g. sigma|E|^2 from the electrical solve -- the
electro-thermal coupling). Returns the temperature field T [K] for the field bundle that
ThermoOpticModel reads. Reduces EXACTLY to carriers.thermal.steady_layered_temperature when Q = 0,
and to the uniform-Joule slab profile T_mean = T_sink + Q L^2/(3k) for a single heated layer.

ALSO provides the TRANSIENT heat equation rho*Cp*dT/dt = div(k grad T) + Q (roadmap R5) via a
theta-method time integrator (solve_thermal_transient_fem). The transient path requires every layer
to carry rho_kg_m3 > 0 and Cp_J_kgK > 0 (mass density and specific heat); the STEADY path never
reads them, so adding them is byte-identical for all existing steady callers (they default 0.0).
Typical material values (NOT stored as constants -- ThermalLayer is the home): Si rho=2329 Cp=700,
SiO2 rho=2200 Cp=730, ITO rho=7140 Cp=340 (kg/m^3, J/(kg K)). Requires NGSolve.

Layers are ordered from the SINK (index 0, at z = 0 / the bottom Dirichlet face) outward; the top
face (z = sum thicknesses) receives `flux_W_m2`. Units: the mesh is built in nm (coordinate =
metres * _S); with k in W/(m K), T in K, the SI weak form maps to mesh coordinates as
  int k gradT.gradv dV'  =  int (Q/_S^2) v dV'  +  int_top (flux/_S) v dS'
(the _S powers convert the SI source [W/m^3] and flux [W/m^2] into the nm-coordinate integrals).
The stiffness/load thus assemble as _S * (K_phys, f_phys); for the transient the mass term must
assemble as _S * M_phys too so the common _S cancels and dt stays in SI seconds. A plain
int rho*Cp*u*v*dV' integral equals _S^3 * M_phys, so the mass coefficient carries 1/_S^2:
  int (rho*Cp/_S^2) u v dV'  =  _S * int rho*Cp u v dV_phys  =  _S * M_phys.   (verified by R5 gates)
"""

from __future__ import annotations

from dynameta.carriers.thermal_fem.common import (_S, ThermalLayer,
                                                  ThermalLayerTwoTemp,
                                                  _add_load_terms,
                                                  _build_layered_mesh,
                                                  _build_thermal_forms,
                                                  _mean_T_per_layer,
                                                  _per_material_cf)
from dynameta.carriers.thermal_fem.steady import ThermalResult, solve_thermal_fem
from dynameta.carriers.thermal_fem.transient import (ThermalTransientResult,
                                                     solve_thermal_transient_fem)
from dynameta.carriers.thermal_fem.twotemp import (ThermalTransientTwoTempResult,
                                                   _twotemp_space_and_forms,
                                                   ThermalTwoTempResult,
                                                   solve_thermal_transient_twotemp_fem,
                                                   solve_thermal_twotemp_fem)
from dynameta.carriers.thermal_fem.kirchhoff import (ThermalKirchhoffLayeredResult,
                                                     ThermalKirchhoffResult,
                                                     invert_kirchhoff,
                                                     kirchhoff_theta,
                                                     solve_thermal_kirchhoff_fem,
                                                     solve_thermal_kirchhoff_layered_1d,
                                                     solve_thermal_transient_kt_fem)
