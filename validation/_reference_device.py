"""Reference gated-ITO metasurface modulator -- a PRIVATE validation/test fixture
(not a showcased example; underscore-prefixed so the runner skips it).

A reflection modulator: metal mirror / oxide-ITO-oxide gate cavity / metal patch --
a square unit cell with a single centred metal-patch inclusion over a full-cell ITO
cavity. ITO is the generic worked ENZ material; its Drude/Kane parameters are
representative near-IR values (see docs/dielectrics.md for the DC-permittivity
sourcing). Exists only to exercise the pipeline across the validation suite.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dynameta.materials import (
    Material, MaterialRegistry, DrudeOptical, ConstantOptical, TransportModel, M_E)
from dynameta.geometry import (
    UnitCell, Stack, Layer, Inclusion, Electrode, Design, centered_square)
from dynameta.sweep import Sweep, BiasPoint
from dynameta.pipeline import run_pipeline


Q_E = 1.602176634e-19

# ITO optical Drude -- representative near-IR ITO Drude fit (eps_inf 4.25, optical m* 0.225,
# Gamma 1.1e14). DOS mass for Stage-1 Nc is a separate, heavier Kane mass.
ITO_N_BG = 4.0e20 * 1e6
ITO_M_OPT = 0.225 * M_E
ITO_GAMMA = 1.1e14
ITO_M_LOW = 0.27 * M_E
ITO_KANE_ALPHA = 0.5
ITO_MOBILITY = 30e-4          # m^2/(V.s); only used by the drift-diffusion path


def ito_dos_mass(n_m3):
    n = np.maximum(np.asarray(n_m3, dtype=np.float64), 1e10)
    HBAR = 1.054571817e-34
    KF = (3.0 * np.pi**2 * n) ** (1.0 / 3.0)
    E_F = HBAR**2 * KF**2 / (2.0 * ITO_M_LOW)
    return ITO_M_LOW * np.sqrt(1.0 + 2.0 * ITO_KANE_ALPHA * E_F / Q_E)


def build_materials(physics: str = "equilibrium") -> MaterialRegistry:
    # Stage-1 carrier model for the ITO. "equilibrium" (default) is the single-
    # variable Poisson + Fermi-Dirac node model -- exact for the gated capacitor's
    # steady state and the fast default. "drift_diffusion" adds the full continuity
    # solve (Scharfetter-Gummel with the degenerate diffusion-enhancement); it
    # needs a mobility and reduces to the equilibrium profile here (no DC current).
    mob = (None if physics == "equilibrium"
           else (lambda n: np.full_like(np.asarray(n, float), ITO_MOBILITY)))
    reg = MaterialRegistry()
    reg.add(Material("air",   ConstantOptical(1.0 + 0j)))
    reg.add(Material("Si",    ConstantOptical(12.0 + 0j)))
    # Dielectrics carry BOTH an optical eps (Stage 2/3) and a DC eps
    # (eps_static_dc) for the Stage-1 gate capacitance, which drives accumulation.
    # HfO2/Al2O3 are high-k: their DC eps (18 / 9, ALD-thin-film values) is far
    # above their optical eps (4 / 2.756). These are measured/film values, NOT the
    # higher crystalline-DFPT numbers a database returns -- see docs/dielectrics.md
    # for the measured-vs-DFPT comparison and DielectricDB (audit + C-V override).
    reg.add(Material("Al2O3", ConstantOptical(2.756 + 0j), eps_static_dc=9.0))
    reg.add(Material("HfO2",  ConstantOptical(4.0 + 0j),   eps_static_dc=18.0))
    reg.add(Material("Al-Nd", ConstantOptical(-180 + 30j), is_metal=True))
    reg.add(Material("Au",    ConstantOptical(-100 + 8j),  is_metal=True))
    reg.add(Material("ITO",
        optical=DrudeOptical(eps_inf=4.25, m_opt_kg=ITO_M_OPT, gamma_rad_s=ITO_GAMMA),
        transport=TransportModel(n_bg_m3=ITO_N_BG, eps_static=9.5,
                                   dos_mass_kg_of_n_m3=ito_dos_mass,
                                   band_gap_eV=3.6, chi_eV=4.5,
                                   physics=physics,
                                   mobility_m2Vs_of_n_m3=mob),
        pretty_name="ITO (indium tin oxide)"))
    return reg


def build_reference_modulator(physics: str = "equilibrium") -> Design:
    cell = UnitCell.square(370e-9)
    layers = [
        Layer("mirror",      70e-9, "Al-Nd"),
        Layer("lower_al2o3",  1e-9, "Al2O3"),
        Layer("lower_hfo2",   7e-9, "HfO2"),
        Layer("ito",          5e-9, "ITO"),
        Layer("upper_hfo2",   7e-9, "HfO2"),
        Layer("upper_al2o3",  1e-9, "Al2O3"),
        Layer("patch",       50e-9, "air",
              inclusions=[Inclusion(centered_square(cell, 175e-9), "Au")]),
    ]
    stack = Stack(layers=layers, superstrate_material="air", substrate_material="Si")
    electrodes = [
        Electrode("bot_contact", "mirror", "full", role="biased"),
        Electrode("top_contact", "patch", centered_square(cell, 175e-9), role="biased"),
        Electrode("ito_gnd_left",  "ito", "x_lo", role="ground", fixed_voltage_V=0.0),
        Electrode("ito_gnd_right", "ito", "x_hi", role="ground", fixed_voltage_V=0.0),
    ]
    return Design(name="reference_modulator", unit_cell=cell, stack=stack,
                    electrodes=electrodes, materials=build_materials(physics),
                    pretty_name="reference gated-ITO metasurface modulator")
