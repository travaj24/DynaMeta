"""Generic gated-film metasurface cell -- a PRIVATE validation/test fixture
(not a showcased example; underscore-prefixed so the runner skips it).

A reflection cell: metal mirror / oxide-film-oxide gate cavity / metal patch -- a square unit
cell with a single centred metal-patch inclusion over a full-cell degenerate semiconductor film.
It is the 3-D (square-patch) form of the library's generic strip cell (tests/_strip_cell_fixture.py):
the same spacer (2 nm oxA / 6 nm oxB / 6 nm film / 6 nm oxB / 2 nm oxA), mirror and patch metals,
400 nm period, 150 nm wide and 40 nm tall patch, glass substrate. The film ("TCO") is a generic
transparent-conductor Drude film (n_bg 3e26 m^-3, optical mass 0.30 m_e, damping 1.2e14 rad/s,
eps_inf 4.0) with a Kane-nonparabolic DOS mass for the Stage-1 N_c. Exists only to exercise the
pipeline across the validation suite.
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

# Film optical Drude (the generic transparent-conductor film of the strip fixture). The DOS mass for
# the Stage-1 N_c is a separate Kane band-edge mass with a non-parabolicity.
FILM_N_BG = 3.0e26
FILM_M_OPT = 0.30 * M_E
FILM_GAMMA = 1.2e14
FILM_M_LOW = 0.25 * M_E
FILM_KANE_ALPHA = 0.4
FILM_MOBILITY = 40e-4         # m^2/(V.s); only used by the drift-diffusion path
FILM_T = 6e-9
PERIOD = 400e-9
PATCH_W = 150e-9


def film_dos_mass(n_m3):
    n = np.maximum(np.asarray(n_m3, dtype=np.float64), 1e10)
    HBAR = 1.054571817e-34
    KF = (3.0 * np.pi**2 * n) ** (1.0 / 3.0)
    E_F = HBAR**2 * KF**2 / (2.0 * FILM_M_LOW)
    return FILM_M_LOW * np.sqrt(1.0 + 2.0 * FILM_KANE_ALPHA * E_F / Q_E)


def build_materials(physics: str = "equilibrium") -> MaterialRegistry:
    # Stage-1 carrier model for the film. "equilibrium" (default) is the single-
    # variable Poisson + Fermi-Dirac node model -- exact for the gated capacitor's
    # steady state and the fast default. "drift_diffusion" adds the full continuity
    # solve (Scharfetter-Gummel with the degenerate diffusion-enhancement); it
    # needs a mobility and reduces to the equilibrium profile here (no DC current).
    mob = (None if physics == "equilibrium"
           else (lambda n: np.full_like(np.asarray(n, float), FILM_MOBILITY)))
    reg = MaterialRegistry()
    reg.add(Material("air",   ConstantOptical(1.0 + 0j)))
    reg.add(Material("glass", ConstantOptical(1.45 ** 2 + 0j)))
    # Dielectrics carry BOTH an optical eps (Stage 2/3) and a DC eps (eps_static_dc)
    # for the Stage-1 gate capacitance, which drives accumulation. High-k gate oxides
    # have a DC eps far above their optical eps; see docs/dielectrics.md.
    reg.add(Material("oxA", ConstantOptical(1.70 ** 2 + 0j), eps_static_dc=9.0))
    reg.add(Material("oxB", ConstantOptical(2.00 ** 2 + 0j), eps_static_dc=20.0))
    reg.add(Material("mirror_metal", ConstantOptical(-120.0 + 20.0j), is_metal=True))
    reg.add(Material("metal", ConstantOptical(-60.0 + 4.0j), is_metal=True))
    reg.add(Material("TCO",
        optical=DrudeOptical(eps_inf=4.0, m_opt_kg=FILM_M_OPT, gamma_rad_s=FILM_GAMMA),
        transport=TransportModel(n_bg_m3=FILM_N_BG, eps_static=9.0,
                                   dos_mass_kg_of_n_m3=film_dos_mass,
                                   band_gap_eV=3.5, chi_eV=4.5,
                                   physics=physics,
                                   mobility_m2Vs_of_n_m3=mob),
        pretty_name="TCO (generic transparent-conductor film)"))
    return reg


def build_reference_modulator(physics: str = "equilibrium") -> Design:
    cell = UnitCell.square(PERIOD)
    layers = [
        Layer("mirror", 80e-9, "mirror_metal"),
        Layer("lo_oxa",  2e-9, "oxA"),
        Layer("lo_oxb",  6e-9, "oxB"),
        Layer("film",  FILM_T, "TCO"),
        Layer("up_oxb",  6e-9, "oxB"),
        Layer("up_oxa",  2e-9, "oxA"),
        Layer("patch",  40e-9, "air",
              inclusions=[Inclusion(centered_square(cell, PATCH_W), "metal")]),
    ]
    stack = Stack(layers=layers, superstrate_material="air", substrate_material="glass")
    electrodes = [
        Electrode("bot_contact", "mirror", "full", role="biased"),
        Electrode("top_contact", "patch", centered_square(cell, PATCH_W), role="biased"),
        Electrode("film_gnd_left",  "film", "x_lo", role="ground", fixed_voltage_V=0.0),
        Electrode("film_gnd_right", "film", "x_hi", role="ground", fixed_voltage_V=0.0),
    ]
    return Design(name="reference_cell", unit_cell=cell, stack=stack,
                    electrodes=electrodes, materials=build_materials(physics),
                    pretty_name="generic gated-film metasurface cell")
