"""
Default Stage-1 builder: a 2D (x, z) DEVSIM device from the layered Design,
implementing the core CarrierSolver Protocol.

Region rule (generalizes the old patch-specific build):
  - A layer's BACKGROUND is meshed as a DC region unless its material is the
    ambient (super/substrate) material -- e.g. the air above a patch is skipped.
  - Each INCLUSION is meshed over its x-extent as its own region.
  - Metals are meshed but INERT (no equation) so their Dirichlet contacts can
    realize (this session's finding); dielectrics/semiconductors get physics.
Contacts:
  - edge-footprint electrodes -> thin x-slab at the cell edge, full layer
    z-range, on the layer's region (the ITO peripheral grounds).
  - metal-gate electrodes -> slab spanning the gate metal's z-extent over the
    footprint x-range, attached to the nearest meshed non-metal neighbour.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
from dynameta.carriers.devsim_env import import_devsim   # D0: the ONE devsim import home

try:
    ds = import_devsim()
except ImportError as _e:                                   # pragma: no cover - optional heavy solver
    raise ImportError("dynameta.carriers.devsim_layered requires the optional 'devsim' package "
                      "(pip install dynameta[solvers]); the pure-TMM / bring-your-own-solver paths "
                      "do not import this module.") from _e

from dynameta.core.carrier_field import (
    CarrierField, CarrierRegion, ELECTRON_DENSITY, POTENTIAL)
from dynameta.core.interfaces import RegionInfo
from dynameta.core.resample import resample_to_grid
from dynameta.carriers import physics_equilibrium as PE
from dynameta.carriers import physics_drift_diffusion as DD
from dynameta.carriers import physics_bipolar_dd as BP
from dynameta.carriers import physics_traps as PT
from dynameta.carriers import physics_heterojunction as HJ
from dynameta.carriers import eq_registry as _R
from dynameta.carriers import extended_precision as XP
from dynameta.carriers import noise_analysis as NA
from dynameta.carriers.dc_solve import solve_dc
from dynameta.geometry.design import Design
from dynameta.materials.band_temperature import band_data_at_T, transport_temperature_K
from dynameta.carriers.tunneling import btbt_for_material as _btbt_for_material
from dynameta.geometry.guards import require_supported_geometry
from dynameta.warning_types import AdvisoryWarning, UnreliableResultWarning

# Width of the thin edge-metal strip inserted at a drift-diffusion semiconductor's grounded edge so
# the ground becomes a region-region INTERFACE (full-line node capture) instead of a weak 2-node
# domain-boundary contact -- the fix that makes GATED DD converge (validation/gated_dd_2d.py). Kept
# thin (~1 nm) so the carrier-side x-seam vs the full-cell optics ITO is negligible: the carrier
# ITO grid then spans [w, P-w], and the optics NGSolve VoxelCoefficient (linear=True) edge-clamps
# the ~1 nm slivers to the adjacent (ohmic-pinned, ~n_bg) value -- the physically correct unbiased
# eps there.
_EDGE_METAL_W_M = 1.5e-9

# the circuit solutions a DC solve writes (DEVSIM creates both on the first circuit solve, values 0.0);
# noise_spectrum restores them with the node solutions
_DC_CIRCUIT_SOLUTIONS = ("dcop", "dcop_prev")


@dataclass
class _RegionSpec:
    name:     str
    material: str
    role:     str            # metal | dielectric | semiconductor
    x_lo:     float
    x_hi:     float
    z_lo:     float
    z_hi:     float


# Band offsets below this (volts) count as ZERO for heterointerface="auto" (a same-crystal split).
_HJ_ZERO_OFFSET_V = 1e-9
# A bipolar heterojunction reads its band gap from its OWN statistics, E_g = kT ln(N_c N_v / n_i^2) (the
# only gap consistent with the bulk model -- any other makes the interface a spurious battery). It must
# agree with TransportModel.band_gap_eV to this tolerance or the build REFUSES (the inputs disagree).
_HJ_EG_TOL_EV = 5e-3


class OhmicContactNoiseError(NotImplementedError):
    """LayeredDevsimBuilder.noise_spectrum / terminal_noise on an electrode that is an OHMIC carrier contact
    (it pins Electrons / Holes as well as Potential). v0.15 drives dielectric (Potential-only) gates only
    (plan 4.2.5, user decision 35): repointing an ohmic contact circuit-driven would move only its Potential
    Dirichlet and leave the carrier pins at the old bias. The follow-on is a set_ssac_contact built on
    physics_bipolar_dd.setup_contact_ohmic_bipolar_circuit (decision D10b)."""


@dataclass(frozen=True)
class LayeredNoiseSpectrum(NA.NoiseSpectrum):
    """noise_analysis.NoiseSpectrum of a LayeredDevsimBuilder gate, in the units it was scaled to.

    units    "A^2/Hz per cell" (per_cell=True: S, every breakdown and every map multiplied by depth_m =
             UnitCell.period_y_m) or "A^2/Hz/m per unit depth" (per_cell=False: DEVSIM's 2-D result as is).
    depth_m  the factor applied (period_y_m, or 1.0).
    gate, vg_V, voltages  the circuit-driven electrode, its bias, and every electrode's bias of the state
             the noise was computed at."""
    units: str = "A^2/Hz/m per unit depth"
    depth_m: float = 1.0
    gate: str = ""
    vg_V: float = 0.0
    voltages: Dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class LayeredTerminalNoise(NA.TerminalNoise):
    """noise_analysis.TerminalNoise of a LayeredDevsimBuilder gate: S_I and Y = G + i omega C both scaled by
    depth_m (per cell: A^2/Hz and S), S_V = S_I / |Y|^2 from the scaled values (V^2/Hz per cell; it scales
    as 1/depth_m). `current` is the LayeredNoiseSpectrum."""
    units: str = "A^2/Hz/m per unit depth"
    depth_m: float = 1.0


class LayeredDevsimBuilder:
    """2-D (x, z) DEVSIM builder over a layered Design.

    heterointerface (v0.14.0, workstream D1a) selects the carrier model on every z-stacked
    semiconductor/semiconductor interface whose two regions BOTH solve carrier continuity
    (physics_heterojunction): "thermionic" (Horio-Yanai / Yang-East-Haddad thermionic emission),
    "continuous" (continuous quasi-Fermi levels, the drift-diffusion limit), or "auto" (default):
    thermionic where the interface carries a band offset (|dE_c| or |dE_v| > 1 nV), continuous where it
    does not (a same-crystal split, e.g. a p/n homojunction cut into two layers). Every semiconductor
    region is placed in ONE Anderson-aligned frame (the reference is the region of the first GROUND
    electrode on a semiconductor, else the first semiconductor region; a gate on a dielectric keeps
    that reference's flat band), so adjacent equilibrium semiconductors now carry their built-in
    potential (audit S1-9 2-D sibling) and every ohmic contact pins Potential at bias + s(region).

    on_chi_temperature (v0.14 row 3.2-10, user decision 3): what a HETEROSTACK (two or more different
    semiconductor materials) does when the device temperature (BipolarOptions.T_K; T_REF otherwise) differs
    from the temperature its affinities were quoted at (TransportModel.chi_reference_T_K, None = T_REF), for
    the materials WITHOUT a band-temperature model (their chi_eV is used unchanged at the device T). A
    material with TransportModel.band_temperature (v0.15, materials.BandTemperature) has chi, E_g and,
    optionally, n_i evaluated at the device T (materials.band_data_at_T) and is never subject to the toggle.
      "refuse" (default): raise physics_heterojunction.AffinityTemperatureError naming the materials, this
               toggle and the field. Use it unless you know your affinities; if chi_eV IS your value at the
               device T, tag the materials (chi_reference_T_K = T_K) instead of switching the check off.
      "warn"   : build, with an AdvisoryWarning stating the size of the assumption (~1.1-2.2 meV in
               Delta E_c per 100 K for Al0.3Ga0.7As/GaAs, more for Si/SiGe) -- exploratory T sweeps.
      "off"    : build silently -- for affinities you have validated at the device T.
    Only materials whose (chi_eV, band_gap_eV) differ count as a heterostack (a p/n homojunction is two
    material names of one semiconductor); single-material devices and the 300 K default are never affected."""

    def __init__(self, design: Design, *, mesh_name: str = "ms_mesh",
                  device_name: str = "ms_device", oxide_tunneling=None,
                  interface_traps=None, heterointerface: str = "auto",
                  extended_precision: Optional[bool] = None, on_chi_temperature: str = "refuse") -> None:
        if heterointerface not in ("auto",) + HJ.HETERO_MODELS:
            raise ValueError("heterointerface must be 'auto', 'thermionic' or 'continuous'; got {!r}"
                             .format(heterointerface))
        if on_chi_temperature not in HJ.CHI_TEMPERATURE_MODES:
            raise ValueError("on_chi_temperature must be one of {}; got {!r}".format(
                HJ.CHI_TEMPERATURE_MODES, on_chi_temperature))
        self.on_chi_temperature = on_chi_temperature        # class docstring; v0.14 row 3.2-10
        # 2026-11-02: _region_specs plans layers + inclusions only -- a Stack.features via came back
        # as the via-free region list (measured), and a tapered inclusion would be a straight box at its
        # TOP section. Refuse before planning (single-sourced message: dynameta.geometry.guards).
        require_supported_geometry(design, "LayeredDevsimBuilder")
        self.design = design
        self.mesh_name = mesh_name
        self.device = device_name
        # D1d: optional gate-oxide tunneling POST-PROCESSOR (carriers.tunneling.OxideTunnelingParams).
        # None (default) -> nothing is computed and CarrierField.extras is unchanged. When set, every
        # solve adds extras["oxide_tunneling"] = {dielectric_region: summary} for each dielectric region
        # sharing an interface with a semiconductor region (the converged field; no feedback).
        # v0.15 (plan 4.1.1): OxideTunnelingParams(coupling="self_consistent", ...) instead INSTALLS the
        # self-consistent term (tunneling.setup_oxide_tunneling) at build_device(), from every bipolar
        # semiconductor region to every gate electrode it reaches through dielectric regions only; the
        # extras then report the term's own kernel call (+ "coupling", "gate", "I_gate_A") and
        # extras["contact_currents_A"] carries the gate's DC tunnelling current.
        self.oxide_tunneling = oxide_tunneling
        # D1c (2026-11-03): interface traps, {key: physics_traps.InterfaceTraps (or a list)} with key an
        # interface name ("<lower region>__<upper region>") or a 2-tuple of region names (any order).
        # Installed on semiconductor(bipolar_dd)/dielectric interfaces after Potential continuity; an
        # unknown key or a non-bipolar side raises at build. None (default) = no interface traps.
        self.interface_traps = dict(interface_traps) if interface_traps else {}
        self.heterointerface = heterointerface
        self._frames: Dict[str, Tuple[float, float]] = {}   # semiconductor region -> (Phi_c_V, s_V)
        self._hetero_ifaces: Dict[str, str] = {}             # interface -> carrier model wired on it
        # D2b (2026-11-03): extended_precision=True builds the device with DEVSIM's 128-bit model
        # evaluation + equation assembly and solves it with the mixed-precision Newton (128-bit
        # residual, double factorization) -- carriers/extended_precision.py. Device-scoped (it never
        # reaches a later default device); False (default) leaves every solve byte-identical.
        # v0.15 (plan 4.2.6, decision 37): None (default) follows Design.carriers.extended_precision
        # (unstated = off); an explicit bool wins, with ONE AdvisoryWarning when it contradicts a stated
        # design value (XP.resolve_design_flag; geometry.specs.CarrierSpec says when to use it).
        self.extended_precision = XP.resolve_design_flag(design, extended_precision,
                                                         "LayeredDevsimBuilder")
        self._specs: List[_RegionSpec] = self._region_specs()
        self._contact_region: Dict[str, str] = {}
        self._iface_pairs: List[tuple] = []                  # (region0, region1) per interface
        # electrode -> circuit source that drives it, once set_ssac_gate repointed it (see _drive)
        self._circuit_gates: Dict[str, str] = {}
        self._circuit_gate_nodes: Dict[str, str] = {}        # electrode -> its circuit node (set_ssac_gate)
        self._built = False

    # ---- region planning ----
    def _ambient(self) -> set:
        s = self.design.stack
        return {s.superstrate_material, s.substrate_material}

    def _metal_material(self) -> Optional[str]:
        """A material name with role 'metal' in the stack (for the inert edge-metal strip). The
        strip's properties are irrelevant (it carries no physics); only role=='metal' matters."""
        d = self.design
        for L in d.stack.layers:
            if d.material_role(L.background_material) == "metal":
                return L.background_material
            for inc in L.inclusions:
                if d.material_role(inc.material) == "metal":
                    return inc.material
        # fallback: ANY registered is_metal material (a bipolar diode has no metal LAYER, but still
        # needs the inert edge-metal strip for its full-edge contacts -- the strip is just a label).
        for nm in d.materials.names():
            if d.material_role(nm) == "metal":
                return nm
        return None

    def _bipolar_edge_contacts(self) -> Dict[str, set]:
        """Layer -> set of x-edge SIDES carrying ANY edge contact (anode/cathode, any role) on a
        bipolar_dd semiconductor layer. Like the DD grounds, each needs a full-edge metal strip so the
        contact is a region-region interface (full-line node capture); a 2-node domain-boundary contact
        cannot anchor the 2D bipolar (3-variable) solve and it diverges."""
        d = self.design
        out: Dict[str, set] = {}
        for E in d.electrodes:
            if not (E.is_edge and E.footprint in ("x_lo", "x_hi")):
                continue
            L = next((ly for ly in d.stack.layers if ly.name == E.layer), None)
            if L is None or L.inclusions:
                continue
            tr = getattr(d.materials.get(L.background_material), "transport", None)
            if tr is not None and getattr(tr, "physics", None) == "bipolar_dd":
                out.setdefault(E.layer, set()).add(E.footprint)
        return out

    def _full_edge_strips(self) -> Dict[str, set]:
        """Union of the electron-DD edge GROUNDS and ALL bipolar_dd edge contacts -> the layers/sides
        that get an adjacent edge-metal strip (full-line node capture). Used by the region carve, the
        mesh lines, and the contact placement so both DD grounds and bipolar anode/cathode are strong."""
        out: Dict[str, set] = {k: set(v) for k, v in self._dd_full_edge_grounds().items()}
        for k, v in self._bipolar_edge_contacts().items():
            out.setdefault(k, set()).update(v)
        return out

    def _dd_full_edge_grounds(self) -> Dict[str, set]:
        """Map layer-name -> set of edge SIDES ({'x_lo','x_hi'}) carrying a GROUND electrode on a
        drift-diffusion semiconductor layer (no inclusions). Each such edge needs a FULL-EDGE ohmic
        ground -- a thin adjacent edge-metal region so the ground is a region-region interface
        (full-line node capture) instead of a weak 2-node domain-boundary contact, which cannot
        anchor the continuity equation (gated DD otherwise does not converge). A layer may be
        grounded on BOTH edges (e.g. the reference ITO). Equilibrium layers are NOT affected."""
        d = self.design
        out: Dict[str, set] = {}
        for E in d.electrodes:
            if not (E.is_edge and E.role == "ground" and E.footprint in ("x_lo", "x_hi")):
                continue
            L = next((ly for ly in d.stack.layers if ly.name == E.layer), None)
            if L is None or L.inclusions:
                continue
            tr = getattr(d.materials.get(L.background_material), "transport", None)
            if tr is not None and getattr(tr, "physics", None) == "drift_diffusion":
                out.setdefault(E.layer, set()).add(E.footprint)
        return out

    def _region_specs(self) -> List[_RegionSpec]:
        d = self.design
        P = d.unit_cell.period_x_m
        z_iv = d.z_intervals()
        ambient = self._ambient()
        fe = self._full_edge_strips()                     # DD grounds + bipolar contacts -> full-edge
        metal_mat = self._metal_material() if fe else None
        if fe and metal_mat is None:
            warnings.warn(
                "layered full-edge contact needed for layer(s) {} but the design has no metal "
                "material to form the edge-metal strip; falling back to the weak 2-node contact "
                "(gated DD / 2D bipolar may not converge).".format(sorted(fe)), AdvisoryWarning, stacklevel=2)
        # y-edge grounds are not resolved in the 2D (x,z) cross-section -> they get no full-edge
        # ohmic ground (and the legacy contact placement mis-locates them at the x-hi edge), so a
        # gated DD device with only a y-edge ground may not converge. Warn rather than fail silently.
        for E in d.electrodes:
            if E.is_edge and E.role == "ground" and E.footprint in ("y_lo", "y_hi"):
                L = next((ly for ly in d.stack.layers if ly.name == E.layer), None)
                tr = getattr(d.materials.get(L.background_material), "transport", None) if L else None
                if tr is not None and getattr(tr, "physics", None) == "drift_diffusion":
                    warnings.warn(
                        "electrode '{}': a y-edge ground on a drift-diffusion layer is not resolved "
                        "in the 2D (x,z) builder -- no full-edge ohmic ground is formed and gated DD "
                        "may not converge. Use an x_lo/x_hi edge ground.".format(E.name), AdvisoryWarning, stacklevel=2)
        specs: List[_RegionSpec] = []
        for L in d.stack.layers:
            zlo, zhi = z_iv[L.name]
            bg_role = d.material_role(L.background_material)
            bg_ambient = L.background_material in ambient
            n_incl = len(L.inclusions)
            # audit C2-3: the 2D interface loop wires only z-ADJACENT region pairs, so an
            # inclusion overlaying a non-ambient background gets coincident DUPLICATED
            # nodes with NO lateral interface -- DEVSIM then gives each side a natural
            # zero-flux wall and the inclusion is electrostatically ISOLATED, silently.
            # The 3D builder creates these lateral interfaces; refuse here until the 2D
            # sidewall wiring is built and validated (matching the 3D raise discipline).
            if not bg_ambient and n_incl:
                raise NotImplementedError(
                    "LayeredDevsimBuilder: layer '{}' has a NON-ambient background WITH "
                    "inclusions -- the 2D builder wires no lateral (sidewall) interfaces, "
                    "so the inclusion would be electrostatically isolated from its "
                    "background with no warning (audit C2-3). Missing: splitting the background "
                    "around the inclusion and wiring the vertical region-region interfaces (Potential "
                    "and, for two transport semiconductors, the v0.14.0 heterointerface carrier "
                    "equations). z-stacked semiconductor heterointerfaces ARE supported. Use the 3D "
                    "builder (Stacked3DSpec.from_design) or make the background ambient.".format(L.name))
            if bg_ambient and n_incl >= 2:
                bbs = [inc.shape.bbox_m() for inc in L.inclusions]
                for i in range(n_incl):
                    for j in range(i + 1, n_incl):
                        if min(bbs[i][1], bbs[j][1]) - max(bbs[i][0], bbs[j][0]) > -1e-15:
                            raise NotImplementedError(
                                "LayeredDevsimBuilder: layer '{}' inclusions {} and {} touch/"
                                "overlap laterally -- the 2D builder wires no lateral "
                                "interface between them, so they would be electrically "
                                "DECOUPLED silently (audit C2-3). Separate them or use the "
                                "3D builder.".format(L.name, i, j))
            if not bg_ambient:
                # background region over the full cell (x split by inclusions is
                # ignored for the background's DC role; inclusions overlay it). For a
                # DD-semiconductor layer with an edge GROUND, carve a thin edge-metal strip
                # at that edge so the ground is a region-region INTERFACE (full-edge ohmic
                # capture) instead of a weak 2-node domain-boundary contact.
                x0, x1 = 0.0, P
                sides = fe.get(L.name, set()) if metal_mat is not None else set()
                w = _EDGE_METAL_W_M
                if "x_lo" in sides:
                    specs.append(_RegionSpec(L.name + "_egnd_lo", metal_mat, "metal",
                                              0.0, w, zlo, zhi))
                    x0 = w
                if "x_hi" in sides:
                    specs.append(_RegionSpec(L.name + "_egnd_hi", metal_mat, "metal",
                                              P - w, P, zlo, zhi))
                    x1 = P - w
                if sides and (x1 - x0) < 4.0 * w:
                    raise ValueError(
                        "layer '{}': the edge-metal ground strip(s) ({:.2g} nm each) leave a "
                        "degenerate semiconductor width {:.2g} nm at period {:.2g} nm -- use a "
                        "larger period or a smaller _EDGE_METAL_W_M.".format(
                            L.name, w * 1e9, (x1 - x0) * 1e9, P * 1e9))
                specs.append(_RegionSpec(L.name, L.background_material, bg_role,
                                          x0, x1, zlo, zhi))
            for i, inc in enumerate(L.inclusions):
                xlo, xhi, _, _ = inc.shape.bbox_m()
                role = d.material_role(inc.material)
                # name: if background is ambient and a single inclusion, the
                # inclusion *is* the layer's region (keeps names like "patch")
                name = L.name if (bg_ambient and n_incl == 1) \
                    else "{}__incl{}".format(L.name, i)
                specs.append(_RegionSpec(name, inc.material, role,
                                          xlo, xhi, zlo, zhi))
        return specs

    def _nearest_nonmetal(self, layer_name: str) -> Tuple[str, float]:
        """For a metal-gate electrode on `layer_name`, find the nearest meshed
        non-metal region and the z where the gate touches it."""
        d = self.design
        names = [L.name for L in d.stack.layers]
        idx = names.index(layer_name)
        z_iv = d.z_intervals()
        nonmetal = {s.name: s for s in self._specs if s.role != "metal"}
        for j in range(idx + 1, len(names)):                 # search upward
            if names[j] in nonmetal:
                return names[j], z_iv[names[j]][0]
        for j in range(idx - 1, -1, -1):                     # search downward
            if names[j] in nonmetal:
                return names[j], z_iv[names[j]][1]
        raise ValueError("gate layer '{}' has no meshed non-metal neighbour"
                          .format(layer_name))

    # ---- mesh build ----
    def build_device(self) -> None:
        d = self.design
        spec = d.mesh_2d
        P = d.unit_cell.period_x_m
        z_iv = d.z_intervals()
        ds.create_2d_mesh(mesh=self.mesh_name)

        # x mesh lines: cell edges + inclusion feature edges + cell midline
        x_lines = {0.0: spec.x_spacing_edge_m, P: spec.x_spacing_edge_m,
                    P / 2.0: spec.x_spacing_feature_mid_m}
        for L in d.stack.layers:
            for inc in L.inclusions:
                xlo, xhi, _, _ = inc.shape.bbox_m()
                x_lines[xlo] = spec.x_spacing_feature_edge_m
                x_lines[xhi] = spec.x_spacing_feature_edge_m
        # edge-metal interface line(s) for full-edge contacts (DD grounds + bipolar) -- strip boundaries
        if self._metal_material() is not None:
            for _ln, _sides in self._full_edge_strips().items():
                if "x_lo" in _sides:
                    x_lines[_EDGE_METAL_W_M] = spec.x_spacing_feature_edge_m
                if "x_hi" in _sides:
                    x_lines[P - _EDGE_METAL_W_M] = spec.x_spacing_feature_edge_m
        for pos in sorted(x_lines):
            ds.add_2d_mesh_line(mesh=self.mesh_name, dir="x", pos=pos,
                                  ns=x_lines[pos], ps=x_lines[pos])

        # y mesh lines: interface-zone refinement over meshed layers
        izone = spec.interface_zone_m
        # map region names back to their parent layer for z lookup; the edge-metal strip
        # ("<layer>_egnd") shares its layer's z-range (already covered), so drop names not in z_iv.
        meshed_layer_names = sorted({nm for nm in (s.name.split("__incl")[0] for s in self._specs)
                                     if nm in z_iv}, key=lambda nm: z_iv[nm][0])
        for nm in meshed_layer_names:
            zlo, zhi = z_iv[nm]
            thk = zhi - zlo
            ds.add_2d_mesh_line(mesh=self.mesh_name, dir="y", pos=zlo,
                                  ns=spec.interface_ps_m, ps=spec.interface_ps_m)
            zone = min(izone, 0.45 * thk)
            if zone > 0:
                ds.add_2d_mesh_line(mesh=self.mesh_name, dir="y", pos=zlo + zone,
                                      ns=spec.coarse_ps_m, ps=spec.coarse_ps_m)
                ds.add_2d_mesh_line(mesh=self.mesh_name, dir="y", pos=zhi - zone,
                                      ns=spec.coarse_ps_m, ps=spec.coarse_ps_m)
            if thk > 4 * izone:
                ds.add_2d_mesh_line(mesh=self.mesh_name, dir="y", pos=0.5 * (zlo + zhi),
                                      ns=spec.coarse_ps_m, ps=spec.coarse_ps_m)
        top_z = max(z_iv[nm][1] for nm in meshed_layer_names)
        ds.add_2d_mesh_line(mesh=self.mesh_name, dir="y", pos=top_z,
                              ns=spec.interface_ps_m, ps=spec.interface_ps_m)

        # regions
        for s in self._specs:
            ds.add_2d_region(mesh=self.mesh_name, region=s.name, material=s.material,
                              xl=s.x_lo, xh=s.x_hi, yl=s.z_lo, yh=s.z_hi, bloat=1e-12)

        # interfaces between z-adjacent meshed regions sharing x-overlap. ONLY
        # between regions that both carry Potential (skip metals -- they are
        # inert with no Potential solution, so a continuity interface would fail;
        # their boundaries are handled by Dirichlet contacts instead).
        for a in self._specs:
            for b in self._specs:
                if a.name >= b.name:
                    continue
                if a.role == "metal" or b.role == "metal":
                    continue
                if abs(a.z_hi - b.z_lo) < 1e-15 or abs(b.z_hi - a.z_lo) < 1e-15:
                    xl = max(a.x_lo, b.x_lo); xh = min(a.x_hi, b.x_hi)
                    if xh <= xl:
                        continue
                    z_if = a.z_hi if abs(a.z_hi - b.z_lo) < 1e-15 else b.z_hi
                    r0, r1 = (a.name, b.name) if a.z_hi <= b.z_lo + 1e-15 else (b.name, a.name)
                    self._iface_pairs.append((r0, r1))       # for the heterointerface wiring below
                    ds.add_2d_interface(mesh=self.mesh_name,
                                          name="{}__{}".format(r0, r1),
                                          region0=r0, region1=r1,
                                          xl=xl, xh=xh, yl=z_if, yh=z_if, bloat=1e-12)

        # contacts
        for E in d.electrodes:
            zlo, zhi = z_iv[E.layer]
            if E.is_edge:
                region = E.layer
                if (self._metal_material() is not None
                        and E.footprint in self._full_edge_strips().get(E.layer, set())):
                    # FULL-EDGE ohmic ground (drift-diffusion): the contact sits at the
                    # semiconductor/edge-metal INTERFACE (x=w, full layer z-range). The adjacent
                    # edge-metal region (_region_specs carved it) makes this an interior region
                    # boundary -> full-line node capture, enough to anchor the continuity equation,
                    # so GATED DD converges (validation/gated_dd_2d.py). The carrier-side ITO grid
                    # is now x in [w, P-w] (both edges grounded) or [w, P] (one); the optics
                    # VoxelCoefficient edge-clamps the ~1nm seam to the adjacent ~n_bg value.
                    xb = _EDGE_METAL_W_M if E.footprint == "x_lo" else (P - _EDGE_METAL_W_M)
                    xlo, xhi = xb - 1e-10, xb + 1e-10
                else:
                    # Thin x-slab at the cell DOMAIN boundary. DEVSIM captures only ~2 box-corner
                    # nodes here (no adjacent region) -- a weak 2-node carrier pin, fine for the
                    # equilibrium solve (n is local) but too weak to anchor a continuity equation.
                    # The full-edge branch above is the DD fix; this remains for equilibrium grounds.
                    xlo = (0.0 - 1e-10) if E.footprint == "x_lo" else (P - 1e-10)
                    xhi = (0.0 + 1e-10) if E.footprint == "x_lo" else (P + 1e-10)
                yl_c, yh_c = zlo, zhi
            else:
                # metal gate: footprint x-range + nearest non-metal neighbour
                region, ybnd = self._nearest_nonmetal(E.layer)
                if E.footprint == "full":
                    xlo, xhi = 0.0, P
                else:
                    bx = E.footprint.bbox_m()
                    xlo, xhi = bx[0], bx[1]
                yl_c, yh_c = min(ybnd, zlo), max(ybnd, zhi)
            ds.add_2d_contact(mesh=self.mesh_name, name=E.name, material="metal",
                                region=region, xl=xlo, xh=xhi, yl=yl_c, yh=yh_c,
                                bloat=1e-10)
            self._contact_region[E.name] = region

        ds.finalize_mesh(mesh=self.mesh_name)
        ds.create_device(mesh=self.mesh_name, device=self.device)
        if self.extended_precision:                 # BEFORE any model: DEVSIM types models at creation
            XP.enable(self.device)

        # per-region physics. Track drift-diffusion semiconductor regions so
        # their ohmic contacts also pin the electron quasi-Fermi level.
        self._dd_regions = set()
        self._bipolar_regions = set()
        for s in self._specs:
            if s.role == "semiconductor":
                tr = self.design.materials.get(s.material).transport
                dos = float(tr.dos_mass_kg_of_n_m3(tr.n_bg_m3))
                if tr.physics == "bipolar_dd":
                    self._setup_bipolar_region(s, tr, dos)
                    self._dd_regions.add(s.name)         # treated as DD for the abs-tol scaling
                    self._bipolar_regions.add(s.name)
                elif tr.physics == "drift_diffusion":
                    # Gated DD needs a strong ohmic ground to anchor the continuity equation. A
                    # FULL-EDGE ground (edge-metal interface, wired automatically when this layer
                    # has an edge GROUND electrode) converges -- validation/gated_dd_2d.py. Without
                    # one, the only ground is a weak 2-node domain-boundary edge contact and a gated
                    # solve may not converge (use equilibrium mode, or add an edge ground here).
                    face = {self._contact_region.get(E.name) for E in self.design.electrodes
                            if not E.is_edge}                    # metal-layer contact on a face: strong
                    if s.name not in face and (s.name not in self._dd_full_edge_grounds()
                                               or self._metal_material() is None):
                        warnings.warn(
                            "layered drift-diffusion on semiconductor '{}': no full-edge ohmic "
                            "ground (only a weak 2-node edge ground) -- a gated device may not "
                            "converge; add an edge GROUND electrode on this layer, or use the "
                            "equilibrium physics mode for DC gate accumulation.".format(s.name), AdvisoryWarning, stacklevel=2)
                    _st = getattr(tr, "carrier_statistics", None)     # v0.15: "fermi_dirac" -> quasi-field
                    DD.setup_semiconductor_region_dd(
                        self.device, s.name, n_bg_m3=tr.n_bg_m3,
                        eps_static=tr.eps_static, dos_mass_kg=dos,
                        mobility_m2Vs=float(tr.mobility_m2Vs_of_n_m3(tr.n_bg_m3)),
                        field_mobility=tr.field_dependent_mobility, v_sat_ms=tr.v_sat_ms,
                        ct_beta=tr.ct_beta, **({"statistics": _st} if _st is not None else {}))
                    self._dd_regions.add(s.name)
                else:
                    PE.setup_semiconductor_region(
                        self.device, s.name, n_bg_m3=tr.n_bg_m3,
                        eps_static=tr.eps_static, dos_mass_kg=dos)
            elif s.role == "dielectric":
                eps_r = self._dielectric_eps_static(s.material)
                PE.setup_dielectric_region(self.device, s.name, eps_r)
            # metals: inert, no setup
        # v0.14.0 (D1a): ONE Anderson-aligned frame for every semiconductor region (the 2-D sibling of
        # audit S1-9 -- each region used to be neutral at psi = 0 in its own frame, so adjacent
        # equilibrium semiconductors had NO built-in potential), then the carrier coupling of each
        # semiconductor/semiconductor interface (thermionic emission or continuous quasi-Fermi level;
        # this replaced the audit-regrade-7.1 refusal that stood here while the interface wired Potential
        # only -- a silently carrier-BLOCKING junction). Contacts below pin Potential at bias + s(r).
        self._frames = self._semiconductor_frames()
        for iface in ds.get_interface_list(device=self.device):
            PE.setup_interface(self.device, iface)
        self._install_interface_traps()
        self._wire_heterointerfaces()
        for c in ds.get_contact_list(device=self.device):
            cr = self._contact_region.get(c)
            off = self._frames.get(cr, (0.0, 0.0))[1]
            if cr in self._bipolar_regions:
                BP.setup_contact_ohmic_bipolar(self.device, c, psi_offset_V=off)  # Potential+Electrons+Holes
            elif cr in self._dd_regions:
                DD.setup_contact_ohmic_dd(self.device, c, psi_offset_V=off)   # Potential + Electrons (=N_D)
            else:
                PE.setup_contact(self.device, c, psi_offset_V=off)            # Potential only
        if getattr(self.oxide_tunneling, "coupling", "post") == "self_consistent":
            self._install_oxide_tunneling()
        self._built = True

    def _install_oxide_tunneling(self) -> None:
        """v0.15: the self-consistent oxide-tunnelling term from each bipolar semiconductor region to each
        gate electrode it reaches through dielectric regions only (tunneling.dielectric_column), in the
        builder's Anderson frame (band_frames) with the materials' affinities."""
        from dynameta.carriers import tunneling as TN
        mats = {sp.name: sp.material for sp in self._specs}
        chi = {}
        for sp in self._specs:
            if sp.role == "semiconductor":
                tr = self.design.materials.get(sp.material).transport
                chi[sp.material] = tr.chi_eV
        n = 0
        for E in self.design.electrodes:
            if E.is_edge or self._contact_region.get(E.name) in self._bipolar_regions | self._dd_regions:
                continue
            for semi in sorted(self._bipolar_regions):
                if TN.dielectric_column(self.device, semi, E.name) is None:
                    continue
                TN.setup_oxide_tunneling(self.device, semi, E.name, self.oxide_tunneling,
                                         semiconductor_material=mats[semi], chi_eV=chi,
                                         phi_c_V=self._frames[semi][0])
                n += 1
        if n == 0:
            raise ValueError("LayeredDevsimBuilder(oxide_tunneling=...coupling='self_consistent'): no gate "
                             "electrode reaches a bipolar_dd semiconductor through dielectric regions")

    def _install_interface_traps(self) -> None:
        """D1c: attach the requested interface traps (physics_traps.install_interface_traps)."""
        if not self.interface_traps:
            return
        by_pair = {frozenset((r0, r1)): "{}__{}".format(r0, r1) for r0, r1 in self._iface_pairs}
        names = set(ds.get_interface_list(device=self.device))
        for key, spec in self.interface_traps.items():
            iface = key if isinstance(key, str) else by_pair.get(frozenset(key))
            if iface not in names:
                raise ValueError("interface_traps key {!r} matches no interface; built interfaces are "
                                 "{}".format(key, sorted(names)))
            PT.install_interface_traps(self.device, iface, spec)
    # ---- heterostacks (v0.14.0, D1a; physics_heterojunction) ----
    def _reference_semiconductor(self, semis: List[str]) -> str:
        """The frame reference: the region of the first GROUND electrode that sits on a semiconductor
        (its contact frame is then unchanged), else the first semiconductor region in plan order."""
        for E in self.design.electrodes:
            if E.role == "ground" and self._contact_region.get(E.name) in semis:
                return self._contact_region[E.name]
        return semis[0]

    def _own_phi_c(self, region: str, tr) -> float:
        """The conduction edge (E_c = q(Phi_c - psi)) the region's physics implies in its own legacy
        frame -- physics_heterojunction module docstring, item 1."""
        if region in self._bipolar_regions:                 # psi = intrinsic level: E_c - E_i = kT ln(N_c/n_i)
            Nc = float(ds.get_parameter(device=self.device, region=region, name="N_dos"))
            ni = float(ds.get_parameter(device=self.device, region=region, name="n_i"))
            return HJ.region_V_t(self.device, region) * float(np.log(Nc / ni))   # kT at the region's T_K
        if region in self._dd_regions:                      # contact pins n = N_D at psi = 0 (FD-SG frame)
            Nc = float(ds.get_parameter(device=self.device, region=region, name="N_c"))
            u = float(tr.n_bg_m3) / Nc
            return -PE.V_T * (float(np.log(u)) + float(BP.fd_potential_shift(u)))
        return float(ds.get_parameter(device=self.device, region=region, name="Phi_c0"))

    def _semiconductor_frames(self) -> Dict[str, Tuple[float, float]]:
        """{region: (Phi_c_V, s_V)} for every semiconductor region (Anderson rule, one frame), and the
        frame shift applied to each region: equilibrium regions get Phi_c0 := Phi_c(r) (their n(psi)
        closure then sits in the common frame); equilibrium + unipolar-DD regions are seeded at
        Potential = s(r) (bipolar: in _solve_bipolar, on top of the built-in seed). s == 0 everywhere ->
        nothing is touched (a single semiconductor, or a same-material same-doping stack)."""
        semis = [sp.name for sp in self._specs if sp.role == "semiconductor"]
        if not semis:
            return {}
        # ONE temperature for the frame (v0.14 D1a follow-up): Fermi-level alignment across regions is an
        # isothermal-equilibrium statement, so every semiconductor region must run at the same V_t (the
        # region parameter the bulk installed from BipolarOptions.T_K; equilibrium / unipolar-DD regions
        # are at T_REF). A mixed-temperature stack is refused, never aligned at 300 K silently.
        vts = {r: HJ.region_V_t(self.device, r) for r in semis}
        if len(semis) > 1 and len(set(vts.values())) > 1:
            raise NotImplementedError(
                "LayeredDevsimBuilder: the semiconductor regions run at different temperatures ({}). Missing: "
                "a NON-ISOTHERMAL device model (heat flow, thermoelectric interface terms); Fermi-level "
                "alignment across regions is defined only at one temperature. Give every semiconductor the "
                "same T (BipolarOptions.T_K; equilibrium and unipolar drift-diffusion regions run at "
                "T_REF = {} K and have no T_K).".format(
                    ", ".join("{} {:.6g} K".format(r, HJ.temperature_of_V_t(v)) for r, v in sorted(vts.items())),
                    PE.T_REF))
        own, chi, mats = {}, {}, {}
        for sp in self._specs:
            if sp.role != "semiconductor":
                continue
            tr = self.design.materials.get(sp.material).transport
            own[sp.name] = self._own_phi_c(sp.name, tr)
            # v0.15: chi at the region's T (TransportModel.band_temperature); no model -> chi_eV unchanged
            chi_T = band_data_at_T(tr, transport_temperature_K(tr)).chi_eV
            chi[sp.name] = float(chi_T) if chi_T is not None else None
            mats[sp.name] = sp.material
        # v0.14 (user decision 2026-11-04): TransportModel.chi_eV defaults to None. Regions of ONE material
        # align with zero offset whatever chi is (bit-identical to the old 4.5 - 4.5 = 0.0); two different
        # materials need both affinities, else the build REFUSES here -- the old "offset zero by default,
        # not by data" AdvisoryWarning (defaults detected by value, 4.5 / 3.6) is gone with the defaults.
        frames = HJ.anderson_alignment(
            own, chi, self._reference_semiconductor(semis), materials=mats,
            hint="LayeredDevsimBuilder: the device has semiconductors of different materials ({}); set "
                 "TransportModel.chi_eV (electron affinity, eV) on every one of them -- it defaults to None "
                 "since v0.14 so a band offset is never zero by default.".format(
                     ", ".join("{}: {} chi_eV={}".format(r, mats[r], chi[r]) for r in sorted(mats))))
        # v0.14 row 3.2-10: the affinities are used unchanged at the device T -- check that they were
        # quoted there (after the chi_eV-is-None refusal above). Only a HETEROstack is checked: materials
        # whose (chi_eV, band_gap_eV) all agree are one semiconductor at different dopings (doping lives
        # in TransportModel, so a p/n homojunction is two material names), with no offset to drift.
        # v0.15: a material with a band-temperature model has chi evaluated at T -> consistent (the tags).
        trs = {m: self.design.materials.get(m).transport for m in set(mats.values())}
        if len({(t.chi_eV, t.band_gap_eV) for t in trs.values()}) > 1:
            T_dev = HJ.temperature_of_V_t(vts[semis[0]])
            HJ.check_affinity_temperature(HJ.affinity_temperature_tags(trs, T_dev), T_dev,
                                          mode=self.on_chi_temperature, owner="LayeredDevsimBuilder")
        for r, (phi_c, s_v) in frames.items():
            if s_v == 0.0:
                continue
            if r not in self._dd_regions:                   # equilibrium closure n = N_c F((psi-Phi_c0)/V_t)
                ds.set_parameter(device=self.device, region=r, name="Phi_c0", value=float(phi_c))
            if r not in self._bipolar_regions:
                nn = len(ds.get_node_model_values(device=self.device, region=r, name="x"))
                ds.set_node_values(device=self.device, region=r, name="Potential", values=[float(s_v)] * nn)
        return frames

    def _hetero_side(self, region: str) -> "HJ.HeteroSide":
        sp = next(x for x in self._specs if x.name == region)
        tr = self.design.materials.get(sp.material).transport
        phi_c = self._frames[region][0]
        m_n = float(tr.dos_mass_kg_of_n_m3(tr.n_bg_m3))
        a_n = float(tr.richardson_n_Am2K2) if tr.richardson_n_Am2K2 else HJ.richardson_constant_Am2K2(m_n)
        if region not in self._bipolar_regions:             # unipolar DD: electrons only, FD g always on
            return HJ.HeteroSide(region, phi_c, "N_c", a_n, fd=True)
        Nc = float(ds.get_parameter(device=self.device, region=region, name="N_dos"))
        Nv = float(ds.get_parameter(device=self.device, region=region, name="N_dos_p"))
        ni = float(ds.get_parameter(device=self.device, region=region, name="n_i"))
        fd = float(ds.get_parameter(device=self.device, region=region, name="fd_enhancement")) > 0.0
        m_p = float(tr.dos_mass_p_kg) if tr.dos_mass_p_kg else m_n
        a_p = float(tr.richardson_p_Am2K2) if tr.richardson_p_Am2K2 else HJ.richardson_constant_Am2K2(m_p)
        eg_eff = HJ.region_V_t(self.device, region) * float(np.log(Nc * Nv / ni ** 2))  # E_g/q at the region's T
        return HJ.HeteroSide(region, phi_c, "N_dos", a_n, fd=fd, phi_v_V=phi_c - eg_eff,
                             p_dos_param="N_dos_p", richardson_p=a_p)

    def _wire_heterointerfaces(self) -> None:
        """Carrier coupling on every z-stacked interface whose two regions BOTH solve carrier continuity.
        Refuses (naming what is missing) a unipolar-DD / bipolar pair and a bipolar heterojunction whose
        band_gap_eV disagrees with its own statistics."""
        semi_cc = self._dd_regions | self._bipolar_regions
        mat = {sp.name: sp.material for sp in self._specs}
        for r0, r1 in self._iface_pairs:
            if r0 not in self._frames or r1 not in self._frames:
                continue                                    # not a semiconductor pair
            t0 = self.design.materials.get(mat[r0]).transport
            t1 = self.design.materials.get(mat[r1]).transport
            if not (r0 in semi_cc and r1 in semi_cc):
                continue                                    # an equilibrium side: Potential continuity only
            if (r0 in self._bipolar_regions) != (r1 in self._bipolar_regions):
                raise NotImplementedError(
                    "LayeredDevsimBuilder: interface '{}'/'{}' joins a unipolar drift-diffusion region and a "
                    "bipolar one. Missing: a hole boundary condition on the unipolar side (it has no Holes "
                    "variable), so the bipolar side's holes would see a silent zero-flux wall. Make both "
                    "regions physics='bipolar_dd' (or both 'drift_diffusion').".format(r0, r1))
            sd0, sd1 = self._hetero_side(r0), self._hetero_side(r1)
            if sd0.holes and mat[r0] != mat[r1]:
                for sd, t, nm in ((sd0, t0, mat[r0]), (sd1, t1, mat[r1])):
                    # the gap AT the region's T: band_gap_eV as given, or the band-temperature model's E_g(T)
                    eg_T = band_data_at_T(t, transport_temperature_K(t)).E_g_eV
                    if eg_T is None:                     # v0.14: no default gap (user decision 2026-11-04)
                        raise ValueError(
                            "LayeredDevsimBuilder: bipolar heterojunction '{}'/'{}': material '{}' has no "
                            "band_gap_eV. The valence-band offset of a bipolar heterojunction needs both "
                            "gaps; set TransportModel.band_gap_eV (eV, at the device temperature) -- it "
                            "defaults to None since v0.14 so an offset is never set by a default.".format(
                                r0, r1, nm))
                    eg = sd.phi_c_V - sd.phi_v_V
                    if abs(eg - float(eg_T)) > _HJ_EG_TOL_EV:
                        raise ValueError(
                            "LayeredDevsimBuilder: bipolar heterojunction material '{}' has band_gap_eV = "
                            "{:.4f} at T (as given, or from its band_temperature model) but its statistics give "
                            "E_g = kT ln(N_c N_v / n_i^2) = {:.4f} eV at T = {:.6g} K (n_i_m3 must be the intrinsic "
                            "density AT that temperature; N_c, N_v follow the DOS masses at T); the "
                            "valence offset would follow the statistics, not the stated gap. Set n_i_m3 = "
                            "sqrt(N_c N_v) exp(-E_g / 2kT) (or correct band_gap_eV; with a band_temperature model, "
                            "derive_n_i=True derives n_i(T) from the n_i at chi_reference_T_K).".format(
                                nm, float(eg_T), eg,
                                HJ.temperature_of_V_t(HJ.region_V_t(self.device, sd.region))))
            model = self.heterointerface
            if model == "auto":
                d_ec = abs(sd0.phi_c_V - sd1.phi_c_V)
                d_ev = abs(sd0.phi_v_V - sd1.phi_v_V) if sd0.holes else 0.0
                model = "thermionic" if max(d_ec, d_ev) > _HJ_ZERO_OFFSET_V else "continuous"
            iface = "{}__{}".format(r0, r1)
            HJ.setup_heterointerface_carriers(self.device, iface, sd0, sd1, model=model)
            self._hetero_ifaces[iface] = model

    def _device_temperature_K(self) -> float:
        """The temperature the semiconductor regions were solved at (their V_t; one value -- the frame
        refuses a mixed-temperature stack), T_REF without semiconductors. It was always reported as
        T_REF, also for a BipolarOptions.T_K device."""
        semis = [sp.name for sp in self._specs if sp.role == "semiconductor"]
        return HJ.temperature_of_V_t(HJ.region_V_t(self.device, semis[0])) if semis else PE.T_REF

    def band_frames(self) -> Dict[str, Tuple[float, float]]:
        """{semiconductor region: (Phi_c_V, s_V)} of the built device: E_c = q (Phi_c - psi) in the common
        Anderson frame and the region's frame shift (its neutral-bulk potential)."""
        return dict(self._frames)

    def heterointerfaces(self) -> Dict[str, str]:
        """{interface: carrier model} for every semiconductor/semiconductor interface wired with carrier
        transport ('thermionic' | 'continuous')."""
        return dict(self._hetero_ifaces)

    def _setup_bipolar_region(self, s, tr, dos: float) -> None:
        """Attach 3-variable bipolar DD (Potential, Electrons, Holes + SRH) to a semiconductor region.
        Sets the signed NetDoping FIRST (an in-region junction expression if given, else uniform
        +n_bg donor / -n_bg acceptor) since the bipolar models reference it, then the FD-enhanced
        Scharfetter-Gummel physics + the charge-neutral equilibrium seed models."""
        if tr.net_doping_expr:                              # a position-dependent junction (e.g. ifelse(x<...))
            ds.node_model(device=self.device, region=s.name, name="NetDoping", equation=tr.net_doping_expr)
        else:                                               # uniform: -n_bg (p / acceptor) or +n_bg (n / donor)
            nd = (-1.0 if tr.acceptor else 1.0) * float(tr.n_bg_m3)
            ds.node_model(device=self.device, region=s.name, name="NetDoping",
                          equation="{:.10e}".format(nd))
        # band DOS at the region's temperature: BipolarOptions.T_K -- the ONE temperature source, the same
        # one that sets the region's V_t -- else T_REF (the default call, bit-identical). The builder
        # derives N_c, N_v from the DOS MASSES, so this is what "supply N_c, N_v at T_K" means here
        # (bipolar_options docstring item 4); n_i_m3 is a direct input and must be given at T_K.
        # ONE helper shared with the 3-D builder (v0.14 row 3.2-7).
        from dynameta.carriers.bipolar_options import band_dos_at_T
        Nc, Nv = band_dos_at_T(dos, tr.dos_mass_p_kg, getattr(tr, "bipolar_options", None))
        # v0.15 carrier statistics (TransportModel.carrier_statistics): None -> the legacy FD g-factor
        # (fd_enhancement=True, statistics unset: byte-identical); "boltzmann" -> g == 1; "fermi_dirac"
        # -> FD transport + FD detailed balance.
        stats = getattr(tr, "carrier_statistics", None)
        # n_i at the region's T: n_i_m3 as given, or derived by the band-temperature model (derive_n_i=True)
        n_i = band_data_at_T(tr, transport_temperature_K(tr)).n_i_m3
        BP.setup_bipolar_region(
            self.device, s.name, eps_static=float(tr.eps_static), n_dos_m3=Nc, n_i_m3=float(n_i),
            mobility_n_m2Vs=float(tr.mobility_m2Vs_of_n_m3(tr.n_bg_m3)),
            mobility_p_m2Vs=float(tr.hole_mobility_m2Vs_of_n_m3(tr.n_bg_m3)),
            tau_n_s=float(tr.tau_srh_s), tau_p_s=float(tr.tau_srh_s),
            fd_enhancement=stats != "boltzmann", statistics=stats, n_dos_p_m3=Nv,
            options=getattr(tr, "bipolar_options", None),       # audit R26/R27 (None -> SRH only)
            # D1d: None -> no BTBT (byte-identical). v0.15: a set with scale_B_with_gap gets B at the region T
            # from the material's band-temperature model (tunneling.btbt_for_material; others pass unchanged)
            btbt=_btbt_for_material(getattr(tr, "btbt", None), tr),
            srh_trap_energy_eV=float(getattr(tr, "srh_trap_energy_eV", 0.0)),
            traps=tuple(getattr(tr, "srh_traps", ())) or None)
        BP.setup_equilibrium_seed_models(self.device, s.name)

    def _dielectric_eps_static(self, material_name: str) -> float:
        mat = self.design.materials.get(material_name)
        eps_dc = mat.dc_permittivity()
        if eps_dc is None:
            # The old code fell back to the OPTICAL eps with only a printed warning --
            # WRONG for a gate dielectric (HfO2 optical ~4 vs DC ~18) and a silent-physics
            # path. RAISE instead, matching the 3D builder (audit F4 / cross-cutting F4).
            raise ValueError(
                "dielectric '{}' has no eps_static_dc; the Stage-1 Poisson solve needs the "
                "DC permittivity (the optical eps would under-predict gate accumulation). "
                "Set Material.eps_static_dc.".format(material_name))
        return float(eps_dc)

    # ---- CarrierSolver Protocol ----
    def regions(self) -> List[RegionInfo]:
        out = []
        py = float(self.design.unit_cell.period_y_m)   # y-bbox is the cell y-period, NOT a copy of x
        for s in self._specs:
            out.append(RegionInfo(name=s.name, role=s.role, material=s.material,
                                    bbox_m=(s.x_lo, s.x_hi, 0.0, py, s.z_lo, s.z_hi),
                                    ndim=2))
        return out

    def solve(self, bias, *, grid_n_x: int = 256, grid_n_z: int = 32,
                rel_tol: float = 1e-5, max_iter: int = 60,
                v_step: float = 0.25, abs_tol: Optional[float] = None,
                verbose: bool = False) -> CarrierField:
        if not self._built:
            self.build_device()
        self._check_circuit_contacts()
        if self._bipolar_regions:
            return self._solve_bipolar(bias, grid_n_x=grid_n_x, grid_n_z=grid_n_z, rel_tol=rel_tol,
                                       max_iter=max_iter, v_step=v_step, abs_tol=abs_tol, verbose=verbose)
        d = self.design
        method = d.mesh_2d.dc_method
        semi = sorted(self._dd_regions)
        # abs_tol=None reaches solve_dc, which resolves dc_solve.default_abs_tol() -- over EVERY live
        # device, as DEVSIM's joint Newton requires -- on every solve's pre-solve state (v0.15 decision
        # 59: the builder's own n_bg * 1e-12 rule is retired -- one kernel; see default_abs_tol for
        # when to pass an explicit float)

        def _targets():
            return {E.name: bias.voltages.get(
                E.name, E.fixed_voltage_V if E.role == "ground" else 0.0)
                for E in d.electrodes}

        def _ramp(from_v):
            for E in d.electrodes:
                target = _targets()[E.name]
                v_now = from_v[E.name]
                n_steps = max(1, int(abs(target - v_now) / v_step + 0.5))
                dv = (target - v_now) / n_steps
                for _ in range(n_steps):
                    v_now += dv
                    self._drive(E.name, v_now)
                    solve_dc(self.device, method=method, abs_tol=abs_tol, rel_tol=rel_tol,
                              max_iter=max_iter, semiconductor_regions=semi, verbose=verbose)

        # audit S6-4 CONTINUATION: the DEVSIM device persists across solve() calls, so a sweep
        # can ramp from the PREVIOUS converged bias instead of re-seeding flat-band and
        # re-ramping 0->V every time (2.6-6.6x fewer coupled-Newton solves on 5-13-point
        # sweeps). Falls back to the historical full flat-band ramp on ANY failure, so a
        # non-converging continuation can never poison the result.
        prev = getattr(self, "_last_converged_bias", None)
        if prev is not None:
            try:
                _ramp(dict(prev))
                self._last_converged_bias = _targets()
                return self._to_carrier_field(bias, grid_n_x, grid_n_z)
            except Exception:
                if verbose:
                    print("[devsim] continuation from previous bias failed; "
                          "falling back to the full flat-band ramp", flush=True)
        # zero-bias seed (grounds at fixed_voltage, biased at 0)
        zero = {}
        for E in d.electrodes:
            v = E.fixed_voltage_V if E.role == "ground" else 0.0
            zero[E.name] = v
            self._drive(E.name, v)
        solve_dc(self.device, method=method, abs_tol=abs_tol, rel_tol=rel_tol,
                  max_iter=max_iter, semiconductor_regions=semi, verbose=verbose)
        _ramp(zero)
        self._last_converged_bias = _targets()
        return self._to_carrier_field(bias, grid_n_x, grid_n_z)

    def _solve_bipolar(self, bias, *, grid_n_x: int, grid_n_z: int, rel_tol: float, max_iter: int,
                       v_step: float, abs_tol: Optional[float], verbose: bool) -> CarrierField:
        """Coupled-Newton solve for a bipolar (3-variable) device: (1) seed Potential/Electrons/Holes
        from the charge-neutral equilibrium node models (built-in potential psi = V_t log(n0/n_i));
        (2) coupled 3-variable Newton at 0 bias DIRECTLY from that seed; (3) ramp the biased electrodes.
        The good built-in seed makes the coupled Newton converge without the 1D diode's separate frozen
        Poisson pre-solve -- which is under-determined on the 2D mesh with the inert edge-metal strips
        (no Potential variable there) and diverges. abs_tol=None resolves dc_solve.default_abs_tol per
        solve (the continuity residual is in carrier-density units ~n_bg; v0.15 decision 59 retired the
        builder's max(n_bg * 1e-12, 1e13) rule)."""
        d = self.design

        # KNOWN LIMITATION (library audit): a GATE electrode on a non-semiconductor layer (a Potential-only
        # Dirichlet on an oxide) does NOT inject the bipolar built-in-potential offset phi_bi the way an
        # ohmic semiconductor contact does, so a GATED bipolar 2D device UNDER-ACCUMULATES (the 3D
        # gated-cap builder applies gate_bias = Vg + phi_bi). Warn LOUDLY rather than return a silently-
        # wrong profile. Bipolar p-n DIODES with ohmic semiconductor contacts are unaffected (validated).
        # audit S5-16 (wave-7, 2026-11-02): the warn used to sit INSIDE `try: ... except Exception:
        # pass` under an `import warnings as _w` alias. A warning escalated to an error (pytest
        # filterwarnings=error, or a caller's filterwarnings('error', category=...)) IS an Exception, so
        # the bare except SWALLOWED the very escalation it was asked for; and the alias hid the site from
        # the V-17 census. Only the electrode inspection is guarded now; the warning is raised outside,
        # as UnreliableResultWarning (the returned profile under-accumulates -- a UserWarning, its old
        # implicit category, so existing filters still match).
        try:
            semi_layers = {L.name for L in d.stack.layers
                           if d.material_role(L.background_material) == "semiconductor"}
            # v0.14.0: an electrode on a METAL layer whose contact lands on a bipolar semiconductor face is
            # an ohmic contact (it carries phi_bi), not a gate -- judge by the contact's region.
            gated = [E.name for E in d.electrodes if getattr(E, "role", "biased") != "ground"
                     and getattr(E, "layer", None) not in semi_layers
                     and self._contact_region.get(E.name) not in self._bipolar_regions]
        except Exception:                                     # noqa: BLE001 (diagnostic only)
            gated = []
        if gated and bias.voltages:
            warnings.warn("LayeredDevsimBuilder bipolar solve: gate electrode(s) {} sit on a non-"
                          "semiconductor layer and do NOT carry the bipolar built-in-potential offset, so a "
                          "GATED bipolar 2D device under-accumulates. Use the 3D gated-cap builder "
                          "(devsim_3d) for a quantitative gated-bipolar result.".format(gated),
                          UnreliableResultWarning, stacklevel=2)

        # (1) seed the built-in (charge-neutral) potential + carriers, in the frame the ohmic
        # contacts PIN -- which carries the Fermi-Dirac degeneracy correction (with the sign the
        # local doping type takes) whenever the region runs the FD g-factor. audit C-10: a bare
        # Boltzmann seed contradicts an FD contact by 0.178 V per degenerate contact at eta = 10, and
        # this 2-D staged solve goes STRAIGHT into the coupled 3-variable Newton from it.
        for s in sorted(self._bipolar_regions):
            seed = BP.equilibrium_seed_psi_expr(BP.fd_shift_model(self.device, s))
            s_v = self._frames.get(s, (0.0, 0.0))[1]        # v0.14.0 heterostack frame shift
            if s_v != 0.0:
                seed = "{} + ({!r})".format(seed, float(s_v))
            ds.node_model(device=self.device, region=s, name="_seed_psi", equation=seed)
            ds.set_node_values(device=self.device, region=s, name="Potential",
                               values=ds.get_node_model_values(device=self.device, region=s, name="_seed_psi"))
            ds.set_node_values(device=self.device, region=s, name="Electrons",
                               values=ds.get_node_model_values(device=self.device, region=s,
                                                               name="IntrinsicElectrons"))
            ds.set_node_values(device=self.device, region=s, name="Holes",
                               values=ds.get_node_model_values(device=self.device, region=s,
                                                               name="IntrinsicHoles"))
        PT.seed_kinetic_traps(self.device, sorted(self._bipolar_regions))   # v0.15: no-op without kinetic traps
        for E in d.electrodes:                              # bias = 0 (grounds at fixed) for equilibrium
            v = E.fixed_voltage_V if E.role == "ground" else 0.0
            self._drive(E.name, v)
        # (2) coupled 3-variable Newton at 0 bias, directly from the built-in seed.
        # solve_dc is the identical ds.solve call plus the C10-d solution-level FD-ceiling
        # check on the converged state (the layered bipolar path previously bypassed it).
        solve_dc(self.device, method="newton", abs_tol=abs_tol, rel_tol=rel_tol,
                 max_iter=max_iter)
        # (5) ramp the biased electrodes to their targets (a fine step -- the diode I-V is exponential,
        # so a coarse step overshoots and the coupled Newton diverges).
        vbip = min(v_step, 0.05)
        for E in d.electrodes:
            target = bias.voltages.get(E.name, E.fixed_voltage_V if E.role == "ground" else 0.0)
            v_now = E.fixed_voltage_V if E.role == "ground" else 0.0
            n_steps = max(1, int(abs(target - v_now) / vbip + 0.5))
            dv = (target - v_now) / n_steps
            for _ in range(n_steps):
                v_now += dv
                self._drive(E.name, v_now)
                solve_dc(self.device, method="newton", abs_tol=abs_tol, rel_tol=rel_tol,
                         max_iter=max_iter)
        if verbose:
            print("[carriers] bipolar staged solve done (bias={})".format(bias.voltages), flush=True)
        # the solved bias (read by noise_spectrum; the bipolar path itself always ramps from 0)
        self._last_converged_bias = {E.name: bias.voltages.get(
            E.name, E.fixed_voltage_V if E.role == "ground" else 0.0) for E in d.electrodes}
        return self._to_carrier_field(bias, grid_n_x, grid_n_z)

    def _drive(self, electrode: str, v: float) -> None:
        """Set electrode `electrode` to `v` volts through whatever ACTUALLY pins its potential.

        v0.14.0 blocker fix (2026-11-04, MEASURED before the fix): after set_ssac_gate the gate's
        Potential Dirichlet reads the circuit node, not the `<name>_bias` parameter, so solve()'s ramp
        -- which only wrote the parameter -- was a no-op for it and a later builder.solve(new bias)
        silently returned the OLD operating point (reference modulator, 1 V -> solve(2 V): every node
        Potential equal to the 1 V state to 1.1e-16 V, 1.0 V off a fresh 2 V solve in upper_al2o3).
        A circuit-driven electrode is now driven by circuit_alter on its source (the parameter is
        still written, so anything reading it stays consistent)."""
        ds.set_parameter(device=self.device, name="{}_bias".format(electrode), value=float(v))
        src = self._circuit_gates.get(electrode)
        if src is not None:
            ds.circuit_alter(name=src, value=float(v))

    def _check_circuit_contacts(self) -> None:
        """Refuse a solve() the builder cannot drive: an electrode repointed circuit-driven OUTSIDE
        set_ssac_gate (a raw ac_analysis.setup_circuit_contact on self.device) has a source the
        builder does not know, so its bias would silently not move (the stale-operating-point defect
        _drive fixes for set_ssac_gate)."""
        unknown = sorted(c for c in _R.circuit_driven_contacts(self.device)
                         if c in self._contact_region and c not in self._circuit_gates)
        if unknown:
            raise RuntimeError(
                "LayeredDevsimBuilder.solve: electrode(s) {} are circuit-driven but were not repointed "
                "with set_ssac_gate, so solve() does not know the source that sets their potential and "
                "would return a stale operating point. Use set_ssac_gate(name, source_name=...) or set "
                "the bias with ds.circuit_alter + a direct DC solve.".format(unknown))

    def set_ssac_gate(self, electrode_name: str, *, source_name: str = "V1",
                      node_name: str = "vac") -> Tuple[str, str]:
        """Repoint an already-built electrode from its bias-parameter Dirichlet to a CIRCUIT-DRIVEN
        gate, so a small-signal AC sweep (ac_analysis.ssac_admittance) or a large-signal transient
        (transient.transient_step) can excite it. Replaces the electrode's bias-Dirichlet
        PotentialEquation with a circuit-node contact carrying an AC voltage source `source_name`
        (circuit node `node_name` -> ground).

        Call AFTER build_device()/solve() (the contact must exist). Then set the DC operating point
        with ds.circuit_alter(name=source_name, value=Vg) + a DC solve before the ssac/transient
        call -- or with builder.solve(BiasPoint({electrode_name: Vg})), which (v0.14.0) ramps the
        circuit source for this electrode instead of its now-unread bias parameter. Returns
        (source_name, node_name).

        This is the first-class form of the post-solve delete_contact_equation + setup_circuit_contact
        reconfiguration. It also keeps the equation registry consistent -- forgets the stale
        bias-contact record and records the circuit one -- so a later Gummel/staged solve restores the
        AC drive, not the bias Dirichlet. Intended for the modulating GATE (a dielectric contact); on
        a drift-diffusion ohmic ground it would repoint only Potential and leave the Electrons pin, so
        a warning is issued in that case.

        For gate NOISE use noise_spectrum / terminal_noise, which call this once and re-settle through the
        library solver. DEVSIM circuits are process-global and cannot be deleted in-process: since v0.15 a
        second call for the same electrode, or a source / node name another circuit already uses, raises
        ValueError BEFORE the bias Dirichlet is removed, and a circuit DEVSIM refuses raises RuntimeError
        naming the fresh-process remedy."""
        if not self._built:
            raise RuntimeError(
                "set_ssac_gate('{}') requires a built device; call build_device() or solve() first "
                "(the electrode contact must exist).".format(electrode_name))
        if electrode_name not in self._contact_region:
            raise ValueError("unknown electrode '{}'; built contacts are {}.".format(
                electrode_name, sorted(self._contact_region)))
        if self._contact_region.get(electrode_name) in getattr(self, "_dd_regions", set()):
            warnings.warn(
                "set_ssac_gate('{}'): this contact is a drift-diffusion ohmic carrier contact -- "
                "repointing it circuit-driven replaces only its Potential Dirichlet and leaves the "
                "Electrons pin in place. Drive the modulating GATE (a dielectric contact), not the "
                "carrier ground.".format(electrode_name), AdvisoryWarning, stacklevel=2)
        from dynameta.carriers import ac_analysis as _AC
        # v0.15 (plan 4.2.5, fix (b)): DEVSIM circuits are process-global and cannot be deleted in-process
        # (tests/_bipolar_calllog.py). Refuse a name clash BEFORE the bias Dirichlet is removed, so the
        # device is never left without a gate equation.
        taken = sorted({node_name, "{}.I".format(source_name)} & set(ds.get_circuit_node_list()))
        if electrode_name in self._circuit_gates or taken:
            raise ValueError(
                "set_ssac_gate('{}'): {} -- DEVSIM circuits are process-global and cannot be removed "
                "in-process; pass another source_name / node_name, or reuse the existing circuit-driven "
                "gate.".format(electrode_name, "this electrode is already circuit-driven by source {!r}"
                               .format(self._circuit_gates.get(electrode_name))
                               if electrode_name in self._circuit_gates
                               else "circuit node(s) {} already exist".format(taken)))
        # drop the bias-Dirichlet contact equation (live) and its stale registry record, then attach
        # the circuit-driven replacement (setup_circuit_contact records the new one).
        ds.delete_contact_equation(device=self.device, contact=electrode_name,
                                   name="PotentialEquation")
        _R.forget(self.device, "PotentialEquation", loc=electrode_name)
        try:
            out = _AC.setup_circuit_contact(self.device, electrode_name, source_name=source_name,
                                            node_name=node_name)
        except ds.error as e:
            raise RuntimeError(
                "set_ssac_gate('{}'): DEVSIM refused the circuit ({}). DEVSIM circuits are process-global: "
                "a circuit node cannot be added once an analysis has run with a circuit present, and none "
                "can be deleted. The electrode's bias Dirichlet is already removed, so rebuild the device "
                "in a fresh process.".format(electrode_name, e)) from e
        self._circuit_gates[electrode_name] = source_name    # solve() now drives this source
        self._circuit_gate_nodes[electrode_name] = node_name
        return out

    # ---- one-call gate noise (v0.15 plan 4.2.5, user decision 35) ----
    def noise_spectrum(self, frequencies, *, gate: str, vg: Optional[float] = None,
                       source_name: str = "V1", node_name: str = "vac",
                       mechanisms: Optional[Sequence[str]] = None,
                       hooge: Union[None, str, float, Mapping] = None,
                       external_generation_shot: bool = True, regions: Optional[Sequence[str]] = None,
                       maps: bool = False, per_cell: bool = True, rel_tol: float = 1e-5,
                       abs_tol: Optional[float] = None, max_iter: int = 60,
                       v_step: float = 0.25) -> LayeredNoiseSpectrum:
        """Short-circuit current noise S_I(f) of the dielectric GATE `gate` (noise_analysis.noise_spectrum
        on '<source_name>.I') at gate bias `vg`, in ONE call. It is the hand recipe (solve, set_ssac_gate,
        circuit_alter, a DC re-settle, noise_analysis.noise_spectrum; validation/devsim_noise.py gate G)
        with its traps removed:
          * the gate is made circuit-driven ONCE (set_ssac_gate, recorded); later calls reuse it. DEVSIM
            circuits are process-global and cannot be removed in-process, so the same gate under another
            source_name, or a name another circuit already uses, raises ValueError.
          * the bias is re-settled through the LIBRARY solver (dc_solve.solve_dc with the bipolar / DD
            branch, tolerances and step solve() uses: extended precision, the density-scaled default
            abs_tol and the well-posedness check all apply), ramping the gate's circuit source from its
            last converged bias to vg in steps of v_step (bipolar: min(v_step, 0.05)). A raw ds.solve
            re-settle bypasses all three.
          * the builder's operating point is LEFT AS IT WAS: every node solution and circuit value is
            restored afterwards, so a result depends only on (state, vg) -- calls at 0, +1, 0 V give a
            third result bitwise equal to the first -- and solve() continues from where it was. (On an
            extended-precision builder the restore is to the double-rounded values.)
          * results are PER CELL (per_cell=True, the default): S, every breakdown and every map times
            UnitCell.period_y_m, the convention of CarrierField.extras["contact_currents_A"]. Exact for
            this builder's device, which is invariant along y: the Green's function is y-independent and
            the sources are spatially white, so they add incoherently along the cell depth,
            S_cell = W S_2D. It is NOT the noise of a cell whose geometry varies along y (a square patch
            modelled by its x-z section: the answer is the y-extruded device's), nor of a source
            correlated over lengths ~W. per_cell=False returns DEVSIM's per-unit-depth result (A^2/Hz/m)
            unscaled, bit for bit the hand recipe's.
        The returned LayeredNoiseSpectrum records units, depth_m, gate, vg_V and voltages.

        Which options: keep mechanisms=None -- it is noise_analysis.DEFAULT_MECHANISMS, read at call
        time, so every default-on source (the thermionic interfaces te_n / te_p since v0.15) is included
        and the total obeys the FDT; name a subset only to split S. hooge: None (default, no 1/f source);
        "material" = each carrier region's TransportModel.hooge_alpha_n / hooge_alpha_p (a region without
        one carries no Hooge source; none set anywhere = no Hooge at all); a float or a {region: alpha |
        (alpha_n, alpha_p)} mapping goes to noise_analysis.noise_spectrum's hooge_alpha unchanged. A
        dielectric gate draws no DC current, so its Hooge noise is that of the round-off equilibrium
        current. external_generation_shot, regions and maps are noise_analysis.noise_spectrum's.

        Scope (decision 35): `gate` must be a Potential-only (dielectric) contact; an OHMIC carrier contact
        raises OhmicContactNoiseError (a NotImplementedError). The builder must be built AND solved
        (RuntimeError otherwise). A bipolar builder at vg != 0 warns UnreliableResultWarning (the gated
        bipolar 2-D device lacks the built-in-potential offset; see solve())."""
        w, units = self._noise_depth(per_cell)
        hooge_alpha = self._resolve_hooge(hooge)
        mech = NA.DEFAULT_MECHANISMS if mechanisms is None else tuple(mechanisms)

        def run():
            return NA.noise_spectrum(self.device, frequencies, output_node="{}.I".format(source_name),
                                     regions=regions, mechanisms=mech, hooge_alpha=hooge_alpha,
                                     external_generation_shot=external_generation_shot, maps=maps)
        sn, v, volts = self._at_gate_bias(gate, vg, source_name, node_name, rel_tol, abs_tol, max_iter,
                                          v_step, run)
        return _scaled_noise(sn, w, units, gate, v, volts)

    def terminal_noise(self, frequencies, *, gate: str, vg: Optional[float] = None,
                       source_name: str = "V1", node_name: str = "vac",
                       mechanisms: Optional[Sequence[str]] = None,
                       hooge: Union[None, str, float, Mapping] = None,
                       external_generation_shot: bool = True, regions: Optional[Sequence[str]] = None,
                       maps: bool = False, per_cell: bool = True, rel_tol: float = 1e-5,
                       abs_tol: Optional[float] = None, max_iter: int = 60,
                       v_step: float = 0.25) -> LayeredTerminalNoise:
        """noise_spectrum (same arguments, same state handling) plus the gate's ssac admittance
        Y = G + i omega C at the SAME operating point (ac_analysis.ssac_admittance, a separate linear
        solve) and the open-circuit S_V = S_I / |Y|^2. Per cell, S_I and Y are both multiplied by
        period_y_m (so S_V scales as 1/period_y_m). The FDT check of an equilibrium gate is
        S_I = 4 kT Re Y."""
        from dynameta.carriers.ac_analysis import ssac_admittance
        w, units = self._noise_depth(per_cell)
        hooge_alpha = self._resolve_hooge(hooge)
        mech = NA.DEFAULT_MECHANISMS if mechanisms is None else tuple(mechanisms)

        def run():
            sn = NA.noise_spectrum(self.device, frequencies, output_node="{}.I".format(source_name),
                                   regions=regions, mechanisms=mech, hooge_alpha=hooge_alpha,
                                   external_generation_shot=external_generation_shot, maps=maps)
            f, C, G = ssac_admittance(sn.frequencies, source_name=source_name)
            return sn, G + 1j * 2.0 * np.pi * f * C
        (sn, Y), v, volts = self._at_gate_bias(gate, vg, source_name, node_name, rel_tol, abs_tol,
                                               max_iter, v_step, run)
        cur = _scaled_noise(sn, w, units, gate, v, volts)
        Yc = Y if w is None else Y * w
        return LayeredTerminalNoise(frequencies=cur.frequencies, S_I=cur.S, S_V=cur.S / np.abs(Yc) ** 2,
                                    Y=Yc, current=cur, units=units, depth_m=1.0 if w is None else w)

    def _noise_depth(self, per_cell: bool):
        """(scale, units) of the gate-noise results: (period_y_m, per cell) or (None, per unit depth)."""
        if per_cell:
            return float(self.design.unit_cell.period_y_m), "A^2/Hz per cell"
        return None, "A^2/Hz/m per unit depth"

    def _resolve_hooge(self, hooge):
        """noise_spectrum's `hooge` -> noise_analysis hooge_alpha. "material" maps every carrier region
        to its material's (TransportModel.hooge_alpha_n, hooge_alpha_p); regions with neither are left
        out (no Hooge source) and an all-empty result is None (no Hooge at all: bit-identical to the
        no-Hooge call). The ONE reader of those TransportModel fields."""
        if not isinstance(hooge, str):
            return hooge
        if hooge != "material":
            raise ValueError("noise_spectrum: hooge must be None, 'material', a float or a mapping; got "
                             "{!r}".format(hooge))
        if not self._built:
            raise RuntimeError("noise_spectrum: the builder must be built AND solved first (call solve(bias))")
        out = {}
        for s in self._specs:
            if s.name not in self._dd_regions and s.name not in self._bipolar_regions:
                continue
            tr = self.design.materials.get(s.material).transport
            pair = (tr.hooge_alpha_n, tr.hooge_alpha_p if s.name in self._bipolar_regions else None)
            if pair != (None, None):
                out[s.name] = pair
        return out or None

    def _check_noise_gate(self, gate: str, source_name: str) -> None:
        if not self._built or getattr(self, "_last_converged_bias", None) is None:
            raise RuntimeError("noise_spectrum: the builder must be built AND solved first (call "
                               "solve(bias)); the noise is taken about a converged DC operating point.")
        if gate not in self._contact_region:
            raise ValueError("noise_spectrum: unknown electrode {!r}; built contacts are {}.".format(
                gate, sorted(self._contact_region)))
        have = self._circuit_gates.get(gate)
        if have is not None and have != source_name:
            raise ValueError("noise_spectrum: electrode {!r} is already circuit-driven by source {!r}; "
                             "DEVSIM circuits cannot be removed in-process, so it cannot be repointed to "
                             "{!r} (pass source_name={!r}).".format(gate, have, source_name, have))
        carrier = sorted(set(ds.get_contact_equation_list(device=self.device, contact=gate))
                         - {"PotentialEquation"})
        region = self._contact_region[gate]
        if carrier or region in self._dd_regions or region in self._bipolar_regions:
            raise OhmicContactNoiseError(
                "noise_spectrum: electrode {!r} is an OHMIC carrier contact on region {!r} (contact "
                "equations {} besides Potential); v0.15 drives dielectric (Potential-only) gates only "
                "(plan 4.2.5, decision 35) -- repointing it would move its Potential and leave the carrier "
                "pins. The ohmic drive is the planned set_ssac_contact follow-on.".format(
                    gate, region, carrier))
        self._check_circuit_contacts()

    def _resettle(self, rel_tol, abs_tol, max_iter) -> None:
        """ONE DC solve through dc_solve.solve_dc with the branch solve() uses (bipolar / DD)."""
        if self._bipolar_regions:
            solve_dc(self.device, method="newton", abs_tol=abs_tol, rel_tol=rel_tol, max_iter=max_iter)
        else:
            solve_dc(self.device, method=self.design.mesh_2d.dc_method, abs_tol=abs_tol, rel_tol=rel_tol,
                     max_iter=max_iter, semiconductor_regions=sorted(self._dd_regions))

    def _solution_state(self):
        """Every node solution a DC solve can change (each region equation's variable) and the circuit
        'dcop' / 'dcop_prev' values -- what noise_spectrum restores."""
        nodes = {}
        for r in ds.get_region_list(device=self.device):
            for eq in ds.get_equation_list(device=self.device, region=r):
                var = ds.get_equation_command(device=self.device, region=r, name=eq)["variable_name"]
                nodes[(r, var)] = list(ds.get_node_model_values(device=self.device, region=r, name=var))
        circ = {}
        for sol in _DC_CIRCUIT_SOLUTIONS:
            if sol in ds.get_circuit_solution_list():
                for nm in ds.get_circuit_node_list():
                    circ[(sol, nm)] = float(ds.get_circuit_node_value(node=nm, solution=sol))
        return nodes, circ

    def _reset_gate_circuit(self, gate: str, source_name: str, keep: dict) -> None:
        """Every circuit value saved in `keep` back; the gate's own circuit unknowns (its node and source
        current) not in `keep` to 0.0, the value DEVSIM gives a new circuit node."""
        mine = {self._circuit_gate_nodes.get(gate), "{}.I".format(source_name)}
        for sol in _DC_CIRCUIT_SOLUTIONS:
            if sol not in ds.get_circuit_solution_list():
                continue
            for nm in ds.get_circuit_node_list():
                if (sol, nm) in keep:
                    ds.set_circuit_node_value(solution=sol, node=nm, value=keep[(sol, nm)])
                elif nm in mine:
                    ds.set_circuit_node_value(solution=sol, node=nm, value=0.0)

    def _at_gate_bias(self, gate, vg, source_name, node_name, rel_tol, abs_tol, max_iter, v_step, fn):
        """fn() with `gate` circuit-driven and re-settled at vg (noise_spectrum's state handling);
        returns (fn(), vg, voltages) and leaves the operating point exactly as found."""
        self._check_noise_gate(gate, source_name)
        start = dict(self._last_converged_bias)
        v0 = float(start[gate])
        v = v0 if vg is None else float(vg)
        if not np.isfinite(v):
            raise ValueError("noise_spectrum: vg must be finite, got {!r}".format(vg))
        if self._bipolar_regions and v != 0.0:
            warnings.warn("LayeredDevsimBuilder.noise_spectrum: gate {!r} at {} V on a bipolar 2-D device "
                          "-- the gated bipolar device does not carry the built-in-potential offset and "
                          "under-accumulates (see solve()); use the 3-D builder for a quantitative result."
                          .format(gate, v), UnreliableResultWarning, stacklevel=3)
        nodes, circ = self._solution_state()
        if gate not in self._circuit_gates:
            self.set_ssac_gate(gate, source_name=source_name, node_name=node_name)
        try:
            # the same starting point on every call: the saved device state, the gate's circuit at 0
            self._reset_gate_circuit(gate, source_name, {})
            step = min(v_step, 0.05) if self._bipolar_regions else v_step
            n_steps = max(1, int(abs(v - v0) / step + 0.5))
            dv = (v - v0) / n_steps
            v_now = v0
            for k in range(n_steps):
                v_now = v if k == n_steps - 1 else v_now + dv
                self._drive(gate, v_now)
                self._resettle(rel_tol, abs_tol, max_iter)
            out = fn()
        finally:
            for (r, var), vals in nodes.items():
                ds.set_node_values(device=self.device, region=r, name=var, values=vals)
            self._reset_gate_circuit(gate, source_name, circ)
            self._drive(gate, v0)
        volts = dict(start)
        volts[gate] = v
        return out, v, volts

    def _to_carrier_field(self, bias, grid_n_x, grid_n_z) -> CarrierField:
        d = self.design
        regions: Dict[str, CarrierRegion] = {}
        n_bg_by_region: Dict[str, float] = {}
        for s in self._specs:
            if s.role != "semiconductor":
                continue
            x = np.array(ds.get_node_model_values(device=self.device, region=s.name, name="x"))
            y = np.array(ds.get_node_model_values(device=self.device, region=s.name, name="y"))
            n = np.array(ds.get_node_model_values(device=self.device, region=s.name, name="Electrons"))
            pot = np.array(ds.get_node_model_values(device=self.device, region=s.name, name="Potential"))
            nodes = np.column_stack([x, y])
            grid = resample_to_grid(nodes, {ELECTRON_DENSITY: n, POTENTIAL: pot},
                                      (grid_n_x, grid_n_z))
            regions[s.name] = CarrierRegion(
                name=s.name, role=s.role, material=s.material, nodes_m=nodes,
                node_fields={ELECTRON_DENSITY: n, POTENTIAL: pot},
                grid_axes_m={"x": grid["axis_0"], "y": grid["axis_1"]},
                grid_fields={ELECTRON_DENSITY: grid[ELECTRON_DENSITY],
                              POTENTIAL: grid[POTENTIAL]})
            n_bg_by_region[s.name] = float(d.materials.get(s.material).transport.n_bg_m3)
        # terminal currents (driver D1): per-unit-depth [A/m] scaled by period_y to [A]; an
        # equilibrium device yields {} -> NO extras key (the byte-identical off-switch). A
        # best-effort diagnostic -- a failure warns, never breaks the solve.
        extras: Dict[str, object] = {}
        try:
            from dynameta.carriers.contact_current import extract_contact_currents
            cc = extract_contact_currents(self.device, depth_m=d.unit_cell.period_y_m)
            if cc:
                extras["contact_currents_A"] = cc
        except Exception as _e:                               # noqa: BLE001 (diagnostic)
            # audit S5-16 (wave-7, 2026-11-02): this was `import warnings as _w; _w.warn(...)` -- no
            # category, no stacklevel, and the alias hid it from the V-17 census (which matches the
            # literal `warnings.warn`). The CarrierField is complete; only the optional extras key is
            # missing -> AdvisoryWarning (a UserWarning, the old implicit category).
            warnings.warn("contact-current extraction unavailable: {}".format(_e),
                          AdvisoryWarning, stacklevel=2)
        if self.oxide_tunneling is not None:                  # D1d: opt-in; None adds no key
            extras["oxide_tunneling"] = self._oxide_tunneling_extras()
        if self.extended_precision:                           # v0.15 provenance; off adds no key
            extras["extended_precision"] = True
        return CarrierField(
            bias_label=bias.label, voltages=dict(bias.voltages), ndim=2,
            temperature_K=self._device_temperature_K(), regions=regions, n_bg_by_region=n_bg_by_region,
            unit_cell_m=(d.unit_cell.period_x_m, d.unit_cell.period_y_m), extras=extras)

    def _oxide_tunneling_extras(self) -> Dict[str, Dict[str, float]]:
        """D1d: post-processed gate-oxide tunneling through every dielectric region that shares an
        interface with a semiconductor region (carriers.tunneling.oxide_tunneling_current), as scalar
        summaries per dielectric region: I_A (scaled by period_y to amperes), J_max_A_m2 (signed, at
        the node of largest |J|), E_ox_max_V_m, V_ox_max_V, t_ox_m."""
        from dynameta.carriers.tunneling import oxide_tunneling_current, oxide_tunneling_terms
        roles = {s.name: s.role for s in self._specs}
        out: Dict[str, Dict[str, float]] = {}
        if getattr(self.oxide_tunneling, "coupling", "post") == "self_consistent":
            for t in oxide_tunneling_terms(self.device):           # the term's own kernel call
                res = oxide_tunneling_current(self.device, t.chain[0], t.semi, self.oxide_tunneling,
                                              depth_m=self.design.unit_cell.period_y_m)
                k = int(np.argmax(np.abs(res["J_A_m2"])))
                out[t.chain[0]] = {"I_A": float(res["I"]), "J_max_A_m2": float(res["J_A_m2"][k]),
                                   "E_ox_max_V_m": float(np.max(res["E_ox_V_m"])),
                                   "V_ox_max_V": float(np.max(res["V_ox_V"])), "t_ox_m": float(res["t_ox_m"]),
                                   "semiconductor_region": t.semi, "model": res["model"],
                                   "coupling": "self_consistent", "gate": t.gate, "I_gate_A": float(res["I_gate"])}
            return out
        for r0, r1 in self._iface_pairs:
            for ox, semi in ((r0, r1), (r1, r0)):
                if roles.get(ox) == "dielectric" and roles.get(semi) == "semiconductor":
                    res = oxide_tunneling_current(self.device, ox, semi, self.oxide_tunneling,
                                                  depth_m=self.design.unit_cell.period_y_m)
                    k = int(np.argmax(np.abs(res["J_A_m2"])))
                    out[ox] = {"I_A": float(res["I"]), "J_max_A_m2": float(res["J_A_m2"][k]),
                               "E_ox_max_V_m": float(np.max(res["E_ox_V_m"])),
                               "V_ox_max_V": float(np.max(res["V_ox_V"])), "t_ox_m": float(res["t_ox_m"]),
                               "semiconductor_region": semi, "model": res["model"]}
        return out

    def teardown(self) -> None:
        _R.clear(self.device)
        for dv in list(ds.get_device_list()):
            ds.delete_device(device=dv)
        for m in list(ds.get_mesh_list()):
            ds.delete_mesh(mesh=m)
        self._circuit_gates = {}
        self._circuit_gate_nodes = {}
        self._built = False


def _scaled_noise(sn, w, units, gate, vg, voltages) -> LayeredNoiseSpectrum:
    """`sn` as a LayeredNoiseSpectrum with every spectral quantity times w (None: unscaled -- the same
    arrays). Node coordinates in maps are not scaled."""
    def sc(a):
        return a if w is None else a * w
    maps = None
    if sn.maps is not None:
        maps = {r: {k: (v if k in ("x", "y", "z") else sc(v)) for k, v in mm.items()}
                for r, mm in sn.maps.items()}
    return LayeredNoiseSpectrum(
        frequencies=sn.frequencies, S=sc(sn.S), output_node=sn.output_node, kind=sn.kind,
        by_mechanism={m: sc(v) for m, v in sn.by_mechanism.items()},
        by_region={r: sc(v) for r, v in sn.by_region.items()},
        by_region_mechanism={r: {m: sc(v) for m, v in d.items()} for r, d in sn.by_region_mechanism.items()},
        temperature_K=dict(sn.temperature_K), maps=maps,
        by_interface={i: {m: sc(v) for m, v in d.items()} for i, d in sn.by_interface.items()},
        units=units, depth_m=1.0 if w is None else w, gate=gate, vg_V=float(vg), voltages=dict(voltages))
