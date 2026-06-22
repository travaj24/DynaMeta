"""EMT (effective-medium) fast SCREEN for sub-wavelength gratings (roadmap v0.5 A5).

When a 1-D lamellar grating's period is well below the wavelength it acts as a HOMOGENEOUS
uniaxial film, so a whole design sweep can be screened in MICROSECONDS by homogenizing each
sub-wavelength layer (Rytov 1956) and solving the resulting uniform-tensor stack with the
Berreman backend -- then validating the shortlist rigorously with the already-bridged RCWA/PMM.
It is an APPROXIMATION (exact only as period -> 0, blind to resonances) -- it SCREENS, it does
not replace the rigorous solve (the homogenized slab keeps an inherent O(period/lambda) interface
error; validation/lumenairy_emt_screen.py pins the monotone convergence onto rcwa as period
shrinks AND the large-period divergence).

Design (no new core code): a lamellar DynaMeta layer is already convertible to PMM segments
[(width_fraction, eps), ...] by layer_to_pmm_segments; those feed straight into Lumenairy's
rytov_segments_tensor (arithmetic mean along the lamellae, harmonic across) -> a (3,3) uniaxial
tensor -> an EpsField(tensor=...) override. The existing make_lumenairy_berreman_solver /
design_to_berreman_layers already consume a uniform-tensor EpsField, so the screen is just
"replace each sub-wavelength lamellar layer's eps_by_region entry with its Rytov tensor, then run
Berreman". Non-lamellar patterned layers are left untouched (they raise downstream -> use RCWA).

Orientation pin: layer_to_pmm_segments partitions along x (the grating period_x), so E_x is
ACROSS the lamellae (TM / perp / harmonic mean = rytov out[0,0]) and E_y is ALONG them (TE / par
/ arithmetic mean = rytov out[1,1]) -- matching DynaMeta's 'x'/'y' polarization rows.

The scalar 2-D mixing rules (maxwell_garnett / bruggeman) are re-exported for a coarse isotropic
screen of dilute 2-D pillar / hole arrays.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from dynameta.core.eps_field import EpsField
from dynameta.optics.lumenairy_bridge.berreman_backend import (_require_berreman,
                                                              make_lumenairy_berreman_solver)
from dynameta.optics.lumenairy_bridge.pmm_backend import layer_to_pmm_segments

__all__ = ["rytov_tensor_for_layer", "homogenize_lamellar_layers",
           "make_lumenairy_emt_screen_solver", "maxwell_garnett_eps", "bruggeman_eps"]


def rytov_tensor_for_layer(layer, design, lambda_m: float, period_x_m: float, period_y_m: float,
                           *, bg_eps=None, order: int = 0) -> np.ndarray:
    """Homogenize a LAMELLAR (1-D, y-invariant) DynaMeta layer into its Rytov (3, 3) uniaxial
    permittivity tensor diag(eps_perp, eps_par, eps_par) -- the period -> 0 effective medium.
    Reuses layer_to_pmm_segments for the x-partition (raises if the layer is not a lamellar
    grating -- the EMT screen is 1-D, like the PMM bridge). order=2 adds the binary
    (period/lambda)^2 bulk-index correction (Lalanne & Lemercier-Lalanne 1996) and is defined
    ONLY for a 2-segment (binary) grating; a multi-region cell with order=2 raises."""
    lum = _require_berreman()                            # lumenairy >= 5.14.4 carries the EMT bridge
    segs = layer_to_pmm_segments(layer, design, lambda_m, period_x_m, period_y_m, bg_eps=bg_eps)
    if order == 2:
        # BINARY = exactly two DISTINCT permittivities (a centered ridge yields 3 segments
        # groove|ridge|groove but is still a binary grating), so collapse by eps VALUE, not
        # segment count; the order=2 correction is symmetric under (swap media, fill -> 1-fill).
        uniq = {}
        for w, e in segs:
            key = next((k for k in uniq if abs(k - complex(e)) <= 1e-12 * (abs(k) + 1.0)), None)
            if key is None:
                key = complex(e)
                uniq[key] = 0.0
            uniq[key] += float(w)
        if len(uniq) != 2:
            raise ValueError("rytov_tensor_for_layer: the order=2 (period/lambda)^2 correction is "
                             "defined only for a BINARY grating (two distinct media); layer {!r} "
                             "has {} distinct media -- use order=0.".format(layer.name, len(uniq)))
        (eA, fA), (eB, _fB) = list(uniq.items())
        return np.asarray(lum.rytov_tensor(eA, eB, fA, period=float(period_x_m),
                                           wavelength=float(lambda_m), order=2), dtype=complex)
    return np.asarray(lum.rytov_segments_tensor(segs, order=0), dtype=complex)


def homogenize_lamellar_layers(design, lambda_m: float, *, order: int = 0,
                               eps_by_region=None) -> dict:
    """Return an eps_by_region dict that REPLACES each LAMELLAR inclusion layer with its
    Rytov-homogenized uniform (3, 3) tensor EpsField, leaving every other entry untouched. Feed
    the result to make_lumenairy_berreman_solver / design_to_berreman_layers: the gratings are
    then solved as homogeneous anisotropic films (the microsecond EMT screen). A layer that
    already has an eps_by_region override is left as-is (an explicit modulation wins); a layer
    whose inclusions are NOT lamellar is left untouched and will raise downstream (-> use RCWA)."""
    out = dict(eps_by_region or {})
    px = float(design.unit_cell.period_x_m)
    py = float(design.unit_cell.period_y_m)
    for L in design.stack.layers:
        if not L.inclusions or L.name in out:
            continue
        try:
            tens = rytov_tensor_for_layer(L, design, lambda_m, px, py, order=order)
        except ValueError:
            # not a lamellar grating (non-rectangle / partial-y / overlap) -- leave it for the
            # rigorous RCWA path to handle; the EMT screen is 1-D-lamellar only.
            continue
        out[L.name] = EpsField(tensor=tens)
    return out


def make_lumenairy_emt_screen_solver(*, order: int = 0, absorption: bool = False,
                                     n_slices: Optional[int] = None):
    """Build an `optical_solver` (run_pipeline seam + solve_sweep) that SCREENS a design: every
    sub-wavelength LAMELLAR layer is Rytov-homogenized into a uniform (3, 3) tensor and the whole
    stack solved by Berreman in microseconds. A SCREEN, not a replacement -- validate the
    down-selected designs with make_lumenairy_rcwa_solver / make_lumenairy_pmm_solver. Non-lamellar
    patterned layers raise (use RCWA). Same signature/semantics as the other bridge solvers."""
    berreman = make_lumenairy_berreman_solver(absorption=absorption, n_slices=n_slices)

    def _solve_at(design, eps_by_region, lambda_m):
        ebr = homogenize_lamellar_layers(design, lambda_m, order=order,
                                         eps_by_region=eps_by_region)
        return berreman(design, None, ebr, lambda_m, None, None)

    def _solve(design, geo, eps_by_region, lambda_m, n_super, n_sub):
        return _solve_at(design, eps_by_region, lambda_m)

    def _solve_sweep(design, geo, assemble_at, lams, n_super, n_sub):
        return [_solve_at(design, assemble_at(lam), lam) for lam in lams]

    _solve.solve_sweep = _solve_sweep
    return _solve


# ---- scalar 2-D mixing rules (coarse isotropic screen for dilute inclusion arrays) ----

def maxwell_garnett_eps(eps_host, eps_inclusion, fill, *, geometry: str = "cylinder") -> complex:
    """Maxwell-Garnett effective (scalar) eps of a DILUTE 2-D inclusion array -- the coarse
    isotropic screen for a pillar / hole array (geometry='cylinder' = field perpendicular to a
    vertical pillar at normal incidence). Thin re-export of lumenairy.maxwell_garnett."""
    return complex(_require_berreman().maxwell_garnett(eps_host, eps_inclusion, float(fill),
                                                       geometry=geometry))


def bruggeman_eps(eps_a, eps_b, fill_a, *, geometry: str = "cylinder") -> complex:
    """Bruggeman (symmetric, percolating) effective eps of a two-phase mixture -- the full-fill
    2-D screen with no host/inclusion asymmetry. Thin re-export of lumenairy.bruggeman."""
    return complex(_require_berreman().bruggeman(eps_a, eps_b, float(fill_a), geometry=geometry))
