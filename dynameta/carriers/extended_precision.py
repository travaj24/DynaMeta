"""
Opt-in EXTENDED-PRECISION (128-bit float) DEVSIM solves -- one switch, scoped per device.

WHAT DEVSIM PROVIDES (DEVSIM 2.11 manual, "Solver" chapter, section 9.3 "Extended precision";
source read at devsim/devsim e3a25f0 = the 2.11.2 line). Three parameters:
  extended_model    -- node/edge/element MODEL EVALUATION in 128-bit. Read when a model is CREATED
                       (create_node_model<..., extended_type>(region->UseExtendedPrecisionModels())),
                       so it fixes the TYPE of every model built after it is set; a model built
                       before keeps double. Looked up region -> device -> global (GlobalData::
                       GetDBEntryOnRegion), so a DEVICE-level setting scopes it to one device.
  extended_equation -- EQUATION ASSEMBLY (residual + Jacobian entries) in 128-bit. Read when the
                       equation is CREATED (EquationCommands.cc), same region -> device -> global
                       lookup.
  extended_solver   -- the solve TEMPLATE: solve / get_matrix_and_rhs / set_initial_condition run
                       Newton<float128> instead of Newton<double> (MathCommands.cc). GLOBAL only,
                       read at every call.
Windows / arm64 wheels use boost cpp_bin_float_quad (113-bit significand, software arithmetic);
Linux x86_64 uses GCC __float128 (manual 9.3.1). get_parameter(name="info")["extended_precision"]
says whether the build has it (True on the 2.11.2 pip wheel, measured). The default geometric
models (EdgeCouple, NodeVolume, ...) are ALWAYS built in the extended type on such a build
(ModelCreate.cc CreateDefaultModels), whatever the parameters say.

THE LINEAR SOLVE IS STILL DOUBLE. Every float128 preconditioner (MKLPardisoPreconditioner,
ExternalPreconditioner = the UMFPACK 5.1 'custom' solver this wheel loads) casts the assembled
matrix and RHS to double, factorizes and solves in double, and casts the solution back. So the
extended solver is a MIXED-PRECISION Newton: residual and Jacobian assembled in 128-bit, the
correction solved in 64-bit. Newton is self-correcting -- the next residual is again 128-bit --
so a DC / transient solve converges to a 128-bit-accurate state (measured: Newton relative updates
fall to ~1e-33 against ~1e-15 in double). A small-signal ssac / noise solve is a SINGLE linear
solve with no outer loop: its answer is a double-precision solve of a (correctly rounded)
128-bit-assembled matrix, and extended precision does not lift its round-off floor (see the
audit note: Re Y(1 Hz) of a p+n diode, -0.271 S/m^2 in double, -0.0727 extended, true +2.58e-5).

USE. ONE switch on the builders -- LayeredDevsimBuilder(..., extended_precision=True),
Stacked3DSpec(extended_precision=True) / Stacked3DSpec.from_design(..., extended_precision=True)
-- or, for a hand-built device, `enable(device)` right after ds.create_device and BEFORE any model
or equation exists. Every DynaMeta solve path (dc_solve, ac_analysis, transient, noise_analysis,
devsim_3d) calls `solve(...)` below, which turns the global extended_solver on for the duration of
the ONE ds.solve call iff a live device opted in, and restores the previous value in a finally --
so nothing leaks into a later device (gate: tests/test_extended_precision.py). With no opted-in
device `solve` is the bare ds.solve call with no parameter write: default-off is byte-identical by
construction. A RAW ds.solve on an opted-in device runs the double Newton over 128-bit models;
wrap it in `extended_solver()` (or call `solve`) to get the extended Newton.

COST (measured, box A, DEVSIM 2.11.2 / UMFPACK 5.1; docs/audit/2026-11-03-devsim-extended-
precision.md), per Newton iteration with IDENTICAL iteration counts: 1-D bipolar diodes 6.3x
(GaN 317 nodes 4.4 -> 27.8 ms; p+n 734 nodes 8.0 -> 51.5 ms); 2-D layered reference metasurface
(76,430 nodes, DD, 9 regions) 18.5-21.8x (0.27-0.36 -> 6.0-6.7 s). All of it is model evaluation
and assembly in software quad arithmetic; the factorization is the same double one.
"""

from __future__ import annotations

import warnings
from contextlib import contextmanager

import devsim as ds

from dynameta.warning_types import AdvisoryWarning

# Device-scoped parameters: model evaluation + equation assembly (read at model / equation creation).
MODEL_PARAMETERS = ("extended_model", "extended_equation")
# Global parameter: the Newton template (read at every solve / get_matrix_and_rhs call).
SOLVER_PARAMETER = "extended_solver"
# ds.solve types that are ONE linear solve (no Newton outer loop): DEVSIM factorizes them in double.
SINGLE_LINEAR_SOLVE_TYPES = ("ac", "noise")
SMALL_SIGNAL_MESSAGE = (
    "extended_precision: a {kind!r} solve on an extended-precision device buys nothing -- DEVSIM "
    "factorizes and solves the small-signal (ssac/noise) system in DOUBLE even with extended_solver "
    "(its float128 preconditioners cast to double), so this answer has the same round-off floor as a "
    "default device (p+n diode: Re Y(1 Hz) -0.073 extended vs -0.27 double, true +2.6e-5 S/m^2). For "
    "the f -> 0 conductance use the DC dI/dV (a +-0.1 mV central difference of solved DC terminal "
    "currents: G0 to 1e-6), and drive the LIGHTLY doped contact (3 decades lower floor than the p+ "
    "end). docs/audit/2026-11-03-devsim-extended-precision.md.")


def available() -> bool:
    """True when this DEVSIM build carries extended precision (info['extended_precision'])."""
    try:
        return bool(ds.get_parameter(name="info").get("extended_precision", False))
    except Exception:                                          # noqa: BLE001 (old builds: no 'info')
        return False


def _get(name: str, device: "str | None" = None, m=ds):
    """A DEVSIM parameter value, or None when it was never set (DEVSIM raises on a missing name;
    every extended_* consumer treats 'missing' exactly like False). `m` is the devsim module (or a
    caller's stand-in for it)."""
    try:
        if device is None:
            return m.get_parameter(name=name)
        return m.get_parameter(device=device, name=name)
    except m.error:
        return None


def enable(device: str) -> None:
    """Opt `device` into 128-bit model evaluation and equation assembly (DEVSIM DEVICE-level
    extended_model + extended_equation; cleared by ds.delete_device with the rest of the device's
    parameters -- DEVSIM 2.1 release note).

    Call right after ds.create_device and BEFORE any physics: DEVSIM fixes a model's precision when
    the model is created, so a model or equation that already exists would silently stay double.
    That ordering is checked here (a region already carrying an equation or the Potential solution
    raises). Raises RuntimeError on a DEVSIM build without extended precision."""
    if not available():
        raise RuntimeError(
            "extended_precision=True: this DEVSIM build has no extended precision "
            "(devsim.get_parameter(name='info')['extended_precision'] is False). The pip wheels "
            "since 2.3.x carry it; see docs/audit/2026-11-03-devsim-extended-precision.md.")
    if device not in ds.get_device_list():
        raise ValueError("extended_precision.enable: no device {!r} (call it right after "
                         "ds.create_device)".format(device))
    for r in ds.get_region_list(device=device):
        if ds.get_equation_list(device=device, region=r) or \
                "Potential" in ds.get_node_model_list(device=device, region=r):
            raise RuntimeError(
                "extended_precision.enable({!r}) called after physics was installed on region {!r}: "
                "DEVSIM fixes a model's precision when it is CREATED, so the existing models and "
                "equations would stay double. Enable it right after ds.create_device."
                .format(device, r))
    for p in MODEL_PARAMETERS:
        ds.set_parameter(device=device, name=p, value=True)


def is_enabled(device: str, m=ds) -> bool:
    """True when `device` assembles in 128-bit (its extended_equation resolves True -- device-level
    from `enable`, or a global setting the caller made by hand)."""
    return bool(_get("extended_equation", device, m))


def any_enabled() -> bool:
    """True when ANY live DEVSIM device opted in (DEVSIM solves all live devices in one Newton)."""
    return any(is_enabled(d) for d in ds.get_device_list())


@contextmanager
def extended_solver(m=ds):
    """Run the enclosed DEVSIM calls with the global extended_solver ON, and restore its previous
    value (or False when it was never set -- every consumer reads 'missing' as False) on ANY exit,
    including a convergence failure. Re-entrant."""
    prev = _get(SOLVER_PARAMETER, None, m)
    m.set_parameter(name=SOLVER_PARAMETER, value=True)
    try:
        yield
    finally:
        m.set_parameter(name=SOLVER_PARAMETER, value=bool(prev) if prev is not None else False)


def solve(ds_module=None, **kwargs):
    """ds.solve(**kwargs), through the extended (mixed-precision) Newton iff a live device opted in.

    Default path (no opted-in device): the bare ds.solve call, no parameter is read-modified-
    written beyond the device scan -- results are byte-identical to a direct ds.solve. Every
    DynaMeta solve site calls this, so DC, transient (transient_dc + transient_bdf1 -- they must
    share one Newton type: DEVSIM keeps SEPARATE TimeData<double> / TimeData<float128> histories),
    ssac and noise all follow the device's switch. An 'ac' or 'noise' solve on an opted-in device
    raises an AdvisoryWarning (SMALL_SIGNAL_MESSAGE): it is one double factorization, so extended
    precision does not lift its floor -- use the DC dI/dV for the f -> 0 conductance.

    `ds_module` (default: devsim) is the module whose `solve` is called -- a caller that resolves
    devsim through its own module attribute (transient.py, whose control-flow gates replace it with a
    stand-in) passes it. A stand-in without `get_device_list` has no live device, so it takes the
    default path."""
    m = ds if ds_module is None else ds_module
    lister = getattr(m, "get_device_list", None)
    devs = list(lister()) if lister is not None else []
    flags = [is_enabled(d, m) for d in devs]
    if not any(flags):
        return m.solve(**kwargs)
    if not all(flags):
        warnings.warn(
            "extended_precision: live devices {} are default (double) while {} opted in; DEVSIM "
            "solves every live device in ONE Newton, so the default devices are solved by the "
            "extended Newton too (their answers move off the double result). Tear the other "
            "devices down first for a byte-identical default answer."
            .format([d for d, f in zip(devs, flags) if not f], [d for d, f in zip(devs, flags) if f]),
            AdvisoryWarning, stacklevel=2)
    if kwargs.get("type") in SINGLE_LINEAR_SOLVE_TYPES:
        # once per call site per process under the default warnings filter (stacklevel -> caller)
        warnings.warn(SMALL_SIGNAL_MESSAGE.format(kind=kwargs.get("type")), AdvisoryWarning,
                      stacklevel=2)
    with extended_solver(m):
        return m.solve(**kwargs)
