"""
PYTHON-ASSEMBLED residual terms through DEVSIM's custom-equation hook (DynaMeta v0.15, plan item 4.0.3).
Opt-in: importing this module does NOTHING to DEVSIM. The one dispatcher is registered on the first
add_custom_term() (or an explicit register_dispatcher()), never at import.

=====================================================================================================
1. WHEN TO USE IT (and when not)
   DEVSIM's node / edge / element models express every term whose value at a node depends on that node,
   its edges or its elements. Those terms go through physics_bipolar_dd.add_recombination_term (the
   ONE additive term mechanism, which the noise analysis and the Gummel bookkeeping read back). Use a
   custom term ONLY for a NONLOCAL term that the model language cannot write:
     * oxide tunnelling: the current at a semiconductor interface node depends on the potential at the
       FAR (gate) face of the oxide, several elements away, and feeds the gate's circuit (KCL) row;
     * nonlocal BTBT: the rate at a path start node depends on the band profile along a path through
       many elements, and the carriers land at different nodes.
   A term that declares itself node-, edge- or element-local is REFUSED (section 3). A thermionic
   interface flux is expressible as a DEVSIM interface model and therefore must NOT come through here.

2. THE MECHANISM (DEVSIM 2.11.2 manual section 5.5 "Custom matrix assembly"; MEASURED here)
   ds.custom_equation(name, procedure) registers procedure(what, timemode) with what in {"MATRIXONLY",
   "RHS", "MATRIXANDRHS"} and timemode in {"DC", "TIME"}. It returns (rcv, rv, permute): rcv a flat list
   of (row, column, value) triplets of the Jacobian, rv a flat list of (row, value) pairs of the
   residual, permute=True to send the entries through DEVSIM's row permutation (bulk rows of interface
   copies are merged into the surviving copy; the bulk row of a CONTACT node is dropped). MEASURED
   (devsim 2.11.2, 2026-11-06): a DC Newton step calls it once with ("MATRIXANDRHS", "DC"); ssac and
   noise call ("MATRIXONLY", "DC") then ("MATRIXONLY", "TIME") -- so a term's static Jacobian IS the
   small-signal matrix (P1 probe: ssac Re Y matched the DC dI/dV to 1.8e-8). An exception in the
   procedure aborts the solve with a devsim error naming the custom equation (DEVSIM stays usable).
   THERE IS NO delete_custom_equation: a registration is process-global and permanent. Hence:
     * ONE dispatcher, DISPATCHER_NAME, registered once per process; it keeps a registry keyed by
       device name and, on every assembly call, concatenates the triplets of every LIVE term.
     * LIVENESS is read back from DEVSIM, never trusted from Python: a term is live only while its
       device is in ds.get_device_list() AND the device carries the marker parameter
       "_ct_<term name>" with this term's token. Device parameters die with ds.delete_device, so a
       device re-created under the same name never inherits a stale term (the stale entry is purged
       at the next assembly).
     * A term whose target equations are currently deleted (eq_registry.delete_by_name in a Gummel /
       Poisson-only pre-solve) reports rows_live() False and contributes nothing.
   The dispatcher always returns permute=True. Terms address the BULK rows of NON-contact nodes
   (node_rows refuses contact nodes, whose bulk rows DEVSIM drops) and circuit rows (circuit_row; the
   permutation leaves them alone -- the P1 probe's gate-KCL entry was conserved to 2e-14 this way).

3. THE TERM PROTOCOL (subclass CustomTerm)
     name              identifier, unique per device ([A-Za-z0-9_]+); the marker is "_ct_<name>".
     locality          MUST be "nonlocal" (the class attribute). "node" / "edge" / "element" are
                       refused with a pointer to add_recombination_term; a missing declaration too.
     jacobian_exact    True when assemble's MATRIX entries are the exact derivative of its RHS
                       entries. noise_analysis refuses a device carrying a live term with False,
                       because the small-signal and noise systems ARE the Jacobian.
     pattern_changes   True when the set of (row, column) entries can change between Newton
                       iterations (nonlocal BTBT paths move with the solution). See section 4.
     equations(device) -> [(region, equation), ...] the rows the term writes (region None: the
                       equation is a circuit node name). The default rows_live checks they all exist.
     assemble(device, what, timemode) -> (rcv, rv), both flat lists as in section 2. Rows and columns
                       must be looked up on EVERY call (equation numbers change whenever an equation
                       is deleted / re-added): node_rows, node_columns, circuit_row. A static term
                       returns ([], []) for timemode "TIME" (its charge part is zero). Only the
                       MATRIX entries are read for "MATRIXONLY" and only rv for "RHS".
     structural_pattern(device) -> [(row, column), ...] or None: the CANDIDATE static pattern
                       (section 4). When given, an emitted DC entry outside it raises. A subclass
                       that OVERRIDES this method counts as declaring a pattern (add-time check).
   Row / column numbers exist only once DEVSIM has numbered the equations (after the first solve or
   get_matrix_and_rhs): ds.get_equation_numbers on a never-solved device hits a DEVSIM assertion
   (MEASURED, Region.cc:1201). The dispatcher calls a term only from inside an assembly, where they
   are valid; noise_sources is called at a solved DC point.
     noise_sources(device) -> [noise_analysis.TwoPointSource, ...] at the current DC point, or None
                       for "no noise model" (noise_analysis then REFUSES the device: a term without
                       its fluctuation would silently break the fluctuation-dissipation theorem).
                       A subclass that does not override it declares "no noise model".
                       Coefficients c_r are the change of residual row r per event, in units of q, in
                       DEVSIM's sign convention for that row (noise_analysis.TwoPointSource).
   THE INVENTORY (v0.15 plan 4.0.2). physics_bipolar_dd.recombination_terms(device, region)["custom"]
   lists every live term whose equations() touch the region -- a term declaring NO region row is listed
   in every region (its footprint is unknown) -- as CustomTermRecord(term, regions, noise_model,
   jacobian_exact); noise_analysis reads its refusals from there, the same inventory as every
   model-expressible term. Declare equations() so a term is attributed to its own regions only.
   SIGN RULE FOR CONSERVATION. A term that moves charge between rows must add EQUAL and OPPOSITE
   amounts to the rows it couples (in each row's own DEVSIM convention: an electron removed from node i
   adds -q x rate to the ElectronContinuityEquation row, like ElectronGeneration = -q U; a hole removed
   adds +q x rate to the HoleContinuityEquation row; the current into a circuit node enters its KCL row
   with the sign the P1 probe pinned, +J for a current J leaving the device into that node). A wrong
   relative sign is not conserved and the gates of every term must include a terminal-current balance
   that the flipped sign breaks (tests/test_custom_terms.py does it for the toy terms).

4. FIXED SPARSITY PATTERN (explicit structural entries)
   DEVSIM's CompressedMatrix DROPS exact zeros (AddEntryImpl returns on v == 0; MEASURED: a (r, c, 0.0)
   triplet leaves nnz unchanged), so "emit a 0.0" does not reserve a slot. A term that declares
   structural_pattern instead gets, on every ("MATRIX*", "DC") call and BEFORE its own entries, the
   pair (r, c, +STRUCTURAL_EPS), (r, c, -STRUCTURAL_EPS) for each candidate entry: the first creates the
   slot, the second returns it to exactly 0.0, and any other value v at that slot is unchanged bit for
   bit for |v| >= 4e-292 or v == 0 (STRUCTURAL_EPS = 2^-1022 is below a quarter ulp of such v; MEASURED:
   nnz +1, every other entry bitwise equal). The static pattern is then the same at every iteration,
   which matters under MKL-PARDISO if DEVSIM_NEW_SYMBOLIC is overridden to "0" (DEVSIM would reuse the
   first iteration's symbolic factorisation; devsim_env sets "1" by default). A pattern_changes term
   WITHOUT a structural_pattern gets an AdvisoryWarning at add time when DEVSIM_NEW_SYMBOLIC == "0".
   (UMFPACK, and PARDISO with the default "1", re-do the symbolic step whenever DEVSIM's matrix grows.)

5. EXTENDED PRECISION (user decision 8, accepted): with a custom term the 128-bit Newton of
   extended_precision.solve CONVERGES, but LINEARLY -- the callback reads the state and returns its
   entries in double, so the term does not follow the sub-double part of each Newton update and every
   step corrects only part of the remaining error. The rate depends on the term's share of its rows'
   Jacobian: P1 probe (a weak FN sink) ~70x per iteration, 7e-30 after ~12; tests/test_custom_terms.py
   toy A (a link as strong as the segment it bridges) ~2.7x per iteration, 34 iterations to 4e-31
   against 4 without the term (MEASURED). The converged state is then exact to the double rounding of
   the term (contact charges cancel exactly, MEASURED). Budget maximum_iterations >= 60 for an extended
   device with custom terms; the double path is unchanged (quadratic).
   WHERE THE TERM DOMINATES ITS ROWS IT STALLS (MEASURED, v0.15 oxide tunnelling, 2026-11-06): the
   self-consistent tunnelling sink of a deep-depleted p-Si MOS (tests/_oxide_tunneling_child.py, O4
   device at 3 V) IS the interface node's electron balance, and the 128-bit Newton then stalls at the
   double floor -- relative error ~7e-16 for 120 iterations, never reaching the 1e-28 criterion -- while
   the same term in accumulation (a weak sink beside ~1e13 A/m^2 SG flux terms) converged in 3 iterations
   to 2e-32 and gave terminal currents exact to 0 ulp. The extended path cannot buy accuracy beyond the
   double value of a term that sets its own row; check the term's share before relying on it (and expect
   a non-converged extended solve there).

PUBLIC API: CustomTerm, add_custom_term, remove_custom_term, live_terms, register_dispatcher,
is_registered, node_rows, node_columns, circuit_row, DISPATCHER_NAME, STRUCTURAL_EPS, MARKER_PREFIX.
Gates: tests/test_custom_terms.py (every registering case in a CHILD process). Audit note:
docs/audit/2026-11-06-v0.15-two-point-source-custom-terms.md.
"""

from __future__ import annotations

import itertools
import math
import os
import re
import warnings
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from dynameta.carriers.eq_registry import ds          # lazy devsim shim: nothing loads at import
from dynameta.warning_types import AdvisoryWarning

__all__ = ["CustomTerm", "add_custom_term", "remove_custom_term", "live_terms", "register_dispatcher",
           "is_registered", "node_rows", "node_columns", "circuit_row", "DISPATCHER_NAME",
           "STRUCTURAL_EPS", "MARKER_PREFIX"]

DISPATCHER_NAME = "dynameta_custom_terms"
MARKER_PREFIX = "_ct_"
STRUCTURAL_EPS = 2.2250738585072014e-308          # 2^-1022, the smallest normal double (section 4)
LOCALITIES_REFUSED = ("node", "edge", "element")

_MATRIX = ("MATRIXONLY", "MATRIXANDRHS")
_RHS = ("RHS", "MATRIXANDRHS")
_NAME_RE = re.compile(r"^[A-Za-z0-9_]+$")

_registered = False                               # process-global: True once the dispatcher exists
_terms: Dict[str, Dict[str, "CustomTerm"]] = {}   # device -> {term name: term}
_tokens: Dict[Tuple[str, str], float] = {}        # (device, term name) -> marker value
_token_counter = itertools.count(1)
_dispatch_calls = 0                               # assembly calls seen (tests prove the hook ran)


class CustomTerm:
    """Base class of a Python-assembled NONLOCAL residual term (module doc, section 3). Subclasses set
    `locality = "nonlocal"` and implement assemble(); override equations / structural_pattern /
    noise_sources as needed."""

    locality: Optional[str] = None
    jacobian_exact: bool = True
    pattern_changes: bool = False

    def __init__(self, name: str):
        self.name = name

    def equations(self, device: str) -> Sequence[Tuple[Optional[str], str]]:
        """[(region, equation)] rows this term writes; region None = a circuit node name."""
        return ()

    def rows_live(self, device: str) -> bool:
        """False while any target equation / circuit node is absent (e.g. a frozen-carrier pre-solve
        deleted the continuity equations): the term then contributes nothing."""
        circuit = None
        for region, eq in self.equations(device):
            if region is None:
                if circuit is None:
                    circuit = set(ds.get_circuit_node_list())
                if eq not in circuit:
                    return False
            elif region not in ds.get_region_list(device=device) or \
                    eq not in ds.get_equation_list(device=device, region=region):
                return False
        return True

    def structural_pattern(self, device: str) -> Optional[Sequence[Tuple[int, int]]]:
        return None

    def assemble(self, device: str, what: str, timemode: str):
        raise NotImplementedError("CustomTerm.assemble must be implemented by the subclass")

    def noise_sources(self, device: str):
        """[noise_analysis.TwoPointSource] at the current DC point; None = no noise model (refused)."""
        return None


# =====================================================================================================
# row / column helpers (look them up on EVERY assembly call)
# =====================================================================================================

def _contact_node_set(device: str, region: str) -> set:
    from dynameta.carriers.noise_analysis import _contact_nodes
    out = set()
    for c in ds.get_contact_list(device=device):
        out.update(int(i) for i in _contact_nodes(device, region, c))
    return out


def node_rows(device: str, region: str, equation: str, nodes) -> np.ndarray:
    """Global row numbers of `equation` at `nodes` of `region` (bulk rows, permuted by DEVSIM). Refuses
    a CONTACT node: DEVSIM drops a contact node's bulk row, so an entry there would vanish silently
    (charge would not be conserved)."""
    idx = np.atleast_1d(np.asarray(nodes, dtype=np.int64))
    bad = sorted(set(idx.tolist()) & _contact_node_set(device, region))
    if bad:
        raise ValueError("custom_terms.node_rows: node(s) {} of region {!r} are contact nodes; DEVSIM "
                         "drops their bulk {} rows, so a custom entry there is silently lost. Couple "
                         "to the contact's circuit row (circuit_row) instead.".format(bad, region,
                                                                                    equation))
    rows = np.asarray(ds.get_equation_numbers(device=device, region=region, equation=equation),
                      dtype=np.int64)
    return rows[idx]


def node_columns(device: str, region: str, variable: str, nodes) -> np.ndarray:
    """Global column numbers of node solution `variable` at `nodes` of `region`."""
    idx = np.atleast_1d(np.asarray(nodes, dtype=np.int64))
    cols = np.asarray(ds.get_equation_numbers(device=device, region=region, variable=variable),
                      dtype=np.int64)
    return cols[idx]


def circuit_row(node: str) -> int:
    """Global row (= column) number of circuit node `node`."""
    return int(ds.get_circuit_equation_number(node=node))


# =====================================================================================================
# registry + dispatcher
# =====================================================================================================

def is_registered() -> bool:
    """True once this process has registered the dispatcher (it can never be unregistered)."""
    return _registered


def register_dispatcher() -> None:
    """Register the ONE process-wide dispatcher (idempotent). add_custom_term calls it; call it
    directly only to test that registration alone changes nothing."""
    global _registered
    if not _registered:
        ds.custom_equation(name=DISPATCHER_NAME, procedure=_dispatch)
        _registered = True


def _marker(name: str) -> str:
    return MARKER_PREFIX + name


def _term_live(device: str, name: str, devices) -> bool:
    if device not in devices:
        return False
    marker = _marker(name)
    if marker not in ds.get_parameter_list(device=device):
        return False
    return float(ds.get_parameter(device=device, name=marker)) == _tokens.get((device, name))


def add_custom_term(device: str, term: CustomTerm) -> CustomTerm:
    """Attach `term` to `device` (replacing a term of the same name) and make sure the dispatcher is
    registered. Validation happens BEFORE any DEVSIM state is touched, so a refused term registers
    nothing. Returns the term."""
    if not isinstance(term, CustomTerm):
        raise TypeError("add_custom_term: expected a custom_terms.CustomTerm, got {!r}".format(term))
    if term.locality in LOCALITIES_REFUSED:
        raise ValueError(
            "add_custom_term: term {!r} declares locality={!r}; a {}-local term is expressible as a "
            "DEVSIM model and must go through physics_bipolar_dd.add_recombination_term (the one term "
            "mechanism the noise analysis and Gummel bookkeeping read back).".format(
                term.name, term.locality, term.locality))
    if term.locality != "nonlocal":
        raise ValueError("add_custom_term: term {!r} must declare locality = 'nonlocal' (got {!r})"
                         .format(getattr(term, "name", None), term.locality))
    if not (isinstance(term.name, str) and _NAME_RE.match(term.name)):
        raise ValueError("add_custom_term: term name {!r} must match [A-Za-z0-9_]+".format(term.name))
    if device not in ds.get_device_list():
        raise ValueError("add_custom_term: no device {!r}".format(device))
    # declared = the subclass overrides structural_pattern (it cannot be CALLED here: a never-solved
    # device has no equation numbers yet)
    declares_pattern = type(term).structural_pattern is not CustomTerm.structural_pattern
    if term.pattern_changes and not declares_pattern and os.environ.get("DEVSIM_NEW_SYMBOLIC") == "0":
        warnings.warn(
            "custom_terms: term {!r} changes the sparsity pattern between Newton iterations and "
            "declares no structural_pattern, while DEVSIM_NEW_SYMBOLIC=0 -- under MKL-PARDISO DEVSIM "
            "would reuse a stale symbolic factorisation. Declare structural_pattern() or unset "
            "DEVSIM_NEW_SYMBOLIC.".format(term.name), AdvisoryWarning, stacklevel=2)
    register_dispatcher()
    token = float(next(_token_counter))
    ds.set_parameter(device=device, name=_marker(term.name), value=token)
    _terms.setdefault(device, {})[term.name] = term
    _tokens[(device, term.name)] = token
    return term


def remove_custom_term(device: str, name: str) -> None:
    """Detach a term (the dispatcher itself stays registered: DEVSIM cannot remove it)."""
    _terms.get(device, {}).pop(name, None)
    _tokens.pop((device, name), None)
    if device in ds.get_device_list():
        ds.set_parameter(device=device, name=_marker(name), value=0.0)


def live_terms(device: str) -> List[CustomTerm]:
    """The terms that the dispatcher would assemble on `device` right now (sorted by name)."""
    devices = set(ds.get_device_list())
    return [t for n, t in sorted(_terms.get(device, {}).items()) if _term_live(device, n, devices)]


def _check_flat(term, rcv, rv):
    if len(rcv) % 3 or len(rv) % 2:
        raise RuntimeError("custom term {!r}: rcv must be flat (row, col, value) triplets and rv flat "
                           "(row, value) pairs; got lengths {} / {}".format(term.name, len(rcv), len(rv)))
    for v in itertools.chain(rcv[2::3], rv[1::2]):
        if not math.isfinite(v):
            raise RuntimeError("custom term {!r}: non-finite entry {!r}".format(term.name, v))


def _dispatch(what, timemode):
    """The ONE registered procedure: concatenate every live term's entries (module doc, section 2)."""
    global _dispatch_calls
    _dispatch_calls += 1
    rcv: list = []
    rv: list = []
    if not _terms:
        return rcv, rv, True
    devices = set(ds.get_device_list())
    for device in sorted(_terms):
        for name in sorted(_terms[device]):
            term = _terms[device][name]
            if not _term_live(device, name, devices):
                del _terms[device][name]                      # stale: device deleted / re-created
                _tokens.pop((device, name), None)
                continue
            if not term.rows_live(device):
                continue
            pattern = term.structural_pattern(device) if timemode == "DC" else None
            if pattern is not None and what in _MATRIX:
                for r, c in pattern:
                    rcv += [int(r), int(c), STRUCTURAL_EPS, int(r), int(c), -STRUCTURAL_EPS]
            a, b = term.assemble(device, what, timemode)
            a, b = list(a), list(b)
            _check_flat(term, a, b)
            if pattern is not None and a:
                allowed = {(int(r), int(c)) for r, c in pattern}
                out = [(int(r), int(c)) for r, c in zip(a[0::3], a[1::3])
                       if (int(r), int(c)) not in allowed]
                if out:
                    raise RuntimeError("custom term {!r}: entries {} lie outside its declared "
                                       "structural_pattern".format(name, out[:5]))
            if what in _MATRIX:
                rcv += a
            if what in _RHS:
                rv += b
        if not _terms[device]:
            del _terms[device]
    return rcv, rv, True
