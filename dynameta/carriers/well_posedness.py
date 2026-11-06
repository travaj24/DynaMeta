"""
DC WELL-POSEDNESS of a DEVSIM device (v0.15, plan 4.4.2; decisions 55-57): detect an EXACTLY
singular DC Jacobian caused by missing Dirichlet data -- a floating body -- before Newton runs, and
explain a NEAR-singular one when Newton fails.

THE NULL MODES (exact, structural):
  * LEFT (a conserved quantity no row fixes). In steady state the continuity rows of one carrier, summed
    over a region whose boundary passes no flux of it, telescope: every edge flux enters two rows with
    opposite signs. What survives is the volume term, q(G - U), and it enters the electron and hole rows
    with opposite signs. So with NO contact replacing a carrier row of the component (no node_model on
    its contact equations -- e.g. a Potential-only contact), w = 1 on every electron AND hole row gives
    w^T J == 0 for every state: the DC Jacobian is rank-deficient by one (two when U == 0: each species
    separately), whatever G, U, mesh or bias. Kinetic trap rows join the same sum with their own
    coefficient (plan 5.7 [C]). The body's charge -- and with it its potential against the contact --
    is then set by round-off and differs between BLAS builds (the R26 fixture until 2026-11-03:
    docs/audit/2026-11-03-devsim-recombination-bgn-ionization.md).
  * RIGHT (an undetermined absolute level). Potential with no Dirichlet anywhere in an electrostatically
    connected component, and no row that depends on the ABSOLUTE potential (drift-diffusion: SG fluxes see
    only differences): v = 1 on its Potential columns gives J v == 0 -- a pure-Neumann Poisson. (An
    equilibrium-physics region, n = n(psi), is NOT singular: n(psi) fixes the level.)
  * LOCAL (a node-only row with no DC coupling). An equation without an edge/element flux and with no
    contact equation -- a kinetic trap's occupancy row TrapEquation_* / ItEquation* (physics_traps,
    v0.15) -- is NEVER part of a conserved sum (no flux to telescope), but it is DC-singular node by node
    when its row vanishes: the trap row is pinned only by its own capture and emission, d(R_n - R_p)/dw =
    -[(n + n1')/tau_n0 + (p + p1')/tau_p0] (1 - f)^2 < 0 (the masked off-interface rows NodeVolume (w - 1)
    are pinned by the mask). Such a row with its rate terms missing (a hand-built or edited trap equation)
    leaves w undetermined: e_i is a left null vector. Every such row is checked individually (an all-zero
    equilibrated row, r = 0).
A NEAR-singular Jacobian (a weakly pinned minority -- the MOS inversion layer, coupled to the body only
through generation ~1e-17 of the matrix scale: plan P4 1.1) is not a setup error and is never raised
here: `diagnose_singular` explains it, and dc_solve.solve_dc appends that diagnosis to a stalled Newton
failure.

HOW THE CHECK DECIDES (error only when both agree, decision 56):
  1. STATIC: from DEVSIM's own equation / contact / interface commands (so raw ds.* calls that bypass
     eq_registry are seen too) build the carrier components (regions joined by interface equations of
     that carrier) and the Potential components; a component none of whose contact equations carries a
     node_model for that equation is a CANDIDATE. No candidate -> no assembly (the common case).
  2. NUMERIC: ONE ds.get_matrix_and_rhs, Ruiz-equilibrated (J_eq = D_r J D_c). The candidate vectors
     are the block indicators of the candidate rows (rows DEVSIM moved or replaced -- interface /
     contact rows, read from the returned row permutation -- excluded), mapped to J_eq's coordinates,
     and the best combination is the SVD of their block residuals. A combination is a null mode when
         r = |y^T J_eq|_2 / (|y|_2 sigma_max(J_eq))      (y = D_r^-1 sum_b c_b 1_b; right: J_eq x)
     is <= tol_singular = 64 eps = 1.4e-14: singular to working precision in the equilibrated 2-norm,
     i.e. within the factorization's own backward error. (A COMPONENTWISE measure was tried first and
     rejected: a column whose only entries are a tiny coupling -- dU/dn of a tau = 1e30 slab -- reads
     r = 1 although the matrix is singular to 1e-37.) Measured: the R26 fixture r = 6.6e-18 (n) /
     4.6e-18 (p); the majority-pinned fix's free-minority candidate r = 4.4e-8 (n) / 1.5e-8 (p) -- the
     SRH coupling ~ h^2 / (D tau), refuted 6 orders above the threshold; a tau = 1e30 slab: two modes
     (rank deficiency 2). In an n slab with nothing pinned the reported mode is the MAJORITY electron
     number (the exact null vector is electrons + holes; numerically the minority coupling dU/dn ~
     p/(tau n) is 1e-25 of the matrix scale, so the majority-only direction is null too). A candidate
     the matrix refutes is reported, never raised: a custom row the static parse cannot read can only
     cause a MISS, not a false raise.
  3. RAISE IllPosedDCError (a ValueError) naming the device, region(s), carrier(s), the conserved
     quantity and the fixes -- unless every block of the mode was acknowledged with `allow_floating`.

WHEN TO USE WHAT:
  * nothing: dc_solve.solve_dc runs `ensure_dc_well_posed` before every DC solve (cached per equation-set
    fingerprint, so a ramp pays the static check only: ~1 ms), the builders route through solve_dc, and
    the test suite wraps raw devsim.solve the same way (tests/conftest.py; opt out with
    @pytest.mark.floating_ok("reason")). Transient / small-signal solves are never checked (C/dt
    regularizes; a floating body is legitimate transiently).
  * `check_dc_well_posed(device, numeric="full")` when debugging: the report, plus the equilibrated
    sigma_min / sigma_max of the whole DC Jacobian.
  * `allow_floating(device, region, reason=...)` for an INTENDED floating body (an isolated well holding a
    fixed charge). DEVSIM has no global-charge constraint row (decision 57 defers one), so the DC answer
    along that mode is then set by round-off: prefer a transient solve from a known initial charge. A
    FLOATING CONTACT (an ohmic contact tied to a circuit node that carries zero current: SOI body, open-
    circuit voltage) is NOT floating in this sense -- its carrier rows are pinned relative to the node's
    unknown potential -- and needs no acknowledgement.
  * `diagnose_singular(device)` after a convergence failure (solve_dc does it for you on a stall).
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from dynameta.carriers.devsim_env import import_devsim

ds = import_devsim()      # D0: devsim via its ONE import home (BLAS pinned before the first import)

from dynameta.carriers import eq_registry as R  # noqa: E402

EPS = float(np.finfo(np.float64).eps)
TOL_SINGULAR = 64 * EPS                 # componentwise residual of an EXACT null mode (round-off)
POTENTIAL = "Potential"
# allow_floating's carrier names -> the equation names DynaMeta's physics modules use
_CARRIER_EQ = {"electrons": "ElectronContinuityEquation", "holes": "HoleContinuityEquation",
               "potential": "PotentialEquation"}
_CONSERVED = {"ElectronContinuityEquation": "electron number",
              "HoleContinuityEquation": "hole number"}
NEAR_SINGULAR = 1e-12                   # equilibrated sigma_min / sigma_max below which a stall is explained


class IllPosedDCError(ValueError):
    """The DC problem has an exactly singular Jacobian (a floating body / pure-Neumann potential):
    its answer along the null mode is set by round-off. The message names the mode and the fixes."""


@dataclass(frozen=True)
class Block:
    """One candidate block: the rows (side='left') or columns (side='right') of `equation`'s variable
    in `region`."""
    device: str
    region: str
    equation: str
    variable: str
    side: str


@dataclass
class NullMode:
    device: str
    side: str
    r: float                                       # componentwise relative residual of the mode
    weights: Dict[Block, float]                    # normalized coefficient per block (support only)

    def describe(self) -> str:
        regs = sorted({b.region for b in self.weights})
        if self.side == "row":
            return ("the DC rows of {} in region(s) {} vanish at {} node(s) (no rate term pins their "
                    "variable {} there: a node-only equation without capture / emission, e.g. a kinetic trap "
                    "row whose terms were removed)".format(sorted({b.equation for b in self.weights}), regs,
                                                            int(next(iter(self.weights.values()))),
                                                            sorted({b.variable for b in self.weights})))
        if self.side == "right":
            return ("Potential of region(s) {} has no Dirichlet anywhere and no row depends on its "
                    "absolute level (pure-Neumann Poisson): J v = 0 for v = 1 on those Potential "
                    "columns".format(regs))
        eqs = sorted({b.equation for b in self.weights})
        carriers = [e for e in eqs if e in _CONSERVED]
        if len(carriers) == 2:
            what = "net carrier charge (holes - electrons)"
        elif carriers:
            what = _CONSERVED[carriers[0]]
        else:
            what = "the sum of {}".format(eqs)
        if any(e not in _CONSERVED for e in eqs) and carriers:
            what += " (with the trapped / auxiliary rows {})".format(
                sorted(e for e in eqs if e not in _CONSERVED))
        w = ", ".join("{}/{} {:+.3g}".format(b.region, b.equation, v)
                      for b, v in sorted(self.weights.items(), key=lambda kv: (kv[0].region, kv[0].equation)))
        return ("{} of region(s) {} is conserved by every DC row and fixed by none (left null vector: "
                "weights {})".format(what, regs, w))


@dataclass
class WellPosedReport:
    devices: Tuple[str, ...]
    candidates: List[Tuple[Block, ...]] = field(default_factory=list)   # static groups
    confirmed: List[NullMode] = field(default_factory=list)            # exact null modes (raised)
    acknowledged: List[NullMode] = field(default_factory=list)         # exact, but allow_floating'd
    refuted: List[NullMode] = field(default_factory=list)              # static candidate, r > tol
    assembled: bool = False
    numeric_error: Optional[str] = None
    sigma_ratio: Optional[float] = None                                # numeric="full" only

    @property
    def well_posed(self) -> bool:
        return not self.confirmed


# =====================================================================================================
# static topology
# =====================================================================================================

class _UF:
    def __init__(self):
        self.p = {}

    def find(self, a):
        self.p.setdefault(a, a)
        while self.p[a] != a:
            self.p[a] = self.p[self.p[a]]
            a = self.p[a]
        return a

    def union(self, a, b):
        self.p[self.find(a)] = self.find(b)


def _static(device: str):
    """(structure key, left candidate groups, right candidate groups) from DEVSIM's commands."""
    eqs = {}                                   # (region, eq) -> (variable, has_flux)
    regions = tuple(ds.get_region_list(device=device))
    for r in regions:
        for e in ds.get_equation_list(device=device, region=r):
            c = ds.get_equation_command(device=device, region=r, name=e)
            eqs[(r, e)] = (c["variable_name"], bool(c.get("edge_model") or c.get("element_model")))
    contacts = []
    for c in ds.get_contact_list(device=device):
        rc = ds.get_region_list(device=device, contact=c)[0]
        rows = []
        for e in ds.get_contact_equation_list(device=device, contact=c):
            cc = ds.get_contact_equation_command(device=device, contact=c, name=e)
            rows.append((e, bool(cc.get("node_model")), cc.get("circuit_node", "")))
        contacts.append((c, rc, tuple(sorted(rows))))
    interfaces = []
    for i in ds.get_interface_list(device=device):
        r0, r1 = ds.get_region_list(device=device, interface=i)
        rows = []
        for e in ds.get_interface_equation_list(device=device, interface=i):
            ic = ds.get_interface_equation_command(device=device, interface=i, name=e)
            rows.append((ic.get("name0") or e, ic.get("name1") or e, ic.get("type", "")))
        interfaces.append((i, r0, r1, tuple(sorted(rows))))
    key = (regions, tuple(sorted(eqs.items())), tuple(contacts), tuple(interfaces))

    uf = _UF()
    for k in eqs:
        uf.find(k)
    for _, r0, r1, rows in interfaces:
        for n0, n1, _ in rows:
            a, b = (r0, n0), (r1, n1)
            if a in eqs and b in eqs and (eqs[a][0] == POTENTIAL) == (eqs[b][0] == POTENTIAL):
                uf.union(a, b)
    pinned = {uf.find((rc, e)) for _, rc, rows in contacts for e, nm, _ in rows
              if nm and (rc, e) in eqs}
    free = {k for k in eqs if uf.find(k) not in pinned}
    left, right = [], []
    flux_free = {k for k in free if eqs[k][0] != POTENTIAL and eqs[k][1]}
    if flux_free:
        host = {r for r, _ in flux_free}
        blocks = sorted(k for k in free if eqs[k][0] != POTENTIAL and k[0] in host)
        left.append(tuple(Block(device, r, e, eqs[(r, e)][0], "left") for r, e in blocks))
    comps: Dict[tuple, list] = {}
    for k in sorted(free):
        if eqs[k][0] == POTENTIAL:
            comps.setdefault(uf.find(k), []).append(k)
    for ks in comps.values():
        right.append(tuple(Block(device, r, e, POTENTIAL, "right") for r, e in ks))
    # LOCAL candidates: every free node-only equation (no flux) -- a kinetic trap row -- row by row
    for r, e in sorted(k for k in free if eqs[k][0] != POTENTIAL and not eqs[k][1]):
        right.append((Block(device, r, e, eqs[(r, e)][0], "row"),))
    return key, left, right


# =====================================================================================================
# numeric confirmation
# =====================================================================================================

def _assemble():
    """(J csc, permutation row, permutation copy) of DEVSIM's DC system (all live devices)."""
    import scipy.sparse as sp
    m = ds.get_matrix_and_rhs(format="csc")
    s = m["static"]
    n = len(s["ap"]) - 1
    J = sp.csc_matrix((np.asarray(s["av"], dtype=np.float64), np.asarray(s["ai"]),
                       np.asarray(s["ap"])), shape=(n, n))
    perm = m.get("permutation") or {}
    row = np.asarray(perm.get("row", np.arange(n)), dtype=np.int64)
    cp = np.asarray(perm.get("copy", np.zeros(n)), dtype=np.int64)
    return J, row, cp


def _ruiz(J, iters: int = 20):
    """Ruiz (inf-norm) equilibration scalings: J_eq = diag(dr) J diag(dc) has every row's and column's
    max |entry| -> 1. Returns (dr, dc, J_eq csc, sigma_max(J_eq) by 30 power iterations)."""
    import scipy.sparse as sp
    A = J.tocsr().astype(np.float64)
    dr, dc = np.ones(A.shape[0]), np.ones(A.shape[1])
    for _ in range(iters):
        r = np.sqrt(abs(A).max(axis=1).toarray().ravel())
        c = np.sqrt(abs(A).max(axis=0).toarray().ravel())
        r[r == 0] = 1.0
        c[c == 0] = 1.0
        A = sp.diags(1.0 / r) @ A @ sp.diags(1.0 / c)
        dr, dc = dr / r, dc / c
    A = A.tocsc()
    x = np.random.default_rng(1).standard_normal(A.shape[1])
    smax = 1.0
    for _ in range(30):
        y = A.T @ (A @ x)
        smax = float(np.sqrt(np.linalg.norm(y) / np.linalg.norm(x)))
        x = y / np.linalg.norm(y)
    return dr, dc, A, smax


def _null_modes(J, row, cp, eq, group: Sequence[Block], tol: float):
    """Exact null modes of the EQUILIBRATED Jacobian within the span of the candidate block indicators.
    `eq` = _ruiz(J). Left: y = diag(dr)^-1 w with w = sum_b c_b 1_b (rows DEVSIM moved / replaced
    excluded), residual |y^T J_eq| = |(w^T J) diag(dc)|; right: x = diag(dc)^-1 v, |J_eq x| =
    |diag(dr) J v|. The best combinations are the singular vectors of the (k x n) block residual matrix
    in the metric |y| (blocks have disjoint support, so the Gram matrix is diagonal). A mode is exact
    when r = |y^T J_eq| / (|y| sigma_max(J_eq)) <= tol -- singular to working precision in the
    equilibrated 2-norm, i.e. within round-off of the factorization's own backward error.
    Returns ([(r, {block: coeff})], best refuted mode or None)."""
    import scipy.sparse as sp
    dr, dc, A_eq, smax = eq
    side = group[0].side
    n = J.shape[0]
    if side == "row":                                   # one node-only block, row by row
        blk = group[0]
        k = np.asarray(ds.get_equation_numbers(device=blk.device, region=blk.region,
                                               equation=blk.equation), dtype=np.int64)
        k = k[(row[k] == k) | (cp[k] != 0)]
        if k.size == 0:
            return [], None
        Ar = A_eq.tocsr()[k]
        rnorm = np.sqrt(np.asarray(Ar.multiply(Ar).sum(axis=1)).ravel()) / smax
        dead = rnorm <= tol
        if np.any(dead):
            return [(float(np.min(rnorm)), {blk: float(np.count_nonzero(dead))})], None
        return [], (float(np.min(rnorm)), {blk: 0.0})
    idx, bid = [], []
    for b, blk in enumerate(group):
        if side == "left":
            k = np.asarray(ds.get_equation_numbers(device=blk.device, region=blk.region,
                                                   equation=blk.equation), dtype=np.int64)
            k = k[(row[k] == k) | (cp[k] != 0)]          # rows DEVSIM moved / replaced carry another eq
        else:
            k = np.asarray(ds.get_equation_numbers(device=blk.device, region=blk.region,
                                                   variable=blk.variable), dtype=np.int64)
        idx.append(k)
        bid.append(np.full(k.size, b))
    idx, bid = np.concatenate(idx), np.concatenate(bid)
    W = sp.csc_matrix((np.ones(idx.size), (idx, bid)), shape=(n, len(group)))
    if side == "left":
        N = np.asarray((J.T @ W).T.todense()) * dc[None, :]
        scale = 1.0 / dr[idx]
    else:
        N = np.asarray((J @ W).T.todense()) * dr[None, :]
        scale = 1.0 / dc[idx]
    gram = np.sqrt(np.bincount(bid, weights=scale * scale, minlength=len(group)))
    gram[gram == 0] = 1.0
    U, sv, _ = np.linalg.svd(N / gram[:, None], full_matrices=False)
    out, best = [], None
    for j in range(U.shape[1]):
        c = U[:, j] / gram
        r = float(sv[j]) / smax
        # participation in the equilibrated metric (|U|), not raw magnitudes: an electron row's
        # Potential derivatives ~ n dwarf a hole row's ~ p, which says nothing about the mode
        part = np.abs(U[:, j])
        sup = part >= 1e-6 * part.max()
        cn = c / c[np.argmax(part)]
        mode = (r, {group[b]: float(cn[b]) for b in np.flatnonzero(sup)})
        if r <= tol:
            out.append(mode)
        elif best is None or r < best[0]:
            best = mode
    return out, best


def _acknowledged(device: str) -> Dict[Tuple[str, str], str]:
    return R.floating_acknowledgements(device)


def check_dc_well_posed(device: "str | None" = None, *, numeric: str = "candidates",
                        tol_singular: float = TOL_SINGULAR, raise_error: bool = True) -> WellPosedReport:
    """Static topology + (only when the static pass proposes a candidate) ONE matrix assembly; see the
    module docstring. `device=None` checks every live device (DEVSIM solves them in one Newton).

    numeric: "candidates" (default) -- confirm the static candidates only; "full" -- additionally the
        equilibrated sigma_min / sigma_max of the whole DC Jacobian (dense SVD up to 5000 unknowns, else
        sparse LU inverse iteration) into report.sigma_ratio (diagnostic; never raises by itself).
    raise_error: raise IllPosedDCError on a confirmed, unacknowledged mode (default) or only report.
    A failing assembly (a model DEVSIM cannot evaluate yet) is recorded in report.numeric_error and does
    not raise: the solve will report its own error."""
    if numeric not in ("candidates", "full"):
        raise ValueError("check_dc_well_posed: numeric must be 'candidates' or 'full', got {!r}"
                         .format(numeric))
    devices = tuple(ds.get_device_list()) if device is None else (device,)
    rep = WellPosedReport(devices=devices)
    for d in devices:
        _, left, right = _static(d)
        rep.candidates.extend(left + right)
    if rep.candidates or numeric == "full":
        _confirm(rep, tol_singular)
    if numeric == "full" and rep.numeric_error is None:
        rep.sigma_ratio = _sigma_ratio()[0]
    if raise_error and rep.confirmed:
        raise IllPosedDCError(_message(rep))
    return rep


def _confirm(rep: WellPosedReport, tol: float) -> None:
    try:
        J, row, cp = _assemble()
    except ds.error as e:                                  # noqa: PERF203
        rep.numeric_error = str(e)
        return
    rep.assembled = True
    eq = _ruiz(J)
    for group in rep.candidates:
        exact, best = _null_modes(J, row, cp, eq, group, tol)
        d = group[0].device
        ack = _acknowledged(d)
        for r, w in exact:
            mode = NullMode(d, group[0].side, r, w)
            if all((b.region, b.equation) in ack for b in w):
                rep.acknowledged.append(mode)
            else:
                rep.confirmed.append(mode)
        if not exact and best is not None:
            rep.refuted.append(NullMode(d, group[0].side, best[0], best[1]))


def _message(rep: WellPosedReport) -> str:
    lines = ["ill-posed DC problem: the DC Jacobian is EXACTLY singular ({} null mode(s); equilibrated "
             "residual r <= {:.1e}, threshold {:.1e} = 64 eps). Along that mode the answer is set by "
             "round-off and differs between BLAS builds.".format(
                 len(rep.confirmed), max(m.r for m in rep.confirmed), TOL_SINGULAR)]
    for m in rep.confirmed:
        lines.append("  device {!r}: {} (r = {:.1e})".format(m.device, m.describe(), m.r))
    if any(m.side == "left" for m in rep.confirmed):
        lines.append(
            "  Fixes: pin the MAJORITY carrier at a contact (a contact equation with a node_model on its "
            "continuity equation -- e.g. physics_bipolar_dd.setup_contact_ohmic_bipolar, or a minority-"
            "blocking majority contact), or make the contact FLOATING through a zero-current circuit "
            "node (physics_bipolar_dd.setup_contact_ohmic_bipolar_circuit), or, if the floating body is "
            "intended, acknowledge it with dynameta.carriers.well_posedness.allow_floating(device, region, "
            "reason=...) and prefer a transient solve from a known charge (there is no DC charge "
            "constraint row: decision 57).")
    if any(m.side == "row" for m in rep.confirmed):
        lines.append(
            "  Fixes: give the node-only equation its rate terms back (a kinetic trap row needs its capture "
            "and emission: physics_traps.install_bulk_traps / install_interface_traps build them), or remove "
            "the equation (and its variable).")
    if any(m.side == "right" for m in rep.confirmed):
        lines.append(
            "  Fixes: give the component a Potential Dirichlet (physics_equilibrium.setup_contact or an "
            "ohmic contact), or acknowledge with allow_floating(device, region, carriers=('potential',), "
            "reason=...).")
    return "\n".join(lines)


def allow_floating(device: str, region: str, *, carriers: Iterable[str] = ("electrons", "holes"),
                   reason: str) -> None:
    """Acknowledge an INTENDED floating body: a confirmed null mode whose every block lies in the
    acknowledged (region, equation) set is reported (report.acknowledged), not raised. `carriers`:
    'electrons', 'holes', 'potential' or raw DEVSIM equation names (a trap equation, ...). `reason` is
    required (it is the record of why a singular DC problem is acceptable). Remember what this means:
    the DC answer along the mode is set by round-off (there is no charge-constraint row; decision 57) --
    a transient solve from a known initial charge is the well-defined alternative. Recorded in
    eq_registry (cleared with eq_registry.clear(device))."""
    if not str(reason).strip():
        raise ValueError("allow_floating: `reason` must say why the floating body is intended")
    for c in carriers:
        R.record_floating(device, region, _CARRIER_EQ.get(str(c).lower(), str(c)), str(reason))
    _CACHE.pop(device, None)


# =====================================================================================================
# the cached hook (dc_solve.solve_dc, tests/conftest.py)
# =====================================================================================================

_CACHE: Dict[str, tuple] = {}


def _region_signature(device: str, regions: Iterable[str]) -> str:
    h = hashlib.sha256()
    for r in sorted(set(regions)):
        x = np.asarray(ds.get_node_model_values(device=device, region=r, name="x"), dtype=np.float64)
        h.update(r.encode())
        h.update(x.tobytes())
    return h.hexdigest()[:16]


# hook counters (read by tests/conftest.py's optional DYNAMETA_WELLPOSED_STATS dump; the CI-time audit)
STATS = {"calls": 0, "static_s": 0.0, "assemblies": 0, "numeric_s": 0.0, "raised": 0, "refuted": {}}


def ensure_dc_well_posed(device: "str | None" = None) -> None:
    """The hook: `check_dc_well_posed` with the verdict cached per device per fingerprint (DEVSIM's
    equation / contact / interface commands, the acknowledgements, eq_registry's generation counter and,
    for a device with candidates, the candidate regions' node coordinates). A device with no candidate
    costs the static pass only; devices with candidates share ONE assembly per new fingerprint. Raises
    IllPosedDCError exactly as check_dc_well_posed does."""
    import time
    t0 = time.perf_counter()
    STATS["calls"] += 1
    devices = tuple(ds.get_device_list()) if device is None else (device,)
    todo, msgs = [], []
    for d in devices:
        key, left, right = _static(d)
        groups = left + right
        fp = (key, R.generation(d), tuple(sorted(_acknowledged(d).items())))
        if groups:
            fp = fp + (_region_signature(d, (b.region for g in groups for b in g)),)
        hit = _CACHE.get(d)
        if hit is not None and hit[0] == fp:
            if hit[1] is not None:
                msgs.append(hit[1])
            continue
        if groups:
            todo.append((d, fp, groups))
        else:
            _CACHE[d] = (fp, None)
    t1 = time.perf_counter()
    STATS["static_s"] += t1 - t0
    if todo:
        rep = WellPosedReport(devices=tuple(d for d, _, _ in todo),
                              candidates=[g for _, _, gs in todo for g in gs])
        _confirm(rep, TOL_SINGULAR)
        STATS["assemblies"] += 1
        STATS["numeric_s"] += time.perf_counter() - t1
        if rep.numeric_error is None:                    # else not cached: retry once models evaluate
            for d, fp, _ in todo:
                mine = [m for m in rep.confirmed if m.device == d]
                msg = _message(WellPosedReport(devices=(d,), confirmed=mine)) if mine else None
                _CACHE[d] = (fp, msg)
                if msg is not None:
                    msgs.append(msg)
            for m in rep.refuted:
                k = "{}: {}".format(m.side, sorted("{}/{}".format(b.region, b.equation) for b in m.weights))
                STATS["refuted"][k] = min(m.r, STATS["refuted"].get(k, np.inf))
    if msgs:
        STATS["raised"] += 1
        raise IllPosedDCError("\n".join(msgs))


# =====================================================================================================
# near-singular diagnosis
# =====================================================================================================

def _sigma_ratio(max_dense: int = 5000):
    """(sigma_min / sigma_max, right singular vector of sigma_min) of the equilibrated DC Jacobian."""
    import scipy.sparse.linalg as spla
    J, _, _ = _assemble()
    _, _, A, _ = _ruiz(J)
    n = A.shape[0]
    if n <= max_dense:
        _, s, vt = np.linalg.svd(A.toarray())
        return float(s[-1] / s[0]), vt[-1]
    smax = float(spla.svds(A, k=1, return_singular_vectors=False)[0])
    try:
        lu = spla.splu(A)
    except RuntimeError:                                 # exactly singular factorization
        return 0.0, np.zeros(n)
    x = np.random.default_rng(0).standard_normal(n)
    x /= np.linalg.norm(x)
    lam = 0.0
    for _ in range(40):                                  # inverse iteration on A^T A
        y = lu.solve(lu.solve(x, trans="T"))
        lam = float(np.linalg.norm(y))
        x = y / lam
    return float(1.0 / np.sqrt(lam)) / smax, x


def diagnose_singular(device: "str | None" = None, *, max_dense: int = 5000) -> str:
    """One-line diagnosis of the DC Jacobian's conditioning for a convergence-failure message: the
    equilibrated (Ruiz) sigma_min / sigma_max and the support of sigma_min's singular vector -- the
    variable block, region and node x-range carrying >= 1 % of its peak amplitude. "near-singular" when
    the ratio is below NEAR_SINGULAR (1e-12; double precision resolves ~1e-16: the Newton correction is
    then noise along that mode). Cost: a dense SVD up to `max_dense` unknowns (804 unknowns < 1 s), else
    a sparse LU + 40 inverse iterations. Reads only; `device` limits the support search."""
    try:
        ratio, v = _sigma_ratio(max_dense)
    except ds.error as e:
        return "conditioning diagnosis unavailable ({})".format(e)
    devices = tuple(ds.get_device_list()) if device is None else (device,)
    best, mass_tot = None, float(np.sum(v * v)) or 1.0
    for d in devices:
        for r in ds.get_region_list(device=d):
            for e in ds.get_equation_list(device=d, region=r):
                var = ds.get_equation_command(device=d, region=r, name=e)["variable_name"]
                k = np.asarray(ds.get_equation_numbers(device=d, region=r, variable=var), dtype=np.int64)
                m = float(np.sum(v[k] ** 2))
                if best is None or m > best[0]:
                    best = (m, d, r, var, k)
    head = ("near-singular DC Jacobian" if ratio < NEAR_SINGULAR else "DC Jacobian conditioning") + \
        " (equilibrated sigma_min/sigma_max = {:.1e})".format(ratio)
    if best is None:
        return head
    m, d, r, var, k = best
    x = np.asarray(ds.get_node_model_values(device=d, region=r, name="x"), dtype=np.float64)
    a = np.abs(v[k])
    sel = a >= 0.01 * a.max() if a.size and a.max() > 0 else np.zeros(a.size, bool)
    span = "x in [{:.3g}, {:.3g}] nm".format(1e9 * x[sel].min(), 1e9 * x[sel].max()) if sel.any() else ""
    tail = (": a weakly pinned carrier / floating mode -- Newton cannot resolve it in double precision "
            "(see dynameta.carriers.well_posedness; for an MOS capacitor in inversion the remedy is "
            "physics_bipolar_dd.setup_quasi_equilibrium_carrier -- close the minority carrier, and the "
            "holes too when they carry a density-gradient potential)" if ratio < NEAR_SINGULAR else "")
    return "{}: mode in {} (device {!r}, region {!r}, {:.3g} of its mass), {}{}".format(
        head, var, d, r, m / mass_tot, span, tail)
