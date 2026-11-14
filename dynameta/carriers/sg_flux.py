"""
THE Scharfetter-Gummel edge flux -- one string builder for every DEVSIM drift-diffusion family
(classical unipolar / bipolar electrons and holes, the unipolar and the bipolar density-gradient
fluxes, and the validation-side frozen-Lambda flux). 1-D / 2-D / 3-D, DC / transient / ssac / noise /
Gummel, contact-current extraction and the impact-ionization current read-back all consume the edge
models this builds, so they share it automatically.

With v = the edge's Bernoulli argument (vdiff_g = (psi@n0 - psi@n1)/(g V_t) for the classical DD,
plus the BGN / quantum-potential quasi-field shifts where those are on) and the DEVSIM sign
convention (residual = q * divergence; docs/implementation_notes.md):

    Jn = +q mu_n EdgeInverseLength V_t g_n * ( n@n1 B(-v) - n@n0 B(v) )
    Jp = -q mu_p EdgeInverseLength V_t g_p * ( p@n1 B(v)  - p@n0 B(-v) )

THE PROBLEM (user decision 2026-11-04; docs/audit/2026-11-04-sg-flux-two-bernoulli.md). Up to
v0.14-pre the classical and unipolar-DG strings were kahan3(n@n1 B(v), n@n1 v, -n@n0 B(v)), i.e. B(-v)
spelled B(v) + v -- exact algebra, but for v << 0 the difference of two O(|v|) numbers leaving
B(-v) ~ |v| e^-|v|, relative error ~ eps |v| e^|v|. MEASURED against mpmath (50 digits) on DEVSIM
2.11.2, 2026-11-04: B(v) + v is off by 1.5e-12 at v = -10, 8.6e-9 at -18, 1.9e-7 at -21, 6.5e-5 at
-27.6, 1.0e-3 at -30, 1.7e-2 at -33 and 100 % from -40 on (its Jacobian dBdx(v) + 1: 2.8e-13 /
1.6e-10 / 3.8e-9 / 1.5e-6 / 2.8e-6 / 1.3e-4 / 34 %), while DEVSIM's own B(+-v) and dBdx(+-v) are
<= 3.1e-16 at every |v| in 0.3..60.

THE FORM: BOTH Bernoullis, B(v) and B(-v), each with DEVSIM's accurate B, combined SIGN-SELECTED --

    electrons:  v >= 0:  kahan3(n@n1 B(v),  n@n1 v,      -n@n0 B(v))      [B(-v) = B(v) + v]
                v <  0:  kahan3(n@n1 B(-v), -n@n0 B(-v), n@n0 v)        [B(v) = B(-v) - v]
    holes:      v >= 0:  kahan3(p@n1 B(v),  -p@n0 B(v),  -p@n0 v)       [B(-v) = B(v) + v]
                v <  0:  kahan3(p@n1 B(-v), -p@n1 v,     -p@n0 B(-v))   [B(v) = B(-v) - v]

so the Bernoulli identity is only ever used with the Bernoulli of the NON-NEGATIVE argument, where
B(|v|) + |v| has no cancellation, and within each branch the n@n1 and n@n0 terms SHARE one Bernoulli.
Why not the plain bracket n@n1 B(-v) - n@n0 B(v) (e36e808's bipolar DG form, the first proposal):
it is exactly as accurate at large |v|, but its two Bernoullis -- and in the Jacobian their two
dBdx -- are rounded INDEPENDENTLY, while near equilibrium (|v| small, n@n1 ~ n@n0, the flux a tiny
difference of two large terms) the shared Bernoulli makes its rounding common-mode, multiplied by
n@n1 - n@n0 instead of n@n1 + n@n0. DEVSIM 2.11.2's dBdx is itself inaccurate near 0 (vs mpmath
2.1e-14 at |v| = 1e-2, 4.1e-13 at 1e-3, 2.4e-10 at 1e-6), so the plain bracket's Jacobian carries that
error at full weight in every quasi-neutral region. MEASURED 2026-11-04 on validation/
extended_precision.py's 1-D Si diode at 0.1 V forward, 40 repeated converged DC passes + ssac each
(std/mean; DC floor = max relative n, p change per extra Newton pass, median):
                         ssac G(1 Hz)   ssac G(1 MHz)   DC floor
    kahan3 (old)         3.2e-4         1.5e-7          1.6e-15
    plain two-Bernoulli  2.2e-3         1.1e-6          3.3e-15
    sign-selected (this) 5.7e-4         2.6e-7          8.9e-16
and the sign-selected form keeps the plain bracket's large-|v| accuracy (tests/test_devsim_sg_flux.py:
value <= 3.3e-16, Jacobian <= 8.9e-16 to |v| = 40, every family). The branches agree analytically
(and in every derivative) at v = 0, so the flux is smooth; numerically they differ by ulps there.
Derivatives are DEVSIM's symbolic diff of the same string (eq_registry.edge_with_derivs; diff of an
ifelse is the ifelse of the branch derivatives) -- exact, no hand-coded Jacobian.

CLAMPING (measured, 2026-11-04, DEVSIM 2.11.2): B and dBdx are finite at EVERY finite argument --
B(+1e300) = 0, B(-1e300) = 1e300, dBdx = -0 / -1, no inf/NaN (B(+-200) = 2.77e-85 / 200,
B(+-700) = 6.9e-302 / 700). The classical DD therefore builds both Bernoullis UNCLAMPED, exactly as
its B(v) always was. The density-gradient fluxes keep upper_clamp=True, B(min(arg, 200)): a no-op
in value (it moves B by < 2.8e-85 where it binds) kept as belt-and-braces for the log-singular
Lambda of a hard wall in Newton transients. The branch selected for |v| > 200 always carries the
linear c v term, so even a clamped Bernoulli never zeroes a Jacobian column (the two-sided clamp
under e36e808's plain bracket did: "Convergence failure!" / "umfpack numeric failed" on the bipolar
DG wall gates, docs/audit/2026-11-03-devsim-bipolar-density-gradient.md sec. 1.6); a lower clamp is
still never used.
"""

from __future__ import annotations

from dynameta.carriers.eq_registry import edge_with_derivs

__all__ = ["bernoulli_expr", "sg_bracket", "install_sg_current"]

_CLAMP = 200        # e^200 ~ 7e86: far from overflow, never reached at convergence (|v| <~ 30)


def bernoulli_expr(arg: str, *, upper_clamp: bool = False) -> str:
    """DEVSIM expression for B(arg) = arg/(e^arg - 1); with upper_clamp, B(min(arg, 200)) (module
    header: the density-gradient fluxes only; never a lower clamp)."""
    if upper_clamp:
        return "B(ifelse({a} > {c}, {c}, {a}))".format(a=arg, c=_CLAMP)
    return "B({})".format(arg)


def sg_bracket(carrier: str, density: str, vdiff: str, bern: str, bern_m: str) -> str:
    """The sign-selected flux bracket (module header) equal to n@n1 B(-v) - n@n0 B(v) (electrons) /
    p@n1 B(v) - p@n0 B(-v) (holes), with `vdiff` the model holding v, `bern` B(v), `bern_m` B(-v)."""
    f = dict(c=density, v=vdiff, b=bern, bm=bern_m)
    if carrier == "electrons":
        return ("ifelse({v} >= 0, kahan3({c}@n1*{b}, {c}@n1*{v}, -{c}@n0*{b}), "
                "kahan3({c}@n1*{bm}, -{c}@n0*{bm}, {c}@n0*{v}))").format(**f)
    if carrier == "holes":
        return ("ifelse({v} >= 0, kahan3({c}@n1*{b}, -{c}@n0*{b}, -{c}@n0*{v}), "
                "kahan3({c}@n1*{bm}, -{c}@n1*{v}, -{c}@n0*{bm}))").format(**f)
    raise ValueError("sg_bracket: carrier must be 'electrons' or 'holes'; got {!r}".format(carrier))


def install_sg_current(device: str, region: str, *, carrier: str, density: str, vdiff: str,
                       bern: str, bern_m: str, current: str, prefactor: str, wrt,
                       current_wrt=None, upper_clamp: bool = False) -> str:
    """Build the edge models bern = B(vdiff), bern_m = B(-vdiff) and current = prefactor * bracket,
    each with its @n0/@n1 derivatives (the Bernoullis w.r.t. `wrt`, the current w.r.t. `current_wrt`,
    default `wrt`). `prefactor` carries the sign, charge, mobility, EdgeInverseLength, V_t and the
    FD g-factor (e.g. "-ElectronCharge*mu_p*EdgeInverseLength*V_t*g_enh_p"). `vdiff` is an edge
    model NAME, so a later redefinition of it (BGN's band-edge quasi-fields) propagates into both
    Bernoullis, the branch selection and every derivative. Returns the current expression."""
    wrt = tuple(wrt)
    edge_with_derivs(device, region, bern, bernoulli_expr(vdiff, upper_clamp=upper_clamp), wrt)
    edge_with_derivs(device, region, bern_m,
                     bernoulli_expr("(-{})".format(vdiff), upper_clamp=upper_clamp), wrt)
    expr = "{}*{}".format(prefactor, sg_bracket(carrier, density, vdiff, bern, bern_m))
    edge_with_derivs(device, region, current, expr,
                     wrt if current_wrt is None else tuple(current_wrt))
    return expr
