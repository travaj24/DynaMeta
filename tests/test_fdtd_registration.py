"""FDTD material registration: every engine registers its materials at the Yee E-NODES (audit S2-14
engine siblings, 2026-11-02; docs/audit/2026-11-02-fdtd-registration.md).

THE DEFECT these gates pin. solve_fdtd_mo_1d sampled the layer eps at the E nodes k*dz (the S2-14
gate in tests/test_fdtd_mo_oblique.py proved that unbiased); solve_fdtd_1d's uniform path,
solve_fdtd_2d(_oblique) and solve_fdtd_3d(_oblique/_mo) sampled the cell centres (k+1/2)*dz. All of
them keep E_k at k*dz and de-embed r0 in that node frame, so the centre rule shifted every
structure -dz/2 on average: a reflection-phase bias of -n_super k0 dz that |R|/|T| cannot see.

HOW THE GATE MEASURES IT. A point-sampled interface sits at the H node between the last outside and
the first inside E node, i.e. within +-dz/2 of the true one (a sawtooth in the sub-cell offset u =
frac(pad/dz)). Sampling u at the MIDPOINT-uniform set (k + 1/2)/N cancels that sawtooth exactly, so
the offset-mean of dphi_r isolates the registration bias (old ~ -1.0 k0dz) from the staircase
spread (+-k0dz) and leaves the O((k dz)^2) Yee dispersion. Slab: n = 2, d = 180 nm (|r| ~ 0.6, near
its quarter-wave maximum -- a small-|r| point would let the |r| error rotate the phase), lambda 1.4
um, band 1.2-1.6 um. Oracle: the exact tangential-field characteristic matrix (cross-checked
against the `tmm` package below). Each bias gate fails on the pre-fix code (measured there: see
the per-test numbers; the doc carries the full before/after table).
"""
import numpy as np
import pytest

from dynameta.constants import C_LIGHT, EPS0, MU0

L1, L2, NS, D, LAM0 = 1.2e-6, 1.6e-6, 2.0, 0.18e-6, 1.4e-6
FI = C_LIGHT / LAM0
K0 = 2.0 * np.pi / LAM0
BIAS_BAR = 0.25          # |mean dphi_r| < 0.25 k0dz (the pre-fix engines sit at -0.9 .. -1.1 k0dz)


def _k0dz(res):
    return K0 * L1 / (res * NS)


def _tangential_r(eps_list, d_list, theta_deg, pol, n_in=1.0, n_out=1.0, lam=LAM0):
    """Exact reflection coefficient of the TANGENTIAL E (Ey for s, Ex for p), exp(-i w t)."""
    k0 = 2.0 * np.pi / lam
    w = k0 * C_LIGHT
    kx = k0 * n_in * np.sin(np.radians(theta_deg))

    def adm(eps):
        kz = np.sqrt(complex(eps) * k0 ** 2 - kx ** 2)
        kz = -kz if kz.imag < 0 else kz
        return (kz / (w * MU0)) if pol == "s" else (w * EPS0 * eps / kz), kz
    M = np.eye(2, dtype=complex)
    for eps, d in zip(eps_list, d_list):
        Y, kz = adm(eps)
        c, s = np.cos(kz * d), np.sin(kz * d)
        M = M @ np.array([[c, -1j * s / Y], [-1j * Y * s, c]])
    Y0, Ys = adm(n_in ** 2)[0], adm(n_out ** 2)[0]
    Yin = (M[1, 0] + M[1, 1] * Ys) / (M[0, 0] + M[0, 1] * Ys)
    return (Y0 - Yin) / (Y0 + Yin)


def _npw_offsets(res, n, base=3.0):
    """n_pad_wave values putting pad/dz at m + (k + 1/2)/n (midpoint-uniform sub-cell offsets)."""
    dz = L1 / (res * NS)
    m = round(base * L2 / dz)
    return [(m + (k + 0.5) / n) * dz / L2 for k in range(n)]


def _at(f, a, fq=FI):
    return complex(np.interp(fq, f, a.real), np.interp(fq, f, a.imag))


def _layer():
    from dynameta.optics.fdtd_nd import FDTDLayer
    return [FDTDLayer(thickness_m=D, eps_inf=NS ** 2)]


# ---------------------------------------------------------------------------------- the helper itself
def test_oracle_matches_tmm_package():
    tmm = pytest.importorskip("tmm")
    for th in (0.0, 30.0):
        ref = tmm.coh_tmm("s", [1, NS, 1], [np.inf, D * 1e9, np.inf], np.radians(th), LAM0 * 1e9)["r"]
        assert _tangential_r([NS ** 2], [D], th, "s") == pytest.approx(complex(ref), abs=1e-12)


def test_material_z_is_the_e_nodes_and_masks_are_half_open():
    from dynameta.optics.fdtd_nd.grid import layer_masks, material_z, pad_masks
    dz = 0.1
    z = material_z(11, dz)
    assert z.tobytes() == (np.arange(11) * dz).tobytes()          # exactly fdtd_mo's old expression

    class _L:
        def __init__(self, t):
            self.thickness_m = t
    (l0, m0), (l1, m1) = layer_masks(z, 0.25, [_L(0.3), _L(0.2)])
    assert np.nonzero(m0)[0].tolist() == [3, 4, 5]                # [0.25, 0.55)
    assert np.nonzero(m1)[0].tolist() == [6, 7]                   # [0.55, 0.75)
    lo, hi = pad_masks(z, 0.25, 0.5)
    assert np.nonzero(lo)[0].tolist() == [0, 1, 2] and np.nonzero(hi)[0].tolist() == [8, 9, 10]


# ---------------------------------------------------------------------------------- 1-D engines
def _r_1d(res, npw, refine=None):
    from dynameta.optics.fdtd import _grid_metrics, _refined_full_edges, solve_fdtd_1d
    from dynameta.optics.fdtd_nd.grid import probe_cells
    lay = _layer()
    r = solve_fdtd_1d(lay, lambda_min_m=L1, lambda_max_m=L2, resolution=res, n_pad_wave=npw,
                      return_time_trace=True, refine=refine)
    tt = r.time_trace
    dz, pad, zs, _Lz, _nz, *_ = _grid_metrics(lay, L1, L2, res, npw)
    if refine is None:
        z_pl = probe_cells(pad, zs, dz)[1] * dz
    else:
        ze = _refined_full_edges(lay, dz, pad, zs, refine, 8)
        z_pl = ze[int(np.searchsorted(ze, 0.7 * pad))]
    f = np.fft.rfftfreq(tt["t"].size, tt["dt"])
    with np.errstate(divide="ignore", invalid="ignore"):          # out-of-band bins only
        rr = np.conj(np.fft.rfft(tt["reflected"]) / np.fft.rfft(tt["incident_left"]))
    return _at(f, rr) * np.exp(-2j * K0 * (pad - z_pl))


def _dphi(r, theta=0.0, pol="s", n_in=1.0):
    return float(np.angle(r / _tangential_r([NS ** 2], [D], theta, pol, n_in=n_in)))


def test_1d_uniform_registration_unbiased_and_converging():
    """solve_fdtd_1d's uniform path. Measured 2026-11-02 (8 offsets): pre-fix -0.95 k0dz (res 30);
    post-fix +0.05 k0dz (res 30), shrinking with dz (res 60 below)."""
    m = {}
    for res in (30, 60):
        m[res] = float(np.mean([_dphi(_r_1d(res, npw)) for npw in _npw_offsets(res, 8)]))
    assert abs(m[30]) < BIAS_BAR * _k0dz(30), m
    assert abs(m[60]) < BIAS_BAR * _k0dz(60), m
    assert abs(m[60]) < 0.5 * abs(m[30]) + 1e-4, m                 # not a dz-proportional offset


def test_1d_cell_centre_rule_is_the_biased_one():
    """Discrimination: the SAME march with the old (k+1/2) map gives ~ -k0dz (proves the gate above
    can see a registration offset; |R| cannot)."""
    from dynameta.optics.fdtd import _grid_metrics, _run
    from dynameta.optics.fdtd_nd.grid import probe_cells, pulse_width, source_pulse
    res = 30
    lay = _layer()
    out, dR = {"node": [], "centre": []}, {"node": [], "centre": []}
    for npw in _npw_offsets(res, 8):
        dz, pad, zs, Lz, nz, _n, f_min, f_max = _grid_metrics(lay, L1, L2, res, npw)
        dt = 0.5 * dz / C_LIGHT
        i_src, i_pL, i_pR = probe_cells(pad, zs, dz)
        tau, t0 = pulse_width(f_min, f_max, 12.0)
        nsteps, _tg, src = source_pulse(0.5 * (f_min + f_max), tau, t0, source_amp=1.0, Lz=Lz, dt=dt)
        one, zero = np.ones(nz), np.zeros(nz)
        eLi, _ = _run(one, zero, zero, zero, dz, dt, nsteps, i_src, i_pL, i_pR, src)
        f = np.fft.rfftfreq(nsteps, dt)
        for key, off in (("node", 0.0), ("centre", 0.5)):
            zc = (np.arange(nz) + off) * dz
            e = np.where((zc >= pad) & (zc < pad + D), NS ** 2, 1.0)
            eLt, _ = _run(e, zero, zero, zero, dz, dt, nsteps, i_src, i_pL, i_pR, src)
            with np.errstate(divide="ignore", invalid="ignore"):  # out-of-band bins only
                rr = np.conj(np.fft.rfft(eLt - eLi) / np.fft.rfft(eLi))
            r = _at(f, rr) * np.exp(-2j * K0 * (pad - i_pL * dz))
            out[key].append(_dphi(r))
            dR[key].append(abs(r) ** 2)
    k = _k0dz(res)
    assert abs(np.mean(out["node"])) < BIAS_BAR * k, out
    assert np.mean(out["centre"]) == pytest.approx(-k, rel=0.25), out    # measured -0.95 k0dz
    # |R| cannot discriminate: a rigid dz/2 translation leaves the offset-mean |R| unchanged
    assert abs(np.mean(dR["node"]) - np.mean(dR["centre"])) < 2e-3


def test_1d_nonuniform_interface_tie_average():
    """solve_fdtd_1d(refine=...): the graded grid SNAPS interfaces onto nodes, and the half-open
    point rule moved each one -h/2 (measured -0.88 k0*h_fine, refine 2/4, res 30/60). The tie nodes
    now take the control-volume arithmetic mean (Hwang & Cangellaris 2001): measured +0.03 k0*h."""
    res = 30
    h_fine = (L1 / (res * NS)) / 2.0
    d = [_dphi(_r_1d(res, npw, refine={0: 2.0})) for npw in _npw_offsets(res, 2)]
    assert abs(np.mean(d)) < BIAS_BAR * K0 * h_fine, d


def test_1d_mo_unchanged_by_the_shared_helper(monkeypatch):
    """fdtd_mo always registered at the nodes; routing it through grid.material_z/layer_masks is
    bit-identical (same expression, same float accumulation)."""
    from dynameta.optics import fdtd_mo as fm
    lay = [fm.MOLayer(thickness_m=D, eps_xx=NS ** 2, eps_yy=NS ** 2)]
    seen = {}
    real = fm._run_mo

    def spy(exx, *a, **k):
        seen.setdefault("exx", []).append(np.array(exx))
        return real(exx, *a, **k)
    monkeypatch.setattr(fm, "_run_mo", spy)
    fm.solve_fdtd_mo_1d(lay, lambda_min_m=L1, lambda_max_m=L2, resolution=12, n_pad_wave=3.0,
                        z_subcell="point")                         # the 2b7a002 rule (opt-out)
    from dynameta.optics.fdtd_nd.grid import z_grid
    dz, pad, _zs, _Lz, nz = z_grid(lay, L1, L2, 12, 3.0, NS)
    zc = np.arange(nz) * dz                                        # the pre-refactor inline rule
    want = np.where((zc >= pad) & (zc < pad + D), NS ** 2, 1.0)
    assert seen["exx"][-1].tobytes() == want.tobytes()


# ---------------------------------------------------------------------------------- 2-D / 3-D engines
def test_2d_normal_r0_phase_unbiased():
    """solve_fdtd_2d's public r0 (front-face de-embedded). Measured 2026-11-02, 8 offsets, res 30:
    pre-fix -0.94 k0dz, post-fix +0.06 k0dz (res 60: -0.97 -> +0.03)."""
    from dynameta.optics.fdtd_nd import solve_fdtd_2d
    res = 30
    d = []
    for npw in _npw_offsets(res, 4):
        r = solve_fdtd_2d(_layer(), period_x_m=0.1e-6, nx=2, lambda_min_m=L1, lambda_max_m=L2,
                          resolution=res, n_pad_wave=npw)
        d.append(_dphi(_at(r.freqs_Hz, r.r0)))
    assert abs(np.mean(d)) < BIAS_BAR * _k0dz(res), d


def test_2d_glass_superstrate_r0_phase_unbiased():
    """n_super = 1.5: the bias scaled with n_super (pre-fix -1.32 k0dz = ~ -1.5 x 0.9); post-fix
    +0.18 k0dz (vacuum k0dz units; res 60 +0.09: O(dz^2) in the glass wavenumber)."""
    from dynameta.optics.fdtd_nd import solve_fdtd_2d
    res = 30
    d = []
    for npw in _npw_offsets(res, 4):
        r = solve_fdtd_2d(_layer(), period_x_m=0.1e-6, nx=2, lambda_min_m=L1, lambda_max_m=L2,
                          resolution=res, n_pad_wave=npw, n_super=1.5)
        d.append(_dphi(_at(r.freqs_Hz, r.r0), n_in=1.5))
    assert abs(np.mean(d)) < 1.5 * BIAS_BAR * _k0dz(res), d


def _obl_front(pol, angle, res, npw, backend, monkeypatch, z_subcell="average", layers=None):
    """r at LAM0 through the PUBLIC solve_fdtd_2d_oblique (its own material fill), by capturing the
    kernel probe lines it computes (the front end itself returns |R0|/|T0| only)."""
    import dynameta.optics.fdtd_nd.solve2d as S2
    from dynameta.optics.fdtd_nd.grid import probe_cells
    cap = []
    real = S2._run_oblique

    def spy(*a, **k):
        out = real(*a, **k)
        cap.append((a, k, out))
        return out
    monkeypatch.setattr(S2, "_run_oblique", spy)
    lay = _layer() if layers is None else layers
    rs = S2.solve_fdtd_2d_oblique(lay, period_x_m=0.1e-6, angle_deg=angle, lambda_min_m=L1,
                                  lambda_max_m=L2, resolution=res, n_pad_wave=npw, nx=4, backend=backend,
                                  pol=pol, z_subcell=z_subcell)
    monkeypatch.setattr(S2, "_run_oblique", real)
    (a_i, _k_i, out_i), (a_t, k_t, out_t) = cap
    dz, pad, zs = a_i[5], npw * L2, float(sum(L.thickness_m for L in lay))   # the solve's own grid
    k_pL = probe_cells(pad, zs, dz)[1]
    dt, nsteps, kx = a_i[6], a_i[7], a_i[13]
    nf = nsteps // 2 + 1
    f = np.fft.rfftfreq(nsteps, dt)
    mean = (lambda x: np.fft.fft(x.mean(axis=1))[:nf])
    with np.errstate(divide="ignore", invalid="ignore"):
        rr = np.conj(mean(out_t[0] - out_i[0]) / mean(out_i[0]))
    th = float(np.degrees(np.arcsin(kx * C_LIGHT / (2.0 * np.pi * FI))))
    r = _at(f, rr) * np.exp(-2j * K0 * np.cos(np.radians(th)) * (pad - k_pL * dz))
    _obl_front.last = rs                                           # the public |R0|/|T0| result
    return r, th, (a_t, k_t)


@pytest.mark.parametrize("pol, angle", [("s", 30.0), ("p", 30.0), ("p", 50.0)])
def test_2d_oblique_phase_unbiased(pol, angle, monkeypatch):
    """solve_fdtd_2d_oblique (numba). Measured 2026-11-02, 8 offsets, res 30: pre-fix s30 -0.82,
    p30 -1.15 k0dz; post-fix s30 +0.04, p30 +0.04, p50 +0.04 k0dz. p-pol needs the per-component
    Ez maps: with the node map on Ez as well, p30 kept -0.29 k0dz (test below)."""
    pytest.importorskip("numba")
    res = 30
    d = []
    for npw in _npw_offsets(res, 4):
        r, th, _ = _obl_front(pol, angle, res, npw, "numba", monkeypatch)
        d.append(_dphi(r, th, pol))
    assert abs(np.mean(d)) < BIAS_BAR * _k0dz(res), d


def test_2d_oblique_ppol_needs_its_own_ez_maps(monkeypatch):
    """Discrimination for the per-component fix: re-run the front end's OWN node maps with Ez sharing
    them (ez=None, the single-array rule) -> a clear negative bias; with the half-node Ez maps the
    front end ships, it is gone. Also checks the front end hands the half-node maps to the kernel."""
    pytest.importorskip("numba")
    from dynameta.optics.fdtd_nd.grid import layer_masks, material_z_half, z_grid
    from dynameta.optics.fdtd_nd.oblique2d import _run_oblique
    res, angle = 30, 30.0
    d_ship, d_single = [], []
    for npw in _npw_offsets(res, 4):
        # the 2b7a002 POINT maps (z_subcell='point'): this gate is about the Ez REGISTRATION
        r, th, (a_t, k_t) = _obl_front("p", angle, res, npw, "numba", monkeypatch, z_subcell="point")
        d_ship.append(_dphi(r, th, "p"))
        ez_i = k_t["ez"][0]
        dz, pad, _zs, _Lz, nz = z_grid(_layer(), L1, L2, res, npw, NS)
        want = np.ones(nz)
        for L, m in layer_masks(material_z_half(nz, dz), pad, _layer()):
            want[m] = L.eps_inf
        assert np.array_equal(ez_i[0], want)                       # Ez at (k+1/2)*dz
        args = list(a_t)
        one = np.ones_like(args[1]); zero = np.zeros_like(args[1])
        out_t = _run_oblique(*args[:14], "p", ez=None)
        out_i = _run_oblique(args[0], one, zero, zero, *args[4:14], "p")
        nsteps, dt = args[7], args[6]
        nf = nsteps // 2 + 1
        f = np.fft.rfftfreq(nsteps, dt)
        mean = (lambda x: np.fft.fft(x.mean(axis=1))[:nf])
        with np.errstate(divide="ignore", invalid="ignore"):
            rr = np.conj(mean(out_t[0] - out_i[0]) / mean(out_i[0]))
        k_pL = args[9]
        rs = _at(f, rr) * np.exp(-2j * K0 * np.cos(np.radians(th)) * (pad - k_pL * dz))
        d_single.append(_dphi(rs, th, "p"))
    k = _k0dz(res)
    assert abs(np.mean(d_ship)) < BIAS_BAR * k, d_ship
    assert np.mean(d_single) < -0.15 * k, d_single                 # measured -0.29 k0dz


def test_2d_oblique_ppol_backends_agree_with_ez_maps():
    """The per-component Ez maps reach all three TM kernels identically (Drude p-pol, numpy/numba/jax)."""
    pytest.importorskip("numba")
    from dynameta.optics.fdtd_nd import FDTDLayer, solve_fdtd_2d_oblique
    ol = [FDTDLayer(thickness_m=250e-9, eps_inf=4.0, drude_wp_rad_s=1.5e15, drude_gamma_rad_s=1.0e14)]
    kw = dict(period_x_m=300e-9, angle_deg=35.0, lambda_min_m=1.4e-6, lambda_max_m=1.6e-6,
              resolution=14, nx=4, pol="p")
    rn = solve_fdtd_2d_oblique(ol, backend="numpy", **kw)
    rb = solve_fdtd_2d_oblique(ol, backend="numba", **kw)
    m = rn.band
    assert float(np.max(np.abs(rn.R0[m] - rb.R0[m]))) < 1e-9
    assert float(np.max(np.abs(rn.T0[m] - rb.T0[m]))) < 1e-9


@pytest.mark.slow
def test_3d_and_3d_mo_phase_unbiased():
    """solve_fdtd_3d r0 and solve_fdtd_3d_mo r_co (probe-referenced; de-embedded here). Measured
    2026-11-02, 8 offsets, res 30: pre-fix -0.94 k0dz both; post-fix +0.06 k0dz both."""
    pytest.importorskip("numba")
    from dynameta.optics.fdtd_mo import MOLayer
    from dynameta.optics.fdtd_nd import solve_fdtd_3d, solve_fdtd_3d_mo
    from dynameta.optics.fdtd_nd.grid import probe_cells, z_grid
    res = 30
    d3, dmo = [], []
    for npw in _npw_offsets(res, 4):
        r = solve_fdtd_3d(_layer(), period_x_m=0.1e-6, period_y_m=0.1e-6, nx=2, ny=2, lambda_min_m=L1,
                          lambda_max_m=L2, resolution=res, n_pad_wave=npw, backend="numba")
        d3.append(_dphi(_at(r.freqs_Hz, r.r0)))
        lay = [MOLayer(thickness_m=D, eps_xx=NS ** 2, eps_yy=NS ** 2)]
        rm = solve_fdtd_3d_mo(lay, period_x_m=0.1e-6, period_y_m=0.1e-6, nx=2, ny=2, lambda_min_m=L1,
                              lambda_max_m=L2, resolution=res, n_pad_wave=npw)
        dz, pad, zs, _Lz, _nz = z_grid(lay, L1, L2, res, npw, NS)
        z_pl = probe_cells(pad, zs, dz)[1] * dz
        dmo.append(_dphi(_at(rm.freqs_Hz, rm.r_co) * np.exp(-2j * K0 * (pad - z_pl))))
    k = _k0dz(res)
    assert abs(np.mean(d3)) < BIAS_BAR * k, d3
    assert abs(np.mean(dmo)) < BIAS_BAR * k, dmo


# =================================================================================================
# z-SUBCELL AVERAGING (the default since 2026-11-02; z_subcell="point" = the 2b7a002 point rule)
# =================================================================================================
class _Lt:
    def __init__(self, t):
        self.thickness_m = t


def test_mix_profile_rules():
    """The shared rule (fdtd_nd.grid): arithmetic (tangential), harmonic (normal), fill-weighted pole
    strengths with kept shapes, pure nodes exact, and the harmonic channel's dispersive fallback."""
    from dynameta.optics.fdtd_nd.grid import mix_profile, point_regions, z_fill_fractions
    z = np.arange(12) * 0.1
    lay = [_Lt(0.33), _Lt(0.2)]
    F = z_fill_fractions(z, 0.25, lay, dz=0.1)
    assert np.allclose(F.sum(axis=0), 1.0, atol=1e-15)
    assert F[1, 6] == pytest.approx(0.3) and F[2, 6] == pytest.approx(0.7)      # [0.55, 0.65] vs 0.58
    m = mix_profile(F, {"e": [1.0, 4.0, 9.0, 1.0], "wp": [0.0, 2.0, 0.0, 0.0], "g": [0.0, 5.0, 0.0, 0.0]},
                    poles=(("wp", 2, ("g",)),))
    assert m["e"][6] == pytest.approx(0.3 * 4.0 + 0.7 * 9.0)                     # arithmetic
    assert m["wp"][6] == pytest.approx(np.sqrt(0.3) * 2.0)                       # wp^2 scales with fill
    assert m["g"][6] == 5.0                                                      # shape kept
    assert m["e"][3] == 4.0 and m["e"][0] == 1.0                                 # pure nodes exact
    h = mix_profile(F, {"e": [1.0, 4.0, 9.0, 1.0], "wp": [0.0] * 4, "g": [0.0] * 4},
                    poles=(("wp", 2, ("g",)),), harmonic=("e",), point_index=point_regions(z, 0.25, lay))
    assert h["e"][6] == pytest.approx(1.0 / (0.3 / 4.0 + 0.7 / 9.0))             # harmonic
    hd = mix_profile(F, {"e": [1.0, 4.0, 9.0, 1.0], "wp": [0.0, 2.0, 0.0, 0.0], "g": [0.0, 5.0, 0.0, 0.0]},
                     poles=(("wp", 2, ("g",)),), harmonic=("e",), point_index=point_regions(z, 0.25, lay))
    assert hd["e"][6] == 9.0 and hd["wp"][6] == 0.0                              # Drude-mixed Ez -> point
    #                                          (z = 0.6 lies in the second layer [0.58, 0.78))
    assert hd["e"][8] == pytest.approx(1.0 / (0.3 / 9.0 + 0.7 / 1.0))            # non-dispersive -> harmonic


def test_two_drude_single_pole_rule_error_is_second_order():
    """A cell mixing two DIFFERENT Drude poles gets one pole: wp^2 = sum f wp_r^2, gamma = the
    wp^2-weighted mean. Against the exact parallel sum sum_r f_r eps_r(w) the error is second order
    in gamma/w: measured here 6.6e-5 relative to the Drude term for an ITO-like pair (gamma 1e14 /
    2e14, w = 1.35e15 rad/s), below the (dgamma/w)^2 / 4 = 1.4e-3 estimate."""
    from dynameta.optics.fdtd_nd.grid import mix_profile
    F = np.array([[0.0], [0.4], [0.6], [0.0]])
    wp1, wp2, g1, g2, w = 2.0e15, 1.5e15, 1.0e14, 2.0e14, 1.35e15
    m = mix_profile(F, {"wp": [0.0, wp1, wp2, 0.0], "g": [0.0, g1, g2, 0.0]}, poles=(("wp", 2, ("g",)),))
    dr = lambda wp, g: -wp ** 2 / (w ** 2 + 1j * g * w)
    exact = 0.4 * dr(wp1, g1) + 0.6 * dr(wp2, g2)
    single = dr(m["wp"][0], m["g"][0])
    rel = abs(single - exact) / abs(exact)
    assert rel < 0.25 * ((g2 - g1) / w) ** 2, rel


def test_point_opt_out_reproduces_the_2b7a002_material_maps(monkeypatch):
    """z_subcell='point' must hand every kernel EXACTLY the 2b7a002 point maps (layer_masks on
    material_z, pads first). Checked by spying on the arrays each engine's kernel receives; the
    end-to-end bit-identity of R/r0/r_co against an exported 2b7a002 package was also measured on all
    seven engines (numpy backends; docs/audit/2026-11-02-fdtd-registration.md)."""
    import dynameta.optics.fdtd as F1
    import dynameta.optics.fdtd_nd.solve2d as S2
    import dynameta.optics.fdtd_nd.solve3d as S3
    from dynameta.optics.fdtd_nd import FDTDLayer
    from dynameta.optics.fdtd_nd.grid import layer_band_index_max, layer_masks, material_z, z_grid
    lay = [FDTDLayer(thickness_m=0.183e-6, eps_inf=4.0),
           FDTDLayer(thickness_m=0.071e-6, eps_inf=2.0, drude_wp_rad_s=1.2e15, drude_gamma_rad_s=1e14)]
    B = dict(lambda_min_m=L1, lambda_max_m=L2, resolution=12, n_pad_wave=3.0)

    def want(nmax, n_super=1.0):
        dz, pad, zs, _Lz, nz = z_grid(lay, L1, L2, 12, 3.0, nmax)
        e = np.ones(nz); zc = material_z(nz, dz)
        e[zc < pad] = n_super ** 2
        for L, m in layer_masks(zc, pad, lay):
            e[m] = L.eps_inf
        return e
    seen = []
    real1 = F1._run
    monkeypatch.setattr(F1, "_run", lambda e, *a, **k: (seen.append(np.array(e)), real1(e, *a, **k))[1])
    F1.solve_fdtd_1d(lay, z_subcell="point", **B)
    n1 = F1._grid_metrics(lay, L1, L2, 12, 3.0)[5]
    assert seen[-1].tobytes() == want(n1).tobytes()
    real2 = S2._dispatch_2d_te
    monkeypatch.setattr(S2, "_dispatch_2d_te",
                        lambda n, e, *a, **k: (seen.append(np.array(e)), real2(n, e, *a, **k))[1])
    S2.solve_fdtd_2d(lay, period_x_m=0.1e-6, nx=2, z_subcell="point", **B)
    wb = 2 * np.pi * np.linspace(C_LIGHT / L2, C_LIGHT / L1, 9)
    n2 = layer_band_index_max(lay, wb, (1.0, 1.0))
    assert seen[-1][0].tobytes() == want(n2).tobytes()
    real3 = S3._dispatch_3d
    monkeypatch.setattr(S3, "_dispatch_3d",
                        lambda n, e, *a, **k: (seen.append(np.array(e)), real3(n, e, *a, **k))[1])
    S3.solve_fdtd_3d(lay, period_x_m=0.1e-6, period_y_m=0.1e-6, nx=2, ny=2, z_subcell="point", **B)
    assert seen[-1][0, 0].tobytes() == want(n2).tobytes()
    # and the DEFAULT really differs at the faces (a mixed cell exists) while matching in the bulk
    S2.solve_fdtd_2d(lay, period_x_m=0.1e-6, nx=2, **B)
    avg, pt = seen[-1][0], want(n2)
    assert np.any(avg != pt) and np.sum(avg != pt) <= 2 * (len(lay) + 1)


def test_z_subcell_rejects_unknown_mode():
    from dynameta.optics.fdtd import solve_fdtd_1d
    with pytest.raises(ValueError, match="z_subcell"):
        solve_fdtd_1d(_layer(), lambda_min_m=L1, lambda_max_m=L2, resolution=8, z_subcell="smooth")


# ---- true-geometry gates: |R|, |T| and the r phase at EVERY sub-cell offset vs exact TMM of the TRUE
# slab (d = 183 nm: 9.15 cells at res 30, NOT commensurate). Measured 2026-11-02 at 8 offsets, res 30
# (res 60) -- point (2b7a002): 1-D max|dR| 7.9e-3 (2.3e-3), max|dphi| 0.073 (0.037) rad; average:
# 1.9e-3 (3.9e-4), 0.0049 (0.0012) rad -- a ~4.8x / 4.1x drop per dz halving, O((k dz)^2). Bars:
# |dR|, |dT| < 3e-3 and |dphi| < 0.1 k0dz per offset at res 30 (point fails both by 2.6x / 8x).
# _ZSUB lets the whole battery be re-run on the point rule to prove each gate fails there:
#   DYNAMETA_TEST_ZSUBCELL=point pytest tests/test_fdtd_registration.py -m "slow or not slow" -k true_geom
import os  # noqa: E402

_ZSUB = os.environ.get("DYNAMETA_TEST_ZSUBCELL", "average")
DT = 0.183e-6
RT_BAR = 3.0e-3
PH_BAR = 0.10                      # x k0dz, per offset


def _rt_true(eps, theta=0.0, pol="s", n_in=1.0):
    """(r, R, T) of the TRUE slab DT (tangential-E characteristic matrix; same in/out media for T)."""
    k0 = 2.0 * np.pi / LAM0
    w = k0 * C_LIGHT
    kx = k0 * n_in * np.sin(np.radians(theta))

    def adm(e):
        kz = np.sqrt(complex(e) * k0 ** 2 - kx ** 2)
        kz = -kz if kz.imag < 0 else kz
        return (kz / (w * MU0)) if pol == "s" else (w * EPS0 * e / kz), kz
    Y, kz = adm(eps)
    c, s = np.cos(kz * DT), np.sin(kz * DT)
    M = np.array([[c, -1j * s / Y], [-1j * Y * s, c]])
    Y0, Ys = adm(n_in ** 2)[0], adm(1.0)[0]
    B, Cc = M[0, 0] + M[0, 1] * Ys, M[1, 0] + M[1, 1] * Ys
    r, t = (Y0 * B - Cc) / (Y0 * B + Cc), 2 * Y0 / (Y0 * B + Cc)
    return r, abs(r) ** 2, abs(t) ** 2 * Ys.real / Y0.real


def _offs(res, n, nmax=NS):
    dz = L1 / (res * nmax)
    m = round(3.0 * L2 / dz)
    return [(m + (k + 0.5) / n) * dz / L2 for k in range(n)]


def _true_geom_1d(res, mode, n_off=8):
    from dynameta.optics.fdtd import _grid_metrics, solve_fdtd_1d
    from dynameta.optics.fdtd_nd import FDTDLayer
    from dynameta.optics.fdtd_nd.grid import probe_cells
    lay = [FDTDLayer(thickness_m=DT, eps_inf=NS ** 2)]
    r_, R_, T_ = _rt_true(NS ** 2)
    out = []
    for npw in _offs(res, n_off):
        r = solve_fdtd_1d(lay, lambda_min_m=L1, lambda_max_m=L2, resolution=res, n_pad_wave=npw,
                          return_time_trace=True, z_subcell=mode)
        tt = r.time_trace
        dz, pad, zs, _Lz, _nz, *_ = _grid_metrics(lay, L1, L2, res, npw)
        f = np.fft.rfftfreq(tt["t"].size, tt["dt"])
        with np.errstate(divide="ignore", invalid="ignore"):
            rr = np.conj(np.fft.rfft(tt["reflected"]) / np.fft.rfft(tt["incident_left"]))
        rc = _at(f, rr) * np.exp(-2j * K0 * (pad - probe_cells(pad, zs, dz)[1] * dz))
        out.append((float(np.interp(FI, r.freqs_Hz, r.R)) - R_, float(np.interp(FI, r.freqs_Hz, r.T)) - T_,
                    float(np.angle(rc / r_))))
    return np.abs(np.asarray(out)).max(axis=0)


def test_true_geom_1d_matches_tmm_at_every_offset_and_converges():
    e30 = _true_geom_1d(30, _ZSUB)
    e60 = _true_geom_1d(60, _ZSUB)
    assert e30[0] < RT_BAR and e30[1] < RT_BAR, e30
    assert e30[2] < PH_BAR * _k0dz(30), e30
    # O((k dz)^2): measured drop 4.8x (|R|), 4.8x (|T|), 4.1x (phase) per halving (point: 3.4x / 2.0x)
    assert e30[0] / e60[0] > 3.0 and e30[1] / e60[1] > 3.0 and e30[2] / e60[2] > 3.0, (e30, e60)


def test_true_geom_1d_point_rule_fails_the_same_bars():
    """Discrimination inside the suite: the 2b7a002 point rule misses both bars at res 30."""
    e = _true_geom_1d(30, "point")
    assert e[0] > RT_BAR and e[2] > PH_BAR * _k0dz(30), e


def _eps_true(L, w=2.0 * np.pi * FI):
    """eps(w) of an FDTDLayer INCLUDING the gain line (== a Lorentz pole of strength
    d_eps = -kappa dN / (eps0 w_a^2) on the kernel's own recursion)."""
    e = L.eps_at(w)
    if L.gain_dN_m3 != 0.0:
        wa, dw = L.gain_w_rad_s, L.gain_dw_rad_s
        e = e + (-L.gain_kappa_C2_kg * L.gain_dN_m3 / (EPS0 * wa ** 2)) * wa ** 2 / (wa ** 2 - w ** 2 - 1j * dw * w)
    return e


def _layer_kind(kind):
    from dynameta.optics.fdtd_nd import FDTDLayer
    W0 = 2.0 * np.pi * FI
    kap = (1.602e-19) ** 2 / 9.109e-31
    return {"diel": FDTDLayer(thickness_m=DT, eps_inf=NS ** 2),
            "dl": FDTDLayer(thickness_m=DT, eps_inf=3.0, drude_wp_rad_s=0.9e15, drude_gamma_rad_s=8e13,
                            lorentz_w0_rad_s=1.25e15, lorentz_gamma_rad_s=1.5e14, lorentz_delta_eps=0.6),
            "dl_vac": FDTDLayer(thickness_m=DT, eps_inf=1.0, drude_wp_rad_s=0.9e15, drude_gamma_rad_s=8e13,
                                lorentz_w0_rad_s=1.25e15, lorentz_gamma_rad_s=1.5e14, lorentz_delta_eps=1.5),
            "gain_vac": FDTDLayer(thickness_m=DT, eps_inf=1.0, gain_w_rad_s=W0, gain_dw_rad_s=1.5e14,
                                  gain_kappa_C2_kg=kap, gain_dN_m3=2e25)}[kind]


def _nmax(lay, ends):
    from dynameta.optics.fdtd_nd.grid import layer_band_index_max
    return layer_band_index_max(lay, 2 * np.pi * np.linspace(C_LIGHT / L2, C_LIGHT / L1, 9), ends)


def _true_geom_2d(kind, mode, res=30, n_off=4, n_super=1.0):
    from dynameta.optics.fdtd_nd import solve_fdtd_2d
    lay = [_layer_kind(kind)]
    r_, R_, T_ = _rt_true(_eps_true(lay[0]), n_in=n_super)
    out = []
    for npw in _offs(res, n_off, _nmax(lay, (n_super, 1.0))):
        r = solve_fdtd_2d(lay, period_x_m=0.1e-6, nx=2, lambda_min_m=L1, lambda_max_m=L2, resolution=res,
                          n_pad_wave=npw, n_super=n_super, z_subcell=mode)
        out.append((float(np.interp(FI, r.freqs_Hz, r.R0)) - R_, float(np.interp(FI, r.freqs_Hz, r.T0)) - T_,
                    float(np.angle(_at(r.freqs_Hz, r.r0) / r_))))
    return np.abs(np.asarray(out)).max(axis=0)


@pytest.mark.slow
@pytest.mark.parametrize("kind, n_super, rt_bar, ph_bar", [
    ("diel", 1.0, RT_BAR, PH_BAR),        # measured avg 2.3e-3 / 0.06 k0dz; point 8.4e-3 / 0.81 k0dz
    ("diel", 1.5, RT_BAR, 0.25),          # glass: avg 1.7e-3 / 0.18 k0dz (n_super-scaled dispersion);
                                          #        point 4.6e-3 / 1.15 k0dz
    ("dl", 1.0, 1.5e-3, PH_BAR),          # Drude+Lorentz on eps_inf 3: avg 6.1e-4 / 0.03; point 2.1e-2 / 0.62
    # index-matched (eps_inf 1) slabs: ONLY the pole/gain STRENGTH fill weighting is tested. |r| is small
    # there (weak contrast), so the r phase is ill-conditioned and only |R|, |T| are gated:
    ("dl_vac", 1.0, 1.5e-3, None),        # ACCURACY only at res 30: avg max|dR| 8.1e-4 (res 60 2.2e-4,
                                          # 3.7x); point 3.3e-4 is offset luck (res 60: 2.2e-3, no
                                          # convergence) -- the dl / gain_vac rows are the discriminators
    ("gain_vac", 1.0, 3.0e-3, None),      # avg max|dT| 1.6e-3 (res 60: 2.3e-4); point 4.4e-2
])
def test_true_geom_2d_matches_tmm_at_every_offset(kind, n_super, rt_bar, ph_bar):
    e = _true_geom_2d(kind, _ZSUB, n_super=n_super)
    assert e[0] < rt_bar and e[1] < rt_bar, (kind, e)
    if ph_bar is not None:
        assert e[2] < ph_bar * _k0dz(30), (kind, e)


@pytest.mark.slow
@pytest.mark.parametrize("pol, angle", [("s", 30.0), ("p", 30.0), ("p", 50.0)])
def test_true_geom_2d_oblique_tangential_and_harmonic_channels(pol, angle, monkeypatch):
    """s: arithmetic Ey; p: arithmetic Ex + HARMONIC Ez. Measured 8 offsets, res 30 -- average:
    max|dR| 2.1e-3 / 2.2e-3 / 1.7e-3, max|dphi| 0.04 / 0.04 / 0.02 k0dz; point: 4.6e-3 / 4.3e-3 /
    2.7e-3 and 0.71 / 1.08 / 2.21 k0dz (the phase bar is the discriminating one at 50 deg)."""
    pytest.importorskip("numba")
    res = 30
    out = []
    for npw in _offs(res, 4):
        r, th, _ = _obl_front(pol, angle, res, npw, "numba", monkeypatch, z_subcell=_ZSUB,
                              layers=[_layer_kind("diel")])
        rs = _obl_front.last
        r_, R_, T_ = _rt_true(NS ** 2, th, pol)
        out.append((float(np.interp(FI, rs.freqs_Hz, rs.R0)) - R_, float(np.interp(FI, rs.freqs_Hz, rs.T0)) - T_,
                    float(np.angle(r / r_))))
    e = np.abs(np.asarray(out)).max(axis=0)
    assert e[0] < RT_BAR and e[1] < RT_BAR, e
    assert e[2] < PH_BAR * _k0dz(res), e


@pytest.mark.slow
def test_true_geom_3d_and_3d_mo():
    """solve_fdtd_3d and solve_fdtd_3d_mo, laterally uniform. Measured 8 offsets, res 30: average
    max|dR| 2.3e-3, max|dphi| 0.06 k0dz (both); point 8.4e-3, 0.81 k0dz."""
    pytest.importorskip("numba")
    from dynameta.optics.fdtd_mo import MOLayer
    from dynameta.optics.fdtd_nd import solve_fdtd_3d, solve_fdtd_3d_mo
    from dynameta.optics.fdtd_nd.grid import probe_cells, z_grid
    res = 30
    r_, R_, T_ = _rt_true(NS ** 2)
    o3, omo = [], []
    for npw in _offs(res, 4):
        kw = dict(period_x_m=0.1e-6, period_y_m=0.1e-6, nx=2, ny=2, lambda_min_m=L1, lambda_max_m=L2,
                  resolution=res, n_pad_wave=npw, z_subcell=_ZSUB)
        r = solve_fdtd_3d([_layer_kind("diel")], backend="numba", **kw)
        o3.append((float(np.interp(FI, r.freqs_Hz, r.R0)) - R_, float(np.interp(FI, r.freqs_Hz, r.T0)) - T_,
                   float(np.angle(_at(r.freqs_Hz, r.r0) / r_))))
        lay = [MOLayer(thickness_m=DT, eps_xx=NS ** 2, eps_yy=NS ** 2)]
        rm = solve_fdtd_3d_mo(lay, **kw)
        dz, pad, zs, _Lz, _nz = z_grid(lay, L1, L2, res, npw, NS)
        rc = _at(rm.freqs_Hz, rm.r_co) * np.exp(-2j * K0 * (pad - probe_cells(pad, zs, dz)[1] * dz))
        omo.append((float(np.interp(FI, rm.freqs_Hz, rm.R)) - R_, float(np.interp(FI, rm.freqs_Hz, rm.T)) - T_,
                    float(np.angle(rc / r_))))
    for o in (o3, omo):
        e = np.abs(np.asarray(o)).max(axis=0)
        assert e[0] < RT_BAR and e[1] < RT_BAR and e[2] < PH_BAR * _k0dz(res), e


@pytest.mark.slow
def test_true_geom_ppol_drude_slab_ez_fallback(monkeypatch):
    """p-pol 50 deg on a DRUDE slab: the Ez cells that mix vacuum and Drude have no exact harmonic ADE
    and fall back to point sampling (fdtd_nd.grid.mix_profile); Ex is still arithmetic-averaged.
    Measured 8 offsets, res 30: average max|dR| 1.7e-3, max|dT| 1.1e-3 (res 60: 2.4e-4 / 5.6e-4);
    point 4.6e-3 / 4.1e-3. The r phase is NOT gated: |r_p| is small near this slab's Brewster angle
    (eps 2.55 -> 58 deg), where the phase is ill-conditioned."""
    pytest.importorskip("numba")
    from dynameta.optics.fdtd_nd import FDTDLayer
    lay = [FDTDLayer(thickness_m=DT, eps_inf=3.0, drude_wp_rad_s=0.9e15, drude_gamma_rad_s=8e13)]
    res = 30
    out = []
    for npw in _offs(res, 4, _nmax(lay, ())):
        r, th, _ = _obl_front("p", 50.0, res, npw, "numba", monkeypatch, z_subcell=_ZSUB, layers=lay)
        rs = _obl_front.last
        r_, R_, T_ = _rt_true(_eps_true(lay[0]), th, "p")
        out.append((float(np.interp(FI, rs.freqs_Hz, rs.R0)) - R_, float(np.interp(FI, rs.freqs_Hz, rs.T0)) - T_))
    e = np.abs(np.asarray(out)).max(axis=0)
    assert e[0] < RT_BAR and e[1] < RT_BAR, e
