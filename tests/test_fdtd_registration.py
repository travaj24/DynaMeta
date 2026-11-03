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
    fm.solve_fdtd_mo_1d(lay, lambda_min_m=L1, lambda_max_m=L2, resolution=12, n_pad_wave=3.0)
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


def _obl_front(pol, angle, res, npw, backend, monkeypatch):
    """r at LAM0 through the PUBLIC solve_fdtd_2d_oblique (its own material fill), by capturing the
    kernel probe lines it computes (the front end itself returns |R0|/|T0| only)."""
    import dynameta.optics.fdtd_nd.solve2d as S2
    from dynameta.optics.fdtd_nd.grid import probe_cells, z_grid
    cap = []
    real = S2._run_oblique

    def spy(*a, **k):
        out = real(*a, **k)
        cap.append((a, k, out))
        return out
    monkeypatch.setattr(S2, "_run_oblique", spy)
    S2.solve_fdtd_2d_oblique(_layer(), period_x_m=0.1e-6, angle_deg=angle, lambda_min_m=L1,
                             lambda_max_m=L2, resolution=res, n_pad_wave=npw, nx=4, backend=backend,
                             pol=pol)
    monkeypatch.setattr(S2, "_run_oblique", real)
    (a_i, _k_i, out_i), (a_t, k_t, out_t) = cap
    dz, pad, zs, _Lz, _nz = z_grid(_layer(), L1, L2, res, npw, NS)
    k_pL = probe_cells(pad, zs, dz)[1]
    dt, nsteps, kx = a_i[6], a_i[7], a_i[13]
    nf = nsteps // 2 + 1
    f = np.fft.rfftfreq(nsteps, dt)
    mean = (lambda x: np.fft.fft(x.mean(axis=1))[:nf])
    with np.errstate(divide="ignore", invalid="ignore"):
        rr = np.conj(mean(out_t[0] - out_i[0]) / mean(out_i[0]))
    th = float(np.degrees(np.arcsin(kx * C_LIGHT / (2.0 * np.pi * FI))))
    r = _at(f, rr) * np.exp(-2j * K0 * np.cos(np.radians(th)) * (pad - k_pL * dz))
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
        r, th, (a_t, k_t) = _obl_front("p", angle, res, npw, "numba", monkeypatch)
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
