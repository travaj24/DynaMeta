"""Shared z-grid sizing, probe placement and source-pulse builders for every FDTD front end.

audit D-15 (2026-11-02): the source-envelope / time-window, probe-placement and z-grid-sizing
blocks were re-typed verbatim in SEVEN front ends -- solve_fdtd_1d (fdtd.py), solve_fdtd_mo_1d
(fdtd_mo.py), solve_fdtd_2d / solve_fdtd_2d_oblique (solve2d.py) and solve_fdtd_3d /
solve_fdtd_3d_oblique / solve_fdtd_3d_mo (solve3d.py) -- so audit D-3 (probe guard), D-10 (window
memory) and D-11 (CFL guard) each needed the same edit in 4-8 places, and two of the copies had
already drifted in spelling (`4.0 * Lz / C_LIGHT` vs `(Lz / C_LIGHT) * 4.0`, `0.35 * pad / dz` vs
`(0.35 * pad) / dz`). The window-memory and CFL halves were single-sourced in wave 5
(`solve2d._window_memory_s`, `spec.courant_guard`); these three helpers close the rest. Each one
is the old inline arithmetic operand-for-operand, so every front end is BIT-IDENTICAL to the
pre-refactor code (checked on all seven, 204 result arrays; the two drifted spellings are the
same float because x4 is exact in binary floating point). The per-front-end sizing INDEX stays a
parameter (`n_max`): the 1-D band-edge Drude index, the 2-D/3-D `eps_at` band maximum, the
oblique copies that omit the end media (audit X-18) and the MO circular-branch bound genuinely
differ and are not unified here.
"""
from __future__ import annotations

import numpy as np

from dynameta.constants import C_LIGHT


def z_grid(layers, lambda_min_m, lambda_max_m, resolution, n_pad_wave, n_max):
    """(dz, pad, z_struct, Lz, nz) of the uniform z grid: `resolution` cells per lambda_min in the
    index-`n_max` medium, `n_pad_wave` * lambda_max of pad each side of the stack."""
    dz = lambda_min_m / (resolution * n_max)
    pad = n_pad_wave * lambda_max_m
    z_struct = float(sum(L.thickness_m for L in layers))
    Lz = 2.0 * pad + z_struct
    nz = int(round(Lz / dz)) + 1
    return dz, pad, z_struct, Lz, nz


def layer_band_index_max(layers, w_band, n_ends=()):
    """max(1, *n_ends, max over layers and w_band of |sqrt(L.eps_at(w))|) -- the grid-sizing index
    of the 2-D/3-D front ends (the oblique ones pass no `n_ends`; see the module note)."""
    return max(1.0, *n_ends, max(max(abs(np.sqrt(L.eps_at(w))) for w in w_band) for L in layers))


def probe_cells(pad, z_struct, dz):
    """(k_src, k_pL, k_pR): the soft source at 0.35 pad, the reflection probe at 0.7 pad (between
    source and structure) and the transmission probe 0.3 pad past the structure. These are pad
    FRACTIONS against a CELL-count CPML -- the reason `solve2d._check_probe_placement` exists."""
    k_src = max(2, int(round((0.35 * pad) / dz)))
    k_pL = int(round((0.7 * pad) / dz))
    k_pR = int(round((pad + z_struct + 0.3 * pad) / dz))
    return k_src, k_pL, k_pR


def pulse_width(f_min, f_max, settle):
    """(tau, t0) of the modulated-Gaussian source covering [f_min, f_max]: tau ~ the inverse
    bandwidth, the centre `settle` pulse widths into the run."""
    tau = 1.0 / (np.pi * (f_max - f_min))
    return tau, settle * tau


def source_pulse(f_c, tau, t0, *, source_amp, Lz, dt, t_ring=0.0, t_end_s=None, tail_tau_s=None):
    """(nsteps, tgrid, src): the run length and the soft-source waveform
    source_amp exp(-((t - t0)/tau)^2) cos(2 pi f_c (t - t0)).

    Run length = the pulse (2 t0) + four box transits + a 200 tau tail + `t_ring` (the material
    memory from `solve2d._window_memory_s`; 0 for the 1-D engines). `t_end_s` / `tail_tau_s`
    (default None = the single-pulse window, bit-identical) let a caller that injects a SECOND,
    later or longer pulse extend the window to cover it: the pulse term becomes max(2 t0, t_end_s)
    and the tail 200 max(tau, tail_tau_s) -- solve_fdtd_1d's bichromatic `second_source`, audit
    D-19."""
    t_pulse = 2.0 * t0 if t_end_s is None else max(2.0 * t0, float(t_end_s))
    tail = 200 * tau if tail_tau_s is None else 200 * max(tau, float(tail_tau_s))
    nsteps = int(round((t_pulse + (Lz / C_LIGHT) * 4.0 + tail + t_ring) / dt))
    tgrid = np.arange(nsteps) * dt
    src = source_amp * np.exp(-((tgrid - t0) / tau) ** 2) * np.cos(2.0 * np.pi * f_c * (tgrid - t0))
    return nsteps, tgrid, src
