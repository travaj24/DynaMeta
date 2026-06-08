"""Plotting helpers for DynaMeta results (matplotlib, an OPTIONAL dependency). Turns a SweepResults into
the figures you actually look at -- the spectrum per bias, the OFF/ON modulation-contrast spectrum, and a
generic 2-D field/eps/density map -- so iterating on a design is interactive instead of hand-extracting
OpticalResult arrays. Every function takes an optional `ax` (compose into your own figure) and an optional
`save` path (write a PNG/PDF/SVG, headless-safe). matplotlib is imported lazily with a clear error."""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np


def _plt():
    try:
        import matplotlib.pyplot as plt
        return plt
    except ImportError as e:                                # pragma: no cover - optional dep
        raise ImportError("dynameta.viz needs the optional 'matplotlib' package "
                          "(pip install dynameta[plot] or pip install matplotlib).") from e


def _ax(ax):
    if ax is not None:
        return ax.figure, ax
    plt = _plt()
    return plt.subplots(figsize=(6.4, 4.0))


def _finish(fig, save):
    if save:
        fig.tight_layout(); fig.savefig(save, dpi=140)
    return fig


def plot_spectra(sweep, metric: str = "R", *, biases: Optional[Sequence[str]] = None, ax=None,
                 save: Optional[str] = None):
    """metric(lambda) as one line per bias (R/T/A/R_flux/...). Returns the Axes."""
    fig, ax = _ax(ax)
    labels = list(biases) if biases is not None else sweep.bias_labels
    data = sweep.fields[metric]
    for lab in labels:
        ax.plot(sweep.wavelengths_nm, data[sweep.bias_labels.index(lab)], marker="o", ms=3, label=str(lab))
    ax.set_xlabel("wavelength (nm)"); ax.set_ylabel(metric)
    ax.set_title("{} spectrum".format(metric)); ax.legend(title="bias", fontsize=8)
    ax.grid(True, alpha=0.3)
    _finish(fig, save)
    return ax


def plot_contrast(sweep, metric: str = "R", *, ref: Optional[str] = None, ax=None, save: Optional[str] = None):
    """The modulation-contrast spectrum |metric(bias) - metric(ref)|(lambda), one line per (non-ref) bias --
    the OFF/ON modulation depth vs wavelength. ref defaults to the first bias."""
    fig, ax = _ax(ax)
    c = sweep.contrast(metric, ref)
    ref_lab = ref if ref is not None else sweep.bias_labels[0]
    for i, lab in enumerate(sweep.bias_labels):
        if lab == ref_lab:
            continue
        ax.plot(sweep.wavelengths_nm, c[i], marker="o", ms=3, label="{} vs {}".format(lab, ref_lab))
    ax.set_xlabel("wavelength (nm)"); ax.set_ylabel("|delta {}|".format(metric))
    ax.set_title("modulation contrast ({})".format(metric)); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    _finish(fig, save)
    return ax


def plot_map(array2d, *, x: Optional[np.ndarray] = None, y: Optional[np.ndarray] = None, ax=None,
             save: Optional[str] = None, cmap: str = "viridis", title: str = "", cbar_label: str = ""):
    """A generic 2-D heatmap (an eps / carrier-density / field map). array2d is (ny, nx); x,y are the axis
    coordinates (else pixel indices). Returns the Axes."""
    fig, ax = _ax(ax)
    a = np.asarray(array2d)
    if x is not None and y is not None:
        im = ax.pcolormesh(np.asarray(x), np.asarray(y), a, cmap=cmap, shading="auto")
    else:
        im = ax.imshow(a, origin="lower", aspect="auto", cmap=cmap)
    cb = fig.colorbar(im, ax=ax)
    if cbar_label:
        cb.set_label(cbar_label)
    if title:
        ax.set_title(title)
    _finish(fig, save)
    return ax


def plot_sweep_summary(sweep, *, metric: str = "R", ref: Optional[str] = None, save: Optional[str] = None):
    """A two-panel figure: the metric spectrum per bias + the modulation-contrast spectrum. Returns the
    Figure (the at-a-glance view of a bias x wavelength sweep)."""
    plt = _plt()
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.0))
    plot_spectra(sweep, metric, ax=axes[0])
    plot_contrast(sweep, metric, ref=ref, ax=axes[1])
    if save:
        fig.tight_layout(); fig.savefig(save, dpi=140)
    return fig
