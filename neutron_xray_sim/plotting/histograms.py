"""
neutron_xray_sim.plotting.histograms
────────────────────────────────────
Figures for bimodal histograms.

* :func:`plot_bimodal_histogram`        — one histogram with marginals / GMM ellipses
* :func:`plot_comparison_grid`          — grid of histograms (e.g. one per artifact)
* :func:`plot_ground_truth_comparison`  — phantom ground truth vs reconstruction
* :func:`plot_cross_algorithm_grid`     — one panel per (X-ray, neutron) algorithm pair
* :func:`plot_artifact_survey`          — figure for ``run_artifact_survey``

All functions return the matplotlib ``Figure`` and never change global
matplotlib state.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse

from ..analysis.gmm import GMMFitResult
from ..analysis.histogram import DEFAULT_GT_ENERGY_IDX, HistogramResult

__all__ = [
    "plot_bimodal_histogram",
    "plot_comparison_grid",
    "plot_ground_truth_comparison",
    "plot_cross_algorithm_grid",
    "plot_artifact_survey",
]


# ──────────────────────────────────────────────────────────────────────────────
# Single histogram and grids
# ──────────────────────────────────────────────────────────────────────────────

def plot_bimodal_histogram(
    hist: HistogramResult,
    gmm: Optional[GMMFitResult] = None,
    material_labels: Optional[Dict[int, str]] = None,
    title: str = "Bimodal Histogram",
    ax: Optional[plt.Axes] = None,
    log_scale: bool = True,
    cmap: str = "cool",
    show_gmm_ellipses: bool = True,
    show_marginals: bool = True,
    figsize: Tuple[float, float] = (8, 7),
) -> plt.Figure:
    """
    Publication-quality bimodal histogram figure.

    Parameters
    ----------
    hist              : HistogramResult
    gmm               : optional GMMFitResult (overlays ellipses)
    material_labels   : optional dict {component_idx: 'material name'}
    title             : figure title
    ax                : existing Axes to plot into; None = create new figure
    log_scale         : use log colour scale on the 2-D panel (linear x/y axes)
    cmap              : matplotlib colourmap name
    show_gmm_ellipses : overlay 2-σ GMM ellipses
    show_marginals    : show marginal distributions with log-count axes
    figsize           : figure size

    Returns
    -------
    matplotlib Figure
    """
    if ax is None:
        if show_marginals:
            # Three-column GridSpec:
            #   col 0 — top marginal (μ_x, log counts)  + 2-D histogram
            #   col 1 — right marginal (μ_n, log counts)
            #   col 2 — colorbar (narrow)
            # Using constrained_layout avoids the tight_layout / sharex
            # misalignment that makes ax_top wider than ax_main.
            fig = plt.figure(figsize=figsize, constrained_layout=True)
            gs  = fig.add_gridspec(
                2, 3,
                width_ratios=[4, 1, 0.18],
                height_ratios=[1, 4],
                hspace=0.0,
                wspace=0.0,
            )
            ax_main  = fig.add_subplot(gs[1, 0])
            ax_top   = fig.add_subplot(gs[0, 0], sharex=ax_main)
            ax_right = fig.add_subplot(gs[1, 1], sharey=ax_main)
            ax_cbar  = fig.add_subplot(gs[1, 2])   # dedicated colorbar column
        else:
            fig, ax_main = plt.subplots(figsize=figsize, constrained_layout=True)
            ax_top = ax_right = ax_cbar = None
    else:
        ax_main  = ax
        fig      = ax.figure
        ax_top = ax_right = ax_cbar = None
        show_marginals = False

    H  = hist.H.T      # transpose: μ_x on horizontal, μ_n on vertical
    extent = [hist.x_edges[0], hist.x_edges[-1],
              hist.n_edges[0], hist.n_edges[-1]]

    # ── 2-D histogram (linear x and y axes; optional log colour scale) ─────────
    H_plot = np.log1p(H) if log_scale else H
    H_plot = np.ma.masked_where(H == 0, H_plot)

    im = ax_main.imshow(
        H_plot,
        origin="lower",
        extent=extent,
        aspect="auto",
        cmap=cmap,
        interpolation="bilinear",
    )

    ax_main.set_xlabel(r"$\mu_x$ [cm$^{-1}$]", fontsize=12)
    ax_main.set_ylabel(r"$\mu_n$ [cm$^{-1}$]", fontsize=12)
    ax_main.set_title(title, fontsize=13)

    # Colorbar: dedicated narrow column when marginals are shown, else float beside main.
    if ax_cbar is not None:
        cbar = plt.colorbar(im, cax=ax_cbar)
    else:
        cbar = plt.colorbar(im, ax=ax_main, fraction=0.046, pad=0.04)
    cbar.set_label("log(1 + counts)" if log_scale else "counts", fontsize=10)

    # ── GMM ellipses ──────────────────────────────────────────────────────────
    if gmm is not None and show_gmm_ellipses:
        colours = plt.cm.Set1(np.linspace(0, 1, gmm.n_components))
        for k in range(gmm.n_components):
            mu_x, mu_n = gmm.means[k]
            cov        = gmm.covariances[k]
            eigvals, eigvecs = np.linalg.eigh(cov)
            angle  = np.degrees(np.arctan2(eigvecs[1, 0], eigvecs[0, 0]))
            width  = 2 * 2.0 * np.sqrt(eigvals[0])  # 2-σ ellipse
            height = 2 * 2.0 * np.sqrt(eigvals[1])
            ellipse = Ellipse(
                (mu_x, mu_n), width=width, height=height,
                angle=angle,
                edgecolor=colours[k], facecolor="none",
                linewidth=1.8, linestyle="--",
            )
            ax_main.add_patch(ellipse)
            ax_main.plot(mu_x, mu_n, "+", color=colours[k],
                         markersize=8, markeredgewidth=2)

            label = (material_labels or {}).get(k, f"#{k}")
            ax_main.annotate(
                label, (mu_x, mu_n),
                xytext=(4, 4), textcoords="offset points",
                fontsize=9, color=colours[k],
                fontweight="bold",
            )

    # ── Marginals (log-count axes; linear position axes) ─────────────────────
    # The 2-D histogram uses linear spatial axes (μ_x, μ_n in cm⁻¹).
    # The *counts* axis of each marginal is log-scaled so that small clusters
    # (water, Fe, Ti) remain visible under the dominant air/HDPE peak.
    # A floor of 0.5 prevents log(0) on empty bins.
    if ax_top is not None:
        x_marg      = hist.H.sum(axis=1).astype(float)   # (bins_x,)
        x_marg_plot = np.maximum(x_marg, 0.5)

        ax_top.fill_between(
            hist.x_centres, 0.5, x_marg_plot,
            step="mid", alpha=0.65, color="steelblue",
        )
        ax_top.step(
            hist.x_centres, x_marg_plot,
            where="mid", color="steelblue", linewidth=0.8, alpha=0.9,
        )
        ax_top.set_yscale("log")
        ax_top.set_ylabel("counts", fontsize=9)
        ax_top.tick_params(labelbottom=False)
        # xlim is locked by sharex — do not call set_xlim here
        ax_top.spines["top"].set_visible(False)
        ax_top.spines["right"].set_visible(False)

    if ax_right is not None:
        n_marg      = hist.H.sum(axis=0).astype(float)   # (bins_n,)
        n_marg_plot = np.maximum(n_marg, 0.5)

        ax_right.fill_betweenx(
            hist.n_centres, 0.5, n_marg_plot,
            step="mid", alpha=0.65, color="tomato",
        )
        ax_right.step(
            n_marg_plot, hist.n_centres,
            where="mid", color="tomato", linewidth=0.8, alpha=0.9,
        )
        ax_right.set_xscale("log")
        ax_right.set_xlabel("counts", fontsize=9)
        ax_right.tick_params(labelleft=False)
        # ylim is locked by sharey — do not call set_ylim here
        ax_right.spines["top"].set_visible(False)
        ax_right.spines["right"].set_visible(False)

    # Hide the empty top-right corner cell and the top colorbar cell
    if show_marginals and ax_cbar is not None:
        fig.add_subplot(gs[0, 1]).set_visible(False)
        fig.add_subplot(gs[0, 2]).set_visible(False)

    return fig


def plot_comparison_grid(
    results: List[Tuple[str, HistogramResult]],
    gmm_results: Optional[List[Optional[GMMFitResult]]] = None,
    ncols: int = 3,
    figsize_per_panel: Tuple[float, float] = (4.5, 4.0),
    log_scale: bool = True,
    cmap: str = "cool",
    suptitle: str = "Artifact Comparison",
) -> plt.Figure:
    """
    Plot a grid of bimodal histograms for easy artifact comparison.

    Parameters
    ----------
    results           : list of (title, HistogramResult) pairs
    gmm_results       : optional list of GMMFitResult for each panel
    ncols             : columns in the grid
    figsize_per_panel : size of each sub-panel
    log_scale         : log colour scale
    cmap              : colourmap
    suptitle          : overall figure title

    Returns
    -------
    matplotlib Figure
    """
    n     = len(results)
    nrows = int(np.ceil(n / ncols))
    fw    = figsize_per_panel[0] * ncols
    fh    = figsize_per_panel[1] * nrows + 0.6  # space for suptitle

    fig, axes = plt.subplots(nrows, ncols, figsize=(fw, fh))
    axes_flat = np.array(axes).ravel()

    gmm_list = gmm_results if gmm_results is not None else [None] * n

    for idx, ((title, hist), gmm_r) in enumerate(zip(results, gmm_list)):
        ax = axes_flat[idx]
        plot_bimodal_histogram(
            hist, gmm=gmm_r, title=title, ax=ax,
            log_scale=log_scale, cmap=cmap,
            show_marginals=False, show_gmm_ellipses=(gmm_r is not None),
        )

    # Hide unused axes
    for idx in range(n, len(axes_flat)):
        axes_flat[idx].set_visible(False)

    fig.suptitle(suptitle, fontsize=14, y=1.01)
    fig.tight_layout()
    return fig


# ──────────────────────────────────────────────────────────────────────────────
# Ground truth vs reconstruction
# ──────────────────────────────────────────────────────────────────────────────

def _gt_points(phantom, energy_idx: int):
    """(μ_x, μ_n) of every phantom material plus its voxel count."""
    mats = phantom.materials
    mu_x = np.array([m._mu_x_table[energy_idx] for m in mats])
    mu_n = np.array([m.mu_n for m in mats])
    counts = np.bincount(phantom.label_vol.ravel(), minlength=len(mats))[:len(mats)]
    return mu_x, mu_n, counts.astype(float)


def _shared_extent(hists: Sequence[HistogramResult], mu_x_gt, mu_n_gt) -> List[float]:
    """Axis limits covering every histogram and every ground-truth point."""
    x_max = max(max(h.x_edges[-1] for h in hists), float(np.max(mu_x_gt)) * 1.08)
    n_max = max(max(h.n_edges[-1] for h in hists), float(np.max(mu_n_gt)) * 1.08)
    return [0.0, x_max, 0.0, n_max]


def _draw_gt_scatter(
    ax,
    phantom,
    extent: List[float],
    energy_idx: int = DEFAULT_GT_ENERGY_IDX,
    fontsize: float = 8,
    size_legend: bool = True,
) -> None:
    """
    Ground-truth "bubble" panel: one labelled bubble per material at its exact
    (μ_x, μ_n), area ∝ √(voxel count) so minor phases stay visible.
    """
    mu_x, mu_n, counts = _gt_points(phantom, energy_idx)
    sqrt_c = np.sqrt(counts)
    s_min, s_range = sqrt_c.min(), sqrt_c.max() - sqrt_c.min() + 1e-9

    def size(c):
        return 60 + 550 * (np.sqrt(c) - s_min) / s_range

    colours = plt.cm.Set1(np.linspace(0, 0.9, len(mu_x)))
    x_max, n_max = extent[1], extent[3]

    ax.set_facecolor("#0d0d0d")
    ax.grid(True, color="#333333", linewidth=0.5, zorder=0)
    for m, mx, mn, c, col in zip(phantom.materials, mu_x, mu_n, counts, colours):
        ax.axvline(mx, color=col, linewidth=0.5, alpha=0.35, zorder=1)
        ax.axhline(mn, color=col, linewidth=0.5, alpha=0.35, zorder=1)
        ax.scatter(mx, mn, s=size(c), color=col, edgecolors="white",
                   linewidths=0.8, zorder=3, alpha=0.92)
        dx = 0.04 * x_max * (1 if mx < x_max * 0.6 else -1)
        dy = 0.04 * n_max * (1 if mn < n_max * 0.6 else -1)
        ax.annotate(
            f"{m.symbol}\n({mx:.3f}, {mn:.3f})",
            xy=(mx, mn), xytext=(mx + dx, mn + dy),
            fontsize=fontsize, color="white", fontweight="bold",
            ha="left" if dx > 0 else "right", va="bottom" if dy > 0 else "top",
            arrowprops=dict(arrowstyle="-", color=col, lw=0.8), zorder=4,
        )

    if size_legend:
        for c in (counts.min(), np.median(counts), counts.max()):
            ax.scatter([], [], s=size(c), color="grey", edgecolors="white",
                       linewidths=0.6, label=f"{int(c):,} vox", alpha=0.8)
        ax.legend(title="Bubble size", loc="lower right", fontsize=7,
                  title_fontsize=7, framealpha=0.3, labelcolor="white",
                  facecolor="#222222", edgecolor="#555555")

    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    ax.set_xlabel(r"$\mu_x$ [cm$^{-1}$]", fontsize=fontsize + 3)
    ax.set_ylabel(r"$\mu_n$ [cm$^{-1}$]", fontsize=fontsize + 3)
    ax.tick_params(colors="white", labelsize=fontsize)
    for spine in ax.spines.values():
        spine.set_edgecolor("#444444")


def _draw_recon_panel(
    subfig,
    hist: HistogramResult,
    title: str,
    log_scale: bool,
    cmap: str,
    show_marginals: bool,
    shared_extent: Optional[List[float]],
    gt_mu_x: Optional[np.ndarray] = None,
    gt_mu_n: Optional[np.ndarray] = None,
    materials: Optional[list] = None,
    fontsize: float = 11,
) -> None:
    """Reconstructed-histogram panel with optional marginals and GT markers."""
    if show_marginals:
        gs = subfig.add_gridspec(2, 3, width_ratios=[4, 1, 0.18],
                                 height_ratios=[1, 4], hspace=0.0, wspace=0.0)
        ax_main = subfig.add_subplot(gs[1, 0])
        ax_top = subfig.add_subplot(gs[0, 0], sharex=ax_main)
        ax_right = subfig.add_subplot(gs[1, 1], sharey=ax_main)
        ax_cbar = subfig.add_subplot(gs[1, 2])
        subfig.add_subplot(gs[0, 1]).set_visible(False)
        subfig.add_subplot(gs[0, 2]).set_visible(False)
    else:
        gs = subfig.add_gridspec(1, 2, width_ratios=[4, 0.18], wspace=0.0)
        ax_main = subfig.add_subplot(gs[0, 0])
        ax_cbar = subfig.add_subplot(gs[0, 1])
        ax_top = ax_right = None

    extent = shared_extent if shared_extent is not None else hist.extent
    H = hist.H.T
    H_plot = np.ma.masked_where(H == 0, np.log1p(H) if log_scale else H)
    im = ax_main.imshow(H_plot, origin="lower", extent=extent, aspect="auto",
                        cmap=cmap, interpolation="bilinear")
    ax_main.set_xlim(extent[0], extent[1])
    ax_main.set_ylim(extent[2], extent[3])
    ax_main.set_xlabel(r"$\mu_x$ [cm$^{-1}$]", fontsize=fontsize)
    ax_main.set_ylabel(r"$\mu_n$ [cm$^{-1}$]", fontsize=fontsize)
    ax_main.set_title(title, fontsize=fontsize + 1, linespacing=1.4)
    ax_main.tick_params(labelsize=fontsize - 3)

    cbar = plt.colorbar(im, cax=ax_cbar)
    cbar.set_label("log(1+counts)" if log_scale else "counts", fontsize=fontsize - 2)
    cbar.ax.tick_params(labelsize=fontsize - 4)

    if gt_mu_x is not None and gt_mu_n is not None:
        for i, (mx, mn) in enumerate(zip(gt_mu_x, gt_mu_n)):
            sym = materials[i].symbol if materials is not None else str(i)
            ax_main.plot(mx, mn, marker="D", color="black", markersize=6,
                         markeredgecolor="black", markeredgewidth=1.5, zorder=5)
            ax_main.annotate(sym, xy=(mx, mn), xytext=(4, 4),
                             textcoords="offset points", fontsize=fontsize - 4,
                             color="black", fontweight="bold", zorder=6)

    if ax_top is not None:
        x_plot = np.maximum(hist.H.sum(axis=1).astype(float), 0.5)
        ax_top.fill_between(hist.x_centres, 0.5, x_plot, step="mid",
                            alpha=0.65, color="steelblue")
        ax_top.step(hist.x_centres, x_plot, where="mid", color="steelblue",
                    linewidth=0.8, alpha=0.9)
        ax_top.set_yscale("log")
        ax_top.set_ylabel("counts", fontsize=fontsize - 3)
        ax_top.tick_params(labelbottom=False, labelsize=fontsize - 4)
        ax_top.set_xlim(extent[0], extent[1])
        ax_top.spines["top"].set_visible(False)
        ax_top.spines["right"].set_visible(False)

    if ax_right is not None:
        n_plot = np.maximum(hist.H.sum(axis=0).astype(float), 0.5)
        ax_right.fill_betweenx(hist.n_centres, 0.5, n_plot, step="mid",
                               alpha=0.65, color="tomato")
        ax_right.step(n_plot, hist.n_centres, where="mid", color="tomato",
                      linewidth=0.8, alpha=0.9)
        ax_right.set_xscale("log")
        ax_right.set_xlabel("counts", fontsize=fontsize - 3)
        ax_right.tick_params(labelleft=False, labelsize=fontsize - 4)
        ax_right.set_ylim(extent[2], extent[3])
        ax_right.spines["top"].set_visible(False)
        ax_right.spines["right"].set_visible(False)


def _metrics_suffix(metrics) -> str:
    """'\\nDB=…  CE=… cm⁻¹' for a ClusterQualityMetrics (empty if None)."""
    if metrics is None:
        return ""
    db, ce = metrics.davies_bouldin, metrics.mean_centroid_error
    db_s = f"{db:.3f}" if db == db else "n/a"
    ce_s = f"{ce:.4f}" if ce == ce else "n/a"
    return f"\nDB={db_s}  CE={ce_s} cm\u207b\u00b9"


def plot_ground_truth_comparison(
    phantom,
    hist_recon: HistogramResult,
    title_gt: str = "Ground Truth",
    title_recon: str = "Reconstructed",
    log_scale: bool = True,
    cmap: str = "cool",
    show_marginals: bool = True,
    figsize: Tuple[float, float] = (15, 7),
    suptitle: str = "Ground truth vs Reconstructed bimodal histogram",
    energy_idx: int = DEFAULT_GT_ENERGY_IDX,
) -> plt.Figure:
    """
    Side-by-side comparison: phantom ground truth (left) vs reconstruction (right).

    **Left** — one labelled bubble per material at its exact (μ_x, μ_n);
    bubble area ∝ √(voxel count).
    **Right** — the reconstructed histogram (log colour scale) with the
    ground-truth positions overlaid as diamonds, so shifts, smearing and
    partial-volume offsets are immediately visible.

    Both panels share identical axis limits.
    """
    mu_x, mu_n, _ = _gt_points(phantom, energy_idx)
    extent = _shared_extent([hist_recon], mu_x, mu_n)

    fig = plt.figure(figsize=figsize, constrained_layout=True)
    if suptitle:
        fig.suptitle(suptitle, fontsize=13)
    sf_gt, sf_recon = fig.subfigures(1, 2, wspace=0.06, width_ratios=[1, 1])

    ax_gt = sf_gt.add_subplot(1, 1, 1)
    _draw_gt_scatter(ax_gt, phantom, extent, energy_idx)
    ax_gt.set_title(title_gt, fontsize=12)

    _draw_recon_panel(sf_recon, hist_recon, title_recon, log_scale=log_scale,
                      cmap=cmap, show_marginals=show_marginals,
                      shared_extent=extent, gt_mu_x=mu_x, gt_mu_n=mu_n,
                      materials=phantom.materials)
    return fig


# ──────────────────────────────────────────────────────────────────────────────
# Multi-panel figures
# ──────────────────────────────────────────────────────────────────────────────

def _subfigure_grid(n_panels: int, ncols: int, figsize_per_panel, suptitle):
    nrows = int(np.ceil(n_panels / ncols))
    fig = plt.figure(figsize=(figsize_per_panel[0] * ncols,
                              figsize_per_panel[1] * nrows),
                     constrained_layout=True)
    if suptitle:
        fig.suptitle(suptitle, fontsize=11)
    subfigs = np.atleast_1d(fig.subfigures(nrows, ncols, wspace=0.05, hspace=0.06)).ravel()
    for sf in subfigs[n_panels:]:
        sf.set_visible(False)
    return fig, subfigs


def plot_cross_algorithm_grid(
    phantom,
    xray_sinos: Dict[str, dict],
    neutron_sinos: Dict[str, dict],
    algorithm_pairs,
    bins: int = 128,
    log_scale: bool = True,
    cmap: str = "cool",
    show_marginals: bool = True,
    show_gt_markers: bool = True,
    show_metrics: bool = True,
    energy_idx: int = DEFAULT_GT_ENERGY_IDX,
    figsize_per_panel: Tuple[float, float] = (5.2, 4.8),
    suptitle: str = "Cross-algorithm bimodal histogram comparison",
    ncols: Optional[int] = None,
    use_astra: bool = True,
):
    """
    One bimodal-histogram panel per (X-ray algorithm, neutron algorithm) pair.

    Build the sinogram dicts with
    :func:`neutron_xray_sim.analysis.make_cross_algorithm_sinos`::

        algs  = ["FBP", "SIRT"]
        pairs = [(ax, an) for ax in algs for an in algs]
        x_sinos, n_sinos = make_cross_algorithm_sinos(phantom, algs)
        fig, hists = plot_cross_algorithm_grid(phantom, x_sinos, n_sinos, pairs)

    Returns
    -------
    (fig, histograms) — histograms is ``{(alg_x, alg_n): HistogramResult}``
    """
    from ..analysis.cross_algorithm import compute_cross_algorithm_histograms

    pairs, hists, metrics = compute_cross_algorithm_histograms(
        phantom, xray_sinos, neutron_sinos, algorithm_pairs, bins=bins,
        compute_metrics=show_metrics, energy_idx=energy_idx, use_astra=use_astra)

    mu_x, mu_n, _ = _gt_points(phantom, energy_idx)
    extent = _shared_extent(list(hists.values()), mu_x, mu_n)
    if ncols is None:
        ncols = max(1, int(np.ceil(np.sqrt(len(pairs)))))
    fig, subfigs = _subfigure_grid(len(pairs), ncols, figsize_per_panel, suptitle)

    for sf, (alg_x, alg_n) in zip(subfigs, pairs):
        title = f"X-ray: {alg_x}  |  Neutron: {alg_n}"
        title += _metrics_suffix(metrics.get((alg_x, alg_n)))
        _draw_recon_panel(
            sf, hists[(alg_x, alg_n)], title, log_scale=log_scale, cmap=cmap,
            show_marginals=show_marginals, shared_extent=extent,
            gt_mu_x=mu_x if show_gt_markers else None,
            gt_mu_n=mu_n if show_gt_markers else None,
            materials=phantom.materials,
        )
    return fig, hists


def plot_artifact_survey(
    results: Dict[str, "object"],
    phantom,
    metrics: Optional[Dict[str, "object"]] = None,
    title: str = "Artifact survey",
    ncols: int = 4,
    figsize_per_panel: Tuple[float, float] = (5.5, 5.0),
    cmap: str = "inferno",
    energy_idx: int = DEFAULT_GT_ENERGY_IDX,
) -> plt.Figure:
    """
    Survey figure: ground-truth bubble panel followed by one histogram panel
    per run (shared axis limits, ground-truth diamonds on every panel).

    Parameters
    ----------
    results : {tag → SimulationResult}
    phantom : the shared PhantomData
    metrics : optional {tag → ClusterQualityMetrics}; DB and CE are appended
              to each panel title
    """
    metrics = metrics or {}
    mu_x, mu_n, _ = _gt_points(phantom, energy_idx)
    extent = _shared_extent([r.histogram for r in results.values()], mu_x, mu_n)
    fig, subfigs = _subfigure_grid(1 + len(results), ncols, figsize_per_panel, title)

    ax = subfigs[0].add_subplot(1, 1, 1)
    _draw_gt_scatter(ax, phantom, extent, energy_idx, fontsize=7, size_legend=False)
    ax.set_title("Ground Truth\n(exact positions)", fontsize=9, color="white")
    subfigs[0].set_facecolor("#0d0d0d")

    for sf, (tag, r) in zip(subfigs[1:], results.items()):
        _draw_recon_panel(
            sf, r.histogram, tag + _metrics_suffix(metrics.get(tag)),
            log_scale=True, cmap=cmap, show_marginals=False,
            shared_extent=extent, gt_mu_x=mu_x, gt_mu_n=mu_n,
            materials=phantom.materials, fontsize=8,
        )
    return fig
