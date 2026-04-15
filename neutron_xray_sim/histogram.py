"""
neutron_xray_sim/histogram.py
──────────────────────────────
Bimodal (2-D joint) histogram analysis for dual-modality CT volumes.

The 2-D histogram H(μ_x, μ_n) is the core diagnostic tool:
  • Each pure material phase → compact Gaussian blob
  • Partial-volume voxels at interfaces → line segments between blobs
  • Beam hardening    → horizontal smearing (μ_x biased, μ_n unchanged)
  • Neutron scatter   → vertical shift of all clusters upward
  • Misalignment      → elongated streaks parallel to μ_n axis (horizontal
                         lines at the nominal μ_x of each phase)
  • Ring artifacts    → vertical striping at specific μ_x values

This module provides:
  1. compute_bimodal_histogram   — 2-D histogram
  2. fit_gmm                     — Gaussian mixture model blob detection
  3. segment_by_gmm              — label each voxel to its nearest GMM component
  4. segment_by_polygon          — manual lasso segmentation on the histogram
  5. detect_artifact_signatures  — quantitative streak / smear metrics
  6. plot_bimodal_histogram       — publication-quality figure
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Ellipse
from matplotlib.path import Path
from matplotlib import cm
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict
import warnings

__all__ = [
    "compute_bimodal_histogram",
    "compute_ground_truth_histogram",
    "fit_gmm",
    "segment_by_gmm",
    "segment_by_polygon",
    "detect_artifact_signatures",
    "evaluate_histogram_quality",
    "compare_algorithms",
    "plot_bimodal_histogram",
    "plot_ground_truth_comparison",
    "plot_cross_algorithm_grid",
    "make_cross_algorithm_sinos",
    "HistogramResult",
    "GMMFitResult",
    "ArtifactSignatures",
    "ClusterQualityMetrics",
]


# ──────────────────────────────────────────────────────────────────────────────
# Data containers
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class HistogramResult:
    """
    Container for a 2-D bimodal histogram.

    Attributes
    ----------
    H            : (bins_x, bins_n) 2-D histogram counts
    x_edges      : X-ray attenuation bin edges  [cm⁻¹]
    n_edges      : neutron attenuation bin edges [cm⁻¹]
    x_centres    : X-ray bin centres
    n_centres    : neutron bin centres
    vol_x_flat   : flattened X-ray voxel values  (for segmentation)
    vol_n_flat   : flattened neutron voxel values
    total_voxels : total number of voxels included
    """
    H:            np.ndarray
    x_edges:      np.ndarray
    n_edges:      np.ndarray
    x_centres:    np.ndarray
    n_centres:    np.ndarray
    vol_x_flat:   np.ndarray
    vol_n_flat:   np.ndarray
    total_voxels: int

    @property
    def extent(self) -> List[float]:
        """matplotlib imshow extent [x_min, x_max, n_min, n_max]."""
        return [self.x_edges[0], self.x_edges[-1],
                self.n_edges[0], self.n_edges[-1]]


@dataclass
class GMMFitResult:
    """
    Result from Gaussian mixture model fitting to the bimodal histogram.

    Attributes
    ----------
    n_components : number of GMM components fitted
    means        : (n_components, 2)  [μ_x, μ_n] cluster centres  [cm⁻¹]
    covariances  : (n_components, 2, 2)  covariance matrices
    weights      : (n_components,)  mixing weights
    labels_flat  : (N³,)  per-voxel cluster assignment (−1 = unassigned)
    bic          : Bayesian information criterion (lower = better fit)
    aic          : Akaike information criterion
    """
    n_components: int
    means:        np.ndarray   # (K, 2)
    covariances:  np.ndarray   # (K, 2, 2)
    weights:      np.ndarray   # (K,)
    labels_flat:  np.ndarray   # (N³,)
    bic:          float
    aic:          float


# ──────────────────────────────────────────────────────────────────────────────
# 1. Compute 2-D histogram
# ──────────────────────────────────────────────────────────────────────────────

def compute_bimodal_histogram(
    vol_x: np.ndarray,
    vol_n: np.ndarray,
    bins: int = 256,
    x_range: Optional[Tuple[float, float]] = None,
    n_range: Optional[Tuple[float, float]] = None,
    mask: Optional[np.ndarray] = None,
) -> HistogramResult:
    """
    Compute the 2-D bimodal (joint) histogram H(μ_x, μ_n).

    Parameters
    ----------
    vol_x    : (N, N, N) X-ray attenuation volume  [cm⁻¹]
    vol_n    : (N, N, N) neutron attenuation volume [cm⁻¹]
    bins     : number of bins per axis
    x_range  : (min, max) for X-ray axis; None = auto
    n_range  : (min, max) for neutron axis; None = auto
    mask     : optional boolean mask; True = include voxel

    Returns
    -------
    HistogramResult
    """
    if vol_x.shape != vol_n.shape:
        raise ValueError(f"Volume shapes must match: {vol_x.shape} vs {vol_n.shape}")

    vx = vol_x.ravel().astype(np.float64)
    vn = vol_n.ravel().astype(np.float64)

    if mask is not None:
        m = mask.ravel()
        vx = vx[m]
        vn = vn[m]

    if x_range is None:
        x_range = (float(np.percentile(vx, 0.1)), float(np.percentile(vx, 99.9)))
    if n_range is None:
        n_range = (float(np.percentile(vn, 0.1)), float(np.percentile(vn, 99.9)))

    H, x_edges, n_edges = np.histogram2d(
        vx, vn,
        bins=[bins, bins],
        range=[x_range, n_range],
    )

    x_centres = 0.5 * (x_edges[:-1] + x_edges[1:])
    n_centres = 0.5 * (n_edges[:-1] + n_edges[1:])

    return HistogramResult(
        H=H,
        x_edges=x_edges,
        n_edges=n_edges,
        x_centres=x_centres,
        n_centres=n_centres,
        vol_x_flat=vx,
        vol_n_flat=vn,
        total_voxels=len(vx),
    )


# ──────────────────────────────────────────────────────────────────────────────
# 2. Gaussian mixture model fitting
# ──────────────────────────────────────────────────────────────────────────────

def fit_gmm(
    hist: HistogramResult,
    n_components: int = 5,
    covariance_type: str = "full",
    n_init: int = 5,
    random_state: int = 0,
    subsample: int = 50_000,
) -> GMMFitResult:
    """
    Fit a Gaussian mixture model to the 2-D bimodal histogram.

    Uses scikit-learn GaussianMixture fitted to a weighted subsample of
    (μ_x, μ_n) voxel pairs drawn from the histogram.

    Parameters
    ----------
    hist            : HistogramResult from compute_bimodal_histogram()
    n_components    : number of Gaussian blobs to fit
    covariance_type : 'full', 'tied', 'diag', 'spherical'
    n_init          : number of random initialisations (best BIC kept)
    random_state    : random seed
    subsample       : max voxels used for fitting (speed)

    Returns
    -------
    GMMFitResult
    """
    try:
        from sklearn.mixture import GaussianMixture
    except ImportError:
        raise ImportError(
            "scikit-learn is required for GMM fitting: pip install scikit-learn"
        )

    vx = hist.vol_x_flat
    vn = hist.vol_n_flat
    pts = np.column_stack([vx, vn])

    # Subsample for speed
    if len(pts) > subsample:
        rng  = np.random.default_rng(random_state)
        idx  = rng.choice(len(pts), subsample, replace=False)
        pts_fit = pts[idx]
    else:
        pts_fit = pts

    gm = GaussianMixture(
        n_components=n_components,
        covariance_type=covariance_type,
        n_init=n_init,
        random_state=random_state,
        max_iter=300,
    )
    gm.fit(pts_fit)

    # Assign labels for all voxels (in chunks to avoid memory issues)
    chunk = 100_000
    labels = np.empty(len(pts), dtype=np.int32)
    for start in range(0, len(pts), chunk):
        labels[start:start+chunk] = gm.predict(pts[start:start+chunk])

    bic = gm.bic(pts_fit)
    aic = gm.aic(pts_fit)

    return GMMFitResult(
        n_components=n_components,
        means=gm.means_,
        covariances=gm.covariances_,
        weights=gm.weights_,
        labels_flat=labels,
        bic=bic,
        aic=aic,
    )


def auto_fit_gmm(
    hist: HistogramResult,
    min_k: int = 2,
    max_k: int = 8,
    random_state: int = 0,
) -> GMMFitResult:
    """
    Fit GMMs for k = min_k..max_k and return the one with the best BIC.

    Parameters
    ----------
    hist         : HistogramResult
    min_k, max_k : range of component counts to try
    random_state : random seed

    Returns
    -------
    best GMMFitResult
    """
    best_bic  = np.inf
    best_fit  = None

    for k in range(min_k, max_k + 1):
        try:
            result = fit_gmm(hist, n_components=k, random_state=random_state)
            print(f"  k={k}: BIC={result.bic:.1f}, AIC={result.aic:.1f}")
            if result.bic < best_bic:
                best_bic = result.bic
                best_fit = result
        except Exception as e:
            warnings.warn(f"GMM fitting failed for k={k}: {e}")

    return best_fit


# ──────────────────────────────────────────────────────────────────────────────
# 3. Segment by GMM labels
# ──────────────────────────────────────────────────────────────────────────────

def segment_by_gmm(
    vol_x: np.ndarray,
    vol_n: np.ndarray,
    gmm_result: GMMFitResult,
    mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Build a 3-D label volume from GMM cluster assignments.

    Parameters
    ----------
    vol_x, vol_n : (N, N, N) reconstructed volumes
    gmm_result   : GMMFitResult from fit_gmm()
    mask         : optional boolean mask

    Returns
    -------
    label_vol : (N, N, N) int32,  values 0 .. n_components−1
    """
    N   = vol_x.shape[0]
    vx  = vol_x.ravel()
    vn  = vol_n.ravel()

    if mask is not None:
        m     = mask.ravel()
        pts   = np.column_stack([vx[m], vn[m]])
    else:
        pts = np.column_stack([vx, vn])

    try:
        from sklearn.mixture import GaussianMixture
    except ImportError:
        raise ImportError("scikit-learn required: pip install scikit-learn")

    # Re-predict using stored GMM parameters
    gm = GaussianMixture(n_components=gmm_result.n_components)
    gm.means_        = gmm_result.means
    gm.covariances_  = gmm_result.covariances
    gm.weights_      = gmm_result.weights
    gm.precisions_chol_ = _compute_precision_chol(gmm_result.covariances)

    labels = np.full(vol_x.size, -1, dtype=np.int32)
    if mask is not None:
        labels[mask.ravel()] = gm.predict(pts)
    else:
        chunk = 100_000
        for start in range(0, len(pts), chunk):
            labels[start:start+chunk] = gm.predict(pts[start:start+chunk])

    return labels.reshape(vol_x.shape)


def segment_by_polygon(
    vol_x: np.ndarray,
    vol_n: np.ndarray,
    polygons: List[np.ndarray],
) -> np.ndarray:
    """
    Segment the volume by assigning each voxel to a manually drawn polygon
    on the bimodal histogram.

    Parameters
    ----------
    vol_x, vol_n : (N, N, N) volumes
    polygons     : list of (M, 2) arrays of (μ_x, μ_n) polygon vertices;
                   the first polygon = label 1, second = label 2, etc.

    Returns
    -------
    label_vol : (N, N, N) int32,  0 = unassigned, 1..K = polygon index
    """
    pts    = np.column_stack([vol_x.ravel(), vol_n.ravel()])
    labels = np.zeros(len(pts), dtype=np.int32)

    for i, verts in enumerate(polygons, start=1):
        path  = Path(verts)
        inside = path.contains_points(pts)
        labels[inside] = i

    return labels.reshape(vol_x.shape)


# ──────────────────────────────────────────────────────────────────────────────
# 4. Artifact signature detection
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class ArtifactSignatures:
    """Quantitative metrics for artifact signatures in the bimodal histogram."""

    horizontal_streak_score: float
    """High value → horizontal streaks (misalignment signature).
       Computed as max normalised marginal variance along n_axis at fixed μ_x."""

    vertical_streak_score: float
    """High value → vertical streaks (ring artifact signature).
       Computed as column-wise variance anisotropy."""

    cluster_elongation: Dict[int, float]
    """Per-cluster elongation ratio (σ_major / σ_minor from GMM covariance)."""

    diagonal_smear_score: float
    """High value → correlated smearing along both axes (beam hardening / X-ray scatter)."""

    marginal_asymmetry_x: float
    """Asymmetry of the X-ray marginal distribution (beam hardening cupping)."""

    marginal_shift_n: float
    """Upward shift of neutron marginal mean relative to clean (scatter build-up)."""


def detect_artifact_signatures(
    hist: HistogramResult,
    gmm: Optional[GMMFitResult] = None,
    ref_hist: Optional[HistogramResult] = None,
) -> ArtifactSignatures:
    """
    Quantify artifact signatures in the bimodal histogram.

    Parameters
    ----------
    hist     : HistogramResult from the (possibly artefacted) run
    gmm      : optional GMMFitResult for per-cluster elongation metrics
    ref_hist : optional clean reference histogram for shift comparison

    Returns
    -------
    ArtifactSignatures
    """
    H  = hist.H.T        # shape (bins_n, bins_x) — neutron on vertical axis
    Hf = H / (H.sum() + 1e-12)   # normalised

    # ── Horizontal streak score ───────────────────────────────────────────────
    # Variance of each row (fixed μ_n) along the μ_x axis, then take the max.
    row_var = Hf.var(axis=1)   # (bins_n,)
    col_var = Hf.var(axis=0)   # (bins_x,)
    horizontal_streak = float(np.max(row_var) / (np.mean(col_var) + 1e-12))

    # ── Vertical streak score ─────────────────────────────────────────────────
    vertical_streak = float(np.max(col_var) / (np.mean(row_var) + 1e-12))

    # ── Cluster elongation ────────────────────────────────────────────────────
    elongation = {}
    if gmm is not None:
        for k in range(gmm.n_components):
            cov = gmm.covariances[k]        # (2, 2)
            eigvals = np.linalg.eigvalsh(cov)
            eigvals = np.sort(np.abs(eigvals))[::-1]
            ratio   = float(np.sqrt(eigvals[0] / (eigvals[1] + 1e-12)))
            elongation[k] = ratio

    # ── Diagonal smear score ──────────────────────────────────────────────────
    # Cross-correlation between row-marginal and column-marginal shifts
    row_mean_x = (Hf * hist.x_centres[np.newaxis, :]).sum(axis=1)  # (bins_n,)
    col_mean_n = (Hf * hist.n_centres[:, np.newaxis]).sum(axis=0)  # (bins_x,)

    # Pearson correlation of the conditional means
    if len(row_mean_x) > 1 and row_mean_x.std() > 0 and col_mean_n.std() > 0:
        diag_smear = float(
            np.corrcoef(row_mean_x, np.interp(
                np.linspace(0, 1, len(row_mean_x)),
                np.linspace(0, 1, len(col_mean_n)), col_mean_n
            ))[0, 1]
        )
    else:
        diag_smear = 0.0

    # ── Marginal asymmetry (X-ray, beam hardening) ────────────────────────────
    x_marginal = Hf.sum(axis=0)   # (bins_x,)
    x_weights  = x_marginal / (x_marginal.sum() + 1e-12)
    x_mean     = float((x_weights * hist.x_centres).sum())
    x_median   = float(hist.x_centres[np.searchsorted(
        np.cumsum(x_weights), 0.5).clip(0, len(hist.x_centres)-1)])
    asymmetry_x = float((x_mean - x_median) / (np.std(hist.vol_x_flat) + 1e-12))

    # ── Neutron marginal shift ────────────────────────────────────────────────
    n_mean_current = float(np.mean(hist.vol_n_flat))
    if ref_hist is not None:
        n_mean_ref = float(np.mean(ref_hist.vol_n_flat))
        shift_n    = n_mean_current - n_mean_ref
    else:
        shift_n = 0.0

    return ArtifactSignatures(
        horizontal_streak_score = horizontal_streak,
        vertical_streak_score   = vertical_streak,
        cluster_elongation      = elongation,
        diagonal_smear_score    = diag_smear,
        marginal_asymmetry_x    = asymmetry_x,
        marginal_shift_n        = shift_n,
    )


# ──────────────────────────────────────────────────────────────────────────────
# 5. Plotting
# ──────────────────────────────────────────────────────────────────────────────

def plot_bimodal_histogram(
    hist: HistogramResult,
    gmm: Optional[GMMFitResult] = None,
    material_labels: Optional[Dict[int, str]] = None,
    title: str = "Bimodal Histogram",
    ax: Optional[plt.Axes] = None,
    log_scale: bool = True,
    cmap: str = "inferno",
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
    cmap: str = "inferno",
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
# Ground-truth histogram helper
# ──────────────────────────────────────────────────────────────────────────────

def compute_ground_truth_histogram(
    phantom,
    bins: int = 256,
    energy_idx: int = 6,
) -> "HistogramResult":
    """
    Build the ideal bimodal histogram directly from phantom attenuation volumes.

    This produces perfectly sharp clusters at each material's exact (μ_x, μ_n)
    values — no reconstruction blur, noise, or partial-volume smearing.
    Comparing this against the reconstructed histogram reveals how much each
    artifact degrades cluster separation.

    Parameters
    ----------
    phantom    : PhantomData object
    bins       : number of histogram bins per axis
    energy_idx : index into XRAY_E_KEV for the X-ray channel.
                 Default 6 corresponds to 80 keV.

    Returns
    -------
    HistogramResult  with vol_x_flat = phantom mu_x at energy_idx,
                         vol_n_flat = phantom mu_n (imaging-effective total)
    """
    mu_x = phantom.mu_x_vols[energy_idx]   # (N, N, N)  [cm⁻¹]
    mu_n = phantom.mu_n_vol                 # (N, N, N)  [cm⁻¹]
    return compute_bimodal_histogram(mu_x, mu_n, bins=bins)


# ──────────────────────────────────────────────────────────────────────────────
# Side-by-side ground-truth vs reconstruction comparison
# ──────────────────────────────────────────────────────────────────────────────

def plot_ground_truth_comparison(
    phantom,
    hist_recon: "HistogramResult",
    title_gt: str = "Ground Truth",
    title_recon: str = "Reconstructed",
    log_scale: bool = True,
    cmap: str = "inferno",
    show_marginals: bool = True,
    figsize: Tuple[float, float] = (15, 7),
    suptitle: str = "Ground truth vs Reconstructed bimodal histogram",
    energy_idx: int = 6,
) -> plt.Figure:
    """
    Side-by-side comparison: phantom ground truth (left) vs reconstruction (right).

    **Left panel — Ground Truth (scatter plot)**
    Each material phase is drawn as a labelled bubble.  Bubble area scales with
    sqrt(voxel_count) so minor phases (Ti, water) remain visible alongside the
    dominant air and HDPE phases.  The exact (mu_x, mu_n) coordinates are
    annotated beside each bubble.  Cross-hairs mark each position so the reader
    can read off values from the axes.

    **Right panel — Reconstructed (2-D histogram + marginals)**
    The standard imshow histogram with log-colour scale.  Ground-truth material
    positions are overlaid as white diamond markers so any shift, smearing, or
    partial-volume offset is immediately apparent.

    Both panels share identical axis limits.

    Parameters
    ----------
    phantom     : PhantomData object (provides material positions and voxel counts)
    hist_recon  : HistogramResult from compute_bimodal_histogram on the
                  reconstructed volumes
    title_gt    : left panel title
    title_recon : right panel title
    log_scale   : log colour scale on the reconstructed 2-D panel
    cmap        : matplotlib colourmap for the reconstructed panel
    show_marginals : show top / right marginal distributions on the recon panel
    figsize     : overall figure size (width, height)
    suptitle    : figure suptitle
    energy_idx  : index into XRAY_E_KEV for the X-ray channel used in mu_x_vols.
                  Default 6 = 80 keV.

    Returns
    -------
    matplotlib Figure
    """
    import matplotlib.colors as _mcolors

    # ── Material positions and counts from phantom ────────────────────────────
    materials  = phantom.materials
    n_mat      = len(materials)
    # mu_x at the chosen energy bin (default index 6 = 80 keV)
    mu_x_vals  = np.array([m._mu_x_table[energy_idx] for m in materials])
    mu_n_vals  = np.array([m.mu_n for m in materials])
    vox_counts = np.array([(phantom.label_vol == i).sum() for i in range(n_mat)],
                           dtype=float)

    # Marker sizes: sqrt-scaled to keep minor phases visible
    sqrt_c  = np.sqrt(vox_counts)
    s_range = sqrt_c.max() - sqrt_c.min()
    sizes   = 60 + 550 * (sqrt_c - sqrt_c.min()) / (s_range + 1e-9)

    # Colours: one per material from a qualitative colormap
    mat_colours = plt.cm.Set1(np.linspace(0, 0.9, n_mat))

    # ── Shared axis limits ────────────────────────────────────────────────────
    # X: 0 to max(recon_max, gt_max).  GT clusters are at exact mu values;
    # recon range may be wider due to FBP artefacts.
    x_max = max(hist_recon.x_edges[-1], float(mu_x_vals.max()) * 1.08)
    n_max = max(hist_recon.n_edges[-1], float(mu_n_vals.max()) * 1.08)
    shared_extent = [0.0, x_max, 0.0, n_max]

    # ── Figure: two subfigures side by side ───────────────────────────────────
    fig = plt.figure(figsize=figsize, constrained_layout=True)
    if suptitle:
        fig.suptitle(suptitle, fontsize=13)

    subfigs = fig.subfigures(1, 2, wspace=0.06, width_ratios=[1, 1])

    # ══ LEFT: ground-truth scatter plot ══════════════════════════════════════
    sf_gt = subfigs[0]
    ax_gt = sf_gt.add_subplot(1, 1, 1)

    # Light grid for readability
    ax_gt.set_facecolor("#0d0d0d")
    ax_gt.grid(True, color="#333333", linewidth=0.5, zorder=0)

    for i, (m, mx, mn, sz, col) in enumerate(
            zip(materials, mu_x_vals, mu_n_vals, sizes, mat_colours)):
        # Skip pure air (0,0) from the scatter body but still annotate
        is_air = (mx < 1e-3 and mn < 1e-3)

        # Cross-hairs so values can be read off the axes
        ax_gt.axvline(mx, color=col, linewidth=0.5, alpha=0.35, zorder=1)
        ax_gt.axhline(mn, color=col, linewidth=0.5, alpha=0.35, zorder=1)

        # Bubble
        ax_gt.scatter(mx, mn, s=sz, color=col, edgecolors="white",
                      linewidths=0.8, zorder=3, alpha=0.92)

        # Label: material symbol + exact coordinates
        label = f"{m.symbol}\n({mx:.3f}, {mn:.3f})"
        # Offset direction: push label away from centre to avoid overlap
        dx = 0.04 * x_max * (1 if mx < x_max * 0.6 else -1)
        dy = 0.04 * n_max * (1 if mn < n_max * 0.6 else -1)
        ax_gt.annotate(
            label,
            xy=(mx, mn),
            xytext=(mx + dx, mn + dy),
            fontsize=8,
            color="white",
            fontweight="bold",
            ha="left" if dx > 0 else "right",
            va="bottom" if dy > 0 else "top",
            arrowprops=dict(arrowstyle="-", color=col, lw=0.8),
            zorder=4,
        )

    ax_gt.set_xlim(shared_extent[0], shared_extent[1])
    ax_gt.set_ylim(shared_extent[2], shared_extent[3])
    ax_gt.set_xlabel(r"$\mu_x$ [cm$^{-1}$]", fontsize=11)
    ax_gt.set_ylabel(r"$\mu_n$ [cm$^{-1}$]", fontsize=11)
    ax_gt.set_title(title_gt, fontsize=12)

    # Size legend (bottom-right)
    legend_counts = [vox_counts.min(), np.median(vox_counts), vox_counts.max()]
    legend_sizes  = [60 + 550 * (np.sqrt(c) - sqrt_c.min()) / (s_range + 1e-9)
                     for c in legend_counts]
    legend_labels = [f"{int(c):,} vox" for c in legend_counts]
    for ls, ll in zip(legend_sizes, legend_labels):
        ax_gt.scatter([], [], s=ls, color="grey", edgecolors="white",
                      linewidths=0.6, label=ll, alpha=0.8)
    ax_gt.legend(title="Bubble size", loc="lower right",
                 fontsize=7, title_fontsize=7,
                 framealpha=0.3, labelcolor="white",
                 facecolor="#222222", edgecolor="#555555")

    ax_gt.tick_params(colors="white", labelsize=8)
    for spine in ax_gt.spines.values():
        spine.set_edgecolor("#444444")

    # ══ RIGHT: reconstructed histogram ═══════════════════════════════════════
    _draw_recon_panel(
        subfigs[1], hist_recon, title_recon,
        log_scale=log_scale, cmap=cmap,
        show_marginals=show_marginals,
        shared_extent=shared_extent,
        gt_mu_x=mu_x_vals,
        gt_mu_n=mu_n_vals,
        gt_colours=mat_colours,
        materials=materials,
    )

    return fig


def _draw_recon_panel(
    subfig: plt.Figure,
    hist: "HistogramResult",
    title: str,
    log_scale: bool,
    cmap: str,
    show_marginals: bool,
    shared_extent: Optional[List[float]],
    gt_mu_x: Optional[np.ndarray] = None,
    gt_mu_n: Optional[np.ndarray] = None,
    gt_colours: Optional[np.ndarray] = None,
    materials: Optional[list] = None,
) -> None:
    """
    Draw the reconstructed histogram panel with optional GT position markers.
    Internal helper for plot_ground_truth_comparison.
    """
    if show_marginals:
        gs = subfig.add_gridspec(
            2, 3,
            width_ratios=[4, 1, 0.18],
            height_ratios=[1, 4],
            hspace=0.0, wspace=0.0,
        )
        ax_main  = subfig.add_subplot(gs[1, 0])
        ax_top   = subfig.add_subplot(gs[0, 0], sharex=ax_main)
        ax_right = subfig.add_subplot(gs[1, 1], sharey=ax_main)
        ax_cbar  = subfig.add_subplot(gs[1, 2])
        subfig.add_subplot(gs[0, 1]).set_visible(False)
        subfig.add_subplot(gs[0, 2]).set_visible(False)
    else:
        gs = subfig.add_gridspec(1, 2, width_ratios=[4, 0.18], wspace=0.0)
        ax_main  = subfig.add_subplot(gs[0, 0])
        ax_cbar  = subfig.add_subplot(gs[0, 1])
        ax_top = ax_right = None

    extent = shared_extent if shared_extent is not None else hist.extent

    # 2-D histogram
    H      = hist.H.T
    H_plot = np.log1p(H) if log_scale else H
    H_plot = np.ma.masked_where(H == 0, H_plot)

    im = ax_main.imshow(
        H_plot, origin="lower", extent=extent,
        aspect="auto", cmap=cmap, interpolation="bilinear",
    )
    ax_main.set_xlim(extent[0], extent[1])
    ax_main.set_ylim(extent[2], extent[3])
    ax_main.set_xlabel(r"$\mu_x$ [cm$^{-1}$]", fontsize=11)
    ax_main.set_ylabel(r"$\mu_n$ [cm$^{-1}$]", fontsize=11)
    ax_main.set_title(title, fontsize=12)

    cbar = plt.colorbar(im, cax=ax_cbar)
    cbar.set_label("log(1+counts)" if log_scale else "counts", fontsize=9)

    # Ground-truth position markers
    if gt_mu_x is not None and gt_mu_n is not None:
        for i, (mx, mn) in enumerate(zip(gt_mu_x, gt_mu_n)):
            col = gt_colours[i] if gt_colours is not None else "white"
            sym = materials[i].symbol if materials is not None else str(i)
            ax_main.plot(mx, mn, marker="D", color="white",
                         markersize=6, markeredgecolor=col,
                         markeredgewidth=1.5, zorder=5)
            ax_main.annotate(
                sym, xy=(mx, mn),
                xytext=(4, 4), textcoords="offset points",
                fontsize=7, color="white", fontweight="bold", zorder=6,
            )

    # Top marginal (log-Y)
    if ax_top is not None:
        x_marg = hist.H.sum(axis=1).astype(float)
        x_plot = np.maximum(x_marg, 0.5)
        ax_top.fill_between(hist.x_centres, 0.5, x_plot,
                            step="mid", alpha=0.65, color="steelblue")
        ax_top.step(hist.x_centres, x_plot, where="mid",
                    color="steelblue", linewidth=0.8, alpha=0.9)
        ax_top.set_yscale("log")
        ax_top.set_ylabel("counts", fontsize=8)
        ax_top.tick_params(labelbottom=False, labelsize=7)
        ax_top.set_xlim(extent[0], extent[1])
        ax_top.spines["top"].set_visible(False)
        ax_top.spines["right"].set_visible(False)

    # Right marginal (log-X)
    if ax_right is not None:
        n_marg = hist.H.sum(axis=0).astype(float)
        n_plot = np.maximum(n_marg, 0.5)
        ax_right.fill_betweenx(hist.n_centres, 0.5, n_plot,
                                step="mid", alpha=0.65, color="tomato")
        ax_right.step(n_plot, hist.n_centres, where="mid",
                      color="tomato", linewidth=0.8, alpha=0.9)
        ax_right.set_xscale("log")
        ax_right.set_xlabel("counts", fontsize=8)
        ax_right.tick_params(labelleft=False, labelsize=7)
        ax_right.set_ylim(extent[2], extent[3])
        ax_right.spines["top"].set_visible(False)
        ax_right.spines["right"].set_visible(False)


# ──────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ──────────────────────────────────────────────────────────────────────────────


# ──────────────────────────────────────────────────────────────────────────────
# Cross-algorithm comparison grid
# ──────────────────────────────────────────────────────────────────────────────

def plot_cross_algorithm_grid(
    phantom,
    xray_sinos,
    neutron_sinos,
    algorithm_pairs,
    bins=128,
    log_scale=True,
    cmap="inferno",
    show_marginals=True,
    show_gt_markers=True,
    show_metrics=True,
    energy_idx=6,
    figsize_per_panel=(5.2, 4.8),
    suptitle="Cross-algorithm bimodal histogram comparison",
    ncols=None,
):
    """
    Plot a grid of bimodal histograms, one panel per (X-ray algorithm, neutron
    algorithm) combination specified in *algorithm_pairs*.

    Each panel reconstructs the X-ray sinogram with *alg_x* and the neutron
    sinogram with *alg_n* independently, then builds ``H(mu_x, mu_n)`` from
    the two resulting volumes.  This isolates the contribution of each
    modality's reconstruction algorithm to the histogram structure.

    Parameters
    ----------
    phantom : PhantomData
        Provides ground-truth material positions, label volume, and voxel_cm.

    xray_sinos : dict {str -> dict}
        Pre-computed X-ray sinogram dicts keyed by algorithm name.
        All values are sinogram dicts as returned by ``project_xray()``
        (keys: ``sino_lam``, ``angles_deg``, ``voxel_cm``, …).
        Typically every value is the **same** sinogram dict — the algorithm
        name is only used to select which reconstruction algorithm to run.
        Use :func:`make_cross_algorithm_sinos` to build this automatically.

    neutron_sinos : dict {str -> dict}
        Same structure for the neutron channel.

    algorithm_pairs : list of (alg_x, alg_n) tuples
        Each tuple defines one grid panel.  Duplicates are ignored.
        Example::

            [("FBP",  "FBP"),
             ("FBP",  "SIRT"),
             ("SIRT", "FBP"),
             ("SIRT", "SIRT"),
             ("SART", "SART")]

    bins : int
        Histogram bins per axis.

    log_scale : bool
        Logarithmic colour scale on the 2-D panels.

    cmap : str
        Matplotlib colourmap.

    show_marginals : bool
        Attach log-count marginal histograms to each panel.

    show_gt_markers : bool
        Overlay white diamond markers at ground-truth material positions on
        every panel.

    show_metrics : bool
        Compute Davies-Bouldin index and mean centroid error for each panel
        and append to the panel title as
        ``"DB=X.XXX  CE=X.XXXX cm^-1"``.

    energy_idx : int
        Energy bin index for the X-ray ground-truth mu_x (default 6 = 80 keV).

    figsize_per_panel : (float, float)
        Size of each histogram panel in inches.

    suptitle : str
        Figure super-title.

    ncols : int or None
        Grid columns.  ``None`` uses ``ceil(sqrt(n_panels))``.

    Returns
    -------
    fig : matplotlib.figure.Figure

    histograms : dict {(alg_x, alg_n) -> HistogramResult}
        All computed histograms for downstream analysis, e.g.
        :func:`evaluate_histogram_quality`.

    Raises
    ------
    KeyError
        If an algorithm pair requests a key absent from *xray_sinos* or
        *neutron_sinos*.
    ValueError
        If *algorithm_pairs* is empty.

    Examples
    --------
    ::

        from neutron_xray_sim import DualModalitySimulation, ArtifactConfig
        from neutron_xray_sim.histogram import (
            make_cross_algorithm_sinos, plot_cross_algorithm_grid
        )

        sim = DualModalitySimulation(preset="composite", N=64, n_angles=120)
        sim._ensure_sinograms()

        algs  = ["FBP", "SIRT", "SART", "CGLS"]
        pairs = [(ax, an) for ax in algs for an in algs]   # full 4x4 grid

        x_sinos, n_sinos = make_cross_algorithm_sinos(sim.phantom, algs)

        fig, hists = plot_cross_algorithm_grid(
            phantom         = sim.phantom,
            xray_sinos      = x_sinos,
            neutron_sinos   = n_sinos,
            algorithm_pairs = pairs,
            suptitle        = "4x4 algorithm comparison — composite phantom",
        )
        fig.savefig("cross_alg_grid.png", dpi=150, bbox_inches="tight")
    """
    import warnings as _warn

    from .reconstructor import reconstruct as _recon

    # ── Deduplicate pairs, preserve order ─────────────────────────────────────
    seen, unique_pairs = set(), []
    for p in algorithm_pairs:
        if p not in seen:
            seen.add(p)
            unique_pairs.append(p)
    pairs = unique_pairs

    n_panels = len(pairs)
    if n_panels == 0:
        raise ValueError("algorithm_pairs is empty.")

    # ── Validate keys ─────────────────────────────────────────────────────────
    for alg_x, alg_n in pairs:
        if alg_x not in xray_sinos:
            raise KeyError(
                f"alg_x='{alg_x}' not found in xray_sinos. "
                f"Available: {list(xray_sinos)}"
            )
        if alg_n not in neutron_sinos:
            raise KeyError(
                f"alg_n='{alg_n}' not found in neutron_sinos. "
                f"Available: {list(neutron_sinos)}"
            )

    # ── GT positions ──────────────────────────────────────────────────────────
    materials  = phantom.materials
    n_mat      = len(materials)
    mu_x_gt    = np.array([m._mu_x_table[energy_idx] for m in materials])
    mu_n_gt    = np.array([m.mu_n for m in materials])
    gt_colours = plt.cm.Set1(np.linspace(0, 0.9, n_mat))

    # ── Reconstruct, caching by (modality, algorithm) ─────────────────────────
    _vol_cache = {}

    def _get_vol(modality, alg, sino):
        key = (modality, alg)
        if key not in _vol_cache:
            _vol_cache[key] = _recon(sino, algorithm=alg,
                                     filter_name="shepp-logan",
                                     remove_rings=True, clip_negative=True,
                                     use_astra=True)
        return _vol_cache[key]

    # ── Build histograms ──────────────────────────────────────────────────────
    pair_hists = {}
    for alg_x, alg_n in pairs:
        vx = _get_vol("X", alg_x, xray_sinos[alg_x])
        vn = _get_vol("N", alg_n, neutron_sinos[alg_n])
        pair_hists[(alg_x, alg_n)] = compute_bimodal_histogram(vx, vn, bins=bins)

    # ── Optional per-panel metrics ────────────────────────────────────────────
    panel_metrics = {}
    if show_metrics:
        # evaluate_histogram_quality may not be defined yet in the uploaded
        # version; catch gracefully
        _eqh = globals().get("evaluate_histogram_quality")
        for pair, hist in pair_hists.items():
            if _eqh is not None:
                try:
                    panel_metrics[pair] = _eqh(
                        hist, phantom,
                        n_components=n_mat,
                        energy_idx=energy_idx,
                        exclude_air=True,
                    )
                except Exception as exc:
                    _warn.warn(f"Metrics failed for {pair}: {exc}")
                    panel_metrics[pair] = None
            else:
                panel_metrics[pair] = None

    # ── Shared axis limits ────────────────────────────────────────────────────
    x_max = max(
        max(h.x_edges[-1] for h in pair_hists.values()),
        float(mu_x_gt.max()) * 1.08,
    )
    n_max = max(
        max(h.n_edges[-1] for h in pair_hists.values()),
        float(mu_n_gt.max()) * 1.08,
    )
    shared_extent = [0.0, x_max, 0.0, n_max]

    # ── Figure grid ───────────────────────────────────────────────────────────
    if ncols is None:
        ncols = max(1, int(np.ceil(np.sqrt(n_panels))))
    nrows = int(np.ceil(n_panels / ncols))

    fig = plt.figure(
        figsize=(figsize_per_panel[0] * ncols, figsize_per_panel[1] * nrows),
        constrained_layout=True,
    )
    if suptitle:
        fig.suptitle(suptitle, fontsize=12)

    subfigs = fig.subfigures(nrows, ncols, wspace=0.05, hspace=0.06)
    # Normalise to 2-D array regardless of shape
    subfigs = np.atleast_2d(subfigs)
    if subfigs.ndim == 1:
        subfigs = subfigs[np.newaxis, :]
    subfigs_flat = subfigs.ravel()

    # ── Draw each panel ───────────────────────────────────────────────────────
    for panel_idx, (alg_x, alg_n) in enumerate(pairs):
        hist = pair_hists[(alg_x, alg_n)]
        m    = panel_metrics.get((alg_x, alg_n))

        # Build title lines
        title = f"X-ray: {alg_x}  |  Neutron: {alg_n}"
        if m is not None:
            nan = float("nan")
            db  = m.davies_bouldin
            ce  = m.mean_centroid_error
            db_s = f"{db:.3f}" if db == db else "n/a"
            ce_s = f"{ce:.4f}" if ce == ce else "n/a"
            title = title + "\nDB=" + db_s + "  CE=" + ce_s + " cm\u207b\u00b9"

        _draw_recon_panel(
            subfigs_flat[panel_idx], hist, title,
            log_scale      = log_scale,
            cmap           = cmap,
            show_marginals = show_marginals,
            shared_extent  = shared_extent,
            gt_mu_x        = mu_x_gt  if show_gt_markers else None,
            gt_mu_n        = mu_n_gt  if show_gt_markers else None,
            gt_colours     = gt_colours,
            materials      = materials,
        )

    # Hide unused slots
    for idx in range(n_panels, len(subfigs_flat)):
        subfigs_flat[idx].set_visible(False)

    return fig, pair_hists


def make_cross_algorithm_sinos(
    phantom,
    algorithms,
    n_angles=120,
    angle_range_deg=180.0,
    kVp=120.0,
    filter_mm_Al=2.0,
    filter_mm_Cu=0.0,
    n_spectrum_bins=12,
    I0=1e5,
    cfg=None,
    use_astra=True,
):
    """
    Convenience wrapper: run one forward projection and return sinogram dicts
    ready for :func:`plot_cross_algorithm_grid`.

    Because reconstruction is the only thing that differs between grid panels,
    a single forward projection is shared by all panels — no extra GPU memory
    or compute is used for duplicate sinograms.

    Parameters
    ----------
    phantom : PhantomData
    algorithms : list of str
        All algorithm names that will appear in any pair.  Only the unique
        set matters; the returned dicts map every listed algorithm to the
        **same** sinogram object.
    n_angles : int
    angle_range_deg : float
    kVp : float
    filter_mm_Al : float
    filter_mm_Cu : float
    n_spectrum_bins : int
    I0 : float
        Incident count for both modalities.
    cfg : ArtifactConfig or None
        If supplied, sinogram-domain artifacts are injected before returning.
    use_astra : bool

    Returns
    -------
    xray_sinos : dict {alg -> xray sinogram dict}
    neutron_sinos : dict {alg -> neutron sinogram dict}

    Examples
    --------
    ::

        phantom = make_composite_phantom(N=64)
        algs = ["FBP", "SIRT", "SART"]

        x_sinos, n_sinos = make_cross_algorithm_sinos(phantom, algs)

        pairs = [("FBP", "SIRT"), ("SIRT", "FBP"), ("SART", "SART")]
        fig, hists = plot_cross_algorithm_grid(phantom, x_sinos, n_sinos, pairs)
    """
    from .projector import make_sinogram_pair
    from .artifacts import inject_sinogram_artifacts

    xray_sino, neutron_sino = make_sinogram_pair(
        phantom,
        n_angles        = n_angles,
        angle_range_deg = angle_range_deg,
        kVp             = kVp,
        filter_mm_Al    = filter_mm_Al,
        filter_mm_Cu    = filter_mm_Cu,
        n_spectrum_bins = n_spectrum_bins,
        I0_xray         = I0,
        I0_neutron      = I0,
        use_astra       = use_astra,
    )

    if cfg is not None:
        import numpy as _np
        xray_sino, neutron_sino = inject_sinogram_artifacts(
            xray_sino, neutron_sino, cfg,
            rng=_np.random.default_rng(seed=0),
        )

    # Share the same dict objects for all algorithms
    unique = list(dict.fromkeys(algorithms))
    xray_sinos    = {alg: xray_sino    for alg in unique}
    neutron_sinos = {alg: neutron_sino for alg in unique}
    return xray_sinos, neutron_sinos


# ──────────────────────────────────────────────────────────────────────────────
# Cluster-quality metrics: centroid error, spread, Davies-Bouldin index
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class ClusterQualityMetrics:
    """
    Quantitative quality metrics for bimodal histogram cluster structure,
    evaluated against known ground-truth material positions from the phantom.

    All per-material fields are dicts keyed by material name (str).

    Attributes
    ----------
    centroid_errors : dict {mat_name -> float}  [cm^-1]
        Euclidean distance in (mu_x, mu_n) space between the GMM component
        matched to this material and its ground-truth position.
        Perfect reconstruction → 0 for all materials.

    sigma_x : dict {mat_name -> float}  [cm^-1]
        Cluster standard deviation along the mu_x axis,
        read from the GMM covariance diagonal: sqrt(Cov[0,0]).

    sigma_n : dict {mat_name -> float}  [cm^-1]
        Cluster standard deviation along mu_n: sqrt(Cov[1,1]).

    mean_centroid_error : float  [cm^-1]
        Mean centroid error across all matched (non-air) materials.
        Lower is better.

    davies_bouldin : float  (dimensionless, ≥ 0)
        Davies-Bouldin index over matched clusters:
            DB = (1/K) * sum_k  max_{j≠k}  (s_k + s_j) / d(c_k, c_j)
        where s_k = sqrt(trace(Cov_k)/2) and d is Euclidean centroid distance.
        Lower is better; 0 = perfect separation.

    overlap_fractions : dict {(mat_a, mat_b) -> float}
        For each pair of neighbouring materials (GT distance < 2 cm^-1),
        the fraction of their combined voxels that are misclassified by the
        GMM relative to the ground-truth label volume.
        Empty dict when phantom.label_vol is unavailable.

    n_matched : int
        Number of material phases successfully matched to a GMM component.

    gmm : GMMFitResult
        The fitted GMM stored for downstream plotting.
    """
    centroid_errors:     Dict[str, float]
    sigma_x:             Dict[str, float]
    sigma_n:             Dict[str, float]
    mean_centroid_error: float
    davies_bouldin:      float
    overlap_fractions:   Dict[Tuple[str, str], float]
    n_matched:           int
    gmm:                 "GMMFitResult"

    def summary(self, indent: str = "  ") -> str:
        """Return a compact human-readable summary string."""
        lines = [
            "ClusterQualityMetrics",
            f"{indent}mean centroid error : {self.mean_centroid_error:.4f} cm^-1",
            f"{indent}Davies-Bouldin index: {self.davies_bouldin:.4f}",
            f"{indent}n_matched           : {self.n_matched}",
            f"{indent}per-material centroid errors (cm^-1):",
        ]
        for name, err in sorted(self.centroid_errors.items()):
            sx = self.sigma_x.get(name, float("nan"))
            sn = self.sigma_n.get(name, float("nan"))
            lines.append(
                f"{indent}  {name:<18}: err={err:.4f}  "
                f"sigma_x={sx:.4f}  sigma_n={sn:.4f}"
            )
        if self.overlap_fractions:
            lines.append(f"{indent}overlap fractions:")
            for (a, b), f in sorted(self.overlap_fractions.items()):
                lines.append(f"{indent}  {a} / {b}: {f:.4f}")
        return "\n".join(lines)


def evaluate_histogram_quality(
    hist: "HistogramResult",
    phantom,
    n_components: Optional[int] = None,
    energy_idx: int = 6,
    exclude_air: bool = True,
    gmm_n_init: int = 5,
    gmm_max_iter: int = 300,
) -> "ClusterQualityMetrics":
    """
    Quantitatively evaluate bimodal histogram cluster quality against the
    known ground-truth material positions from *phantom*.

    Three complementary metrics are computed:

    **1. Centroid error** (accuracy)
        A Gaussian Mixture Model is fitted to the voxel-level
        ``(mu_x, mu_n)`` pairs.  Each GMM component is matched to the
        nearest unmatched ground-truth material by greedy nearest-neighbour
        assignment.  The per-material centroid error is the Euclidean distance
        between the matched GMM centroid and the true material position::

            eps_k = sqrt((mu_x_hat - mu_x_GT)^2 + (mu_n_hat - mu_n_GT)^2)

    **2. Cluster spread** (compactness)
        sigma_x = sqrt(Cov[0,0]),  sigma_n = sqrt(Cov[1,1])
        for each matched cluster.

    **3. Davies-Bouldin index** (separability)
        DB = (1/K) * sum_k  max_{j≠k}  (s_k + s_j) / d(c_k, c_j)
        where s_k = sqrt(trace(Cov_k)/2).  Lower DB = better.

    Parameters
    ----------
    hist          : HistogramResult from compute_bimodal_histogram()
    phantom       : PhantomData (provides GT positions and label_vol)
    n_components  : GMM components to fit.  Defaults to len(phantom.materials).
    energy_idx    : energy bin for mu_x GT lookup (default 6 = 80 keV)
    exclude_air   : exclude the air phase from centroid-error and DB metrics
    gmm_n_init    : GMM random initialisations (higher = more robust)
    gmm_max_iter  : maximum EM iterations

    Returns
    -------
    ClusterQualityMetrics
    """
    try:
        from sklearn.mixture import GaussianMixture
    except ImportError:
        raise ImportError(
            "scikit-learn is required: pip install scikit-learn"
        )

    # ── Ground-truth positions ────────────────────────────────────────────────
    materials    = phantom.materials
    gt_positions = {}
    for mat in materials:
        if exclude_air and mat.name.lower() in ("air",):
            continue
        gt_positions[mat.name] = (
            float(mat._mu_x_table[energy_idx]),
            float(mat.mu_n),
        )

    n_mat_total = len(materials)
    if n_components is None:
        n_components = n_mat_total

    # ── Fit GMM on voxel pairs ────────────────────────────────────────────────
    vx = hist.vol_x_flat
    vn = hist.vol_n_flat
    MAX_SAMPLES = 500_000
    if len(vx) > MAX_SAMPLES:
        rng = np.random.default_rng(seed=42)
        idx = rng.choice(len(vx), size=MAX_SAMPLES, replace=False)
        vx_fit, vn_fit = vx[idx], vn[idx]
    else:
        vx_fit, vn_fit = vx, vn

    pts = np.column_stack([vx_fit, vn_fit])

    gm = GaussianMixture(
        n_components=n_components,
        covariance_type="full",
        n_init=gmm_n_init,
        max_iter=gmm_max_iter,
        random_state=42,
    )
    gm.fit(pts)

    gmm_means = gm.means_
    gmm_covs  = gm.covariances_
    gmm_wts   = gm.weights_

    # ── Greedy nearest-neighbour matching ─────────────────────────────────────
    gt_names  = list(gt_positions.keys())
    gt_coords = np.array([gt_positions[n] for n in gt_names])

    dist_mat = np.sqrt(
        ((gt_coords[:, np.newaxis, :] - gmm_means[np.newaxis, :, :]) ** 2).sum(axis=2)
    )

    matched_gt: Dict[str, int] = {}
    remaining_gt  = list(range(len(gt_names)))
    remaining_gmm = list(range(n_components))
    while remaining_gt and remaining_gmm:
        sub      = dist_mat[np.ix_(remaining_gt, remaining_gmm)]
        flat_idx = int(np.argmin(sub))
        r, c     = divmod(flat_idx, len(remaining_gmm))
        gt_idx   = remaining_gt[r]
        gmm_idx  = remaining_gmm[c]
        matched_gt[gt_names[gt_idx]] = gmm_idx
        remaining_gt.remove(gt_idx)
        remaining_gmm.remove(gmm_idx)

    # ── Per-material metrics ──────────────────────────────────────────────────
    centroid_errors: Dict[str, float] = {}
    sigma_x_dict:   Dict[str, float] = {}
    sigma_n_dict:   Dict[str, float] = {}

    for name, gmm_idx in matched_gt.items():
        mx_hat, mn_hat = gmm_means[gmm_idx]
        mx_gt,  mn_gt  = gt_positions[name]
        centroid_errors[name] = float(
            np.sqrt((mx_hat - mx_gt) ** 2 + (mn_hat - mn_gt) ** 2)
        )
        sigma_x_dict[name] = float(np.sqrt(max(gmm_covs[gmm_idx][0, 0], 0.0)))
        sigma_n_dict[name] = float(np.sqrt(max(gmm_covs[gmm_idx][1, 1], 0.0)))

    mean_ce = float(np.mean(list(centroid_errors.values()))) if centroid_errors else 0.0

    # ── Davies-Bouldin index ──────────────────────────────────────────────────
    matched_idx = list(matched_gt.values())
    K_db = len(matched_idx)
    if K_db >= 2:
        spreads = np.array([
            float(np.sqrt(np.trace(gmm_covs[k]) / 2.0))
            for k in matched_idx
        ])
        centres = gmm_means[matched_idx]
        db_sum  = 0.0
        for i in range(K_db):
            worst = 0.0
            for j in range(K_db):
                if i == j:
                    continue
                d_ij = float(np.sqrt(((centres[i] - centres[j]) ** 2).sum()))
                r_ij = (spreads[i] + spreads[j]) / (d_ij + 1e-12)
                if r_ij > worst:
                    worst = r_ij
            db_sum += worst
        davies_bouldin = db_sum / K_db
    else:
        davies_bouldin = float("nan")

    # ── Overlap fractions ─────────────────────────────────────────────────────
    overlap_fractions: Dict[Tuple[str, str], float] = {}
    if hasattr(phantom, "label_vol") and phantom.label_vol is not None:
        try:
            pts_full = np.column_stack([vx, vn])
            chunk    = 200_000
            gmm_labels = np.empty(len(pts_full), dtype=np.int32)
            for start in range(0, len(pts_full), chunk):
                gmm_labels[start:start+chunk] = gm.predict(
                    pts_full[start:start+chunk]
                )
            gt_labels = phantom.label_vol.ravel()

            mat_name_to_idx = {m.name: i for i, m in enumerate(materials)}
            gmm_to_mat_idx  = {
                v: mat_name_to_idx[k]
                for k, v in matched_gt.items()
                if k in mat_name_to_idx
            }

            for name_a, gmm_idx_a in matched_gt.items():
                mat_idx_a = mat_name_to_idx.get(name_a, -1)
                for name_b, gmm_idx_b in matched_gt.items():
                    if name_b <= name_a:
                        continue
                    mat_idx_b = mat_name_to_idx.get(name_b, -1)
                    xa, na = gt_positions[name_a]
                    xb, nb = gt_positions[name_b]
                    if np.sqrt((xa-xb)**2 + (na-nb)**2) > 2.0:
                        continue
                    mask = np.isin(gt_labels, [mat_idx_a, mat_idx_b])
                    if mask.sum() == 0:
                        continue
                    pred = gmm_labels[mask]
                    true = gt_labels[mask]
                    wrong = sum(
                        gmm_to_mat_idx.get(int(pred[i]), -1) != int(true[i])
                        for i in range(len(true))
                    )
                    overlap_fractions[(name_a, name_b)] = wrong / len(true)
        except Exception as exc:
            warnings.warn(f"Overlap fraction computation failed: {exc}")

    # ── Package GMMFitResult ──────────────────────────────────────────────────
    labels_full = np.full(len(vx), -1, dtype=np.int32)
    try:
        pts_full = np.column_stack([vx, vn])
        chunk = 200_000
        for start in range(0, len(pts_full), chunk):
            labels_full[start:start+chunk] = gm.predict(pts_full[start:start+chunk])
    except Exception:
        pass

    gmm_result = GMMFitResult(
        n_components = n_components,
        means        = gmm_means,
        covariances  = gmm_covs,
        weights      = gmm_wts,
        labels_flat  = labels_full,
        bic          = float(gm.bic(pts)),
        aic          = float(gm.aic(pts)),
    )

    return ClusterQualityMetrics(
        centroid_errors      = centroid_errors,
        sigma_x              = sigma_x_dict,
        sigma_n              = sigma_n_dict,
        mean_centroid_error  = mean_ce,
        davies_bouldin       = davies_bouldin,
        overlap_fractions    = overlap_fractions,
        n_matched            = len(matched_gt),
        gmm                  = gmm_result,
    )


def compare_algorithms(
    results: List,
    phantom,
    energy_idx: int = 6,
    exclude_air: bool = True,
    print_table: bool = True,
) -> Dict[str, "ClusterQualityMetrics"]:
    """
    Evaluate cluster quality for a list of SimulationResult objects and
    optionally print a formatted ASCII comparison table.

    Parameters
    ----------
    results     : list of SimulationResult objects (must have .histogram and .tag)
    phantom     : PhantomData (shared ground truth)
    energy_idx  : X-ray energy index for mu_x lookup (default 6 = 80 keV)
    exclude_air : exclude air from metrics
    print_table : print a formatted table to stdout

    Returns
    -------
    dict {result.tag -> ClusterQualityMetrics}

    Examples
    --------
    ::

        metrics = compare_algorithms([r_fbp, r_sirt, r_sart], phantom=phantom)
    """
    n_mat = len(phantom.materials)
    metrics_dict: Dict[str, "ClusterQualityMetrics"] = {}

    for r in results:
        tag  = getattr(r, "tag", str(r))
        hist = r.histogram if hasattr(r, "histogram") else r
        m    = evaluate_histogram_quality(
            hist, phantom,
            n_components=n_mat,
            energy_idx=energy_idx,
            exclude_air=exclude_air,
        )
        metrics_dict[tag] = m

    if print_table:
        _print_quality_table(metrics_dict)

    return metrics_dict


def _print_quality_table(
    metrics_dict: Dict[str, "ClusterQualityMetrics"],
) -> None:
    """Print a formatted ASCII comparison table of cluster quality metrics."""
    if not metrics_dict:
        return

    all_mats = sorted({
        name
        for m in metrics_dict.values()
        for name in m.centroid_errors
    })

    col_w  = 22
    mat_w  = 14
    sep    = "\u2500" * (col_w + 2 + (mat_w + 2) * len(all_mats) + 14 + 10)
    hdr    = "".join(f"  {n[:mat_w]:<{mat_w}}" for n in all_mats)

    print()
    print("  Cluster quality \u2014 centroid error per material [cm^-1]")
    print(sep)
    _h1 = "Algorithm / Tag"
    _h2 = "Mean err"
    _h3 = "DB index"
    print(f"  {_h1:<{col_w}}{hdr}  {_h2:>10}  {_h3:>10}")
    print(sep)
    for tag, m in metrics_dict.items():
        row = f"  {tag[:col_w]:<{col_w}}"
        for name in all_mats:
            err = m.centroid_errors.get(name, float("nan"))
            row += f"  {err:>{mat_w}.4f}"
        row += f"  {m.mean_centroid_error:>10.4f}  {m.davies_bouldin:>10.4f}"
        print(row)
    print(sep)

    print()
    print("  Cluster spread \u2014 sigma_x / sigma_n [cm^-1]")
    print(sep)
    print(f"  {_h1:<{col_w}}{hdr}")
    print(sep)
    for tag, m in metrics_dict.items():
        row = f"  {tag[:col_w]:<{col_w}}"
        for name in all_mats:
            sx = m.sigma_x.get(name, float("nan"))
            sn = m.sigma_n.get(name, float("nan"))
            row += f"  {sx:.3f}/{sn:.3f}  "
        print(row)
    print(sep)
    print()


def _compute_precision_chol(covariances: np.ndarray) -> np.ndarray:
    """Compute precision Cholesky for sklearn GaussianMixture warm-start."""
    from scipy.linalg import cholesky
    K = covariances.shape[0]
    prec_chol = np.zeros_like(covariances)
    for k in range(K):
        try:
            cov_chol = cholesky(covariances[k], lower=True)
            prec_chol[k] = np.linalg.inv(cov_chol).T
        except Exception:
            prec_chol[k] = np.eye(2)
    return prec_chol
