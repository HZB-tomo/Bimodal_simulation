"""
neutron_xray_sim.analysis.gmm
─────────────────────────────
Gaussian-mixture modelling of the bimodal histogram and histogram-based
segmentation of the reconstructed volumes.

* :func:`fit_gmm`           — fit a K-component GMM to the (μ_x, μ_n) voxel pairs
* :func:`auto_fit_gmm`      — choose K by BIC
* :func:`predict_gmm`       — assign points to components of a fitted GMM
* :func:`segment_by_gmm`    — label volume from a fitted GMM
* :func:`segment_by_polygon`— label volume from hand-drawn histogram polygons
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
from matplotlib.path import Path

from .histogram import HistogramResult

__all__ = [
    "GMMFitResult",
    "fit_gmm",
    "auto_fit_gmm",
    "predict_gmm",
    "segment_by_gmm",
    "segment_by_polygon",
]


@dataclass
class GMMFitResult:
    """
    Result of a Gaussian-mixture fit to the bimodal histogram.

    Attributes
    ----------
    n_components : number of components
    means        : (K, 2)  [μ_x, μ_n] component centres  [cm⁻¹]
    covariances  : (K, 2, 2) full covariance matrices (always full, whatever
                   ``covariance_type`` was used for fitting)
    weights      : (K,) mixing weights
    labels_flat  : per-voxel component index, aligned with the histogram's
                   ``vol_x_flat`` / ``vol_n_flat`` (−1 = unassigned)
    bic, aic     : information criteria of the fit (lower = better)
    """
    n_components: int
    means:        np.ndarray
    covariances:  np.ndarray
    weights:      np.ndarray
    labels_flat:  np.ndarray
    bic:          float
    aic:          float


def _require_sklearn():
    try:
        from sklearn.mixture import GaussianMixture
    except ImportError as exc:  # pragma: no cover - sklearn is a core dependency
        raise ImportError(
            "scikit-learn is required for GMM fitting: pip install scikit-learn"
        ) from exc
    return GaussianMixture


def _full_covariances(gm) -> np.ndarray:
    """Return sklearn covariances as a (K, d, d) array for any covariance_type."""
    cov = np.asarray(gm.covariances_)
    K, d = gm.means_.shape
    if gm.covariance_type == "full":
        return cov
    if gm.covariance_type == "tied":
        return np.broadcast_to(cov, (K, d, d)).copy()
    if gm.covariance_type == "diag":
        return np.stack([np.diag(c) for c in cov])
    if gm.covariance_type == "spherical":
        return np.stack([np.eye(d) * c for c in cov])
    raise ValueError(f"Unknown covariance_type {gm.covariance_type!r}")


def predict_gmm(points: np.ndarray, gmm: GMMFitResult, chunk: int = 200_000) -> np.ndarray:
    """
    Maximum-posterior component index for each point.

    Parameters
    ----------
    points : (n, 2) array of (μ_x, μ_n)
    gmm    : fitted GMMFitResult

    Returns
    -------
    (n,) int32 component indices
    """
    points = np.asarray(points, dtype=np.float64)
    K = gmm.n_components
    log_w = np.log(np.clip(gmm.weights, 1e-300, None))
    prec = np.empty_like(gmm.covariances, dtype=np.float64)
    log_det = np.empty(K)
    for k in range(K):
        cov = gmm.covariances[k] + 1e-12 * np.eye(points.shape[1])
        prec[k] = np.linalg.inv(cov)
        log_det[k] = np.linalg.slogdet(cov)[1]

    labels = np.empty(len(points), dtype=np.int32)
    for start in range(0, len(points), chunk):
        p = points[start:start + chunk]
        score = np.empty((len(p), K))
        for k in range(K):
            d = p - gmm.means[k]
            maha = np.einsum("ni,ij,nj->n", d, prec[k], d)
            score[:, k] = log_w[k] - 0.5 * (maha + log_det[k])
        labels[start:start + chunk] = np.argmax(score, axis=1)
    return labels


def fit_gmm(
    hist: HistogramResult,
    n_components: int = 5,
    covariance_type: str = "full",
    n_init: int = 5,
    random_state: int = 0,
    subsample: int = 50_000,
    means_init: Optional[np.ndarray] = None,
    max_iter: int = 300,
) -> GMMFitResult:
    """
    Fit a Gaussian mixture model to the (μ_x, μ_n) voxel pairs of *hist*.

    Parameters
    ----------
    hist            : HistogramResult from ``compute_bimodal_histogram``
    n_components    : number of Gaussian components
    covariance_type : 'full', 'tied', 'diag' or 'spherical'
    n_init          : number of EM initialisations (the best likelihood is kept)
    random_state    : random seed (subsampling and EM initialisation)
    subsample       : maximum number of voxels used for fitting (speed)
    means_init      : optional (K, 2) initial means (e.g. ground-truth positions)
    max_iter        : maximum EM iterations

    Returns
    -------
    GMMFitResult (covariances are always returned as full 2×2 matrices)
    """
    GaussianMixture = _require_sklearn()

    pts = np.column_stack([hist.vol_x_flat, hist.vol_n_flat])
    if len(pts) == 0:
        raise ValueError(
            "Histogram has no voxel samples (vol_x_flat is empty); histograms "
            "re-loaded from a SimCache must be recomputed from the volumes."
        )
    if len(pts) > subsample:
        rng = np.random.default_rng(random_state)
        pts_fit = pts[rng.choice(len(pts), subsample, replace=False)]
    else:
        pts_fit = pts

    gm = GaussianMixture(
        n_components=n_components,
        covariance_type=covariance_type,
        n_init=1 if means_init is not None else n_init,
        means_init=means_init,
        random_state=random_state,
        max_iter=max_iter,
    )
    gm.fit(pts_fit)

    result = GMMFitResult(
        n_components=n_components,
        means=gm.means_,
        covariances=_full_covariances(gm),
        weights=gm.weights_,
        labels_flat=np.empty(0, dtype=np.int32),
        bic=float(gm.bic(pts_fit)),
        aic=float(gm.aic(pts_fit)),
    )
    result.labels_flat = predict_gmm(pts, result)
    return result


def auto_fit_gmm(
    hist: HistogramResult,
    min_k: int = 2,
    max_k: int = 8,
    random_state: int = 0,
    verbose: bool = True,
) -> Optional[GMMFitResult]:
    """
    Fit GMMs for k = min_k..max_k and return the one with the lowest BIC.

    Returns None only if every fit failed (a warning is issued per failure).
    """
    best_fit: Optional[GMMFitResult] = None
    for k in range(min_k, max_k + 1):
        try:
            result = fit_gmm(hist, n_components=k, random_state=random_state)
        except Exception as exc:  # noqa: BLE001 - report and keep trying other k
            warnings.warn(f"GMM fitting failed for k={k}: {exc}", stacklevel=2)
            continue
        if verbose:
            print(f"  k={k}: BIC={result.bic:.1f}, AIC={result.aic:.1f}")
        if best_fit is None or result.bic < best_fit.bic:
            best_fit = result
    return best_fit


def segment_by_gmm(
    vol_x: np.ndarray,
    vol_n: np.ndarray,
    gmm_result: GMMFitResult,
    mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Label every voxel with its most likely GMM component.

    Parameters
    ----------
    vol_x, vol_n : reconstructed volumes (same shape)
    gmm_result   : GMMFitResult from :func:`fit_gmm`
    mask         : optional boolean mask; voxels outside it get label −1

    Returns
    -------
    label volume, int32, same shape as *vol_x*, values 0..K−1 (−1 = masked)
    """
    if vol_x.shape != vol_n.shape:
        raise ValueError(f"Volume shapes must match: {vol_x.shape} vs {vol_n.shape}")
    pts = np.column_stack([vol_x.ravel(), vol_n.ravel()])
    labels = np.full(vol_x.size, -1, dtype=np.int32)
    if mask is None:
        labels[:] = predict_gmm(pts, gmm_result)
    else:
        m = np.asarray(mask, dtype=bool).ravel()
        labels[m] = predict_gmm(pts[m], gmm_result)
    return labels.reshape(vol_x.shape)


def segment_by_polygon(
    vol_x: np.ndarray,
    vol_n: np.ndarray,
    polygons: List[np.ndarray],
) -> np.ndarray:
    """
    Segment the volume with hand-drawn polygons on the bimodal histogram.

    Parameters
    ----------
    vol_x, vol_n : volumes (same shape)
    polygons     : list of (M, 2) arrays of (μ_x, μ_n) polygon vertices;
                   the first polygon is label 1, the second label 2, …

    Returns
    -------
    label volume, int32, 0 = unassigned, 1..K = polygon index (later polygons
    win where polygons overlap)
    """
    pts = np.column_stack([vol_x.ravel(), vol_n.ravel()])
    labels = np.zeros(len(pts), dtype=np.int32)
    for i, verts in enumerate(polygons, start=1):
        labels[Path(verts).contains_points(pts)] = i
    return labels.reshape(vol_x.shape)
