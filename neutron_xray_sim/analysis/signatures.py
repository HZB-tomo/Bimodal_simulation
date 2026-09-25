"""
neutron_xray_sim.analysis.signatures
────────────────────────────────────
Ground-truth-free scores that quantify characteristic artifact signatures in a
bimodal histogram (streaks, diagonal smear, marginal asymmetry / shift).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np

from .gmm import GMMFitResult
from .histogram import HistogramResult

__all__ = ["ArtifactSignatures", "detect_artifact_signatures"]


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
