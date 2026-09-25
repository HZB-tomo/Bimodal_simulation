"""
neutron_xray_sim.analysis.histogram
───────────────────────────────────
The 2-D bimodal (joint) histogram H(μ_x, μ_n) — the core diagnostic.

  • Each pure material phase → compact Gaussian blob
  • Partial-volume voxels at interfaces → line segments between blobs
  • Beam hardening    → horizontal smearing (μ_x biased, μ_n unchanged)
  • Neutron scatter   → vertical shift of all clusters
  • Misalignment      → streaks between the clusters of neighbouring phases
  • Ring artifacts    → striping at specific μ values

Related modules:
  ``analysis.gmm``        — Gaussian-mixture fitting and segmentation
  ``analysis.signatures`` — artifact-signature scores
  ``analysis.quality``    — ground-truth-anchored cluster-quality metrics
  ``plotting.histograms`` — figures
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

__all__ = [
    "DEFAULT_GT_ENERGY_IDX",
    "HistogramResult",
    "compute_bimodal_histogram",
    "compute_ground_truth_histogram",
]

#: Index into ``XRAY_E_KEV`` used for ground-truth μ_x (``XRAY_E_KEV[6]`` = 80 keV,
#: a typical effective energy of a 120 kVp / 2 mm Al spectrum).
DEFAULT_GT_ENERGY_IDX = 6


# ──────────────────────────────────────────────────────────────────────────────
# Data container
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
    mask_flat    : flattened boolean voxel mask, or None if all voxels are used
    """
    H:            np.ndarray
    x_edges:      np.ndarray
    n_edges:      np.ndarray
    x_centres:    np.ndarray
    n_centres:    np.ndarray
    vol_x_flat:   np.ndarray
    vol_n_flat:   np.ndarray
    total_voxels: int
    mask_flat:    Optional[np.ndarray] = None
    """Flattened boolean mask used to select voxels (None = all voxels).
    Needed to align ``vol_*_flat`` with a ground-truth label volume."""

    @property
    def extent(self) -> List[float]:
        """matplotlib imshow extent [x_min, x_max, n_min, n_max]."""
        return [self.x_edges[0], self.x_edges[-1],
                self.n_edges[0], self.n_edges[-1]]



# ──────────────────────────────────────────────────────────────────────────────
# Compute 2-D histogram
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

    mask_flat = None
    if mask is not None:
        mask_flat = np.asarray(mask, dtype=bool).ravel()
        if mask_flat.size != vx.size:
            raise ValueError(f"mask has {mask_flat.size} voxels, volumes have {vx.size}")
        vx = vx[mask_flat]
        vn = vn[mask_flat]

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
        mask_flat=mask_flat,
    )


def compute_ground_truth_histogram(
    phantom,
    bins: int = 256,
    energy_idx: int = DEFAULT_GT_ENERGY_IDX,
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
