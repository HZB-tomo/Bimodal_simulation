"""
neutron_xray_sim.analysis.quality
─────────────────────────────────
Ground-truth-anchored cluster-quality metrics for bimodal histograms.

Shared building blocks (used by this module, ``metrics_table`` and
``metrics_morphology`` so that every metric family computes things the same
way):

* :func:`is_air`                    — is a material the air/background phase?
* :func:`ground_truth_positions`    — exact (μ_x, μ_n) of each phantom material
* :func:`match_components`          — greedy nearest-neighbour cluster matching
* :func:`davies_bouldin`            — Davies–Bouldin separability index
* :func:`pairwise_overlap`          — misclassification rate per material pair
* :func:`ground_truth_labels_for`   — phantom labels aligned with a histogram

High-level API:

* :class:`ClusterQualityMetrics` / :func:`evaluate_histogram_quality`
* :func:`compare_algorithms`
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .gmm import GMMFitResult, fit_gmm
from .histogram import DEFAULT_GT_ENERGY_IDX, HistogramResult

__all__ = [
    "is_air",
    "ground_truth_positions",
    "match_components",
    "davies_bouldin",
    "pairwise_overlap",
    "ground_truth_labels_for",
    "ClusterQualityMetrics",
    "evaluate_histogram_quality",
    "compare_algorithms",
]


# ──────────────────────────────────────────────────────────────────────────────
# Shared building blocks
# ──────────────────────────────────────────────────────────────────────────────

def is_air(material, energy_idx: int = DEFAULT_GT_ENERGY_IDX) -> bool:
    """True for the air / vacuum background phase (by name or by μ ≈ 0)."""
    if material.name.lower() == "air":
        return True
    return bool(material._mu_x_table[energy_idx] < 1e-3 and material.mu_n < 1e-3)


def ground_truth_positions(
    phantom,
    energy_idx: int = DEFAULT_GT_ENERGY_IDX,
    skip_air: bool = True,
    key: str = "symbol",
) -> Tuple[List[int], List[str], np.ndarray]:
    """
    Exact (μ_x, μ_n) position of every material of *phantom*.

    Parameters
    ----------
    energy_idx : index into ``XRAY_E_KEV`` for μ_x (default 80 keV)
    skip_air   : leave out the air phase
    key        : material attribute used as the name (``"symbol"`` or ``"name"``)

    Returns
    -------
    labels : phantom label index of each retained material
    names  : material names (``getattr(material, key)``)
    coords : (n, 2) array of [μ_x, μ_n] in cm⁻¹
    """
    labels, names, coords = [], [], []
    for i, m in enumerate(phantom.materials):
        if skip_air and is_air(m, energy_idx):
            continue
        labels.append(i)
        names.append(getattr(m, key))
        coords.append((float(m._mu_x_table[energy_idx]), float(m.mu_n)))
    return labels, names, np.array(coords, dtype=float).reshape(-1, 2)


def match_components(gt_coords: np.ndarray, comp_coords: np.ndarray) -> Dict[int, int]:
    """
    Greedy nearest-neighbour matching of ground-truth points to components.

    Repeatedly pairs the closest remaining (ground truth, component) pair.

    Returns
    -------
    dict {ground-truth row → component row}
    """
    gt_coords = np.asarray(gt_coords, dtype=float)
    comp_coords = np.asarray(comp_coords, dtype=float)
    D = np.linalg.norm(gt_coords[:, None, :] - comp_coords[None, :, :], axis=-1)
    rem_gt = list(range(len(gt_coords)))
    rem_c = list(range(len(comp_coords)))
    matches: Dict[int, int] = {}
    while rem_gt and rem_c:
        sub = D[np.ix_(rem_gt, rem_c)]
        i, j = np.unravel_index(np.argmin(sub), sub.shape)
        matches[rem_gt[i]] = rem_c[j]
        del rem_gt[i]
        del rem_c[j]
    return matches


def davies_bouldin(centres: np.ndarray, spreads: np.ndarray) -> float:
    """
    Davies–Bouldin index  DB = (1/K) Σ_k max_{j≠k} (s_k + s_j) / d(c_k, c_j).

    Pairs of coincident centres (d ≤ 1e-9) are ignored.  Returns NaN when fewer
    than two clusters are given.  Lower is better; 0 = perfectly separated.
    """
    centres = np.asarray(centres, dtype=float)
    spreads = np.asarray(spreads, dtype=float)
    K = len(centres)
    if K < 2:
        return float("nan")
    D = np.linalg.norm(centres[:, None, :] - centres[None, :, :], axis=-1)
    R = (spreads[:, None] + spreads[None, :]) / np.where(D > 1e-9, D, np.inf)
    np.fill_diagonal(R, -np.inf)
    worst = R.max(axis=1)
    worst = worst[np.isfinite(worst)]
    return float(worst.mean()) if worst.size else float("nan")


def pairwise_overlap(
    true_labels: np.ndarray,
    pred_labels: np.ndarray,
    pairs: Sequence[Tuple[int, int]],
) -> Dict[Tuple[int, int], float]:
    """
    Misclassification rate for each pair of material labels.

    For a pair (a, b), considers the voxels whose true label is a or b and
    returns the fraction whose predicted label differs from the true one.
    Pairs with no voxels are omitted.
    """
    out: Dict[Tuple[int, int], float] = {}
    for a, b in pairs:
        mask = (true_labels == a) | (true_labels == b)
        n = int(mask.sum())
        if n == 0:
            continue
        out[(a, b)] = float(np.count_nonzero(pred_labels[mask] != true_labels[mask]) / n)
    return out


def ground_truth_labels_for(hist: HistogramResult, phantom) -> Optional[np.ndarray]:
    """
    Phantom labels aligned with ``hist.vol_x_flat`` — or None if the histogram
    was not computed on the phantom grid (so voxel-wise comparison is invalid).
    """
    label_vol = getattr(phantom, "label_vol", None)
    if label_vol is None:
        return None
    labels = label_vol.ravel()
    if hist.mask_flat is not None:
        if hist.mask_flat.size != labels.size:
            return None
        labels = labels[hist.mask_flat]
    if labels.size != hist.vol_x_flat.size:
        return None
    return labels


# ──────────────────────────────────────────────────────────────────────────────
# GMM-based cluster-quality metrics
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
        Empty when the histogram is not on the phantom grid.

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
    gmm:                 GMMFitResult

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
    hist: HistogramResult,
    phantom,
    n_components: Optional[int] = None,
    energy_idx: int = DEFAULT_GT_ENERGY_IDX,
    exclude_air: bool = True,
    gmm_n_init: int = 5,
    gmm_max_iter: int = 300,
    overlap_threshold: float = 2.0,
) -> ClusterQualityMetrics:
    """
    Evaluate bimodal-histogram cluster quality against the phantom.

    A GMM is fitted to the (μ_x, μ_n) voxel pairs and each ground-truth
    material is matched greedily to its nearest component.  Then:

    **1. Centroid error** (accuracy)
        ε_k = ‖(μ̂_x, μ̂_n) − (μ_x, μ_n)_GT‖ per material; CE = mean ε_k.
    **2. Cluster spread** (compactness)
        σ_x = √Cov[0,0], σ_n = √Cov[1,1] of the matched component.
    **3. Davies–Bouldin index** (separability) over matched components,
        with s_k = √(trace(Cov_k)/2).  Lower is better.
    **4. Overlap fractions** for neighbouring materials (GT distance below
        *overlap_threshold*), when the histogram is on the phantom grid.

    Parameters
    ----------
    hist          : HistogramResult from compute_bimodal_histogram()
    phantom       : PhantomData (ground-truth positions and label volume)
    n_components  : GMM components.  Defaults to ``len(phantom.materials)``.
    energy_idx    : energy bin for ground-truth μ_x (default 80 keV)
    exclude_air   : leave air out of the metrics
    gmm_n_init    : GMM random initialisations
    gmm_max_iter  : maximum EM iterations

    Returns
    -------
    ClusterQualityMetrics (per-material dicts are keyed by ``Material.name``)
    """
    if n_components is None:
        n_components = len(phantom.materials)

    gmm = fit_gmm(hist, n_components=n_components, n_init=gmm_n_init,
                  random_state=42, subsample=500_000, max_iter=gmm_max_iter)

    gt_label_idx, gt_names, gt_coords = ground_truth_positions(
        phantom, energy_idx, skip_air=exclude_air, key="name")
    matches = match_components(gt_coords, gmm.means)   # gt row → component

    centroid_errors: Dict[str, float] = {}
    sigma_x: Dict[str, float] = {}
    sigma_n: Dict[str, float] = {}
    for g, c in matches.items():
        name = gt_names[g]
        centroid_errors[name] = float(np.linalg.norm(gmm.means[c] - gt_coords[g]))
        sigma_x[name] = float(np.sqrt(max(gmm.covariances[c][0, 0], 0.0)))
        sigma_n[name] = float(np.sqrt(max(gmm.covariances[c][1, 1], 0.0)))
    mean_ce = float(np.mean(list(centroid_errors.values()))) if centroid_errors else 0.0

    comps = list(matches.values())
    spreads = [np.sqrt(np.trace(gmm.covariances[c]) / 2.0) for c in comps]
    db = davies_bouldin(gmm.means[comps], spreads)

    overlap: Dict[Tuple[str, str], float] = {}
    true_labels = ground_truth_labels_for(hist, phantom)
    if true_labels is not None:
        comp_to_label = np.full(gmm.n_components, -1, dtype=np.int64)
        for g, c in matches.items():
            comp_to_label[c] = gt_label_idx[g]
        pred_labels = comp_to_label[gmm.labels_flat]
        pairs, names = [], {}
        for a in range(len(gt_names)):
            for b in range(a + 1, len(gt_names)):
                if np.linalg.norm(gt_coords[a] - gt_coords[b]) > overlap_threshold:
                    continue
                la, lb = gt_label_idx[a], gt_label_idx[b]
                pairs.append((la, lb))
                names[(la, lb)] = tuple(sorted((gt_names[a], gt_names[b])))
        for pair, frac in pairwise_overlap(true_labels, pred_labels, pairs).items():
            overlap[names[pair]] = frac

    return ClusterQualityMetrics(
        centroid_errors=centroid_errors,
        sigma_x=sigma_x,
        sigma_n=sigma_n,
        mean_centroid_error=mean_ce,
        davies_bouldin=db,
        overlap_fractions=overlap,
        n_matched=len(matches),
        gmm=gmm,
    )


def compare_algorithms(
    results: List,
    phantom,
    energy_idx: int = DEFAULT_GT_ENERGY_IDX,
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
    energy_idx  : X-ray energy index for mu_x lookup (default 80 keV)
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
