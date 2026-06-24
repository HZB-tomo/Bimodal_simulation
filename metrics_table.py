"""
compute_histogram_metrics  (v2)
─────────────────────────────────
Unified computation of the bimodal-histogram shape and quality metrics
defined in the math companion document (§1 and §2).

This v2 release adds three optional capabilities that the v1 release lacked,
all motivated by the empirical failure mode observed on the realistic 18650
battery phantom:

  (1) Rare-class-aware GMM fit
      A new keyword `gt_seeded_gmm=True` initialises the GMM with one
      component seeded at every ground-truth material position and uses
      class-balanced subsampling so that rare phases (e.g. the Al current
      collector at 0.3 % voxel fraction) are not absorbed by abundant ones
      (graphite, NMC, separator).

  (2) Bin-aspect-normalised shape metrics
      A new keyword `normalize_shape_axes=True` rescales the histogram so
      both axes have unit empirical std before computing S_h and S_v,
      removing the bin-aspect dependency that gives spuriously large streak
      scores on clean runs of phantoms whose μ_x range vastly exceeds
      their μ_n range (battery cells: μ_x up to 23 cm⁻¹, μ_n up to 3 cm⁻¹).

  (3) Automated pathology detection
      After matching, the function checks for five known degeneracies and
      emits warnings via `table.notes`. The user can also pass
      `raise_on_pathology=True` to make warnings fatal in batch pipelines.

All three additions are OFF by default; v1 behaviour is preserved
bit-for-bit when the new keywords are not supplied.

Usage
─────
    >>> # v1-equivalent call (unchanged behaviour)
    >>> table = compute_histogram_metrics(phantom, hist)
    >>>
    >>> # Recommended for the realistic battery phantom
    >>> table = compute_histogram_metrics(
    ...     phantom, hist,
    ...     gt_seeded_gmm=True,
    ...     normalize_shape_axes=True,
    ...     ref_hist=hist_clean,
    ... )

Author: neutron_xray_sim contributors.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Dict, Tuple, List, Any

import numpy as np

from neutron_xray_sim.histogram import (
    HistogramResult,
    GMMFitResult,
    fit_gmm,
    detect_artifact_signatures,
)


# ──────────────────────────────────────────────────────────────────────────────
#  Pathology-detection thresholds (overridable through kwargs)
# ──────────────────────────────────────────────────────────────────────────────
DEFAULT_PATHOLOGY_THRESHOLDS = {
    "max_centroid_error_cm":      3.0,   # ε_k above this is implausible
    "min_component_weight":       0.005, # weight below this = degenerate
    "max_elongation":              30.0, # E_k above this = partial-volume line
    "duplicate_component_eps_cm": 0.20,  # two components closer than this = duplicate
    "uncovered_gt_distance_cm":    1.0,  # GT material with no component within this = coverage gap
}


# ──────────────────────────────────────────────────────────────────────────────
#  Metric metadata
# ──────────────────────────────────────────────────────────────────────────────
_METRIC_INFO: Dict[str, Dict[str, str]] = {
    "S_h": {"family": "shape", "name": "Horizontal streak score",
            "unit": "dimensionless", "eq": "(1.2)",
            "diagnoses": "rigid-body misalignment between modalities",
            "ground_truth_required": "no"},
    "S_v": {"family": "shape", "name": "Vertical streak score",
            "unit": "dimensionless", "eq": "(1.3)",
            "diagnoses": "ring artifacts (especially X-ray channel)",
            "ground_truth_required": "no"},
    "S_d": {"family": "shape", "name": "Diagonal smear score",
            "unit": "Pearson ρ ∈ [-1, 1]", "eq": "(1.6)",
            "diagnoses": "polychromatic X-ray beam hardening, X-ray scatter",
            "ground_truth_required": "no"},
    "A_x": {"family": "shape", "name": "X-ray marginal asymmetry",
            "unit": "dimensionless ((mean − median)/σ)", "eq": "(1.8)",
            "diagnoses": "X-ray cupping (a beam-hardening sub-effect)",
            "ground_truth_required": "no"},
    "Delta_n": {"family": "shape", "name": "Neutron marginal shift",
                "unit": "cm⁻¹", "eq": "(1.9)",
                "diagnoses": "neutron scatter build-up (requires reference)",
                "ground_truth_required": "no (but requires a reference run)"},
    "E_k": {"family": "shape (per-cluster)", "name": "Cluster elongation",
            "unit": "dimensionless (≥ 1)", "eq": "(1.10)",
            "diagnoses": "anisotropic per-cluster broadening",
            "ground_truth_required": "no (uses GMM only)"},
    "eps_k": {"family": "quality (per-cluster)",
              "name": "Per-material centroid error",
              "unit": "cm⁻¹", "eq": "(2.1)",
              "diagnoses": "quantitative fidelity of recovered cluster position",
              "ground_truth_required": "yes"},
    "CE":    {"family": "quality", "name": "Mean centroid error",
              "unit": "cm⁻¹", "eq": "(2.2)",
              "diagnoses": "global cluster-position fidelity",
              "ground_truth_required": "yes"},
    "sigma_x_k": {"family": "quality (per-cluster)",
                  "name": "Cluster spread along μ_x",
                  "unit": "cm⁻¹", "eq": "(2.3)",
                  "diagnoses": "axial cluster compactness in the μ_x direction",
                  "ground_truth_required": "yes (matching only)"},
    "sigma_n_k": {"family": "quality (per-cluster)",
                  "name": "Cluster spread along μ_n",
                  "unit": "cm⁻¹", "eq": "(2.3)",
                  "diagnoses": "axial cluster compactness in the μ_n direction",
                  "ground_truth_required": "yes (matching only)"},
    "DB":    {"family": "quality", "name": "Davies–Bouldin index",
              "unit": "dimensionless", "eq": "(2.5)",
              "diagnoses": "cluster separability (lower is better)",
              "ground_truth_required": "yes (matching only)"},
    "O_ab":  {"family": "quality (pairwise)",
              "name": "Pairwise material overlap fraction",
              "unit": "fraction ∈ [0, 1]", "eq": "(2.7)",
              "diagnoses": "downstream segmentation error rate, per neighbour pair",
              "ground_truth_required": "yes (label volume)"},
}


# ──────────────────────────────────────────────────────────────────────────────
#  Result container
# ──────────────────────────────────────────────────────────────────────────────
@dataclass
class HistogramMetricsTable:
    """
    Output of `compute_histogram_metrics`. Three views of the same data:

      * `scalars`     : dict[name → float | None]
      * `per_cluster` : dict[name → dict[material → float]]
      * `pairwise`    : dict[(material_a, material_b) → float]

    Plus metadata recording how the metrics were computed.

    The `pathologies` field is a list of structured diagnostics emitted by
    the v2 pathology detector; `notes` is a flat list of human-readable
    strings (the v1 field, retained for compatibility).
    """
    scalars:     Dict[str, Optional[float]] = field(default_factory=dict)
    per_cluster: Dict[str, Dict[str, float]] = field(default_factory=dict)
    pairwise:    Dict[Tuple[str, str], float] = field(default_factory=dict)

    has_ground_truth: bool = False
    has_reference:    bool = False
    n_components:     Optional[int] = None
    energy_idx:       Optional[int] = None
    notes:            List[str] = field(default_factory=list)
    pathologies:      List[Dict[str, Any]] = field(default_factory=list)

    # ────── pretty-printing ──────
    def __str__(self) -> str:
        lines = []
        sep = "─" * 78
        lines.append(sep)
        lines.append(" Bimodal histogram — shape and quality metrics ")
        lines.append(sep)
        lines.append(f"   ground truth: {'yes' if self.has_ground_truth else 'no'}")
        lines.append(f"   reference run: {'yes' if self.has_reference else 'no'}")
        if self.n_components is not None:
            lines.append(f"   K (GMM): {self.n_components}")
        if self.energy_idx is not None:
            lines.append(f"   X-ray energy bin: {self.energy_idx}")
        lines.append("")

        # Pathologies first — always visible
        if self.pathologies:
            lines.append(" ⚠  Pathology warnings  ⚠")
            lines.append(sep)
            for p in self.pathologies:
                tag = p.get("severity", "warning").upper()
                msg = p.get("message", "<no message>")
                lines.append(f"   [{tag:7}] {msg}")
            lines.append("")

        # Shape
        lines.append(" Shape metrics  (no ground truth required)")
        lines.append(sep)
        for k in ("S_h", "S_v", "S_d", "A_x", "Delta_n"):
            v = self.scalars.get(k)
            info = _METRIC_INFO[k]
            cell = f"{v:>10.4f}" if v is not None else f"{'N/A':>10}"
            lines.append(f"  {info['eq']:>7} {k:<10}  {cell}     {info['name']}")
        if self.per_cluster.get("E_k"):
            lines.append("")
            lines.append("   (1.10) E_k       per-cluster elongation:")
            for cl, val in self.per_cluster["E_k"].items():
                lines.append(f"           {cl:>20}: {val:>8.3f}")

        # Quality
        if self.has_ground_truth:
            lines.append("")
            lines.append(" Quality metrics  (ground-truth-anchored)")
            lines.append(sep)
            for k in ("CE", "DB"):
                v = self.scalars.get(k)
                info = _METRIC_INFO[k]
                cell = f"{v:>10.4f}" if v is not None else f"{'N/A':>10}"
                lines.append(
                    f"  {info['eq']:>7} {k:<10}  {cell}     {info['name']}  "
                    f"[{info['unit']}]"
                )
            for metric_name in ("eps_k", "sigma_x_k", "sigma_n_k"):
                d = self.per_cluster.get(metric_name, {})
                if not d:
                    continue
                eq = _METRIC_INFO[metric_name]["eq"]
                lines.append("")
                lines.append(f"   {eq} {metric_name}     "
                             f"{_METRIC_INFO[metric_name]['name']} [cm⁻¹]:")
                for cl, val in d.items():
                    lines.append(f"           {cl:>20}: {val:>8.4f}")
            if self.pairwise:
                lines.append("")
                lines.append("   (2.7) O_ab     pairwise overlap fractions "
                             "(neighbours only, < 2 cm⁻¹ apart):")
                for (a, b), val in self.pairwise.items():
                    lines.append(f"           {a:>10} ↔ {b:<10}: {val:>7.4f}")
        else:
            lines.append("")
            lines.append(" Quality metrics  : skipped — no ground truth.")

        if self.notes:
            lines.append("")
            lines.append(" Notes")
            lines.append(sep)
            for n in self.notes:
                lines.append(f"   • {n}")
        lines.append(sep)
        return "\n".join(lines)

    __repr__ = __str__

    # ────── exports ──────
    def to_records(self) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        for k in ("S_h", "S_v", "S_d", "A_x", "Delta_n"):
            if k not in self.scalars:
                continue
            info = _METRIC_INFO[k]
            rows.append({**{kk: info[kk] for kk in
                            ("family", "name", "eq", "unit", "diagnoses")},
                         "metric": k, "cluster_or_pair": "—",
                         "value": self.scalars[k]})
        for cl, val in self.per_cluster.get("E_k", {}).items():
            info = _METRIC_INFO["E_k"]
            rows.append({**{kk: info[kk] for kk in
                            ("family", "name", "eq", "unit", "diagnoses")},
                         "metric": "E_k", "cluster_or_pair": cl, "value": val})
        for k in ("CE", "DB"):
            if k not in self.scalars:
                continue
            info = _METRIC_INFO[k]
            rows.append({**{kk: info[kk] for kk in
                            ("family", "name", "eq", "unit", "diagnoses")},
                         "metric": k, "cluster_or_pair": "—",
                         "value": self.scalars[k]})
        for k in ("eps_k", "sigma_x_k", "sigma_n_k"):
            for cl, val in self.per_cluster.get(k, {}).items():
                info = _METRIC_INFO[k]
                rows.append({**{kk: info[kk] for kk in
                                ("family", "name", "eq", "unit", "diagnoses")},
                             "metric": k, "cluster_or_pair": cl, "value": val})
        info = _METRIC_INFO["O_ab"]
        for (a, b), val in self.pairwise.items():
            rows.append({**{kk: info[kk] for kk in
                            ("family", "name", "eq", "unit", "diagnoses")},
                         "metric": "O_ab", "cluster_or_pair": f"{a} ↔ {b}",
                         "value": val})
        return rows

    def to_dataframe(self):
        try:
            import pandas as pd
        except ImportError as e:
            raise ImportError("pandas is required for to_dataframe()") from e
        return pd.DataFrame(self.to_records())

    def to_csv(self, path: str) -> None:
        import csv
        records = self.to_records()
        if not records:
            raise ValueError("No metrics to write.")
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(records[0].keys()))
            writer.writeheader()
            writer.writerows(records)


# ──────────────────────────────────────────────────────────────────────────────
#  v2 internals — rare-class-aware fit and pathology detection
# ──────────────────────────────────────────────────────────────────────────────
def _fit_gmm_seeded(
    hist:               HistogramResult,
    phantom,
    keep_idx:           List[int],
    energy_idx:         int,
    n_components:       int,
    balanced_subsample: bool,
    subsample_per_class: int,
    random_state:       int,
) -> GMMFitResult:
    """
    Ground-truth-seeded GMM fit (v2 only).

    Initialises the GMM means at the ground-truth positions of every retained
    material, optionally with extra components for partial-volume tails.
    Optionally draws class-balanced samples to prevent abundant phases from
    monopolising the EM optimisation.
    """
    from sklearn.mixture import GaussianMixture

    materials = list(phantom.materials)
    gt_means = np.array([
        [float(materials[i]._mu_x_table[energy_idx]),
         float(materials[i].mu_n)]
        for i in keep_idx
    ])
    n_gt = len(gt_means)

    # If the user asked for more components than ground-truth materials, pad
    # the means_init with random points drawn from the histogram bulk so the
    # extra components can model partial-volume tails / unmodelled phases.
    if n_components > n_gt:
        rng = np.random.default_rng(random_state)
        # draw extra seed points from the central 50% of the empirical
        # distribution to avoid placing seeds far in the tails
        vx, vn = hist.vol_x_flat, hist.vol_n_flat
        qx = np.quantile(vx, [0.25, 0.75])
        qn = np.quantile(vn, [0.25, 0.75])
        extra = rng.uniform(
            low=[qx[0], qn[0]], high=[qx[1], qn[1]],
            size=(n_components - n_gt, 2),
        )
        means_init = np.vstack([gt_means, extra])
    elif n_components < n_gt:
        # User asked for *fewer* components than materials — fall back to the
        # n_components closest-to-centroid materials to avoid seed mismatch.
        # This is unusual; warn via the caller.
        means_init = gt_means[:n_components]
    else:
        means_init = gt_means

    # Class-balanced subsample of voxel pairs.
    pts_x = hist.vol_x_flat
    pts_n = hist.vol_n_flat
    pts   = np.column_stack([pts_x, pts_n])

    if balanced_subsample and hasattr(phantom, "label_vol") \
            and phantom.label_vol is not None:
        labels_flat = phantom.label_vol.ravel()
        rng = np.random.default_rng(random_state)
        chunks = []
        for mat_idx in keep_idx:
            mask_idx = np.where(labels_flat == mat_idx)[0]
            if mask_idx.size == 0:
                continue
            n_take = min(subsample_per_class, mask_idx.size)
            chosen = rng.choice(mask_idx, n_take, replace=False)
            chunks.append(pts[chosen])
        pts_fit = np.vstack(chunks) if chunks else pts
    else:
        pts_fit = pts

    # Cap the fit set; class-balanced or not, very large arrays slow down EM
    if len(pts_fit) > 200_000:
        rng = np.random.default_rng(random_state)
        idx = rng.choice(len(pts_fit), 200_000, replace=False)
        pts_fit = pts_fit[idx]

    gm = GaussianMixture(
        n_components=n_components,
        covariance_type="full",
        n_init=1,                # we have a good init; no need to randomise
        means_init=means_init,
        max_iter=300,
        random_state=random_state,
    )
    gm.fit(pts_fit)

    # Predict labels for ALL voxels
    chunk = 100_000
    labels = np.empty(len(pts), dtype=np.int32)
    for start in range(0, len(pts), chunk):
        labels[start:start+chunk] = gm.predict(pts[start:start+chunk])

    return GMMFitResult(
        n_components=n_components,
        means=gm.means_,
        covariances=gm.covariances_,
        weights=gm.weights_,
        labels_flat=labels,
        bic=gm.bic(pts_fit),
        aic=gm.aic(pts_fit),
    )


def _detect_pathologies(
    gmm:                 GMMFitResult,
    sigma:               Dict[str, int],
    gt_positions:        Dict[str, np.ndarray],
    eps_dict:            Dict[str, float],
    elongations:         Dict[int, float],
    thresholds:          Dict[str, float],
    overlap_threshold:   float,
) -> List[Dict[str, Any]]:
    """Five-condition pathology check; returns list of structured warnings."""
    paths: List[Dict[str, Any]] = []

    # 1. Implausibly large centroid errors
    bad_eps = {n: e for n, e in eps_dict.items()
               if e > thresholds["max_centroid_error_cm"]}
    if bad_eps:
        for name, e in sorted(bad_eps.items(), key=lambda kv: -kv[1]):
            paths.append({
                "severity": "warning",
                "kind": "large_centroid_error",
                "message": (
                    f"Material '{name}' has ε_k = {e:.2f} cm⁻¹, exceeding the "
                    f"plausibility threshold of "
                    f"{thresholds['max_centroid_error_cm']:.1f} cm⁻¹. "
                    f"Likely cause: GMM matching defect (no nearby component) "
                    f"rather than reconstruction error. Try gt_seeded_gmm=True "
                    f"or increase n_components."
                ),
                "data": {"material": name, "eps_k": e},
            })

    # 2. Degenerate component weights
    weak = [k for k, w in enumerate(gmm.weights)
            if w < thresholds["min_component_weight"]]
    if weak:
        paths.append({
            "severity": "info",
            "kind": "degenerate_component_weight",
            "message": (
                f"GMM components {weak} have mixing weight < "
                f"{thresholds['min_component_weight']}. "
                f"They model rare/empty regions and may inflate DB."
            ),
            "data": {"components": weak,
                     "weights": [float(gmm.weights[k]) for k in weak]},
        })

    # 3. Extreme elongation: GMM fitting partial-volume lines as clusters
    big_elong = {k: e for k, e in elongations.items()
                 if e > thresholds["max_elongation"]}
    if big_elong:
        paths.append({
            "severity": "warning",
            "kind": "elongated_components",
            "message": (
                f"GMM components {sorted(big_elong)} have elongation > "
                f"{thresholds['max_elongation']:.0f}. They are fitting "
                f"partial-volume *lines* rather than physical clusters; CE "
                f"and DB inherit the elongated covariance and may be "
                f"misleading. Consider a Student-t mixture, isotropic "
                f"covariance, or excluding partial-volume voxels."
            ),
            "data": big_elong,
        })

    # 4. Duplicate components (two close means) — overcoverage of one phase
    K = gmm.n_components
    centres = gmm.means
    eps_dup = thresholds["duplicate_component_eps_cm"]
    duplicates = []
    for i in range(K):
        for j in range(i + 1, K):
            if np.linalg.norm(centres[i] - centres[j]) < eps_dup:
                duplicates.append((i, j))
    if duplicates:
        paths.append({
            "severity": "info",
            "kind": "duplicate_components",
            "message": (
                f"GMM components are co-located (closer than {eps_dup} cm⁻¹) "
                f"in pairs {duplicates}. Implies one physical phase is "
                f"over-fitted; reduce n_components or use BIC selection."
            ),
            "data": {"pairs": duplicates},
        })

    # 5. Coverage gaps: ground-truth materials with no component within range
    uncovered = []
    cov_thr = thresholds["uncovered_gt_distance_cm"]
    for name, gt in gt_positions.items():
        d_min = np.min(np.linalg.norm(centres - gt, axis=1))
        if d_min > cov_thr:
            uncovered.append((name, float(d_min)))
    if uncovered:
        for name, d in uncovered:
            paths.append({
                "severity": "warning",
                "kind": "uncovered_ground_truth",
                "message": (
                    f"Ground-truth material '{name}' has no GMM component "
                    f"within {cov_thr:.1f} cm⁻¹ (nearest = {d:.2f} cm⁻¹). "
                    f"This material is invisible to the current fit. "
                    f"Try gt_seeded_gmm=True or increase n_components."
                ),
                "data": {"material": name, "nearest_component_distance": d},
            })

    return paths


def _normalised_streak_scores(hist: HistogramResult) -> Tuple[float, float]:
    """
    Compute S_h and S_v on a histogram whose two axes have been rescaled
    so each has unit empirical std.  This removes the bin-aspect-ratio
    dependency that makes raw S_h, S_v dependent on the (μ_x, μ_n) plane
    aspect ratio of the phantom.
    """
    sx = float(np.std(hist.vol_x_flat) + 1e-12)
    sn = float(np.std(hist.vol_n_flat) + 1e-12)
    # rebuild the histogram on rescaled coordinates
    x = hist.vol_x_flat / sx
    n = hist.vol_n_flat / sn
    nb = hist.H.shape[0]            # use the same bin count
    H, _, _ = np.histogram2d(
        x, n,
        bins=[nb, nb],
        range=[[x.min(), x.max()], [n.min(), n.max()]],
    )
    Hf = H.T / (H.sum() + 1e-12)    # neutron on vertical axis, normalised
    row_var = Hf.var(axis=1)
    col_var = Hf.var(axis=0)
    S_h = float(np.max(row_var) / (np.mean(col_var) + 1e-12))
    S_v = float(np.max(col_var) / (np.mean(row_var) + 1e-12))
    return S_h, S_v


# ──────────────────────────────────────────────────────────────────────────────
#  Main entry point
# ──────────────────────────────────────────────────────────────────────────────
def compute_histogram_metrics(
    phantom,
    hist_recon: HistogramResult,
    *,
    ref_hist:               Optional[HistogramResult] = None,
    gmm:                    Optional[GMMFitResult]    = None,
    n_components:           Optional[int]             = None,
    energy_idx:             int                       = 6,
    overlap_threshold:      float                     = 2.0,
    skip_air:               bool                      = True,
    gmm_kwargs:             Optional[Dict[str, Any]]  = None,
    # ───── v2 additions ─────
    gt_seeded_gmm:          bool                      = False,
    subsample_per_class:    int                       = 20_000,
    normalize_shape_axes:   bool                      = False,
    detect_pathologies:     bool                      = True,
    pathology_thresholds:   Optional[Dict[str, float]] = None,
    raise_on_pathology:     bool                      = False,
) -> HistogramMetricsTable:
    """
    Compute every shape and quality metric on a 2-D bimodal histogram.

    See module docstring for the full discussion of v2 additions.

    New v2 keyword arguments
    ────────────────────────
    gt_seeded_gmm : bool, default False
        Initialise the GMM with means seeded at ground-truth positions and
        use class-balanced subsampling. Recommended for phantoms with
        rare phases (e.g. battery current collectors).

    subsample_per_class : int, default 20_000
        Voxels drawn per ground-truth material when gt_seeded_gmm=True.

    normalize_shape_axes : bool, default False
        Rescale the histogram so both axes have unit std before computing
        S_h and S_v. Recommended for phantoms with strongly anisotropic
        bimodal range (e.g. battery cells: μ_x ≈ 23 cm⁻¹, μ_n ≈ 3 cm⁻¹).

    detect_pathologies : bool, default True
        Run the five-condition pathology checker. Results are stored in
        `table.pathologies` as a list of structured warnings.

    pathology_thresholds : dict[str → float], optional
        Override the defaults in DEFAULT_PATHOLOGY_THRESHOLDS.

    raise_on_pathology : bool, default False
        If True and any pathology with severity='warning' is detected,
        raise a RuntimeError instead of returning a table. Useful in
        batch sweep pipelines that should fail loudly on degenerate fits.
    """
    thresholds = dict(DEFAULT_PATHOLOGY_THRESHOLDS)
    if pathology_thresholds is not None:
        thresholds.update(pathology_thresholds)

    # ── 1. Pull ground-truth positions ───────────────────────────────────────
    materials = list(phantom.materials)
    if skip_air:
        keep_idx = [
            i for i, m in enumerate(materials)
            if not (m._mu_x_table[energy_idx] < 1e-3 and m.mu_n < 1e-3)
        ]
    else:
        keep_idx = list(range(len(materials)))
    gt_positions: Dict[str, np.ndarray] = {
        materials[i].symbol: np.array([
            float(materials[i]._mu_x_table[energy_idx]),
            float(materials[i].mu_n),
        ])
        for i in keep_idx
    }
    n_materials = len(gt_positions)
    if n_materials == 0:
        raise ValueError("No non-air materials retained.")

    # ── 2. Fit GMM if not supplied ───────────────────────────────────────────
    if gmm is None:
        K = n_components if n_components is not None else n_materials
        if gt_seeded_gmm:
            gmm = _fit_gmm_seeded(
                hist_recon, phantom, keep_idx, energy_idx,
                n_components=K, balanced_subsample=True,
                subsample_per_class=subsample_per_class,
                random_state=0,
            )
        else:
            kw = {"n_components": K} | (gmm_kwargs or {})
            gmm = fit_gmm(hist_recon, **kw)

    # ── 3. Shape metrics ─────────────────────────────────────────────────────
    sig = detect_artifact_signatures(hist_recon, gmm=gmm, ref_hist=ref_hist)

    if normalize_shape_axes:
        S_h, S_v = _normalised_streak_scores(hist_recon)
    else:
        S_h, S_v = float(sig.horizontal_streak_score), \
                   float(sig.vertical_streak_score)

    scalars: Dict[str, Optional[float]] = {
        "S_h":     S_h,
        "S_v":     S_v,
        "S_d":     float(sig.diagonal_smear_score),
        "A_x":     float(sig.marginal_asymmetry_x),
        "Delta_n": float(sig.marginal_shift_n) if ref_hist is not None else None,
    }
    per_cluster: Dict[str, Dict[str, float]] = {
        "E_k": {f"comp_{k}": float(v) for k, v in sig.cluster_elongation.items()},
    }
    notes: List[str] = []
    if ref_hist is None:
        notes.append("Δ_n (Eq. 1.9) requires a clean reference; reported as None.")
    if normalize_shape_axes:
        notes.append("S_h, S_v computed on axis-normalised histogram (v2 mode).")

    # ── 4. Quality metrics ───────────────────────────────────────────────────
    has_gt = hasattr(phantom, "label_vol") and phantom.label_vol is not None
    pathologies: List[Dict[str, Any]] = []

    if has_gt:
        gt_names = list(gt_positions.keys())
        gt_pts = np.stack([gt_positions[n] for n in gt_names])
        gmm_pts = gmm.means
        D = np.sum((gt_pts[:, None, :] - gmm_pts[None, :, :]) ** 2, axis=-1)
        rem_gt: List[int] = list(range(len(gt_names)))
        rem_gm: List[int] = list(range(len(gmm_pts)))
        sigma: Dict[str, int] = {}
        while rem_gt and rem_gm:
            sub = D[np.ix_(rem_gt, rem_gm)]
            ki, jj = np.unravel_index(np.argmin(sub), sub.shape)
            sigma[gt_names[rem_gt[ki]]] = rem_gm[jj]
            del rem_gt[ki]
            del rem_gm[jj]

        eps_dict, sigma_x_dict, sigma_n_dict = {}, {}, {}
        for name, comp in sigma.items():
            disp = gmm.means[comp] - gt_positions[name]
            eps_dict[name]     = float(np.linalg.norm(disp))
            cov                = gmm.covariances[comp]
            sigma_x_dict[name] = float(np.sqrt(max(cov[0, 0], 0.0)))
            sigma_n_dict[name] = float(np.sqrt(max(cov[1, 1], 0.0)))

        per_cluster["eps_k"]     = eps_dict
        per_cluster["sigma_x_k"] = sigma_x_dict
        per_cluster["sigma_n_k"] = sigma_n_dict
        scalars["CE"] = float(np.mean(list(eps_dict.values()))) if eps_dict else None

        # DB
        matched_components = list(sigma.values())
        if len(matched_components) >= 2:
            s = np.array([
                np.sqrt(max(np.trace(gmm.covariances[c]) / 2.0, 1e-12))
                for c in matched_components
            ])
            centres = gmm.means[matched_components]
            db_terms = []
            for i in range(len(matched_components)):
                vals = []
                for j in range(len(matched_components)):
                    if i == j: continue
                    d_ij = np.linalg.norm(centres[i] - centres[j])
                    if d_ij > 1e-9:
                        vals.append((s[i] + s[j]) / d_ij)
                if vals:
                    db_terms.append(max(vals))
            scalars["DB"] = float(np.mean(db_terms)) if db_terms else None
        else:
            scalars["DB"] = None
            notes.append("DB (2.5) needs ≥ 2 matched components.")

        # Pairwise overlap
        pairwise: Dict[Tuple[str, str], float] = {}
        if (gmm.labels_flat is not None
                and phantom.label_vol is not None):
            inv_sigma = {c: n for n, c in sigma.items()}
            name_to_label = {materials[i].symbol: i for i in keep_idx}
            label_flat = phantom.label_vol.ravel()
            comp_flat  = gmm.labels_flat
            if comp_flat.shape == label_flat.shape:
                comp_to_idx = np.full(gmm.n_components, len(gt_names), dtype=int)
                for c, name in inv_sigma.items():
                    comp_to_idx[c] = gt_names.index(name)
                pred_idx = comp_to_idx[comp_flat]
                for ia, na in enumerate(gt_names):
                    for ib in range(ia + 1, len(gt_names)):
                        nb = gt_names[ib]
                        if np.linalg.norm(
                            gt_positions[na] - gt_positions[nb]
                        ) > overlap_threshold:
                            continue
                        la = name_to_label[na]
                        lb = name_to_label[nb]
                        mask = (label_flat == la) | (label_flat == lb)
                        if not mask.any():
                            continue
                        true_idx = np.where(label_flat == la, ia, ib)
                        true_idx = np.where(mask, true_idx, -1)
                        wrong = mask & (pred_idx != true_idx)
                        pairwise[(na, nb)] = float(
                            wrong.sum() / max(mask.sum(), 1)
                        )

        # ── 4b. Pathology detection (v2) ─────────────────────────────────
        if detect_pathologies:
            pathologies = _detect_pathologies(
                gmm, sigma, gt_positions, eps_dict,
                {k: v for k, v in sig.cluster_elongation.items()},
                thresholds, overlap_threshold,
            )
    else:
        notes.append("Phantom has no label_vol; quality metrics skipped.")
        pairwise = {}

    table = HistogramMetricsTable(
        scalars=scalars,
        per_cluster=per_cluster,
        pairwise=pairwise,
        has_ground_truth=has_gt,
        has_reference=ref_hist is not None,
        n_components=gmm.n_components,
        energy_idx=energy_idx,
        notes=notes,
        pathologies=pathologies,
    )

    if raise_on_pathology and any(p.get("severity") == "warning"
                                   for p in pathologies):
        raise RuntimeError(
            f"compute_histogram_metrics: {len(pathologies)} pathology "
            f"warnings detected. First: {pathologies[0]['message']}"
        )

    return table
