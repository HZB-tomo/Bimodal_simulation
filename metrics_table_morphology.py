"""
metrics_table_morphology  (v3)
─────────────────────────────────
Morphology-aware computation of bimodal-histogram metrics.

This v3 design is significantly simpler and more honest than the v2 GMM
seeded approach. It computes per-material centroid errors directly from
the GROUND-TRUTH LABEL VOLUME — no clustering, no GMM, no k-means, no
seeded matching that can mask reconstruction biases.

Two modes
─────────
mode='label_anchored' (default)
    Per-material ε_k = ‖ centroid(eroded_GT_mask) − GT_position ‖
    on the (μ_x, μ_n) plane. CE = mean ε_k. DB computed from per-material
    centroids and per-axis spreads. Pairwise overlap fractions O_ab use
    label-anchored centroids as Voronoi seeds.
    The most honest centroid metric in any simulation context.

mode='morphology_explore'
    Same scalar metrics, plus per-(material, region) RegionSummary
    objects produced by spatial connected-component analysis. Use for
    detecting anomalous regions (one graphite turn behaving differently
    from the others, etc.) — the principal academic value of the
    morphology-aware approach.

Why label-anchored ε_k is more honest than GMM-seeded ε_k
──────────────────────────────────────────────────────────
A seeded GMM can place a tiny low-weight component near the ground-truth
position of a material whose voxels actually reconstruct elsewhere
(typical for high-contrast inclusions in low-resolution reconstructions
where FBP cannot resolve them — Fe in a 64³ reconstruction may
reconstruct to the dominant background HDPE μ values). The seeded GMM
will then report ε_k(Fe) ≈ 0.4 cm⁻¹ when the real Fe voxels are
3+ cm⁻¹ away. Label-anchored ε_k cannot be fooled this way: it asks
"where do the Fe voxels actually reconstruct to?" and reports the
honest answer.

Author: neutron_xray_sim contributors.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict, Tuple, List

import numpy as np
from scipy.ndimage import label as nd_label, binary_erosion

from neutron_xray_sim.metrics_table import (
    HistogramMetricsTable,
    _normalised_streak_scores,
)
from neutron_xray_sim.histogram import (
    HistogramResult,
    detect_artifact_signatures,
)


@dataclass
class RegionSummary:
    """Statistics for one spatial connected component of one material."""
    material:          str
    region_id:         int
    n_voxels_total:    int
    n_voxels_interior: int
    centroid_attn:     Tuple[float, float]
    sigma_attn:        Tuple[float, float]
    centroid_spatial:  Tuple[float, float, float]
    eps_k:             float


# ─────────────────────────────────────────────────────────────────────────
def _per_material_label_anchored(
    phantom, vol_x: np.ndarray, vol_n: np.ndarray,
    keep_idx: List[int], energy_idx: int, erosion_iter: int,
) -> Dict[str, dict]:
    """Per-material centroid + spread from eroded GT-label mask."""
    out: Dict[str, dict] = {}
    for i in keep_idx:
        m = phantom.materials[i]
        mask = (phantom.label_vol == i)
        if not mask.any():
            continue
        eroded = binary_erosion(mask, iterations=erosion_iter) \
                 if erosion_iter > 0 else mask
        if eroded.sum() < 4:
            eroded = mask    # too thin to erode
        gx = float(m._mu_x_table[energy_idx])
        gn = float(m.mu_n)
        rx = vol_x[eroded]
        rn = vol_n[eroded]
        cx, cn = float(np.mean(rx)), float(np.mean(rn))
        out[m.symbol] = dict(
            eps_k    = float(np.hypot(cx - gx, cn - gn)),
            sigma_x  = float(np.std(rx)),
            sigma_n  = float(np.std(rn)),
            n_voxels = int(eroded.sum()),
            centroid = (cx, cn),
            gt       = (gx, gn),
        )
    return out


def _connected_components(
    phantom, vol_x: np.ndarray, vol_n: np.ndarray,
    keep_idx: List[int], energy_idx: int,
    connectivity: int, min_component_voxels: int, erosion_iter: int,
) -> List[RegionSummary]:
    """Connected-component analysis of each material's GT mask."""
    if connectivity == 6:
        struct = np.zeros((3, 3, 3), dtype=bool)
        for ax in range(3):
            for off in (-1, 1):
                idx = [1, 1, 1]
                idx[ax] = 1 + off
                struct[tuple(idx)] = True
        struct[1, 1, 1] = True
    else:
        struct = np.ones((3, 3, 3), dtype=bool)

    summaries: List[RegionSummary] = []
    for i in keep_idx:
        m = phantom.materials[i]
        gx = float(m._mu_x_table[energy_idx])
        gn = float(m.mu_n)
        mat_mask = (phantom.label_vol == i)
        if not mat_mask.any():
            continue
        comp_lbl, n_c = nd_label(mat_mask, structure=struct)
        for c in range(1, n_c + 1):
            comp = (comp_lbl == c)
            n_total = int(comp.sum())
            if n_total < min_component_voxels:
                continue
            comp_int = binary_erosion(comp, iterations=erosion_iter) \
                       if erosion_iter > 0 else comp
            if comp_int.sum() < 4:
                comp_int = comp
            n_int = int(comp_int.sum())
            cx = float(np.mean(vol_x[comp_int]))
            cn = float(np.mean(vol_n[comp_int]))
            sx = float(np.std(vol_x[comp_int]))
            sn = float(np.std(vol_n[comp_int]))
            zs, ys, xs = np.where(comp_int)
            summaries.append(RegionSummary(
                material=m.symbol, region_id=int(c),
                n_voxels_total=n_total, n_voxels_interior=n_int,
                centroid_attn=(cx, cn), sigma_attn=(sx, sn),
                centroid_spatial=(float(np.mean(xs)), float(np.mean(ys)),
                                   float(np.mean(zs))),
                eps_k=float(np.hypot(cx - gx, cn - gn)),
            ))
    return summaries


# ─────────────────────────────────────────────────────────────────────────
def compute_histogram_metrics_morphology_aware(
    phantom,
    hist_recon: HistogramResult,
    vol_x:      np.ndarray,
    vol_n:      np.ndarray,
    *,
    ref_hist:             Optional[HistogramResult] = None,
    energy_idx:           int                       = 6,
    skip_air:             bool                      = True,
    overlap_threshold:    float                     = 2.0,
    erosion_iter:         int                       = 1,
    mode:                 str                       = "label_anchored",
    connectivity:         int                       = 6,
    min_component_voxels: int                       = 8,
    normalize_shape_axes: bool                      = False,
) -> Tuple[HistogramMetricsTable, List[RegionSummary]]:
    """
    Returns (HistogramMetricsTable, list of RegionSummary).

    region_summaries is empty in 'label_anchored' mode and populated in
    'morphology_explore' mode.

    All quality metrics (ε_k, CE, σ_x^(k), σ_n^(k), DB, O_ab) are
    computed by anchoring on the ground-truth label volume rather than
    by clustering — see module docstring for rationale.
    """
    if mode not in ("label_anchored", "morphology_explore"):
        raise ValueError(f"mode must be 'label_anchored' or "
                         f"'morphology_explore', got {mode!r}")
    if vol_x.shape != vol_n.shape:
        raise ValueError("vol_x and vol_n must have identical shape")
    if vol_x.shape != phantom.label_vol.shape:
        raise ValueError(
            f"label_vol shape {phantom.label_vol.shape} does not match "
            f"vol_x shape {vol_x.shape}; reconstruct on the same grid as "
            f"the phantom."
        )

    materials = list(phantom.materials)
    if skip_air:
        keep_idx = [
            i for i, m in enumerate(materials)
            if not (m._mu_x_table[energy_idx] < 1e-3 and m.mu_n < 1e-3)
        ]
    else:
        keep_idx = list(range(len(materials)))

    per_mat = _per_material_label_anchored(
        phantom, vol_x, vol_n, keep_idx, energy_idx, erosion_iter,
    )
    eps_dict     = {n: d["eps_k"]   for n, d in per_mat.items()}
    sigma_x_dict = {n: d["sigma_x"] for n, d in per_mat.items()}
    sigma_n_dict = {n: d["sigma_n"] for n, d in per_mat.items()}
    centroids    = {n: d["centroid"] for n, d in per_mat.items()}

    CE = float(np.mean(list(eps_dict.values()))) if eps_dict else None

    # DB
    names = list(centroids.keys())
    if len(names) >= 2:
        s_arr = np.array([
            np.sqrt((sigma_x_dict[n]**2 + sigma_n_dict[n]**2) / 2.0)
            for n in names
        ])
        c_arr = np.array([centroids[n] for n in names])
        db_terms = []
        for i in range(len(names)):
            vals = []
            for j in range(len(names)):
                if i == j: continue
                d_ij = float(np.linalg.norm(c_arr[i] - c_arr[j]))
                if d_ij > 1e-9:
                    vals.append((s_arr[i] + s_arr[j]) / d_ij)
            if vals:
                db_terms.append(max(vals))
        DB = float(np.mean(db_terms)) if db_terms else None
    else:
        DB = None

    # Pairwise overlap by Voronoi assignment to label-anchored centroids
    name_to_label = {materials[i].symbol: i for i in keep_idx}
    label_flat = phantom.label_vol.ravel()
    vx_flat    = vol_x.ravel()
    vn_flat    = vol_n.ravel()
    pairwise: Dict[Tuple[str, str], float] = {}
    for ia, na in enumerate(names):
        for ib in range(ia + 1, len(names)):
            nb = names[ib]
            if np.linalg.norm(np.array(centroids[na]) - np.array(centroids[nb])) \
                    > overlap_threshold:
                continue
            la, lb = name_to_label[na], name_to_label[nb]
            mask = (label_flat == la) | (label_flat == lb)
            if not mask.any():
                continue
            ca = np.array(centroids[na])
            cb = np.array(centroids[nb])
            pts_x = vx_flat[mask]; pts_n = vn_flat[mask]
            d_a = (pts_x - ca[0])**2 + (pts_n - ca[1])**2
            d_b = (pts_x - cb[0])**2 + (pts_n - cb[1])**2
            pred  = np.where(d_a < d_b, la, lb)
            wrong = (pred != label_flat[mask])
            pairwise[(na, nb)] = float(wrong.sum() / mask.sum())

    # Per-region (morphology_explore only)
    region_summaries: List[RegionSummary] = []
    if mode == "morphology_explore":
        region_summaries = _connected_components(
            phantom, vol_x, vol_n, keep_idx, energy_idx,
            connectivity, min_component_voxels, erosion_iter,
        )

    # Shape metrics
    sig = detect_artifact_signatures(hist_recon, gmm=None, ref_hist=ref_hist)
    if normalize_shape_axes:
        S_h, S_v = _normalised_streak_scores(hist_recon)
    else:
        S_h = float(sig.horizontal_streak_score)
        S_v = float(sig.vertical_streak_score)

    scalars: Dict[str, Optional[float]] = {
        "S_h":     S_h,
        "S_v":     S_v,
        "S_d":     float(sig.diagonal_smear_score),
        "A_x":     float(sig.marginal_asymmetry_x),
        "Delta_n": float(sig.marginal_shift_n) if ref_hist is not None else None,
        "CE":      CE,
        "DB":      DB,
    }
    per_cluster: Dict[str, Dict[str, float]] = {
        "eps_k":     eps_dict,
        "sigma_x_k": sigma_x_dict,
        "sigma_n_k": sigma_n_dict,
        "E_k": {
            n: max(sigma_x_dict[n], sigma_n_dict[n])
                 / max(min(sigma_x_dict[n], sigma_n_dict[n]), 1e-12)
            for n in names
        },
    }
    notes = [
        f"Morphology-aware metrics (mode='{mode}'). "
        "ε_k anchored on the GT label volume — no clustering.",
    ]
    if mode == "morphology_explore":
        notes.append(f"{len(region_summaries)} connected components reported "
                     "via region_summaries.")
    if ref_hist is None:
        notes.append("Δ_n needs a clean reference; reported as None.")

    table = HistogramMetricsTable(
        scalars=scalars, per_cluster=per_cluster, pairwise=pairwise,
        has_ground_truth=True, has_reference=ref_hist is not None,
        n_components=len(names), energy_idx=energy_idx,
        notes=notes, pathologies=[],
    )
    return table, region_summaries
