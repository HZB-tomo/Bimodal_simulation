"""Ground-truth and particle-shape metrics for the NMC fusion simulation."""
from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd
from scipy import ndimage
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from skimage import measure, segmentation
from skimage.feature import peak_local_max
from skimage.filters import threshold_otsu


def normalized_rmse(reference: np.ndarray, estimate: np.ndarray) -> float:
    ref = np.asarray(reference, dtype=float)
    est = np.asarray(estimate, dtype=float)
    return float(np.sqrt(np.mean((est - ref) ** 2)) / (np.sqrt(np.mean(ref ** 2)) + 1e-12))


def segment_particles(
    volume: np.ndarray,
    voxel_um_zyx: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    threshold: float | None = None,
    threshold_scale: float = 1.0,
    smooth_sigma_um: float = 1.2,
    seed_min_distance_um: float = 4.0,
    min_equivalent_diameter_um: float = 8.0,
    max_equivalent_diameter_um: float = 60.0,
    remove_border: bool = True,
) -> np.ndarray:
    """Segment approximately spherical particles with a distance watershed.

    The previous implementation used a fixed 5-voxel maximum filter and a
    20-voxel size threshold.  In noisy laminography this produced tens of
    thousands of tiny fragments.  This implementation uses physical-unit
    smoothing, physical seed spacing, and equivalent-diameter filtering.
    """
    v = np.asarray(volume, dtype=np.float32)
    finite = np.isfinite(v)
    if not finite.any():
        raise ValueError("Volume contains no finite values.")
    fill = float(np.nanmin(v[finite]))
    v = np.where(finite, v, fill)

    vz, vy, vx = map(float, voxel_um_zyx)
    sigma_px = (
        max(smooth_sigma_um / vz, 0.0),
        max(smooth_sigma_um / vy, 0.0),
        max(smooth_sigma_um / vx, 0.0),
    )
    smoothed = ndimage.gaussian_filter(v, sigma=sigma_px)

    if threshold is None:
        vals = smoothed[np.isfinite(smoothed)]
        threshold = float(threshold_otsu(vals))
    threshold *= float(threshold_scale)

    mask = smoothed > threshold
    structure = ndimage.generate_binary_structure(3, 1)
    mask = ndimage.binary_opening(mask, structure=structure, iterations=1)
    mask = ndimage.binary_closing(mask, structure=structure, iterations=1)
    mask = ndimage.binary_fill_holes(mask)

    # Remove connected components too small to represent the minimum particle.
    voxel_volume = vz * vy * vx
    min_volume = np.pi / 6.0 * min_equivalent_diameter_um ** 3
    min_voxels = max(1, int(np.floor(min_volume / voxel_volume)))
    cc = measure.label(mask)
    mask = np.zeros_like(mask, dtype=bool)
    for region in measure.regionprops(cc):
        if region.area >= min_voxels:
            mask[cc == region.label] = True

    distance = ndimage.distance_transform_edt(mask, sampling=(vz, vy, vx))
    min_distance_px = max(
        1,
        int(np.ceil(seed_min_distance_um / min(vz, vy, vx))),
    )
    coordinates = peak_local_max(
        distance,
        labels=mask,
        min_distance=min_distance_px,
        exclude_border=False,
    )
    markers = np.zeros(mask.shape, dtype=np.int32)
    if len(coordinates):
        markers[tuple(coordinates.T)] = np.arange(1, len(coordinates) + 1)
    else:
        markers = measure.label(mask)

    labels = segmentation.watershed(-distance, markers, mask=mask)
    labels = measure.label(labels > 0) if labels.max() == 0 else labels

    # Filter by equivalent diameter and border contact.
    keep = np.zeros_like(labels, dtype=np.int32)
    next_label = 1
    nz, ny, nx = labels.shape
    for region in measure.regionprops(labels):
        volume_um3 = region.area * voxel_volume
        deq = (6.0 * volume_um3 / np.pi) ** (1.0 / 3.0)
        if not (min_equivalent_diameter_um <= deq <= max_equivalent_diameter_um):
            continue
        if remove_border:
            z0, y0, x0, z1, y1, x1 = region.bbox
            if z0 == 0 or y0 == 0 or x0 == 0 or z1 == nz or y1 == ny or x1 == nx:
                continue
        keep[labels == region.label] = next_label
        next_label += 1
    return keep


def particle_shape_table(
    labels: np.ndarray,
    voxel_um_zyx: Tuple[float, float, float],
) -> pd.DataFrame:
    """Calculate physical particle metrics and centred physical centroids."""
    vz, vy, vx = map(float, voxel_um_zyx)
    nz, ny, nx = labels.shape
    origin = np.array(
        [-nz * vz / 2.0, -ny * vy / 2.0, -nx * vx / 2.0],
        dtype=float,
    )
    spacing = np.array([vz, vy, vx], dtype=float)
    rows = []
    for region in measure.regionprops(labels):
        # Voxel-centre coordinates in the same centred system as particle truth.
        coords = origin + (region.coords.astype(float) + 0.5) * spacing
        V = float(region.area * vz * vy * vx)
        deq = float((6.0 * V / np.pi) ** (1.0 / 3.0))
        centered = coords - coords.mean(axis=0)
        if len(coords) >= 4:
            cov = np.cov(centered, rowvar=False)
            eig = np.sort(np.maximum(np.linalg.eigvalsh(cov), 0))[::-1]
            axes = 4.0 * np.sqrt(eig)
            major, middle, minor = map(float, axes)
        else:
            major = middle = minor = np.nan

        minc = np.maximum(np.asarray(region.bbox[:3]) - 1, 0)
        maxc = np.minimum(np.asarray(region.bbox[3:]) + 1, labels.shape)
        block = labels[tuple(slice(a, b) for a, b in zip(minc, maxc))] == region.label
        try:
            verts, faces, _, _ = measure.marching_cubes(
                block.astype(np.float32), 0.5, spacing=(vz, vy, vx)
            )
            A = float(measure.mesh_surface_area(verts, faces))
            psi = float(np.pi ** (1.0 / 3.0) * (6.0 * V) ** (2.0 / 3.0) / max(A, 1e-12))
        except ValueError:
            A = psi = np.nan

        centroid = coords.mean(axis=0)
        rows.append({
            "particle_id": int(region.label),
            "centroid_z_um": float(centroid[0]),
            "centroid_y_um": float(centroid[1]),
            "centroid_x_um": float(centroid[2]),
            "volume_um3": V,
            "equivalent_diameter_um": deq,
            "surface_area_um2": A,
            "sphericity": psi,
            "major_axis_um": major,
            "middle_axis_um": middle,
            "minor_axis_um": minor,
            "major_to_minor_ratio": major / max(minor, 1e-12),
            "axial_to_inplane_rms": major / max(np.sqrt((middle**2 + minor**2) / 2.0), 1e-12),
        })
    return pd.DataFrame(rows)


def match_reconstruction_to_truth(
    reconstructed: pd.DataFrame,
    truth: pd.DataFrame,
    max_distance_um: float = 10.0,
) -> pd.DataFrame:
    """One-to-one centroid matching using the Hungarian assignment."""
    if reconstructed.empty or truth.empty:
        return reconstructed.iloc[0:0].copy()

    rxyz = reconstructed[
        ["centroid_z_um", "centroid_y_um", "centroid_x_um"]
    ].to_numpy(float)
    txyz = truth[
        ["center_z_um", "center_y_um", "center_x_um"]
    ].to_numpy(float)

    distances = cdist(rxyz, txyz)
    # Large penalty outside the admissible radius.
    cost = distances.copy()
    penalty = max(max_distance_um * 1000.0, 1e6)
    cost[cost > max_distance_um] = penalty
    row_ind, col_ind = linear_sum_assignment(cost)
    valid = distances[row_ind, col_ind] <= max_distance_um
    row_ind = row_ind[valid]
    col_ind = col_ind[valid]

    out = reconstructed.iloc[row_ind].reset_index(drop=True).copy()
    tr = truth.iloc[col_ind].reset_index(drop=True)
    out["truth_particle_id"] = tr["particle_id"].to_numpy()
    out["match_distance_um"] = distances[row_ind, col_ind]
    out["truth_diameter_um"] = tr["diameter_um"].to_numpy()
    out["diameter_relative_error"] = (
        out["equivalent_diameter_um"].to_numpy() - out["truth_diameter_um"].to_numpy()
    ) / out["truth_diameter_um"].to_numpy()
    out["sphericity_error"] = out["sphericity"] - 1.0
    return out
