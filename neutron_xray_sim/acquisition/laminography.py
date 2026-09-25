"""Full 3-D ASTRA projection and reconstruction utilities.

Geometry coordinates are in micrometres.  Volumes are stored as ``(z, y, x)``
arrays and reconstructed on an explicitly windowed ASTRA volume geometry.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import numpy as np

from .cone3d_geometry import ConeGeometryConfig, astra_projection_geometry


def _require_astra():
    try:
        import astra
    except ImportError as exc:
        raise RuntimeError("ASTRA Toolbox with CUDA support is required.") from exc
    return astra


def create_volume_geometry(
    shape_zyx: Tuple[int, int, int],
    voxel_um_zyx: Tuple[float, float, float],
):
    """Create an ASTRA volume geometry whose physical windows are in µm."""
    astra = _require_astra()
    nz, ny, nx = map(int, shape_zyx)
    vz, vy, vx = map(float, voxel_um_zyx)
    if min(vz, vy, vx) <= 0:
        raise ValueError("Voxel sizes must be positive.")
    return astra.create_vol_geom(
        ny, nx, nz,
        -nx * vx / 2.0, nx * vx / 2.0,
        -ny * vy / 2.0, ny * vy / 2.0,
        -nz * vz / 2.0, nz * vz / 2.0,
    )


def forward_project_cone3d(
    volume_zyx: np.ndarray,
    geometry: ConeGeometryConfig,
    voxel_um_zyx: Tuple[float, float, float],
    gpu_index: int = 0,
    output_memmap: Optional[str | Path] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """GPU forward-project a 3-D attenuation volume.

    The input attenuation must be in µm⁻¹ because geometry lengths are in µm.
    ASTRA projection order is ``(detector_v, view, detector_u)``.
    """
    astra = _require_astra()
    vol = np.asarray(volume_zyx, dtype=np.float32, order="C")
    vol_geom = create_volume_geometry(vol.shape, voxel_um_zyx)
    proj_geom, translations = astra_projection_geometry(geometry)

    vol_id = astra.data3d.create("-vol", vol_geom, vol)
    sino_id = astra.data3d.create("-sino", proj_geom)
    cfg = astra.astra_dict("FP3D_CUDA")
    cfg["VolumeDataId"] = vol_id
    cfg["ProjectionDataId"] = sino_id
    cfg.setdefault("option", {})["GPUindex"] = int(gpu_index)
    alg_id = astra.algorithm.create(cfg)
    try:
        astra.algorithm.run(alg_id)
        data = astra.data3d.get(sino_id).astype(np.float32, copy=False)
    finally:
        astra.algorithm.delete(alg_id)
        astra.data3d.delete(vol_id)
        astra.data3d.delete(sino_id)

    if output_memmap is not None:
        path = Path(output_memmap)
        path.parent.mkdir(parents=True, exist_ok=True)
        mm = np.memmap(path, dtype=np.float32, mode="w+", shape=data.shape)
        mm[:] = data
        mm.flush()
        data = mm
    return data, translations


def line_integrals_to_noisy_log(
    line_integrals: np.ndarray,
    incident_photons: float = 2e5,
    detector_sigma_px: float = 0.7,
    seed: int = 42,
) -> np.ndarray:
    """Convert dimensionless line integrals to noisy log projections."""
    from scipy.ndimage import gaussian_filter

    if incident_photons <= 0:
        raise ValueError("incident_photons must be positive.")
    rng = np.random.default_rng(seed)
    p = np.asarray(line_integrals, dtype=np.float32)
    transmission = np.exp(-np.clip(p, 0, 80))
    if detector_sigma_px > 0:
        # Blur only detector axes, not the projection/view axis.
        transmission = gaussian_filter(
            transmission, sigma=(detector_sigma_px, 0.0, detector_sigma_px)
        )
    expected = np.maximum(transmission * incident_photons, 0.0)
    counts = rng.poisson(expected).astype(np.float32)
    counts = np.maximum(counts, 1.0)
    return (-np.log(counts / float(incident_photons))).astype(np.float32)


def reconstruct_cone3d(
    projections_vau: np.ndarray,
    geometry: ConeGeometryConfig,
    volume_shape_zyx: Tuple[int, int, int],
    voxel_um_zyx: Tuple[float, float, float],
    algorithm: str = "CGLS3D_CUDA",
    iterations: int = 30,
    gpu_index: int = 0,
    min_constraint: Optional[float] = 0.0,
) -> np.ndarray:
    """Iteratively reconstruct onto the requested physical grid."""
    astra = _require_astra()
    proj_geom, _ = astra_projection_geometry(geometry)
    vol_geom = create_volume_geometry(volume_shape_zyx, voxel_um_zyx)
    sino = np.asarray(projections_vau, dtype=np.float32, order="C")
    sino_id = astra.data3d.create("-sino", proj_geom, sino)
    rec_id = astra.data3d.create("-vol", vol_geom, 0)
    cfg = astra.astra_dict(algorithm)
    cfg["ProjectionDataId"] = sino_id
    cfg["ReconstructionDataId"] = rec_id
    cfg.setdefault("option", {})["GPUindex"] = int(gpu_index)
    if min_constraint is not None:
        cfg["option"]["MinConstraint"] = float(min_constraint)
    alg_id = astra.algorithm.create(cfg)
    try:
        astra.algorithm.run(alg_id, int(iterations))
        rec = astra.data3d.get(rec_id).astype(np.float32, copy=False)
    finally:
        astra.algorithm.delete(alg_id)
        astra.data3d.delete(sino_id)
        astra.data3d.delete(rec_id)
    return rec
