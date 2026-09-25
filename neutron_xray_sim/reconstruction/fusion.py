"""Multiresolution Fourier fusion for tomography and laminography volumes."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np


@dataclass(frozen=True)
class FusionConfig:
    voxel_um_zyx: Tuple[float, float, float] = (1.77, 1.5, 1.5)
    laminography_axis_zyx: Tuple[float, float, float] = (1.0, 0.0, 0.0)
    missing_cone_half_angle_deg: float = 67.0
    tomography_native_voxel_um: float = 3.0
    radial_transition_fraction: float = 0.15
    angular_transition_deg: float = 4.0
    preserve_dc_from: str = "tomography"


def _smoothstep(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, 0, 1)
    return x * x * (3 - 2 * x)


def fusion_weight_tomography(shape_zyx: Tuple[int, int, int], cfg: FusionConfig) -> np.ndarray:
    nz, ny, nx = shape_zyx
    vz, vy, vx = cfg.voxel_um_zyx
    kz = np.fft.fftfreq(nz, d=vz)
    ky = np.fft.fftfreq(ny, d=vy)
    kx = np.fft.fftfreq(nx, d=vx)
    KZ, KY, KX = np.meshgrid(kz, ky, kx, indexing="ij")
    K = np.sqrt(KZ**2 + KY**2 + KX**2)

    axis = np.asarray(cfg.laminography_axis_zyx, dtype=float)
    axis /= np.linalg.norm(axis)
    dot = np.abs(KZ * axis[0] + KY * axis[1] + KX * axis[2])
    angle = np.rad2deg(np.arccos(np.clip(dot / np.maximum(K, 1e-12), 0, 1)))

    # Inside cone around ±axis: angle <= half-angle.
    a0 = cfg.missing_cone_half_angle_deg
    da = max(cfg.angular_transition_deg, 1e-6)
    angular = 1.0 - _smoothstep((angle - (a0 - da)) / (2 * da))

    nyq_tomo = 1.0 / (2.0 * cfg.tomography_native_voxel_um)
    width = max(cfg.radial_transition_fraction * nyq_tomo, 1e-9)
    radial = 1.0 - _smoothstep((K - (nyq_tomo - width)) / width)
    W = angular * radial
    W[K == 0] = 1.0 if cfg.preserve_dc_from == "tomography" else 0.0
    return W.astype(np.float32)


def fuse_volumes_fourier(
    laminography_zyx: np.ndarray,
    tomography_zyx: np.ndarray,
    cfg: FusionConfig,
) -> tuple[np.ndarray, np.ndarray]:
    if laminography_zyx.shape != tomography_zyx.shape:
        raise ValueError("Volumes must already share shape and physical grid.")
    lam = np.asarray(laminography_zyx, dtype=np.float32)
    tomo = np.asarray(tomography_zyx, dtype=np.float32)
    W = fusion_weight_tomography(lam.shape, cfg)
    FL = np.fft.fftn(lam)
    FT = np.fft.fftn(tomo)
    FF = (1.0 - W) * FL + W * FT
    fused = np.fft.ifftn(FF).real.astype(np.float32)
    return fused, W
