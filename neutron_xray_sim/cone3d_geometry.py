"""ASTRA ``cone_vec`` geometries for circular tomography and translated planar laminography.

All geometry vectors and volume windows use micrometres.  The array convention
outside ASTRA is ``(z, y, x)``; ASTRA coordinates are ``(x, y, z)``.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Tuple

import numpy as np


@dataclass(frozen=True)
class ConeGeometryConfig:
    detector_shape_vu: Tuple[int, int] = (384, 384)
    detector_pixel_um: float = 150.0
    sod_um: float = 2400.0
    sdd_um: float = 120000.0
    n_views: int = 720
    rotation_range_deg: float = 360.0
    trajectory: Literal["tomography", "translated_laminography"] = "tomography"
    translation_pattern: Literal["fermat", "raster", "circular"] = "fermat"
    translation_radius_um: float = 150.0
    rotation_during_laminography: bool = False
    seed: int = 42

    @property
    def magnification(self) -> float:
        return self.sdd_um / self.sod_um

    @property
    def effective_detector_pixel_um(self) -> float:
        return self.detector_pixel_um / self.magnification


def _translation_positions_xy(cfg: ConeGeometryConfig) -> np.ndarray:
    """Return sample translations in ASTRA xyz order, restricted to the sample plane."""
    n = cfg.n_views
    radius = cfg.translation_radius_um
    if cfg.translation_pattern == "fermat":
        golden = np.pi * (3.0 - np.sqrt(5.0))
        i = np.arange(n)
        r = radius * np.sqrt((i + 0.5) / n)
        a = i * golden
        return np.c_[r * np.cos(a), r * np.sin(a), np.zeros(n)]
    if cfg.translation_pattern == "circular":
        a = np.linspace(0, 2 * np.pi, n, endpoint=False)
        return np.c_[radius * np.cos(a), radius * np.sin(a), np.zeros(n)]
    if cfg.translation_pattern == "raster":
        side = int(np.ceil(np.sqrt(n)))
        xx, yy = np.meshgrid(
            np.linspace(-radius, radius, side),
            np.linspace(-radius, radius, side),
            indexing="xy",
        )
        pts = np.c_[xx.ravel(), yy.ravel(), np.zeros(side * side)]
        return pts[:n]
    raise ValueError(f"Unsupported translation pattern: {cfg.translation_pattern}")


def make_cone_vec(cfg: ConeGeometryConfig) -> tuple[np.ndarray, np.ndarray]:
    """Return ASTRA cone vectors and physical sample translations.

    For tomography, the source rotates around the ASTRA z-axis.  For planar
    laminography, the source is fixed on the negative z-axis, the detector is
    parallel to the x-y sample plane, and source/detector are shifted together
    by the negative sample translation.  This makes the laminography missing
    direction coincide with array axis 0 (z).
    """
    if cfg.sod_um <= 0 or cfg.sdd_um <= cfg.sod_um:
        raise ValueError("Require 0 < SOD < SDD.")
    rows, cols = cfg.detector_shape_vu
    if rows <= 0 or cols <= 0 or cfg.detector_pixel_um <= 0:
        raise ValueError("Detector dimensions and pixel size must be positive.")

    odd_um = cfg.sdd_um - cfg.sod_um
    vectors = np.zeros((cfg.n_views, 12), dtype=np.float32)

    if cfg.trajectory == "tomography":
        angles = np.linspace(
            0.0, np.deg2rad(cfg.rotation_range_deg), cfg.n_views, endpoint=False
        )
        translations_xyz_um = np.zeros((cfg.n_views, 3), dtype=np.float64)
        for i, angle in enumerate(angles):
            radial = np.array([np.cos(angle), np.sin(angle), 0.0])
            source = -cfg.sod_um * radial
            detector = odd_um * radial
            u = cfg.detector_pixel_um * np.array([-np.sin(angle), np.cos(angle), 0.0])
            v = cfg.detector_pixel_um * np.array([0.0, 0.0, 1.0])
            vectors[i] = np.r_[source, detector, u, v]
    elif cfg.trajectory == "translated_laminography":
        translations_xyz_um = _translation_positions_xy(cfg)
        angles = (
            np.linspace(
                0.0, np.deg2rad(cfg.rotation_range_deg), cfg.n_views, endpoint=False
            )
            if cfg.rotation_during_laminography
            else np.zeros(cfg.n_views)
        )
        for i, angle in enumerate(angles):
            ca, sa = np.cos(angle), np.sin(angle)
            source = np.array([0.0, 0.0, -cfg.sod_um])
            detector = np.array([0.0, 0.0, odd_um])
            u = cfg.detector_pixel_um * np.array([ca, sa, 0.0])
            v = cfg.detector_pixel_um * np.array([-sa, ca, 0.0])
            shift = translations_xyz_um[i]
            source = source - shift
            detector = detector - shift
            vectors[i] = np.r_[source, detector, u, v]
    else:
        raise ValueError(f"Unsupported trajectory: {cfg.trajectory}")

    return vectors, translations_xyz_um


def astra_projection_geometry(cfg: ConeGeometryConfig):
    import astra

    vectors, translations = make_cone_vec(cfg)
    rows, cols = cfg.detector_shape_vu
    return astra.create_proj_geom("cone_vec", rows, cols, vectors), translations
