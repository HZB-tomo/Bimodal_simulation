"""NMC particle-slab phantom generation for DIANA.

The generator returns a DIANA-compatible ``PhantomData`` and a particle truth
``pandas.DataFrame``. Rasterisation is local to each sphere bounding box, so
thousands of particles are practical without evaluating a full-volume mask per
particle.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional, Tuple

import numpy as np
import pandas as pd

from .materials import MATERIALS, Material, material_from_formula
from .phantom import PhantomData


@dataclass(frozen=True)
class NMCParticleConfig:
    shape_zyx: Tuple[int, int, int] = (192, 512, 512)
    voxel_um_zyx: Tuple[float, float, float] = (1.77, 1.5, 1.5)
    n_particles: int = 1200
    diameter_mean_um: float = 28.0
    diameter_sigma_um: float = 7.0
    diameter_min_um: float = 8.0
    diameter_max_um: float = 60.0
    distribution: Literal["gaussian", "lognormal"] = "gaussian"
    slab_thickness_um: float = 180.0
    packing_margin_um: float = 0.5
    max_attempts_per_particle: int = 2000
    seed: int = 42
    composition: str = "LiNi0.8Mn0.1Co0.1O2"
    density_gcc: float = 4.8
    material_name: str = "NMC811"
    material_symbol: str = "NMC"


def make_nmc_material(config: NMCParticleConfig) -> Material:
    return material_from_formula(
        name=config.material_name,
        symbol=config.material_symbol,
        formula=config.composition,
        density_gcc=config.density_gcc,
        color="#6b8e23",
    )


def _sample_diameters(cfg: NMCParticleConfig, rng: np.random.Generator) -> np.ndarray:
    if cfg.distribution == "gaussian":
        d = rng.normal(cfg.diameter_mean_um, cfg.diameter_sigma_um, cfg.n_particles * 3)
    elif cfg.distribution == "lognormal":
        sigma2 = np.log(1 + (cfg.diameter_sigma_um / cfg.diameter_mean_um) ** 2)
        sigma = np.sqrt(sigma2)
        mu = np.log(cfg.diameter_mean_um) - sigma2 / 2
        d = rng.lognormal(mu, sigma, cfg.n_particles * 3)
    else:
        raise ValueError(f"Unsupported distribution: {cfg.distribution}")
    d = d[(d >= cfg.diameter_min_um) & (d <= cfg.diameter_max_um)]
    if len(d) < cfg.n_particles:
        raise RuntimeError("Could not sample enough diameters within truncation limits.")
    return d[: cfg.n_particles]


def generate_nmc_particle_slab(
    config: NMCParticleConfig,
    material: Optional[Material] = None,
) -> tuple[PhantomData, pd.DataFrame]:
    """Generate a non-overlapping planar particle slab.

    Storage order is ``(z, y, x)``. DIANA currently documents ``(z, x, y)`` in
    parts of the package; for isotropic x-y sampling the distinction is benign,
    but the truth table stores explicit physical x/y/z coordinates.
    """
    cfg = config
    nz, ny, nx = map(int, cfg.shape_zyx)
    vz, vy, vx = map(float, cfg.voxel_um_zyx)
    if not (np.isclose(vx, vy) and np.isclose(vx, vz)):
        # PhantomData stores one scalar voxel size. Use geometric mean while
        # retaining the true anisotropic spacing in the truth table metadata.
        voxel_cm = (vx * vy * vz) ** (1 / 3) * 1e-4
    else:
        voxel_cm = vx * 1e-4

    material = material or make_nmc_material(cfg)
    rng = np.random.default_rng(cfg.seed)
    diameters = _sample_diameters(cfg, rng)

    full_um = np.array([nz * vz, ny * vy, nx * vx], dtype=float)
    slab_thickness = min(cfg.slab_thickness_um, full_um[0])

    centers: list[np.ndarray] = []
    radii: list[float] = []
    accepted_diameters: list[float] = []

    for diameter in diameters:
        r = diameter / 2
        accepted = False
        for _ in range(cfg.max_attempts_per_particle):
            z = rng.uniform(-slab_thickness / 2 + r, slab_thickness / 2 - r)
            y = rng.uniform(-full_um[1] / 2 + r, full_um[1] / 2 - r)
            x = rng.uniform(-full_um[2] / 2 + r, full_um[2] / 2 - r)
            c = np.array([z, y, x])
            if centers:
                C = np.vstack(centers)
                R = np.asarray(radii)
                if np.any(np.linalg.norm(C - c, axis=1) < (R + r + cfg.packing_margin_um)):
                    continue
            centers.append(c)
            radii.append(r)
            accepted_diameters.append(diameter)
            accepted = True
            break
        if not accepted:
            break

    labels = np.zeros((nz, ny, nx), dtype=np.uint16)
    z0, y0, x0 = -full_um / 2

    records = []
    for pid, (center, radius, diameter) in enumerate(zip(centers, radii, accepted_diameters), start=1):
        cz, cy, cx = center
        iz0 = max(0, int(np.floor((cz - radius - z0) / vz)))
        iz1 = min(nz, int(np.ceil((cz + radius - z0) / vz)))
        iy0 = max(0, int(np.floor((cy - radius - y0) / vy)))
        iy1 = min(ny, int(np.ceil((cy + radius - y0) / vy)))
        ix0 = max(0, int(np.floor((cx - radius - x0) / vx)))
        ix1 = min(nx, int(np.ceil((cx + radius - x0) / vx)))

        zz = z0 + (np.arange(iz0, iz1) + 0.5) * vz
        yy = y0 + (np.arange(iy0, iy1) + 0.5) * vy
        xx = x0 + (np.arange(ix0, ix1) + 0.5) * vx
        Z, Y, X = np.meshgrid(zz, yy, xx, indexing="ij")
        mask = (Z - cz) ** 2 + (Y - cy) ** 2 + (X - cx) ** 2 <= radius**2
        block = labels[iz0:iz1, iy0:iy1, ix0:ix1]
        block[mask] = 1

        volume = 4.0 / 3.0 * np.pi * radius**3
        area = 4.0 * np.pi * radius**2
        records.append(
            {
                "particle_id": pid,
                "center_z_um": cz,
                "center_y_um": cy,
                "center_x_um": cx,
                "radius_um": radius,
                "diameter_um": diameter,
                "volume_um3": volume,
                "surface_area_um2": area,
                "sphericity": 1.0,
            }
        )

    phantom = PhantomData(
        Nz=nz,
        Nx=ny,
        Ny=nx,
        voxel_cm=voxel_cm,
        label_vol=labels.astype(np.uint8),
        materials=[MATERIALS["air"], material],
        name="NMC particle slab",
    )
    truth = pd.DataFrame.from_records(records)
    truth.attrs.update(
        {
            "voxel_um_zyx": cfg.voxel_um_zyx,
            "shape_zyx": cfg.shape_zyx,
            "requested_particles": cfg.n_particles,
            "accepted_particles": len(records),
            "composition": cfg.composition,
            "density_gcc": cfg.density_gcc,
        }
    )
    return phantom, truth
