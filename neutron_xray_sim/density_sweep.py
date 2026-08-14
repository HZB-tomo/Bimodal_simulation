"""Density-sweep helpers for planar NMC particle simulations."""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Sequence
import json
import numpy as np
import pandas as pd

from .nmc_phantom import NMCParticleConfig, generate_nmc_particle_slab


def expected_sphere_volume_um3(mean_diameter_um: float, sigma_diameter_um: float) -> float:
    mu = float(mean_diameter_um)
    sigma = float(sigma_diameter_um)
    return float(np.pi / 6.0 * (mu**3 + 3.0 * mu * sigma**2))


def sample_region_volume_um3(config: NMCParticleConfig) -> float:
    nz, ny, nx = config.shape_zyx
    vz, vy, vx = config.voxel_um_zyx
    slab = min(float(config.slab_thickness_um), nz * vz)
    return float(slab * ny * vy * nx * vx)


def estimate_particle_count_for_fraction(
    config: NMCParticleConfig,
    target_solid_fraction: float,
) -> int:
    if not 0 < target_solid_fraction < 1:
        raise ValueError("target_solid_fraction must be between zero and one.")
    mean_volume = expected_sphere_volume_um3(
        config.diameter_mean_um, config.diameter_sigma_um
    )
    return max(
        1,
        int(round(target_solid_fraction * sample_region_volume_um3(config) / mean_volume)),
    )


def analytical_solid_fraction(truth: pd.DataFrame, config: NMCParticleConfig) -> float:
    if truth.empty:
        return 0.0
    return float(truth["volume_um3"].sum() / sample_region_volume_um3(config))


def voxelized_solid_fraction(label_volume: np.ndarray, config: NMCParticleConfig) -> float:
    nz, ny, nx = config.shape_zyx
    vz, _, _ = config.voxel_um_zyx
    slab_voxels = min(nz, int(np.ceil(config.slab_thickness_um / vz)))
    z0 = max(0, (nz - slab_voxels) // 2)
    z1 = min(nz, z0 + slab_voxels)
    return float(np.count_nonzero(label_volume[z0:z1]) / label_volume[z0:z1].size)


def build_density_config(
    base_config: NMCParticleConfig,
    target_solid_fraction: float,
    seed: int,
    particle_count_scale: float = 1.05,
) -> NMCParticleConfig:
    n_est = estimate_particle_count_for_fraction(base_config, target_solid_fraction)
    return replace(
        base_config,
        n_particles=max(1, int(np.ceil(n_est * particle_count_scale))),
        seed=int(seed),
    )


def phantom_density_preflight(
    base_config: NMCParticleConfig,
    target_fractions: Sequence[float],
    output_dir: str | Path,
    seeds: Sequence[int] | None = None,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if seeds is None:
        seeds = [base_config.seed + i for i in range(len(target_fractions))]
    if len(seeds) != len(target_fractions):
        raise ValueError("seeds must have the same length as target_fractions.")

    rows = []
    phantoms = {}
    for target, seed in zip(target_fractions, seeds):
        cfg = build_density_config(base_config, float(target), int(seed))
        phantom, truth = generate_nmc_particle_slab(cfg)
        analytical = analytical_solid_fraction(truth, cfg)
        voxelized = voxelized_solid_fraction(phantom.label_vol, cfg)
        accepted = len(truth)
        requested = cfg.n_particles

        tag = f"phi_{target:.3f}".replace(".", "p")
        case_dir = output_dir / tag
        case_dir.mkdir(parents=True, exist_ok=True)
        truth.to_csv(case_dir / "particle_truth.csv", index=False)
        np.save(case_dir / "ground_truth_labels.npy", phantom.label_vol)
        with open(case_dir / "phantom_config.json", "w") as f:
            json.dump(cfg.__dict__, f, indent=2)

        rows.append({
            "case": tag,
            "target_solid_fraction": float(target),
            "requested_particles": requested,
            "accepted_particles": accepted,
            "acceptance_fraction": accepted / max(requested, 1),
            "analytical_solid_fraction": analytical,
            "voxelized_solid_fraction": voxelized,
            "mean_accepted_diameter_um": float(truth["diameter_um"].mean()) if accepted else np.nan,
            "median_accepted_diameter_um": float(truth["diameter_um"].median()) if accepted else np.nan,
            "diameter_std_um": float(truth["diameter_um"].std()) if accepted > 1 else np.nan,
            "seed": int(seed),
        })
        phantoms[float(target)] = (cfg, phantom, truth)

    summary = pd.DataFrame(rows)
    summary.to_csv(output_dir / "density_preflight_summary.csv", index=False)
    return summary, phantoms
