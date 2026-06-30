#!/usr/bin/env python3
"""
examples/01_artifact_comparison.py
────────────────────────────────────
Run 8 simulation configurations and produce a comparison grid showing
how each artifact type (and their combination) deforms the bimodal histogram.

Output files
────────────
  01_histogram_grid.png    — 2×4 grid of bimodal histograms
  01_slice_comparison.png  — X-ray and neutron slice grid
  01_signatures.txt        — ASCII table of quantitative artifact metrics
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import matplotlib.pyplot as plt

from neutron_xray_sim import DualModalitySimulation, ArtifactConfig


# ──────────────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────────────

PHANTOM   = "composite"   # 'composite' | 'battery' | 'bone_implant' | 'industrial'
N         = 64            # voxel grid size  (increase to 128 for publication quality)
N_ANGLES  = 120           # projection angles
ALGORITHM = "FBP"         # 'FBP' | 'SIRT' | 'CGLS'


# ──────────────────────────────────────────────────────────────────────────────
# Build simulation instance
# ──────────────────────────────────────────────────────────────────────────────

sim = DualModalitySimulation(
    preset    = PHANTOM,
    N         = N,
    n_angles  = N_ANGLES,
    algorithm = ALGORITHM,
    auto_gmm  = False,    # set True to overlay GMM ellipses (slower)
    verbose   = True,
)


# ──────────────────────────────────────────────────────────────────────────────
# Define 8 experimental configurations
# ──────────────────────────────────────────────────────────────────────────────

CONFIGS = [
    # (tag,  ArtifactConfig)
    (
        "1. Clean",
        ArtifactConfig.clean()
    ),
    (
        "2. Poisson noise\n(I₀=5×10⁴)",
        ArtifactConfig(
            photon_noise=True,
            I0_xray=5e4, I0_neutron=5e4,
        )
    ),
    (
        "3. Low dose noise\n(I₀=5×10³)",
        ArtifactConfig(
            photon_noise=True,
            I0_xray=5e3, I0_neutron=5e3,
        )
    ),
    (
        "4. Beam hardening\n(no BHC)",
        ArtifactConfig(
            apply_bh_correction=False,    # BH emerges from polychromatic proj.
        )
    ),
    (
        "5. Neutron scatter\n(f=0.08)",
        ArtifactConfig(
            neutron_scatter=True,
            scatter_fraction=0.08,
            scatter_sigma_pixels=10.0,
        )
    ),
    (
        "6. Misalignment\n(3 vx, 1.5°)",
        ArtifactConfig(
            misalignment=True,
            translation_voxels=(3.0, 0.0, 0.0),
            rotation_deg=(0.0, 1.5, 0.0),
        )
    ),
    (
        "7. Ring artifacts",
        ArtifactConfig(
            ring_artifacts=True,
            n_bad_columns=4,
            ring_amplitude=0.06,
        )
    ),
    (
        "8. All realistic",
        ArtifactConfig.realistic()
    ),
]


# ──────────────────────────────────────────────────────────────────────────────
# Run all configurations
# ──────────────────────────────────────────────────────────────────────────────

results = []
ref = None

for tag, cfg in CONFIGS:
    result = sim.run(cfg, tag=tag, ref_result=ref)
    results.append(result)
    if ref is None:
        ref = result        # first result is the clean reference


# ──────────────────────────────────────────────────────────────────────────────
# Figures
# ──────────────────────────────────────────────────────────────────────────────

print("\n[ex01] Generating figures …")

# ── Histogram grid ────────────────────────────────────────────────────────────
fig_hist = sim.comparison_grid(results, ncols=4,
                               suptitle=f"Bimodal Histogram — Artifact Comparison "
                                        f"({PHANTOM} phantom, {N}³ voxels)")
fig_hist.savefig("01_histogram_grid.png", dpi=150, bbox_inches="tight")
print("  saved 01_histogram_grid.png")

# ── Slice comparison ──────────────────────────────────────────────────────────
# Show only clean, noise, misalign, realistic to keep figure manageable
subset = [results[0], results[1], results[5], results[7]]
fig_slices = sim.comparison_slices(subset)
fig_slices.savefig("01_slice_comparison.png", dpi=150, bbox_inches="tight")
print("  saved 01_slice_comparison.png")

# ── Artifact-signature table ──────────────────────────────────────────────────
table = sim.signature_table(results)
print("\n" + table)
with open("01_signatures.txt", "w") as f:
    f.write(table)
print("  saved 01_signatures.txt")

plt.show()
print("[ex01] Done.")
