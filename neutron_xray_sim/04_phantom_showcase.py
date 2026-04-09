#!/usr/bin/env python3
"""
examples/04_phantom_showcase.py
─────────────────────────────────
Run all four preset phantoms in clean mode and compare their bimodal
histograms and central slices.  Shows how material placement in (μ_x, μ_n)
space differs across phantom types.

Output files
────────────
  04_phantom_histograms.png  — 4-panel bimodal histogram grid
  04_phantom_slices.png      — 4×2 slice panel (X-ray + neutron)
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import matplotlib.pyplot as plt

from neutron_xray_sim import (
    DualModalitySimulation, ArtifactConfig,
    make_composite_phantom, make_battery_phantom,
    make_bone_implant_phantom, make_industrial_phantom,
)
from neutron_xray_sim.histogram import plot_comparison_grid


N        = 64
N_ANGLES = 120

PHANTOM_FACTORIES = {
    "Composite":    make_composite_phantom,
    "Battery":      make_battery_phantom,
    "Bone+Implant": make_bone_implant_phantom,
    "Industrial":   make_industrial_phantom,
}


results = {}
for name, factory in PHANTOM_FACTORIES.items():
    print(f"\n{'═'*50}")
    print(f"  Phantom: {name}")
    phantom = factory(N=N, voxel_cm=10.0/N)

    sim = DualModalitySimulation(
        phantom  = phantom,
        n_angles = N_ANGLES,
        verbose  = True,
    )
    results[name] = sim.run(ArtifactConfig.clean(), tag=name)


# ── Histogram grid ────────────────────────────────────────────────────────────
pairs = [(name, r.histogram) for name, r in results.items()]
fig_h = plot_comparison_grid(
    pairs, ncols=2,
    suptitle="Bimodal Histograms — Four Phantom Presets (Clean)",
    figsize_per_panel=(5.5, 5.0),
)
fig_h.savefig("04_phantom_histograms.png", dpi=150, bbox_inches="tight")
print("saved 04_phantom_histograms.png")


# ── Slice comparison ──────────────────────────────────────────────────────────
n_ph = len(results)
fig, axes = plt.subplots(2, n_ph, figsize=(4 * n_ph, 8))

for col, (name, r) in enumerate(results.items()):
    si = N // 2
    vmin_x = np.percentile(r.vol_xray, 1)
    vmax_x = np.percentile(r.vol_xray, 99)
    vmin_n = np.percentile(r.vol_neutron, 1)
    vmax_n = np.percentile(r.vol_neutron, 99)

    axes[0, col].imshow(r.vol_xray[si], cmap="gray",
                        vmin=vmin_x, vmax=vmax_x)
    axes[0, col].set_title(f"X-ray\n{name}", fontsize=9)
    axes[0, col].axis("off")

    axes[1, col].imshow(r.vol_neutron[si], cmap="gray",
                        vmin=vmin_n, vmax=vmax_n)
    axes[1, col].set_title(f"Neutron\n{name}", fontsize=9)
    axes[1, col].axis("off")

fig.suptitle("Central Slices — All Phantom Presets (Clean)", fontsize=13)
fig.tight_layout()
fig.savefig("04_phantom_slices.png", dpi=150, bbox_inches="tight")
print("saved 04_phantom_slices.png")

plt.show()
print("[ex04] Done.")
