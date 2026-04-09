#!/usr/bin/env python3
"""
examples/03_gmm_segmentation.py
─────────────────────────────────
Fit a Gaussian mixture model (GMM) to the bimodal histogram and use it to
segment the reconstructed volume back into material phases.

Demonstrates:
  • Auto BIC-based component selection
  • 2-D histogram with overlaid GMM ellipses
  • Recovered label volume compared to ground truth

Output files
────────────
  03_gmm_histogram.png    — bimodal histogram + GMM ellipses
  03_gmm_segmentation.png — ground-truth vs GMM segmentation slices
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from neutron_xray_sim import DualModalitySimulation, ArtifactConfig
from neutron_xray_sim.histogram import (
    auto_fit_gmm, segment_by_gmm, plot_bimodal_histogram
)


# ──────────────────────────────────────────────────────────────────────────────
# Setup
# ──────────────────────────────────────────────────────────────────────────────

sim = DualModalitySimulation(
    preset   = "composite",
    N        = 64,
    n_angles = 180,
    auto_gmm = False,   # we'll call GMM manually below
    verbose  = True,
)


# ──────────────────────────────────────────────────────────────────────────────
# Run: clean + moderate noise
# ──────────────────────────────────────────────────────────────────────────────

cfg = ArtifactConfig(
    photon_noise=True,
    I0_xray=2e4, I0_neutron=2e4,
)
result = sim.run(cfg, tag="moderate noise")

# Number of true material phases (air + 5 materials in composite phantom = 6)
n_true = len(result.phantom.materials)
print(f"\n[ex03] True materials: {[m.name for m in result.phantom.materials]}")


# ──────────────────────────────────────────────────────────────────────────────
# GMM fitting with BIC selection
# ──────────────────────────────────────────────────────────────────────────────

print(f"\n[ex03] Auto-fitting GMM (BIC selection, k=2..{n_true+1}) …")
gmm = auto_fit_gmm(result.histogram, min_k=2, max_k=n_true + 1)
print(f"  → selected k={gmm.n_components}  BIC={gmm.bic:.1f}")

# Build material label lookup from GMM cluster centres
# Sort clusters by μ_x so label 0 = air, increasing density
order  = np.argsort(gmm.means[:, 0])
labels_by_mux = {int(order[i]): f"Phase {i}" for i in range(gmm.n_components)}
# Try to match to phantom materials by proximity
for k, mean in enumerate(gmm.means):
    mu_x, mu_n = mean
    # Find closest phantom material by L2 distance in (μ_x, μ_n) space
    best_mat = min(result.phantom.materials, key=lambda m: (
        (m.mu_x_at(80) - mu_x) ** 2 + (m.mu_n - mu_n) ** 2
    ))
    labels_by_mux[k] = best_mat.symbol


# ──────────────────────────────────────────────────────────────────────────────
# Plot histogram with GMM ellipses
# ──────────────────────────────────────────────────────────────────────────────

fig_hist = plot_bimodal_histogram(
    result.histogram,
    gmm=gmm,
    material_labels=labels_by_mux,
    title="Bimodal Histogram + GMM Segmentation (2-σ ellipses)",
    show_marginals=True,
)
fig_hist.savefig("03_gmm_histogram.png", dpi=150, bbox_inches="tight")
print("saved 03_gmm_histogram.png")


# ──────────────────────────────────────────────────────────────────────────────
# Volume segmentation
# ──────────────────────────────────────────────────────────────────────────────

print("[ex03] Segmenting volume …")
seg_vol = segment_by_gmm(result.vol_xray, result.vol_neutron, gmm)

# Ground-truth label volume from phantom
gt_vol = result.phantom.label_vol


# ──────────────────────────────────────────────────────────────────────────────
# Comparison figure
# ──────────────────────────────────────────────────────────────────────────────

N  = sim.N
si = N // 2

cmap_gt  = plt.cm.get_cmap("tab10", n_true)
cmap_seg = plt.cm.get_cmap("tab10", gmm.n_components)

fig, axes = plt.subplots(2, 3, figsize=(12, 8))

# Row 1: X-ray, neutron, ground-truth
axes[0, 0].imshow(result.vol_xray[si],   cmap="gray")
axes[0, 0].set_title("X-ray recon (central slice)")
axes[0, 0].axis("off")

axes[0, 1].imshow(result.vol_neutron[si], cmap="gray")
axes[0, 1].set_title("Neutron recon (central slice)")
axes[0, 1].axis("off")

im_gt = axes[0, 2].imshow(gt_vol[si], cmap=cmap_gt,
                            vmin=0, vmax=n_true - 1)
axes[0, 2].set_title("Ground-truth labels")
axes[0, 2].axis("off")
cbar = plt.colorbar(im_gt, ax=axes[0, 2], ticks=range(n_true))
cbar.ax.set_yticklabels([m.symbol for m in result.phantom.materials], fontsize=7)

# Row 2: segmentation for XY, XZ, YZ
for col, (slice_gt, slice_seg, title) in enumerate([
    (gt_vol[si],       seg_vol[si],       "GMM labels — XY"),
    (gt_vol[:, si, :], seg_vol[:, si, :], "GMM labels — XZ"),
    (gt_vol[:, :, si], seg_vol[:, :, si], "GMM labels — YZ"),
]):
    im_seg = axes[1, col].imshow(slice_seg, cmap=cmap_seg,
                                  vmin=0, vmax=gmm.n_components - 1)
    axes[1, col].set_title(title)
    axes[1, col].axis("off")

fig.suptitle("GMM Histogram Segmentation vs Ground Truth", fontsize=13)
fig.tight_layout()
fig.savefig("03_gmm_segmentation.png", dpi=150, bbox_inches="tight")
print("saved 03_gmm_segmentation.png")

plt.show()
print("[ex03] Done.")
