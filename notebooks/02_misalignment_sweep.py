#!/usr/bin/env python3
"""
examples/02_misalignment_sweep.py
───────────────────────────────────
Sweep the inter-modality translation misalignment from 0 to 8 voxels and
show how the bimodal histogram clusters progressively smear into horizontal
streaks — the key diagnostic signature of misregistration.

Output files
────────────
  02_misalign_sweep.png   — N-panel histogram grid (one per displacement)
  02_misalign_metrics.png — quantitative streak score vs displacement
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import matplotlib.pyplot as plt

from neutron_xray_sim import DualModalitySimulation, ArtifactConfig
from neutron_xray_sim.histogram import detect_artifact_signatures


# ──────────────────────────────────────────────────────────────────────────────
# Setup
# ──────────────────────────────────────────────────────────────────────────────

sim = DualModalitySimulation(
    preset   = "composite",
    N        = 64,
    n_angles = 120,
    verbose  = True,
)

DISPLACEMENTS = [0.0, 0.5, 1.0, 2.0, 3.0, 5.0, 8.0]   # voxels


# ──────────────────────────────────────────────────────────────────────────────
# Run sweep
# ──────────────────────────────────────────────────────────────────────────────

results = []
hscores = []

ref = sim.run(ArtifactConfig.clean(), tag="0 vx (clean)")
results.append(ref)
hscores.append(ref.signatures.horizontal_streak_score)

for d in DISPLACEMENTS[1:]:
    cfg = ArtifactConfig(
        misalignment=True,
        translation_voxels=(d, 0.0, 0.0),
        rotation_deg=(0.0, 0.0, 0.0),
    )
    r = sim.run(cfg, tag=f"{d:.1f} vx", ref_result=ref)
    results.append(r)
    hscores.append(r.signatures.horizontal_streak_score)


# ──────────────────────────────────────────────────────────────────────────────
# Histogram grid
# ──────────────────────────────────────────────────────────────────────────────

fig_hist = sim.comparison_grid(
    results, ncols=4,
    suptitle="Misalignment Sweep: bimodal histogram smearing",
)
fig_hist.savefig("02_misalign_sweep.png", dpi=150, bbox_inches="tight")
print("saved 02_misalign_sweep.png")


# ──────────────────────────────────────────────────────────────────────────────
# Metric plot
# ──────────────────────────────────────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(DISPLACEMENTS, hscores, "o-", color="tomato", lw=2, ms=7)
ax.set_xlabel("Translation misalignment [voxels]", fontsize=12)
ax.set_ylabel("Horizontal streak score", fontsize=12)
ax.set_title("Misalignment severity vs histogram streak score", fontsize=12)
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig("02_misalign_metrics.png", dpi=150, bbox_inches="tight")
print("saved 02_misalign_metrics.png")

plt.show()
