# Example scripts

Four runnable scripts under `examples/` reproduce the package's main figures. Each adds
the repository root to `sys.path` and writes its output files to the working directory.
Run them from the repo root, e.g. `python examples/01_artifact_comparison.py`.

## 01 — Artifact comparison

`examples/01_artifact_comparison.py`

Runs eight configurations on one phantom — clean, two noise levels, beam hardening,
neutron scatter, misalignment, ring artifacts, and the full realistic combination —
and shows how each one deforms the bimodal histogram.

Outputs:

- `01_histogram_grid.png` — a 2×4 grid of bimodal histograms, one per configuration.
- `01_slice_comparison.png` — X-ray and neutron central slices for a representative subset.
- `01_signatures.txt` — a table of quantitative artifact-signature metrics.

The clean run is used as the reference for the signature comparison. Edit the top of
the script to change the phantom, grid size `N`, projection count, or algorithm.

## 02 — Misalignment sweep

`examples/02_misalignment_sweep.py`

Sweeps the inter-modality translation from 0 to 8 voxels and shows the histogram
clusters progressively smearing into horizontal streaks — the diagnostic signature of
misregistration between the two modalities.

Outputs:

- `02_misalign_sweep.png` — one histogram panel per displacement.
- `02_misalign_metrics.png` — the streak score plotted against displacement.

This is the clearest demonstration of why registration matters for the bimodal method,
and of what the streak metric in `detect_artifact_signatures` measures.

## 03 — GMM segmentation

`examples/03_gmm_segmentation.py`

Fits a Gaussian mixture model to the bimodal histogram (with automatic BIC-based
component selection) and uses it to segment the reconstructed volume back into material
phases, then compares the recovered labels to ground truth.

Outputs:

- `03_gmm_histogram.png` — the bimodal histogram with GMM ellipses overlaid.
- `03_gmm_segmentation.png` — ground-truth versus GMM-recovered segmentation slices.

This shows the analysis half of the package end to end: histogram → GMM → segmentation
→ comparison.

## 04 — Phantom showcase

`examples/04_phantom_showcase.py`

Runs all four core preset phantoms (composite, battery, bone+implant, industrial) in
clean mode and compares how their materials occupy different regions of the (μₓ, μₙ)
plane.

Outputs:

- `04_phantom_histograms.png` — a four-panel histogram grid.
- `04_phantom_slices.png` — a 4×2 panel of X-ray and neutron slices.

A good first script to run to get a feel for what each phantom looks like in both
modalities and in the bimodal histogram.

## Adapting the examples

All four follow the same pattern:

```python
sim = DualModalitySimulation(preset=..., N=..., n_angles=..., algorithm=...)
result = sim.run(ArtifactConfig(...), tag="...")
fig = sim.comparison_grid([...])
fig.savefig("...png", dpi=150, bbox_inches="tight")
```

To turn them into a study, wrap `sim.run` in a loop over the quantity you care about
(dose `I0`, `n_angles`, `algorithm`, or beam mode), collect the
[`ClusterQualityMetrics`](histogram-analysis.md), and plot a metric-versus-parameter
curve — the `diana_plots` module has ready-made helpers for exactly these curves.
