# Histogram analysis

This is where the simulation pays off: the reconstructed X-ray and neutron volumes are
combined into a joint histogram, modelled with a Gaussian mixture, segmented back into
phases, and scored against ground truth. Everything here lives in `histogram.py`.

## The 2-D bimodal histogram

```python
from neutron_xray_sim import compute_bimodal_histogram

hist = compute_bimodal_histogram(
    vol_x, vol_n,
    bins=200,            # bins per axis
    x_range=None,        # (min, max) on the X-ray axis; None = auto (0.1–99.9 pct)
    n_range=None,        # (min, max) on the neutron axis; None = auto
    mask=None,           # optional boolean mask; e.g. exclude air voxels
)
```

Returns a `HistogramResult` with the 2-D count array `H`, the bin edges and centres for
both axes, the flattened voxel value arrays (used for segmentation), and an `extent`
property for `imshow`. Pass `mask=(label_vol != 0)` to drop background and zoom the
histogram onto the materials.

### Ground-truth reference histogram

```python
from neutron_xray_sim import compute_ground_truth_histogram
hist_gt = compute_ground_truth_histogram(phantom, ...)
```

This builds the histogram every material *should* produce from the phantom's known
attenuation values — the target the reconstructed histogram is compared against.

## Fitting a Gaussian mixture model

Each material cluster is modelled as a 2-D Gaussian. You can fix the number of
components or let BIC choose it:

```python
from neutron_xray_sim import fit_gmm, auto_fit_gmm

gmm = fit_gmm(hist, n_components=5)             # fixed K
gmm = auto_fit_gmm(hist, min_k=2, max_k=7)      # BIC-selected K
```

`GMMFitResult` carries the component `means` `(K, 2)` in (μₓ, μₙ), the `covariances`
`(K, 2, 2)`, the mixing `weights`, the per-voxel `labels_flat`, and the `bic` / `aic`
scores. The fit is done on a weighted subsample of voxel pairs for speed.

## Segmentation

Turn a fitted GMM (or a hand-drawn polygon) into a label volume:

```python
from neutron_xray_sim import segment_by_gmm, segment_by_polygon

labels = segment_by_gmm(vol_x, vol_n, gmm)            # assign each voxel to a component
labels = segment_by_polygon(vol_x, vol_n, polygon)    # manual region in (μ_x, μ_n) space
```

`segment_by_gmm` recovers a material map purely from the bimodal data, which you can
compare to the phantom's ground-truth labels — the basis of the overlap metric below.
[Example 03](examples.md) walks through this.

## Quality metrics (ground-truth-anchored)

The headline analysis. Given the phantom, the reconstructed histogram, and the GMM,
`evaluate_histogram_quality` matches each GMM component to a known material and reports
a `ClusterQualityMetrics` object:

| Metric | Meaning | Good value |
|---|---|---|
| `centroid_errors[mat]` | distance the material's cluster moved from its true (μₓ, μₙ) position (cm⁻¹) | → 0 |
| `sigma_x[mat]`, `sigma_n[mat]` | cluster spread along each axis (cm⁻¹) | small |
| `mean_centroid_error` | mean centroid error over all matched non-air materials | low |
| `davies_bouldin` | overall cluster separability index | low (0 = perfect) |
| `overlap_fractions[(a, b)]` | for neighbouring materials, fraction of voxels misclassified vs ground truth | low |
| `n_matched` | how many phases were matched to a component | = number of materials |

`metrics.summary()` prints a compact, readable table. Because these are anchored to the
phantom's exact composition, they let you plot "how separable are my materials?" as a
single number against dose, projection count, algorithm, or beam mode.

A tabulated form across many conditions is provided by `metrics_table.py`
(`compute_histogram_metrics`, `HistogramMetricsTable`) and a morphology-aware variant in
`metrics_table_morphology.py` — see [the API reference](api-reference.md#metrics_table).

## Artifact signatures

A complementary, *reference-free* view: `detect_artifact_signatures` scores the
histogram for the geometric distortions specific artifacts leave behind, optionally
comparing to a clean reference histogram:

```python
from neutron_xray_sim import detect_artifact_signatures
sigs = detect_artifact_signatures(hist, gmm=gmm, ref_hist=hist_clean)
```

`ArtifactSignatures` reports horizontal-streak, vertical-streak, and diagonal-smear
scores, X-axis asymmetry, and the neutron-axis shift relative to the reference. These
are the numbers `SimulationResult.signatures` carries and that
`DualModalitySimulation.signature_table()` tabulates. A rising horizontal-streak score,
for instance, is the fingerprint of inter-modality misalignment.

## Comparing algorithms or conditions

`compare_algorithms` and `make_cross_algorithm_sinos` help reconstruct one sinogram set
with several algorithms and lay their histograms side by side; `plot_comparison_grid`,
`plot_bimodal_histogram`, `plot_ground_truth_comparison`, and
`plot_cross_algorithm_grid` are the plotting entry points. See
[the API reference](api-reference.md#histogram) for the full list.

## A note on the bimodal histogram and registration

The whole method assumes the two volumes are registered: voxel *(i, j, k)* in the X-ray
volume is the same physical point as in the neutron volume. The `misalignment` artifact
deliberately breaks this, and the centroid-error and streak metrics quantify the
resulting damage — which is why measuring registration sensitivity is one of the
package's primary use cases.
