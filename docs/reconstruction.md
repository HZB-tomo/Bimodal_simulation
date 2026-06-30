# Reconstruction

Reconstruction turns the projection sinograms back into 3-D attenuation volumes. The
package exposes one analytic and several iterative algorithms behind a single
`reconstruct` function, with `reconstruct_pair` doing both modalities at once.

## Available algorithms

`AVAILABLE_ALGORITHMS`:

| Name | Type | Backend | Notes |
|---|---|---|---|
| `FBP` | analytic | ASTRA GPU or scikit-image CPU | filtered back-projection; the universal default |
| `GRIDREC` | analytic | TomoPy (CPU) | Fourier gridding; falls back to FBP if TomoPy is absent |
| `SIRT` | iterative | ASTRA | simultaneous iterative reconstruction |
| `SART` | iterative | ASTRA | simultaneous algebraic reconstruction |
| `CGLS` | iterative | ASTRA | conjugate-gradient least squares; converges fast |
| `EM` | iterative | ASTRA | expectation–maximisation (a.k.a. MLEM) |
| `OSSART` | iterative | ASTRA | ordered-subset SART |
| `TV_MIN` | iterative | ASTRA | total-variation minimisation (edge-preserving) |
| `NESTEROV_SIRT` | iterative | ASTRA | SIRT with Nesterov acceleration |

Names are case-insensitive and several aliases are accepted (e.g. `ram-lak` → `FBP`,
`mlem` → `EM`, `total_variation` → `TV_MIN`, `nesterov` → `NESTEROV_SIRT`).

**Backend behaviour.** FBP runs on either backend. The iterative algorithms require
ASTRA; if ASTRA is unavailable the reconstructor warns and falls back to scikit-image
FBP. `GRIDREC` requires TomoPy and otherwise falls back to FBP.

## `reconstruct`

```python
from neutron_xray_sim import reconstruct

vol = reconstruct(
    sino_dict,                 # a projector output dict
    algorithm="FBP",
    filter_name="shepp-logan", # ramp filter for FBP / gridrec
    n_iter=50,                 # iterations for iterative algorithms
    n_subsets=10,              # OSSART only
    lambda_tv=0.02,            # TV_MIN only (higher = smoother)
    remove_rings=True,         # Vo (2018) ring removal before reconstruction
    ring_snr=3.0,
    center_offset=0.0,         # rotation-centre correction (pixels)
    use_astra=True,
    clip_negative=True,        # clip result to >= 0
    clip_threshold=0.0,        # zero out voxels below this value (cm^-1)
)
```

It reads `sino_lam`, `angles_deg`, and `voxel_cm` from the dictionary and divides the
result by `voxel_cm`, so the returned volume is in **cm⁻¹**. Output shape is
`(N_slice, N_det, N_det)` float32.

### Parameter notes

| Parameter | When it matters |
|---|---|
| `filter_name` | FBP/gridrec only: `ram-lak`, `shepp-logan`, `cosine`, `hann`, `hamming`. Sharper filters (ram-lak) preserve edges but pass more noise. |
| `n_iter` | All iterative algorithms. CGLS typically needs far fewer iterations than SIRT for similar quality. |
| `lambda_tv` | `TV_MIN` only; typical range 0.005–0.1. Higher suppresses noise but can flatten real structure. |
| `remove_rings` | Applies Vo ring-removal to the sinogram first; lower `ring_snr` is more aggressive. |
| `center_offset` | Correct a known rotation-axis offset (pixels). |
| `clip_threshold` | Set near-zero background (air) to 0; ~0.01–0.05 cm⁻¹ is reasonable for solid objects. Applied after `clip_negative`. |

## `reconstruct_pair`

```python
from neutron_xray_sim import reconstruct_pair

vol_x, vol_n = reconstruct_pair(
    xray_sino, neutron_sino,
    algorithm="FBP", filter_name="shepp-logan",
    n_iter=50, remove_rings=True, use_astra=True,
)
```

Returns both volumes with identical settings; this is what the orchestrator calls.

## Choosing an algorithm

- **Exploration / many runs:** `FBP`. Fast, runs without a GPU, good enough to see how
  artifacts move the clusters.
- **Few projections / limited-angle:** an iterative method (`SIRT`, `CGLS`, or
  `TV_MIN`) reconstructs noticeably better from sparse data, at the cost of GPU time.
- **Noise suppression with edge preservation:** `TV_MIN`.
- **Fast iterative convergence:** `CGLS` or `NESTEROV_SIRT`.

Because every algorithm produces the same kind of volume, you can hold the phantom and
artifacts fixed and sweep `algorithm` to study how the reconstruction method itself
affects bimodal-cluster quality — feed the resulting histograms to
[`evaluate_histogram_quality`](histogram-analysis.md) and compare the metrics.
