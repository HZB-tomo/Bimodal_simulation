# The simulation pipeline

This page describes the end-to-end flow, the orchestrator that drives it, and the data
contracts that connect the stages. If you want to call the stages by hand instead of
using `DualModalitySimulation`, this is the reference.

## Overview

```
make_phantom / PhantomBuilder / phantom_from_segmented_volume
        │  PhantomData (label volume + μ volumes, ground truth)
        ▼
make_sinogram_pair  ──►  (xray_sino_dict, neutron_sino_dict)
        │                project_xray (polychromatic) + project_neutron (thermal)
        ▼
inject_sinogram_artifacts  ──►  noisy / scattered / ringed / blurred sinograms
        ▼
reconstruct_pair  ──►  (vol_xray, vol_neutron)   both (N, N, N) float32, cm⁻¹
        ▼
inject_volume_artifacts  ──►  misaligned / corrupted volumes
        ▼
compute_bimodal_histogram  ──►  HistogramResult  H(μ_x, μ_n)
        ▼
fit_gmm / auto_fit_gmm  ──►  GMMFitResult
        ▼
evaluate_histogram_quality / detect_artifact_signatures  ──►  metrics
```

## The orchestrator: `DualModalitySimulation`

`DualModalitySimulation` wires all of the above together and caches intermediate
results so that running many artifact configurations on one phantom is cheap — the
expensive clean projection is computed once and reused.

```python
from neutron_xray_sim import DualModalitySimulation, ArtifactConfig

sim = DualModalitySimulation(
    preset="composite",   # which phantom
    N=64,                 # cubic grid size
    n_angles=180,         # projection angles
    kVp=120.0,            # X-ray tube voltage
    algorithm="FBP",      # reconstruction method
    histogram_bins=200,   # bins per histogram axis
    auto_gmm=False,       # fit a GMM after each run?
    use_astra=True,       # GPU if available
    cache_dir=None,       # set a path to persist every stage to disk
)
```

Key constructor parameters:

| Parameter | Meaning |
|---|---|
| `preset` | one of the [phantom presets](phantoms.md) (ignored if you pass `phantom=`) |
| `N` | cubic grid size (N×N×N voxels) |
| `n_angles`, `angle_range_deg` | projection count and angular range (180 = half scan) |
| `kVp`, `filter_mm_Al`, `filter_mm_Cu`, `n_spectrum_bins` | X-ray source and spectrum |
| `algorithm`, `filter_name`, `n_iter` | reconstruction method and its settings |
| `histogram_bins` | bins per axis in the bimodal histogram |
| `auto_gmm`, `max_gmm_k` | automatic GMM fitting and the max components for BIC selection |
| `use_astra`, `verbose` | GPU backend and progress printing |
| `phantom` | supply your own `PhantomData`, overriding `preset` |
| `cache_dir`, `overwrite_cache` | on-disk caching of every pipeline stage |

### Running

`sim.run(cfg, tag=...)` executes one full pass with the given
[`ArtifactConfig`](artifacts.md) and returns a `SimulationResult`. Because the clean
sinograms are cached internally, the first `run` is the slow one; subsequent runs only
re-inject artifacts, reconstruct, and analyse.

```python
r0 = sim.run(ArtifactConfig.clean(),     tag="clean")
r1 = sim.run(ArtifactConfig.realistic(), tag="realistic", ref_result=r0)
```

Passing `ref_result=r0` lets the artifact-signature analysis measure cluster *shifts*
relative to the clean reference.

Batch helper:

```python
sim.run_batch([
    ("clean",     ArtifactConfig.clean()),
    ("noise",     ArtifactConfig.noise_only()),
    ("realistic", ArtifactConfig.realistic()),
], ref_tag="clean")
```

### `SimulationResult`

Each run returns a dataclass with everything the run produced:

| Field | Contents |
|---|---|
| `tag`, `cfg`, `elapsed_s` | label, the config used, wall-clock time |
| `phantom` | the ground-truth `PhantomData` |
| `vol_xray`, `vol_neutron` | reconstructed volumes, `(N, N, N)` float32, cm⁻¹ |
| `xray_sino`, `neutron_sino` | the (post-artifact) sinogram dicts |
| `histogram` | a `HistogramResult` |
| `gmm` | an optional `GMMFitResult` |
| `signatures` | optional `ArtifactSignatures` (streak/smear/shift scores) |

Convenience methods: `summary()` (text overview), `plot_histogram()`,
`plot_slices()`. On the simulation object: `comparison_grid()`,
`comparison_slices()`, and `signature_table()` compare several results at once.

## Calling the stages directly

If you want fine control, bypass the orchestrator:

```python
import numpy as np
from neutron_xray_sim import (
    make_phantom, make_sinogram_pair, ArtifactConfig,
    inject_sinogram_artifacts, reconstruct_pair,
    inject_volume_artifacts, compute_bimodal_histogram,
)

phantom = make_phantom("composite", N=64)

xray_sino, neutron_sino = make_sinogram_pair(
    phantom, n_angles=180, kVp=120.0, I0_xray=1e5, I0_neutron=1e5,
)

cfg = ArtifactConfig(photon_noise=True, I0_xray=5e4, I0_neutron=5e4)
rng = np.random.default_rng(0)
xray_sino, neutron_sino = inject_sinogram_artifacts(xray_sino, neutron_sino, cfg, rng=rng)

vol_x, vol_n = reconstruct_pair(xray_sino, neutron_sino, algorithm="FBP")
vol_x, vol_n = inject_volume_artifacts(vol_x, vol_n, cfg, rng=rng)

hist = compute_bimodal_histogram(vol_x, vol_n, bins=200)
```

## Data contracts

### `PhantomData`

Carries the integer **label volume** and the derived attenuation volumes:

- `label_vol` — `(Nz, Nx, Ny)` integer array; value *i* selects `materials[i]`.
- `materials` — ordered list of `Material`; index 0 is air by convention.
- `voxel_cm` — voxel side length in cm.
- `mu_n_vol`, `mu_n_abs_vol`, `mu_n_coh_vol`, `mu_n_inc_vol` — neutron attenuation
  volumes (total and the three components), cm⁻¹.
- `mu_x_vols` — X-ray attenuation, shape `(13, Nz, Nx, Ny)`, one slab per `XRAY_E_KEV`
  energy, cm⁻¹.

These derived volumes are built automatically from `label_vol` + `materials`.

### Sinogram dictionaries

Both projectors return a `dict`. The reconstructor only requires three keys, but the
extra fields are useful for artifact injection and bookkeeping.

X-ray (`project_xray`):

| Key | Meaning |
|---|---|
| `sino_lam` | `(n_angles, n_slices, n_det)` line integrals (optical depth) — **the reconstruction input** |
| `sino_trans` | transmitted-intensity sinogram, `exp(-line integral)` summed over the spectrum |
| `angles_deg` | projection angles |
| `spectrum` | `{energies_keV, weights}` of the polychromatic source |
| `I0`, `voxel_cm`, `geometry`, `SDD`, `SOD` | acquisition metadata |

Neutron (`project_neutron`):

| Key | Meaning |
|---|---|
| `sino_lam` | total line-integral sinogram (absorption + coherent + incoherent) |
| `sino_trans` | total transmission |
| `sino_abs_lam` | absorption-only line integrals |
| `sino_scatter_lam` | coherent + incoherent line integrals |
| `angles_deg`, `I0`, `scatter_D_over_L`, `voxel_cm` | metadata |

`reconstruct` reads `sino_lam`, `angles_deg`, and `voxel_cm`, and divides the
reconstructed values by `voxel_cm` to return cm⁻¹.

### Reconstructed volumes

`reconstruct_pair` returns `(vol_xray, vol_neutron)`, each `(N_slice, N_det, N_det)`
float32 in cm⁻¹, non-negativity-clipped by default.

## Caching

If you pass `cache_dir=` to `DualModalitySimulation`, a `SimCache` persists the
phantom, the raw sinograms, and every run to disk under that directory (subfolders
`phantom/`, `sinograms/`, `runs/<tag>/`). On the next session the simulation reloads
the cached phantom and raw sinograms instead of recomputing them. Use
`overwrite_cache=True` to force regeneration. See [the API reference](api-reference.md#io)
for the cache layout.
