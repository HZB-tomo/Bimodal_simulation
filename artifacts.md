# Artifacts

Every acquisition imperfection in the simulator is controlled through a single
dataclass, `ArtifactConfig`. Setting a flag to `False` (or an amplitude to 0) disables
that artifact completely, so you can study one effect at a time or any combination.

Artifacts are applied in two domains:

- **Sinogram-domain** artifacts are injected before reconstruction (noise, scatter,
  detector blur, rings, beam-hardening correction).
- **Volume-domain** artifacts are applied after reconstruction (inter-modality
  misalignment, salt-and-pepper voxel corruption).

`inject_sinogram_artifacts(...)` and `inject_volume_artifacts(...)` perform the two
stages; `DualModalitySimulation.run` calls them for you.

## Factory presets

| Constructor | What it gives you |
|---|---|
| `ArtifactConfig.clean()` | all artifacts off — the reference run |
| `ArtifactConfig.noise_only(I0=5e4)` | Poisson counting noise only |
| `ArtifactConfig.beam_hardening_only()` | polychromatic beam-hardening artifact, no correction |
| `ArtifactConfig.scatter_only()` | neutron + X-ray scatter only |
| `ArtifactConfig.misalignment_only(translation, rotation)` | inter-modality misregistration only |
| `ArtifactConfig.realistic()` | all physical artifacts at moderate, realistic levels |

`cfg.summary()` returns a one-line description of the active artifacts.

## Field reference

### 1. Photon / neutron counting noise

| Field | Default | Meaning |
|---|---|---|
| `photon_noise` | `False` | enable Poisson counting noise on both sinograms |
| `I0_xray` | `1e5` | incident X-ray photons per pixel per projection |
| `I0_neutron` | `1e5` | incident neutrons per pixel per projection |

Lower `I0` means lower dose and noisier sinograms; this is the cleanest knob for a
dose study. Noise widens every histogram cluster.

### 2. X-ray beam hardening

Beam hardening is **not injected** — it emerges automatically from polychromatic
projection (lower-energy photons are absorbed preferentially, so the effective spectrum
hardens through the object). The config controls whether you *correct* it:

| Field | Default | Meaning |
|---|---|---|
| `apply_bh_correction` | `False` | apply a polynomial beam-hardening correction to the X-ray sinogram |
| `bh_correction_order` | `3` | polynomial order (2–4 typical) |

Leave `apply_bh_correction=False` to keep the artifact and study its effect; set it
`True` to correct it.

### 3. Neutron scatter build-up

| Field | Default | Meaning |
|---|---|---|
| `neutron_scatter` | `False` | add a scattered-neutron background (Gaussian halo) |
| `scatter_fraction` | `0.05` | fraction of unscattered intensity that becomes scatter |
| `scatter_sigma_pixels` | `8.0` | Gaussian blur σ of the scatter halo (detector pixels) |
| `scatter_D_over_L` | `100.0` | beam collimation ratio; larger → more geometric scatter |

### 4. X-ray scatter

| Field | Default | Meaning |
|---|---|---|
| `xray_scatter` | `False` | add scattered X-ray background |
| `xray_scatter_fraction` | `0.03` | fraction of primary intensity that becomes scatter |
| `xray_scatter_sigma_pixels` | `20.0` | Gaussian blur σ of the X-ray scatter halo |

### 5. Detector point-spread function

| Field | Default | Meaning |
|---|---|---|
| `detector_psf` | `False` | convolve each projection with a Gaussian PSF (scintillator blur) |
| `psf_sigma_xray_pixels` | `0.8` | X-ray detector PSF σ (pixels) |
| `psf_sigma_neutron_pixels` | `1.5` | neutron detector PSF σ (pixels; scintillators are coarser) |

Blur mixes neighbouring voxels and smears clusters along the line that joins them.

### 6. Ring artifacts

| Field | Default | Meaning |
|---|---|---|
| `ring_artifacts` | `False` | introduce ring/band artifacts from bad detector columns |
| `n_bad_columns` | `3` | number of bad columns |
| `ring_amplitude` | `0.05` | offset added to bad columns (log-attenuation units) |
| `ring_seed` | `42` | RNG seed for which columns are bad |

### 7. Inter-modality misalignment (volume domain)

| Field | Default | Meaning |
|---|---|---|
| `misalignment` | `False` | apply a rigid-body transform to the neutron volume |
| `translation_voxels` | `(0, 0, 0)` | translation (Δy, Δx, Δz) in voxels |
| `rotation_deg` | `(0, 0, 0)` | Euler angles (ry, rx, rz) in degrees |

This is the most damaging artifact for the bimodal method specifically: each voxel's
X-ray value is paired with a *displaced* neutron value, so clusters smear into
horizontal streaks. [Example 02](examples.md) sweeps the translation to map this out.

### 8. Salt-and-pepper voxel noise (volume domain)

| Field | Default | Meaning |
|---|---|---|
| `salt_pepper` | `False` | randomly corrupt a fraction of voxels in both volumes |
| `salt_pepper_fraction` | `0.001` | fraction of voxels corrupted |
| `salt_pepper_seed` | `99` | RNG seed |

## Putting it together

A custom combination — moderate low-dose noise plus a small misregistration:

```python
from neutron_xray_sim import ArtifactConfig

cfg = ArtifactConfig(
    photon_noise=True, I0_xray=1e4, I0_neutron=1e4,
    neutron_scatter=True, scatter_fraction=0.08,
    misalignment=True, translation_voxels=(3.0, 0.0, 0.0),
    rotation_deg=(0.0, 1.5, 0.0),
)
print(cfg.summary())
```

Pass it to `sim.run(cfg, tag="...")`. To make runs reproducible, `run` takes an
`rng_seed`; the volume-domain ring/salt seeds are fixed separately in the config so
that toggling other artifacts does not change which columns or voxels are affected.
