# API reference

A compact listing of the public surface, subpackage by subpackage. Every name in the
tables is importable from the top level (`from neutron_xray_sim import ...`) unless
marked *(module only)*. Docstrings in the source are the authoritative reference.

```
neutron_xray_sim/
├── physics/          elements, materials, neutron spectra, Bragg edges
├── phantoms/         PhantomData, PhantomBuilder, presets, importers
├── acquisition/      projectors, artifacts, noise, cone-beam geometry
├── reconstruction/   reconstruction algorithms, Fourier fusion
├── analysis/         histograms, GMM, signatures, quality metrics
├── plotting/         histogram figures, publication figure set
├── simulation.py     DualModalitySimulation, run_artifact_survey
├── io.py             SimCache
└── gui/              desktop GUI (PyQt5)
```

> **Old import paths still work.** Code written for 1.x (`neutron_xray_sim.histogram`,
> `neutron_xray_sim.artifacts`, `neutron_xray_sim.diana_plots`, …) keeps working through
> aliases defined in `neutron_xray_sim/_compat.py`. New code should use the paths below.

---

## physics

| Name | Module | Summary |
|---|---|---|
| `ATOMIC_MASS`, `NEUTRON_XS` | `physics.elements` *(module only)* | the single table of per-element data |
| `xray_mass_atten(el)` | `physics.elements` *(module only)* | NIST μ/ρ of one element on `XRAY_E_KEV` (lazy) |
| `Material` | `physics.materials` | one material's X-ray and neutron attenuation |
| `MATERIALS` | `physics.materials` | the material database, keyed by id |
| `XRAY_E_KEV` | `physics.materials` | the 13-point X-ray energy grid (keV) |
| `material_from_formula(...)` | `physics.materials` | material from a chemical formula |
| `make_composite_material(...)` | `physics.materials` | material from a mixture of phases |
| `register_material(key, mat)` | `physics.materials` | add a material to `MATERIALS` |
| `xray_spectrum(kVp, ...)` | `physics.materials` | Kramers bremsstrahlung spectrum |
| `AIR, WATER, ALUMINUM, …` | top level | handles to built-in materials |
| `NEUTRON_MODES`, `NeutronBeam`, `thermal_beam`, `cold_mono_beam`, `cold_poly_beam`, `ill_next_beam` | `physics.neutron_spectra` | neutron beam models |
| `mu_n_lut_for_beam`, `mu_n_spectrum_lut`, `plot_spectra` | `physics.neutron_spectra` | beam-specific attenuation |
| `load_phase`, `mu_n_bragg_lut`, `set_phantom_neutron_mu`, `run_bragg_edge_scan`, `classify_phases` | `physics.ncrystal_bragg` *(module only)* | NCrystal Bragg-edge tools |

## phantoms

| Name | Module | Summary |
|---|---|---|
| `PhantomData` | `phantoms.base` | label volume + derived attenuation volumes |
| `PhantomBuilder` | `phantoms.base` | compose a phantom from primitives; `paint(mat, mask)` for anything else |
| `make_phantom(preset, N=, Nx=, Ny=, Nz=, voxel_cm=, **kw)` | `phantoms.presets` | build a named preset |
| `PHANTOM_PRESETS`, `register_preset` | `phantoms.presets` | the preset registry |
| `make_composite_phantom`, `make_battery_phantom`, … | `phantoms.presets` | individual preset factories |
| `phantom_from_array`, `phantom_from_segmented_volume` | `phantoms.importer` | phantoms from segmented data |
| `NMCParticleConfig`, `generate_nmc_particle_slab` | `phantoms.nmc` *(module only)* | particle-slab phantoms |

## acquisition

| Name | Module | Summary |
|---|---|---|
| `make_sinogram_pair(phantom, ...)` | `acquisition.projector` | X-ray + neutron sinograms |
| `project_xray`, `project_xray_monochromatic`, `project_neutron` | `acquisition.projector` | single-modality projectors |
| `line_integrals(vol, angles, ...)` | `acquisition.projector` *(module only)* | the core projection operator |
| `ArtifactConfig`, `PRESET_CONFIGS` | `acquisition.artifacts` | artifact configuration |
| `inject_sinogram_artifacts`, `inject_volume_artifacts` | `acquisition.artifacts` | apply artifacts |
| `add_poisson_noise`, `predicted_sigma_lambda`, `cnr`, `d_prime`, `rose_dose_threshold`, … | `acquisition.noise` | counting statistics and dose metrics |
| `ConeGeometryConfig`, `forward_project_cone3d`, `reconstruct_cone3d` | `acquisition.cone3d_geometry`, `acquisition.laminography` *(module only)* | 3-D cone-beam / laminography (ASTRA) |

## reconstruction

| Name | Summary |
|---|---|
| `reconstruct(sino_dict, algorithm="FBP", ...)` | reconstruct one volume (cm⁻¹) |
| `reconstruct_pair(xray_sino, neutron_sino, ...)` | reconstruct both modalities |
| `AVAILABLE_ALGORITHMS` | `FBP, GRIDREC, SIRT, SART, CGLS, EM, OSSART, TV_MIN, NESTEROV_SIRT` |
| `FusionConfig`, `fuse_volumes_fourier` | `reconstruction.fusion` *(module only)* |

## analysis

| Name | Module | Summary |
|---|---|---|
| `HistogramResult`, `compute_bimodal_histogram`, `compute_ground_truth_histogram` | `analysis.histogram` | the bimodal histogram |
| `DEFAULT_GT_ENERGY_IDX` | `analysis.histogram` | energy bin used for ground-truth μₓ (80 keV) |
| `GMMFitResult`, `fit_gmm`, `auto_fit_gmm`, `predict_gmm` | `analysis.gmm` | Gaussian mixtures |
| `segment_by_gmm`, `segment_by_polygon` | `analysis.gmm` | histogram-based segmentation |
| `ArtifactSignatures`, `detect_artifact_signatures` | `analysis.signatures` | ground-truth-free artifact scores |
| `ClusterQualityMetrics`, `evaluate_histogram_quality`, `compare_algorithms` | `analysis.quality` | GMM-based quality metrics |
| `ground_truth_positions`, `match_components`, `davies_bouldin`, `pairwise_overlap` | `analysis.quality` *(module only)* | shared metric building blocks |
| `HistogramMetricsTable`, `compute_histogram_metrics` | `analysis.metrics_table` | full metric table with pathology checks |
| `compute_histogram_metrics_morphology_aware` | `analysis.metrics_morphology` | label-anchored metrics |
| `make_cross_algorithm_sinos` | `analysis.cross_algorithm` | shared sinograms for algorithm comparisons |

## plotting

| Name | Module | Summary |
|---|---|---|
| `plot_bimodal_histogram` | `plotting.histograms` | one histogram with marginals / GMM ellipses |
| `plot_comparison_grid` | `plotting.histograms` | grid of histograms |
| `plot_ground_truth_comparison` | `plotting.histograms` | ground truth vs reconstruction |
| `plot_cross_algorithm_grid` | `plotting.histograms` | one panel per algorithm pair |
| `plot_artifact_survey` | `plotting.histograms` | survey figure |
| `plot_CE_vs_nprojections`, `plot_recovery_heatmap`, … | `plotting.publication` | the paper's figure set (saves PDFs; formerly `diana_plots`) |

## simulation & io

| Name | Summary |
|---|---|
| `DualModalitySimulation` | the orchestrator: `run`, `run_batch`, `comparison_grid`, `comparison_slices`, `signature_table` |
| `SimulationResult` | everything one run produced |
| `run_artifact_survey(...)` | returns `(results, figure, metrics)` |
| `SimCache` | on-disk cache: `save_*`, `load_*`, `load_phantom`, `list_run_tags` |
| `tag_to_slug` | run tag → directory name |
