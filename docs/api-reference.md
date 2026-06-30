# API reference

A compact listing of the public surface, module by module. Names below are importable
from the top-level package (`from neutron_xray_sim import ...`) unless noted otherwise.
For full parameter lists, read the docstrings in the source — they are the
authoritative reference.

---

## materials

Module: `neutron_xray_sim.materials`

| Name | Kind | Summary |
|---|---|---|
| `Material` | dataclass | one material's X-ray and neutron attenuation; properties `mu_n`, `mu_n_scatter`, `rho`; methods `mu_x_at`, `mu_x_array` |
| `MATERIALS` | dict | the material database, keyed by name |
| `XRAY_E_KEV` | array | the 13-point X-ray energy grid (keV) |
| `xray_spectrum(kVp, filter_mm_Al, filter_mm_Cu, n_bins)` | function | normalised bremsstrahlung spectrum → `(energies_keV, weights)` |
| `material_from_formula(name, symbol, formula, density_gcc, ...)` | function | build a `Material` from a chemical formula |
| `make_composite_material(name, symbol, bulk_density_gcc, components, ...)` | function | build a `Material` from a mixture of phases |
| `AIR, WATER, ALUMINUM, HDPE, IRON, TITANIUM, COPPER, LEAD, BONE, TUNGSTEN, ZINC` | constants | convenience handles to built-in materials |

Also exposed in `materials`: `XRAY_MASS_ATTEN`, `ATOMIC_MASS`, `NEUTRON_XS`,
`parse_formula`, `neutron_components_from_formula`.

See [Materials](materials.md).

---

## phantom

Module: `neutron_xray_sim.phantom`

| Name | Kind | Summary |
|---|---|---|
| `PhantomData` | dataclass | labelled volume + derived attenuation volumes (the ground truth) |
| `PhantomBuilder` | class | compose a phantom from geometric primitives |
| `make_phantom(preset, N=..., Nx=, Ny=, Nz=, voxel_cm=)` | function | load a named preset (cubic or non-cubic) |
| `PHANTOM_PRESETS` | dict | preset name → factory function |
| `make_composite_phantom`, `make_battery_phantom`, `make_bone_implant_phantom`, `make_industrial_phantom`, `make_custom_cylindrical_battery_phantom`, `make_hdpe_composite_phantom` | functions | the individual preset builders |

See [Phantoms](phantoms.md).

---

## projector

Module: `neutron_xray_sim.projector`

| Name | Summary |
|---|---|
| `make_sinogram_pair(phantom, n_angles, angle_range_deg, xray_mode, kVp, ..., I0_xray, I0_neutron, use_astra, geometry, SDD, SOD)` | project both modalities → `(xray_sino_dict, neutron_sino_dict)` |
| `project_xray(phantom, angles_deg, kVp, filter_mm_Al, filter_mm_Cu, n_spectrum_bins, use_astra, I0, geometry, SDD, SOD)` | polychromatic X-ray sinogram dict |
| `project_neutron(phantom, angles_deg, use_astra, I0, scatter_D_over_L)` | thermal-neutron sinogram dict (with absorption / scatter components) |

`xray_mode` is `"polychromatic"` (default) or `"monochromatic"` (needs `xray_energy_keV`).
X-ray geometry can be `"parallel"` or `"cone"`; neutron projection is always parallel.
See [the pipeline data contracts](pipeline.md#data-contracts).

---

## artifacts

Module: `neutron_xray_sim.artifacts`

| Name | Summary |
|---|---|
| `ArtifactConfig` | master dataclass of all artifact switches and amplitudes; factories `clean`, `noise_only`, `beam_hardening_only`, `scatter_only`, `misalignment_only`, `realistic`; method `summary` |
| `inject_sinogram_artifacts(xray_sino, neutron_sino, cfg, rng)` | apply sinogram-domain artifacts |
| `inject_volume_artifacts(vol_x, vol_n, cfg, rng)` | apply volume-domain artifacts |
| `PRESET_CONFIGS` | dict of named example configurations |

See [Artifacts](artifacts.md) for every field.

---

## reconstructor

Module: `neutron_xray_sim.reconstructor`

| Name | Summary |
|---|---|
| `reconstruct(sino_dict, algorithm, filter_name, n_iter, n_subsets, lambda_tv, remove_rings, ring_snr, center_offset, use_astra, clip_negative, clip_threshold)` | reconstruct one volume (cm⁻¹) |
| `reconstruct_pair(xray_sino, neutron_sino, ...)` | reconstruct both volumes with identical settings |
| `AVAILABLE_ALGORITHMS` | list of supported algorithm names |

See [Reconstruction](reconstruction.md).

---

## histogram

Module: `neutron_xray_sim.histogram`

| Name | Kind | Summary |
|---|---|---|
| `HistogramResult` | dataclass | a 2-D bimodal histogram and its bin geometry |
| `GMMFitResult` | dataclass | fitted GMM (means, covariances, weights, labels, BIC/AIC) |
| `ClusterQualityMetrics` | dataclass | ground-truth-anchored cluster-quality metrics; method `summary` |
| `ArtifactSignatures` | dataclass | reference-free distortion scores |
| `compute_bimodal_histogram(vol_x, vol_n, bins, x_range, n_range, mask)` | function | build `H(μ_x, μ_n)` |
| `compute_ground_truth_histogram(phantom, ...)` | function | the target histogram from known materials |
| `fit_gmm(hist, n_components, ...)` | function | fit a GMM with fixed K |
| `auto_fit_gmm(hist, min_k, max_k, ...)` | function | BIC-selected GMM |
| `segment_by_gmm(vol_x, vol_n, gmm)` | function | label volume from a GMM |
| `segment_by_polygon(vol_x, vol_n, polygon)` | function | label volume from a manual region |
| `detect_artifact_signatures(hist, gmm, ref_hist)` | function | compute `ArtifactSignatures` |
| `evaluate_histogram_quality(phantom, hist, gmm, ...)` | function | compute `ClusterQualityMetrics` |
| `compare_algorithms(...)`, `make_cross_algorithm_sinos(...)` | functions | reconstruct/compare across algorithms |
| `plot_bimodal_histogram`, `plot_comparison_grid`, `plot_ground_truth_comparison`, `plot_cross_algorithm_grid` | functions | plotting |

See [Histogram analysis](histogram-analysis.md).

---

## simulation

Module: `neutron_xray_sim.simulation`

| Name | Kind | Summary |
|---|---|---|
| `DualModalitySimulation` | class | the orchestrator; methods `run`, `run_batch`, `comparison_grid`, `comparison_slices`, `signature_table` |
| `SimulationResult` | dataclass | everything one run produced; methods `summary`, `plot_histogram`, `plot_slices` |
| `run_artifact_survey(...)` | function | run a structured one-artifact-at-a-time survey |

See [The simulation pipeline](pipeline.md).

---

## neutron_spectra

Module: `neutron_xray_sim.neutron_spectra`

| Name | Summary |
|---|---|
| `NeutronBeam` | dataclass describing a neutron beam (energy grid, flux, mean energy/wavelength) |
| `NEUTRON_MODES` | dict of pre-built beams: `thermal`, `cold_5meV`, `cold_2meV`, `ill_next` |
| `thermal_beam()`, `cold_mono_beam(E_meV)`, `cold_poly_beam(...)`, `ill_next_beam(...)` | beam constructors |
| `mu_n_lut_for_beam(materials, beam)` | flux-weighted neutron attenuation per material |
| `mu_n_spectrum_lut(materials, beam, n_bins)` | per-energy-bin attenuation table |
| `mu_n_at_energy(...)`, `mu_n_for_beam(...)` | single-material attenuation helpers |
| `plot_spectra({name: beam})` | overlay beam spectra |

See [Neutron spectra](neutron-spectra.md).

---

## volume_importer

Module: `neutron_xray_sim.volume_importer`

| Name | Summary |
|---|---|
| `phantom_from_segmented_volume(volume_source, metadata, name=None)` | main constructor from a segmented volume (`phantom_from_files` is an alias) |
| `phantom_from_array(seg, metadata, name=None)` | build from an in-memory array |
| `SegmentationMetadata` | metadata dataclass (`name`, `voxel_cm`, `class_map`, `axis_order`) |
| `metadata_from_dict`, `load_segmentation_metadata`, `resolve_segmentation_metadata` | metadata loaders |
| `load_segmented_array`, `remap_labels_to_material_indices` | volume loaders/remappers |

See [Importing real segmented data](importing-data.md).

---

## io

Module: `neutron_xray_sim.io`

| Name | Summary |
|---|---|
| `SimCache(root, overwrite=False)` | on-disk cache of phantom, raw sinograms, and runs |
| `tag_to_slug(tag, max_len=64)` | filesystem-safe slug from a run tag |

Cache layout under `root/`: `phantom/`, `sinograms/`, `runs/<tag>/`, `survey/<slug>/`.
Pass `cache_dir=` to `DualModalitySimulation` to use it automatically.

---

## metrics_table

Modules: `neutron_xray_sim.metrics_table`, `neutron_xray_sim.metrics_table_morphology`

| Name | Summary |
|---|---|
| `HistogramMetricsTable` | tabulated cluster metrics across conditions |
| `compute_histogram_metrics(...)` | populate the table from a run |
| `compute_histogram_metrics_morphology_aware(...)` | morphology-aware variant (label- or morphology-anchored) |

---

## diana_plots

Module: `neutron_xray_sim.diana_plots`

Publication-figure helpers, useful for parameter studies. They plot metrics against
projection count and across geometries/beam modes, e.g.
`plot_CE_vs_nprojections`, `plot_DB_vs_nprojections`,
`plot_sigma_x_vs_nprojections`, `plot_sigma_n_vs_nprojections`,
`plot_epsilon_vs_nprojections`, `plot_pairwise_overlap_vs_nprojections`,
`plot_CE_all_geometries`, `plot_DB_all_geometries`, `plot_cross_sweep_heatmap`,
`plot_xray_attenuation_spectra`, `plot_phantom_label_map`,
`plot_reconstruction_slice`, plus `plot_bimodal_histogram`, `plot_xray_marginal`,
`plot_neutron_marginal`. (CE = centroid error, DB = Davies–Bouldin.)
