# Changelog

All notable changes to DIANA (`neutron_xray_sim`). Dates are ISO-8601.

## [2.0.0] — 2026-09-25

A reorganisation for collaboration plus a round of bug fixes. **Public imports are
backwards compatible**: `from neutron_xray_sim import ...` and the old module paths
(`neutron_xray_sim.histogram`, `.artifacts`, `.diana_plots`, …) keep working.

### ⚠ Changes that alter simulated numbers

* **CPU projector geometry fixed.** Without ASTRA, the NumPy projector rotated in the
  opposite sense to the CPU (scikit-image) FBP, with a half-pixel centre offset, so CPU
  reconstructions came out **mirrored** relative to the phantom (correlation with the
  phantom ≈ 0 instead of ≈ 0.9). Anything that compared CPU reconstructions voxel-wise with
  the phantom — label-anchored metrics, overlap fractions, GT-seeded GMMs, difference
  images — was affected. The projector now matches `skimage.transform.radon` to machine
  precision. The ASTRA (GPU) path is unchanged.
* **Detection-chain order fixed.** Scatter and detector PSF are now applied *before* the
  Poisson noise (previously the noisy data were blurred, which smoothed the noise away).
* **Scatter and PSF blur are 2-D.** They previously acted along detector rows only; they now
  blur each projection image (slice × detector). Runs with `neutron_scatter`,
  `xray_scatter` or `detector_psf` enabled differ from 1.x.
* **Single element table.** `neutron_spectra` had its own copy of the cross-section table
  that had drifted from `materials` (e.g. Mn coherent 2.15 b vs the corrected 1.75 b; In
  missing). Both now read `physics/elements.py`, so energy-dependent μₙ changes slightly
  for affected elements.

### Fixed

* `segment_by_gmm` crashed with current scikit-learn (`precisions_cholesky_`); GMM
  prediction is now a small NumPy routine independent of sklearn internals, and
  `fit_gmm` always returns full covariance matrices (for `diag`/`tied`/`spherical` too).
* The top-level `plot_bimodal_histogram` was silently replaced by the publication variant
  from `diana_plots` (a duplicate import); it is the documented function again.
* Importing the package changed global matplotlib style (`classic`) and created an
  `outputs_disc_sweep/` directory in the working directory. It no longer has side effects.
* Importing the package emitted ten warnings about missing X-ray data files; element data
  is now loaded lazily with a clear error only when an element is actually used.
* `make_phantom("jellyroll_battery", Nx=..., Ny=..., Nz=...)` raised `TypeError`; presets
  that do not support non-cubic grids now raise a clear `ValueError`, and extra keyword
  arguments are forwarded to the preset (`make_phantom(..., n_jellyroll_turns=1)`).
* `project_xray_monochromatic` assumed a cubic phantom (wrong sinogram shape otherwise).
* `DualModalitySimulation.run` used a shared mutable default `ArtifactConfig`.
* `SimCache` silently reused stale sinograms when projection settings (e.g. `n_angles`)
  changed, and silently mixed phantoms in one directory; both are now detected.
* `SimCache.load_phantom()` was documented but did not exist; it now restores the phantom
  (and the GUI uses it instead of rebuilding a preset by name).
* `SimCache` read methods created empty run directories as a side effect.
* The cached sinogram metadata always recorded `kVp = 0`.
* `evaluate_histogram_quality` silently returned no overlap fractions for masked
  histograms (label/voxel misalignment); masks are now tracked in `HistogramResult`.
  The overlap loop is vectorised (it was a Python loop over voxels).
* `compute_histogram_metrics(gt_seeded_gmm=True)` indexed ground-truth labels
  incorrectly for masked histograms.
* `RegionSummary.centroid_spatial` had x and y swapped.
* `run_bragg_edge_scan` left the caller's phantom holding the last wavelength's μ.
* `PhantomBuilder` material lookup could raise "truth value of an array is ambiguous";
  materials are now compared by identity.
* `neutron_spectra` monkey-patched `numpy.trapezoid` globally.
* `make_composite_material` validated input with `assert` (skipped under `python -O`).
* `pyproject.toml` named a non-existent build backend (`pip install .` failed); `pandas`
  was used but not declared.
* `examples/03_gmm_segmentation.py` used `plt.cm.get_cmap`, removed in matplotlib 3.9.

### Changed — structure

* The flat package is split into subpackages: `physics`, `phantoms`, `acquisition`,
  `reconstruction`, `analysis`, `plotting`, `gui` (see `docs/api-reference.md`).
* `histogram.py` (1.8k lines) → `analysis.histogram`, `analysis.gmm`,
  `analysis.signatures`, `analysis.quality`, `analysis.cross_algorithm`,
  `plotting.histograms`. `phantom.py` → `phantoms.base` + `phantoms.presets`.
* The three metric families share one implementation of ground-truth positions, component
  matching, the Davies–Bouldin index and overlap fractions (`analysis.quality`).
* The projector's three copies of the ASTRA/NumPy loop are one `line_integrals` function;
  `artifacts` uses `noise.add_poisson_noise` instead of its own Poisson model.
* The two copies of the ground-truth bubble panel and the survey drawing code moved out of
  `simulation.py` into `plotting.histograms` (`plot_artifact_survey`).
* The GUI moved into the package (`diana-gui` / `python -m neutron_xray_sim.gui`);
  `launch_gui.py` still works.
* Removed duplicates: example scripts inside the package, duplicate notebooks
  (`01_real_battery_simulation`, tutorial v2, volume-pair analyser v1), `setup.py`,
  the package-level `requirements.txt`, and 29 committed `.pyc` files.
* Notebook outputs are no longer committed (repository size of `notebooks/` 20 MB → 0.3 MB).

### Added

* `tutorials/`: seven executable, CPU-friendly tutorial notebooks.
* `tests/`: a pytest suite (65 tests) including regression tests for the fixes above.
* GitHub Actions CI (lint, tests on Python 3.9/3.12, tutorial execution), `pre-commit`
  (nbstripout, ruff), issue and PR templates, `CONTRIBUTING.md`.
* `register_material`, `register_preset`, `PhantomBuilder.paint` and coordinate grids
  `PhantomBuilder.X/Y/Z`, `predict_gmm`, `line_integrals`, `DEFAULT_GT_ENERGY_IDX`,
  `verbose=` switches on `make_sinogram_pair` / `reconstruct_pair`, preset `li_ion_spiral`.

## [1.1.0]

Previous release (flat module layout).
