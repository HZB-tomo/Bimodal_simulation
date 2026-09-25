# Contributing to DIANA

Thanks for helping! This page explains how the code is organised, how to set up a
development environment, and the few conventions that keep collaboration smooth.

## 1. Set up

```bash
git clone https://github.com/OriolSansPlanell/DIANA.git
cd DIANA
python -m venv .venv && source .venv/bin/activate      # or a conda env (needed for ASTRA)
pip install -e ".[dev,notebooks]"
pre-commit install                                     # strips notebook outputs, runs ruff
pytest                                                 # ~1 min on CPU
```

Everything runs without a GPU. If you have one, `conda install -c astra-toolbox -c nvidia
astra-toolbox` enables the ASTRA backend; the tests use the CPU backends.

## 2. Where does my code go?

The package is split by pipeline stage. Each subpackage has one job and depends only on the
ones above it in this list:

| Subpackage | Put here… | Examples |
|---|---|---|
| `physics/` | element data, materials, beam spectra | a new element, a material, a neutron source model |
| `phantoms/` | ground-truth volumes | a new preset, an importer for a new file format |
| `acquisition/` | anything that turns a phantom into measured sinograms | a projector, a new artifact, a noise model |
| `reconstruction/` | sinograms → volumes | a reconstruction algorithm, volume fusion |
| `analysis/` | volumes → numbers | histogram metrics, segmentation, signatures |
| `plotting/` | figures (never computes anything expensive) | a new plot |
| `simulation.py` | orchestration of the stages | new `DualModalitySimulation` options |
| `io.py` | persistence | new cached artifacts |

Recipes for the most common additions:

* **Element** — add `ATOMIC_MASS` and `NEUTRON_XS` entries in `physics/elements.py`, and the
  NIST XCOM export as `physics/data/xray/<El>.txt`.
* **Material** — `register_material("key", material_from_formula(...))` in your own code, or
  add it to `MATERIALS` in `physics/materials.py` if it is generally useful.
* **Phantom preset** — write `make_<name>_phantom(N=64, voxel_cm=None, Nx=None, Ny=None,
  Nz=None)` in `phantoms/presets.py` using `resolve_grid` + `PhantomBuilder`, and add it to
  `PHANTOM_PRESETS`.
* **Artifact** — add the fields to `ArtifactConfig` (default *off*), implement a private
  `_apply_<name>` function in `acquisition/artifacts.py`, and call it at the right place in
  the detection chain inside `inject_sinogram_artifacts` / `inject_volume_artifacts`.
* **Reconstruction algorithm** — add it to `AVAILABLE_ALGORITHMS` and the dispatch in
  `reconstruction/reconstructor.py`; provide a CPU fallback or a clear warning.
* **Metric** — build on the shared helpers in `analysis/quality.py`
  (`ground_truth_positions`, `match_components`, `davies_bouldin`, `pairwise_overlap`)
  rather than re-implementing them.

If something should be importable as `neutron_xray_sim.<name>`, export it from the
subpackage `__init__.py` **and** the top-level `__init__.py` (`__all__`).

## 3. Conventions

* **Units:** linear attenuation in cm⁻¹, lengths in cm (voxel sizes `voxel_cm`), X-ray
  energies in keV, neutron energies in meV.
* **Array order:** volumes are `(Nz, Nx, Ny)`; coordinates `(z, x, y)` from the volume
  centre; sinograms are `(n_angles, n_slices, n_detector)`.
* **No global side effects on import:** no `plt.style.use`, no directories created, no
  warnings. Functions that save files take an output-directory argument.
* **Randomness** goes through an explicit `numpy.random.Generator` or a seed argument.
* **Optional dependencies** (ASTRA, TomoPy, NCrystal, tifffile, PyQt5) are imported inside
  the functions that need them, with a helpful error or a CPU fallback.
* **Docstrings** use the NumPy style already used in the code; state units.
* **Style:** follow the surrounding code. `ruff` checks for real problems (undefined
  names, unused imports, mutable defaults), not formatting.

## 4. Tests

* Every bug fix gets a regression test; every feature gets a test.
* Keep tests fast: `N ≤ 48`, a few dozen angles, `use_astra=False`, `verbose=False`.
  Mark anything slower with `@pytest.mark.slow` (`pytest -m "not slow"` skips them).
* Shared fixtures live in `tests/conftest.py`.
* If your change **alters simulated numbers**, say so in the PR and in `CHANGELOG.md` —
  people compare against published results.

## 5. Notebooks

* `tutorials/` — teaching notebooks. They must run top to bottom on a CPU in a few minutes
  (CI executes all of them). Keep them small, explained, and ending with an exercise.
* `notebooks/` — research / paper notebooks. They may need a GPU, large memory or data.
* `examples/` — non-interactive scripts.
* **Never commit outputs.** `pre-commit` runs `nbstripout` automatically; CI rejects
  notebooks with outputs. Put figures you want to keep in `results/` or a release.
* Don't copy helper functions between notebooks — if two notebooks need it, it belongs in
  the package.

## 6. Pull requests

1. Branch from `main` (`git switch -c my-feature`).
2. Keep PRs focused; one topic per PR is much easier to review.
3. `pytest` and `pre-commit run --all-files` must pass; CI runs them on Python 3.9 and 3.12
   and executes the tutorials.
4. Update docs (`docs/`) and `CHANGELOG.md` for user-visible changes.
