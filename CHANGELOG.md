# Changelog

All notable changes are documented here. Entries marked **[numerical]** change
results a previous version would have produced — check them before re-using old
figures alongside new ones.

## 1.2.0

Repository restructured for multi-person development, material creation
reworked around declarative specs, and a set of correctness bugs fixed.

### Fixed — correctness

- **Beam-hardening correction was silently discarded.** `inject_sinogram_artifacts`
  tracked `sino_lam` and `sino_trans` as two independent arrays. The scatter
  models rebuilt their output from the *raw* transmission, so any step applied
  before them — in practice the polynomial BHC — was thrown away. The chain now
  carries one running state and derives transmission where it is needed.
  **[numerical]** for any run combining `apply_bh_correction` with either
  scatter model.
- **Ring removal cancelled the ring artifact.** `DualModalitySimulation.run`
  passed `remove_rings=True` unconditionally, and the Vo filter subtracts
  exactly the constant column offsets that `ArtifactConfig(ring_artifacts=True)`
  injects. The "ring artifacts" survey run was therefore nearly identical to the
  clean reference. Ring removal is now disabled automatically for runs that
  inject rings, and is configurable per simulation and per run.
  **[numerical]** for every ring-artifact run.
- **Half the reconstruction aliases were unreachable.** `_resolve_algorithm`
  normalised `-` to `_` before looking up an alias table whose keys still
  contained hyphens, so `'ram-lak'`, `'ml-em'` and `'os-sart'` all raised
  "Unknown algorithm".
- **`make_phantom` could not build two of its own presets.** The battery
  factories did not accept the `Nx`/`Ny`/`Nz` keywords the dispatcher passes, so
  `make_phantom("jellyroll_battery", Nx=…, Ny=…, Nz=…)` raised `TypeError` and
  the cubic path passed `N=None` into a division. Every preset now shares one
  signature, enforced at registration time.
- **Monochromatic X-ray projection assumed a cubic phantom.** It sized its
  output from `phantom.N` (which is `Nz`) on all three axes, producing a shape
  mismatch for any non-cubic phantom.
- **`Material` equality crashed on NumPy fields.** The generated `__eq__`
  compared the attenuation table, so `material in list` raised "truth value of
  an array is ambiguous" whenever two materials agreed on their leading fields.
  Materials now compare by identity, which is also the correct semantics.
- **`ArtifactConfig.clean().summary()` reported `"BH_artifact(no_correction)"`,**
  never `"clean (no artifacts)"` — the fallback branch was unreachable. Since
  `run()` derives the default run tag from `summary()`, clean runs were
  mislabelled. Added `ArtifactConfig.is_clean()`.
- **`run()` had a mutable default argument** (`cfg=ArtifactConfig.clean()`),
  evaluated once at import and shared by every call.
- **`DualModalitySimulation(phantom=…)` left `self.N` at the constructor
  default,** so slice indices derived from it could fall outside a supplied
  phantom.
- **`SimCache.load_phantom()` was documented but did not exist.** It now does,
  and restores materials from the metadata rather than the current database.
- `ring_artifacts` with `n_bad_columns` greater than the detector width raised
  inside `rng.choice`; it is now clamped.
- `_remove_rings_vo` accepted an `snr` parameter it never used; replaced with
  the median-filter width it actually applies.

### Fixed — accuracy

- **X-ray attenuation is now interpolated log-log** when resampling NIST tables
  and when evaluating `mu_x_at` / `mu_x_array`, which is what NIST recommends
  and what the documentation already claimed. The previous linear interpolation
  differed by up to ~9 % for medium-Z elements at grid points that fall between
  tabulated NIST energies. Absorption edges (duplicate energies) are handled
  explicitly. **[numerical]** for all formula-derived materials.

### Added — data quality

- `MATERIALS.audit()` re-validates every spec and reports implausible data.
  It currently flags three shipped X-ray tables as inconsistent with NIST:
  `hdpe` (rises across 60→80 keV), `bone` (rises across 90→100 keV), and `lead`
  (K-edge jump at 70–80 keV rather than 88.0 keV). **The numbers are unchanged**
  so existing results stay reproducible; see `docs/materials.md` for the fix
  procedure. `hdpe` matters most — it is the matrix of three built-in phantoms.
- `element_data_status()` and `available_elements()` report which NIST element
  tables are installed. A formula needing a missing element now raises
  `MissingElementDataError` naming the file, the download URL, and the available
  elements, instead of a bare `KeyError('Si')`.
- Import no longer emits ten warnings about missing element files; tables are
  read lazily on first use.

### Changed — material system

`materials.py` became the `materials/` package: `elements` (per-element data and
the NIST reader), `core` (`Material`, `MaterialSpec`, the builder and validator),
`registry` (`MaterialRegistry`), `database` (the built-in catalogue as data).
All previous imports still resolve.

- Every material — built-in, contributed, or ad hoc — is now a `MaterialSpec`
  built by `build_material`, so all of them are validated identically. The two
  parallel construction paths that shared no validation are gone.
- A spec declares its X-ray and neutron sources **independently**, so a material
  can take X-ray from a measured table and neutron from its formula. That is how
  the built-in metals work, and it removes the duplicated hand-entered neutron
  numbers.
- `MATERIALS` is a `MaterialRegistry` — still a `Mapping`, so every existing
  usage works — with `register()`, `load_file()`, `save_file()`, `search()`,
  `names(tag=…)`, `table()`, `spec()` and `audit()`.
- Materials are built lazily and cached.
- Specs carry `reference` and `tags`. Every built-in material now cites its
  source.
- Contributors can add materials from a JSON/YAML file without touching the
  package.
- Elements added: Ti, Cu, W, Pb (neutron cross sections and atomic masses).

### Changed — phantoms

- Presets are registered with `@register_phantom(name, description=…)`, which
  enforces the shared signature at import time. `PHANTOM_PRESETS` is populated by
  the decorator; `PHANTOM_DESCRIPTIONS` gives menus and docs something to show.
- `resolve_grid()` replaces the "either N or all of Nx/Ny/Nz" block that was
  copy-pasted into five factories and had drifted between them.
- `make_phantom` forwards extra keyword arguments to the preset, so
  `make_phantom("jellyroll_battery", n_jellyroll_turns=3)` works.
- `make_li_ion_battery_phantom` (Archimedean-spiral jellyroll) was unreachable
  dead code; it is now the `spiral_battery` preset. `jellyroll_battery` still
  resolves to the concentric-ring factory it always did.
- **Attenuation volumes are derived lazily.** `PhantomData` built all 13 X-ray
  energy volumes eagerly in `__post_init__` — 13× the label volume in float32,
  about 7 GB for a 512³ phantom, and the projector never used them (it rebuilds
  per spectrum energy anyway). `mu_x_vols` is now a cached property, with
  `mu_x_at_index(i)` and `mu_x_at_energy(E)` for one energy at a time.
- Label lookup is by identity through a dict, not a linear scan with `==`
  (see the `Material` equality fix), and rejects more than 255 materials
  explicitly rather than overflowing `uint8`.

### Changed — other

- `SimCache.save_phantom` no longer writes the derived attenuation volumes by
  default (16 extra copies of the volume on disk). They are reproducible from
  the label volume plus the materials, both of which are saved; pass
  `save_derived_volumes=True` for the old behaviour. Cache schema is now 1.1 —
  `phantom/meta.json` carries the full material definitions, so a cached phantom
  is self-describing.
- `inject_sinogram_artifacts` applies Gaussian blur to the whole sinogram stack
  in one call instead of looping `n_angles × n_slices` times in Python.
- The scatter models were two copies of the same function; they now share
  `_add_scatter_halo`.
- The two projectors shared a copy-pasted ASTRA-availability block; it is now
  `_astra_usable()`, which also explains *why* it fell back to CPU.
- `neutron_xray_sim/__init__.py` no longer re-exports `plot_bimodal_histogram`
  from both `histogram` and `diana_plots` — the second silently shadowed the
  first. Import the module you want. `__all__` now matches what is exported.
- `FBP` was listed as requiring ASTRA despite having a validated scikit-image
  CPU path.

### Added — infrastructure

- `tests/` with 99 CPU-only tests covering materials, phantoms, artifacts,
  reconstruction, caching and the orchestrator. Regression tests name the bug
  they pin.
- GitHub Actions CI: lint and tests on Python 3.9 and 3.12, plus a job that
  installs *without* optional extras and imports the package, to catch a stray
  top-level `import astra`.
- `CONTRIBUTING.md`, PR and issue templates, `CODEOWNERS`.
- Ruff configuration; the package is lint-clean.

### Removed

- `setup.py` — metadata lives in `pyproject.toml` alone. The two had drifted
  (different author, different URL).
- Duplicated copies of the four example scripts inside the package directory
  (identical to those in `notebooks/`).
- A second `requirements.txt` inside the package directory.
- 29 committed `__pycache__/*.pyc` files across three Python versions.
- Dead code: `_astra_project_2d`, unused imports.
- Superseded notebook copies; `notebooks/` is now the single home for them.

### Fixed — packaging

- `build-backend` was `setuptools.backends.legacy:build`, which does not exist —
  `pip install .` failed. It is now `setuptools.build_meta`.
- `pandas` is a hard import in four modules but was not declared; it is now a
  dependency. `tifffile`, `ncrystal` and `PyQt5` are declared as extras.
- `lib/xray_data/*.txt` is declared as package data, so it survives packaging.
