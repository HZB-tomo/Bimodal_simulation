# Roadmap

Suggested directions for DIANA, ordered by what unblocks the most work per unit
of effort. Nothing here is committed to; it is a menu for planning discussions.

Each item states **why** it matters for this package specifically, not just that
it is good practice.

---

## Near term — remove the sharp edges

### 1. Replace the three suspect X-ray tables

`MATERIALS.audit()` flags `hdpe`, `bone` and `lead` as inconsistent with NIST.
HDPE is the matrix of three built-in phantoms, so its ~9 % bias near 80 keV
propagates into every composite result.

The fix is a data task, not a code task: export Al, Ti, Cu, Zn, W, Pb, Si, Ca
and N from NIST XCOM into `lib/xray_data/`, then convert those specs from the
tabulated path to the derived one. The infrastructure is already in place —
`element_data_status()` tells you what is missing, and the derived path is
already used by twelve materials. **This is the single highest-value item on the
list**, because everything downstream inherits it.

### 2. Pin the physics with reference tests

The current tests check behaviour (shapes, invariants, that an artifact does
something). They do not check that the numbers are *right*. Add:

- known attenuation values at 80 keV for water, HDPE and iron, against NIST;
- a Beer-Lambert round trip: a uniform cylinder of known μ, projected and
  reconstructed, should return μ to within the discretisation error;
- cluster positions in the bimodal histogram of a clean run should sit on the
  ground-truth material coordinates to within a stated tolerance.

Those turn "the pipeline runs" into "the pipeline is correct", which is what a
collaboration actually needs before people trust each other's runs.

### 3. A geometry/convention smoke test

The `(Nz, Nx, Ny)` storage order and `(z, x, y)` coordinate order are the most
common source of silent errors, and the axis-order confusion is already visible
in the codebase (a `PhantomBuilder` default height that passes the full length
where a half-length is meant; a misalignment docstring that names `(Δy, Δx, Δz)`
while the code applies `(z, x, y)`). A test that places an asymmetric marker and
asserts where it lands in each projection would catch a whole class of these.

---

## Phantom creation tools

The `PhantomBuilder` primitives (sphere, box, cylinder, rod, layer) cover
idealised samples well. Real samples are rougher than that, and the gap shows up
as unrealistically clean histogram clusters.

### 4. Microstructure primitives

- **Porosity fields** — `add_porosity(material, fraction, correlation_length)`
  using thresholded Gaussian random fields, rather than hand-placed spheres.
  Partial-volume effects at pore boundaries are a major driver of cluster
  spread, and are currently unrepresented.
- **Rough interfaces** — perturb a primitive's boundary with band-limited noise.
  A perfectly smooth HDPE/water interface makes segmentation look easier than
  it is.
- **Packed-particle generators** — `nmc_phantom.py` already does sphere packing
  for NMC. Generalising it (arbitrary size distributions, non-spherical shapes,
  a target packing fraction) would serve powders, ceramics and sediments alike.
- **Cracks and delamination** — thin, high-aspect-ratio features are exactly
  where the two modalities disagree most, and are the interesting case for
  bimodal analysis.

### 5. Phantoms from a specification file

Presets currently require writing Python. A YAML phantom description would let
a collaborator build a sample without touching the package — and would make the
phantom archivable next to the results:

```yaml
name: sintered_pellet
grid: {N: 256, extent_cm: 0.8}
primitives:
  - {type: cylinder, material: aluminum, radius_cm: 0.3, axis: z}
  - {type: sphere,   material: water,    center_cm: [0, 0.1, 0], radius_cm: 0.05}
```

This mirrors what `MaterialSpec` now does for materials, and reuses the same
validate-then-build discipline.

### 6. Import improvements

`volume_importer` handles segmented TIFF/NPZ volumes. Worth adding: DICOM,
HDF5/NeXus (the standard at neutron facilities), and a *downsampling* path so a
2048³ tomogram can be explored at 256³ before committing to a full run.

### 7. A CLI

```bash
diana run --phantom composite --artifacts realistic --n-angles 360 -o results/
diana survey --phantom battery --algorithms FBP,SIRT,SART
diana materials table --tag battery
```

Batch jobs on a cluster are far easier to launch from a command line than from a
notebook, and a CLI forces the configuration to be a serialisable object — which
in turn makes runs reproducible.

---

## Analytics and quantification

This is where the package's scientific value concentrates, and where it is
currently thinnest relative to its ambitions.

### 8. Provenance in every result

A `SimulationResult` should carry enough metadata to regenerate itself: package
version, git commit, phantom spec, material specs, artifact config, algorithm
and parameters, RNG seeds, and whether ASTRA was used. The cache already stores
material specs as of schema 1.1 — extending that to the whole run makes results
self-describing, which matters as soon as more than one person is producing
them.

### 9. Uncertainty on every metric

Metrics like Davies-Bouldin and centroid error are currently single numbers from
a single noise realisation. With Poisson noise in the loop they are random
variables. Running *n* seeds and reporting a mean with a confidence interval
would change conclusions — some of the differences between artifact scenarios
are likely within noise. `diana_plots.plot_grouped_bars_with_ci` already expects
intervals; the pipeline does not produce them yet.

### 10. Detectability rather than separability

`noise.py` has `d_prime`, `joint_d_prime` and `rose_dose_threshold` but they are
not wired into the survey. The question a beamline scientist actually asks is
"can I distinguish phase A from phase B at this dose?", which is a detectability
question. Reporting d′ per material pair alongside DB and CE would connect the
histogram analysis to an experimental decision.

### 11. Dose-normalised comparison

Comparing modalities or acquisition settings at equal *number of projections* is
not the same as comparing them at equal *dose*. A dose-normalised sweep — fix
total fluence, vary the number of projections against the counts per projection
— is a natural and cheap extension of the existing sweep machinery, and answers
a question the current sweeps cannot.

### 12. Segmentation quality against ground truth

The phantom's label volume is exact ground truth, and it is currently used only
to place markers on plots. Per-material Dice/IoU and a confusion matrix between
the GMM segmentation and the true labels would turn "the clusters look separated"
into a number. This is the most direct measure of what the whole pipeline is
for.

### 13. Report generation

One call producing an HTML or PDF report — phantom slices, histograms, metric
tables with uncertainty, and the provenance block — would make results shareable
between collaborators without each person rebuilding the same figures.

---

## Engineering

### 14. Cache the projection, not the reconstruction

Forward projection dominates runtime and is identical across artifact
configurations; `DualModalitySimulation` already exploits that in memory. A
content-addressed on-disk cache keyed by (phantom hash, geometry, spectrum)
would extend the benefit across processes and users on a shared filesystem.

### 15. Chunked processing for large volumes

Several arrays are materialised at full size. Making the projector and histogram
operate slice-wise or in chunks would lift the practical ceiling from a few
hundred voxels per side to over a thousand — which is where real tomograms live.

### 16. Parallelism

Slice loops in `reconstruct` and angle loops in the NumPy projector are
embarrassingly parallel. `joblib` or a process pool would give a near-linear
speed-up on the CPU path, which is what most contributors will be using.

### 17. Split `diana_plots.py` and `histogram.py`

Both are over 1,600 lines and mix analysis with rendering. Splitting the
plotting into `plots/` submodules by figure family, and separating the histogram
*computation* from its *drawing*, would let several people work on figures
without conflicting — and would remove the duplicate `plot_bimodal_histogram`
that made the two modules unsafe to star-import together.

### 18. Published documentation

`docs/` is already substantial. Wiring it to MkDocs or Sphinx and publishing on
GitHub Pages, with the API reference generated from the docstrings, would keep
it honest — generated docs rot visibly, hand-written ones rot silently.

---

## Physics extensions

### 19. Finish the Bragg-edge path

`ncrystal_bragg.py` computes wavelength-resolved neutron attenuation but is not
integrated into `DualModalitySimulation`. Energy-resolved neutron imaging is a
genuinely distinguishing capability — it would let the package address phase and
strain mapping, not only attenuation contrast.

### 20. Validate the cone-beam and laminography paths

`cone3d_geometry.py` and `laminography_projector.py` exist but are neither
tested nor reachable from the orchestrator. Either wire them in with tests, or
mark them explicitly as experimental so nobody builds on an unvalidated path.

### 21. A physically grounded beam-hardening correction

`_apply_bh_correction` uses hard-coded polynomial coefficients described as
"reasonable defaults for 120 kVp with 2 mm Al". Fitting the correction to the
actual spectrum in use — which the package already computes — would make it
correct for the settings someone is simulating rather than for one setting.

### 22. Detector response beyond a Gaussian PSF

Real scintillators have non-Gaussian tails, and neutron detectors have
significant afterglow and gamma sensitivity. Since detector blur strongly
affects partial-volume mixing, and partial-volume mixing is what fills in the
space between histogram clusters, a more faithful model would change
conclusions about cluster separability.
