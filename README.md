# DIANA

**A simulation toolkit for dual-modality neutron / X-ray tomography and bimodal-histogram analysis.**

Python package import name: `neutron_xray_sim` · Version 2.0.0 ·
[Tutorials](tutorials/README.md) · [Documentation](docs/index.md) ·
[Contributing](CONTRIBUTING.md) · [Changelog](CHANGELOG.md)

---

## What this is

DIANA is an end-to-end simulator for **combined neutron and X-ray
computed tomography**. The two modalities are complementary: X-rays are strongly
attenuated by heavy elements (metals, bone mineral), while thermal neutrons are
attenuated mainly by light, hydrogen-rich materials (water, polymers, organics).
Plotting every voxel as a point in the joint space *(μₓ, μₙ)* — its X-ray linear
attenuation against its neutron linear attenuation — produces a **bimodal
histogram** in which each material forms a distinct cluster.

This package lets you:

- build voxelised 3-D **phantoms** from a database of realistic materials (or import
  your own segmented volumes);
- forward-project them with a **polychromatic X-ray** model and a **thermal-neutron**
  model (GPU-accelerated via ASTRA, with a NumPy CPU fallback);
- inject controllable **acquisition artifacts** (Poisson noise, beam hardening,
  scatter, detector blur, ring artifacts, inter-modality misalignment, …);
- **reconstruct** with a choice of analytic and iterative CT algorithms;
- compute the **2-D bimodal histogram**, fit a **Gaussian mixture model**, segment the
  volume back into phases, and **quantify** how acquisition conditions degrade the
  separability of material clusters.

The intended use is methodological: understanding *how* each artifact and each
acquisition choice deforms the bimodal histogram, and measuring the effect with
reproducible, ground-truth-anchored metrics.

The package is developed in the tomography group at Helmholtz-Zentrum Berlin (HZB).

---

## The pipeline at a glance

```
 PhantomData ──► forward projection ──► sinogram artifacts ──► reconstruction
 (ground truth)   X-ray (polychromatic)   noise / scatter /      FBP / SIRT /
                  + neutron (thermal)      rings / PSF / BHC      CGLS / TV / …
                                                                       │
                                                                       ▼
   quality metrics ◄── GMM fit / segmentation ◄── bimodal histogram ◄── volume
   (centroid error,      H(μ_x, μ_n)               H(μ_x, μ_n)           artifacts
    Davies–Bouldin,                                                      (misalign,
    cluster overlap)                                                      salt&pepper)
```

Every stage is a plain function with explicit inputs and outputs, and the whole
chain is wrapped by the `DualModalitySimulation` orchestrator.

---

## Installation

Requires Python ≥ 3.9.

```bash
git clone https://github.com/OriolSansPlanell/DIANA.git
cd DIANA
pip install -e ".[notebooks]"          # add ",dev" if you want to contribute
```

The core dependencies (NumPy, SciPy, scikit-image, scikit-learn, matplotlib, pandas) are
all pip-installable and give you the **full pipeline on CPU**.

For GPU-accelerated projection and iterative reconstruction, install the optional
**ASTRA Toolbox** (needs a CUDA GPU and conda):

```bash
conda install -c astra-toolbox -c nvidia astra-toolbox
```

If ASTRA is not present the package automatically falls back to a NumPy/scikit-image
implementation, and iterative algorithms degrade gracefully to FBP. See
[`docs/installation.md`](docs/installation.md) for details, including the optional extras
and the NIST X-ray data files the material database depends on.

---

## Quick start

```python
from neutron_xray_sim import DualModalitySimulation, ArtifactConfig
import matplotlib.pyplot as plt

# Build the simulation around a preset phantom
sim = DualModalitySimulation(preset="composite", N=64, n_angles=120)

# A clean reference run and a fully realistic run
r_clean = sim.run(ArtifactConfig.clean(),     tag="clean")
r_real  = sim.run(ArtifactConfig.realistic(), tag="realistic")

# Compare their bimodal histograms side by side
fig = sim.comparison_grid([r_clean, r_real])
plt.show()
```

`r_clean` and `r_real` are `SimulationResult` objects that carry the reconstructed
volumes, the histogram, an optional GMM fit, and quantitative artifact signatures.
Call `print(r_real.summary())` for a one-screen overview.

## Learn the software

Start with the **[tutorials](tutorials/README.md)** — seven notebooks that run on a laptop
CPU in a few minutes each:

1. Getting started · 2. Materials and phantoms · 3. Projection and artifacts ·
4. Reconstruction · 5. Histogram analysis · 6. Artifact studies and caching ·
7. Neutron spectra

Then: compact scripts in [`examples/`](docs/examples.md), full research notebooks in
[`notebooks/`](notebooks/) (these usually need a GPU), and the reference in [`docs/`](docs/index.md).

---

## Phantom presets

| Preset | Sample | Highlights |
|---|---|---|
| `composite` | HDPE matrix + water / Fe / Ti inclusions (~1 cm) | Hydrogen contrast: HDPE is bright to neutrons, dim to X-rays |
| `battery` | Alkaline AAA cell cross-section (~1.4 cm) | Electrolyte and separator visible only with neutrons |
| `bone_implant` | Cortical bone + Ti implant (~1 cm) | Neutrons resolve the bone–metal interface where X-rays starve |
| `industrial` | Multi-material part with W and Fe inserts | Beam hardening and neutron complementarity in NDE |
| `jellyroll_battery` | Wound cylindrical cell | Layered electrode structure |
| `HDPE_composite` | HDPE block with steel rod, Al/Fe cubes, voids | Mixed-density inclusions and air bubbles |
| `li_ion_spiral` | Spiral-wound Li-ion cell | Archimedean jellyroll with realistic layer thicknesses |

You can also build phantoms primitive-by-primitive with `PhantomBuilder`, or import a
real segmented volume — see [`docs/phantoms.md`](docs/phantoms.md) and
[`docs/importing-data.md`](docs/importing-data.md).

---

## Reconstruction algorithms

`FBP`, `GRIDREC`, `SIRT`, `SART`, `CGLS`, `EM`, `OSSART`, `TV_MIN`, `NESTEROV_SIRT`.

FBP runs everywhere (ASTRA GPU or scikit-image CPU); the iterative algorithms require
ASTRA. See [`docs/reconstruction.md`](docs/reconstruction.md).

---

## Artifacts

All artifacts are switched on and parameterised through a single `ArtifactConfig`
dataclass, so you can run them one at a time or in combination:

`photon_noise`, beam hardening (`apply_bh_correction`), `neutron_scatter`,
`xray_scatter`, `detector_psf`, `ring_artifacts`, `misalignment`, `salt_pepper`.

Factory presets: `ArtifactConfig.clean()`, `.noise_only()`, `.beam_hardening_only()`,
`.scatter_only()`, `.misalignment_only()`, `.realistic()`. Full field reference in
[`docs/artifacts.md`](docs/artifacts.md).

---

## Repository layout

```
DIANA/
├── neutron_xray_sim/          # the Python package
│   ├── physics/               # element data, materials, X-ray & neutron spectra, Bragg edges
│   ├── phantoms/              # PhantomData, PhantomBuilder, presets, segmented-volume import
│   ├── acquisition/           # projectors, artifacts, noise, cone-beam/laminography geometry
│   ├── reconstruction/        # FBP / iterative reconstruction, Fourier fusion
│   ├── analysis/              # histograms, GMM, signatures, quality metrics
│   ├── plotting/              # histogram figures, publication figure set
│   ├── simulation.py          # DualModalitySimulation orchestrator, artifact survey
│   ├── io.py                  # SimCache: on-disk cache of every pipeline stage
│   └── gui/                   # desktop GUI (diana-gui)
├── tutorials/                 # step-by-step executable notebooks — start here
├── examples/                  # non-interactive scripts reproducing the main figures
├── notebooks/                 # research / paper notebooks (GPU, large N)
├── tests/                     # pytest suite (run: pytest)
├── docs/                      # documentation
├── results/                   # selected published figures
└── pyproject.toml             # packaging, dependencies, tool config
```

Old flat import paths from 1.x (e.g. `neutron_xray_sim.histogram`) still work.
The GUI starts with `diana-gui` (after `pip install -e ".[gui]"`) or `python launch_gui.py`.

---

## Contributing

Contributions are welcome — new materials, phantoms, artifact models, algorithms, metrics or
tutorials. See **[CONTRIBUTING.md](CONTRIBUTING.md)** for the development setup, where each
kind of code belongs, and the test / notebook conventions. In short:

```bash
pip install -e ".[dev,notebooks]" && pre-commit install && pytest
```

---

## Documentation

Full documentation lives in [`docs/`](docs/index.md) and can be served as GitHub Pages
or pasted into the GitHub wiki:

- [Documentation home](docs/index.md)
- [Installation](docs/installation.md)
- [Concepts: physics and the bimodal histogram](docs/concepts.md)
- [The simulation pipeline](docs/pipeline.md)
- [Phantoms](docs/phantoms.md)
- [Materials](docs/materials.md)
- [Artifacts](docs/artifacts.md)
- [Reconstruction](docs/reconstruction.md)
- [Histogram analysis](docs/histogram-analysis.md)
- [Neutron spectra and beam modes](docs/neutron-spectra.md)
- [Importing real segmented data](docs/importing-data.md)
- [Tutorials](tutorials/README.md)
- [Example scripts](docs/examples.md)
- [API reference](docs/api-reference.md)

---

## Citing and license

Author: Built and maintained by Dr. Oriol Sans-Planell and Dr. Shahabeddin Dayani. Helmholtz-Zentrum Berlin.
Under MIT licence. 
If you use this work or want to cite us, please use the following DOI: https://doi.org/10.21203/rs.3.rs-9724333/v1 
