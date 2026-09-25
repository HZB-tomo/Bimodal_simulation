# Installation

## Requirements

- Python ≥ 3.9
- Core dependencies, installed automatically (declared in `pyproject.toml`):

  | Package | Minimum | Role |
  |---|---|---|
  | numpy | 1.24 | arrays, the whole pipeline |
  | scipy | 1.10 | filtering, affine transforms, interpolation |
  | scikit-image | 0.21 | CPU filtered back-projection |
  | scikit-learn | 1.3 | Gaussian mixture model fitting |
  | matplotlib | 3.7 | all plotting |
  | pandas | 1.5 | metric tables, NMC particle phantoms |

- Optional extras:

  | Extra | Installs | Enables |
  |---|---|---|
  | `gpu` | ASTRA Toolbox (prefer conda, see below) | GPU projection and iterative reconstruction |
  | `gridrec` | TomoPy | the `GRIDREC` algorithm |
  | `bragg` | NCrystal | Bragg-edge cross sections (`physics.ncrystal_bragg`) |
  | `tiff` | tifffile | importing `.tif` segmentations |
  | `gui` | PyQt5 | the desktop GUI (`diana-gui`) |
  | `notebooks` | JupyterLab, ipykernel, tifffile | running the tutorials and notebooks |
  | `dev` | pytest, ruff, nbmake, nbstripout, pre-commit | contributing |

## Standard install (CPU)

```bash
git clone https://github.com/OriolSansPlanell/DIANA.git
cd DIANA
pip install -e ".[notebooks]"        # or: pip install -r requirements.txt
```

The `-e` (editable) install means changes you make to the source are picked up
immediately — no `sys.path` manipulation is needed in scripts or notebooks.

This gives you the complete pipeline. Projection and FBP reconstruction run through a
NumPy / scikit-image implementation, which is geometrically identical to
`skimage.transform.radon` / `iradon`, so CPU reconstructions line up voxel-for-voxel
with the phantom. `N = 64` is comfortable on a laptop; `N = 128` is feasible but slow.

## GPU install (ASTRA)

ASTRA is distributed through conda and needs a CUDA-capable GPU:

```bash
conda install -c astra-toolbox -c nvidia astra-toolbox
```

When ASTRA is importable, projection and the iterative reconstruction algorithms
(`SIRT`, `SART`, `CGLS`, `EM`, `OSSART`, `TV_MIN`, `NESTEROV_SIRT`) run on the GPU.
Select the backend per call with `use_astra=True` (the default).

### Graceful fallback

The package never hard-fails on a missing GPU:

- `use_astra=True` without ASTRA → a warning, and projection falls back to NumPy.
- An iterative algorithm without ASTRA → a warning, and reconstruction falls back to FBP.
- The ASTRA projection path assumes **square** 2-D slices; for phantoms with `Nx ≠ Ny`
  the NumPy path is used even when ASTRA is present.

## Checking the install

```python
import neutron_xray_sim as nxs
print(nxs.__version__)            # 2.0.0
print(nxs.AVAILABLE_ALGORITHMS)   # ['FBP', 'GRIDREC', 'SIRT', ...]
```

and, for contributors, `pytest` from the repository root (about a minute on CPU).

## The X-ray data files

X-ray attenuation is derived from **NIST XCOM** mass-attenuation tables stored under
`neutron_xray_sim/physics/data/xray/`, one file per element (e.g. `Fe.txt`). They are
loaded lazily: an element's file is only read when a material containing it is built.
If the file is missing you get an error naming the element and the expected path.

Files currently shipped: C, Co, F, Fe, H, Li, Mn, Ni, O, P. Elements that have neutron
data but still need an XCOM export: Al, Ca, Cl, Cu, In, K, Mg, N, Na, Pb, S, Si, Ti, W,
Zn. (Built-in materials such as `aluminum` or `zinc` use tabulated μ values and do not
need these files; only `material_from_formula` / `make_composite_material` do.)

To add an element, export its table from NIST XCOM (energies in MeV; the parser reads
the "total attenuation with coherent scattering" column, cm²/g) and save it as
`neutron_xray_sim/physics/data/xray/<Symbol>.txt`. See `physics/elements.py`.

## Verifying GPU vs CPU at runtime

Top-level functions log the backend they use, e.g.
`[projector] Projecting 120 angles (ASTRA GPU) …` or `(NumPy CPU)`. Pass
`verbose=False` to `DualModalitySimulation` (or `make_sinogram_pair`) to silence the log.
