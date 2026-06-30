# Installation

## Requirements

- Python ≥ 3.9
- The packages listed in `requirements.txt`:

  | Package | Minimum | Role |
  |---|---|---|
  | numpy | 1.24 | arrays, the whole pipeline |
  | scipy | 1.10 | filtering, affine transforms, interpolation (artifacts) |
  | scikit-image | 0.21 | CPU filtered back-projection fallback |
  | matplotlib | 3.7 | all plotting |
  | scikit-learn | 1.3 | Gaussian mixture model fitting (recommended) |

- **Optional, for GPU acceleration:** the [ASTRA Toolbox](https://www.astra-toolbox.com/)
  (CUDA GPU required).
- **Optional, for the `GRIDREC` algorithm:** [TomoPy](https://tomopy.readthedocs.io/).

## Standard install (CPU)

```bash
git clone https://github.com/HZB-tomo/Bimodal_simulation.git
cd Bimodal_simulation
pip install -r requirements.txt
```

This gives you the complete pipeline. Projection and FBP reconstruction run through a
NumPy / scikit-image implementation; everything in
[Concepts](concepts.md) and [Histogram analysis](histogram-analysis.md) works without
a GPU. The CPU path is slower and is best used at modest grid sizes (`N = 64` is
comfortable; `N = 128` is feasible but slow).

## GPU install (ASTRA)

ASTRA is distributed through conda and needs a CUDA-capable GPU:

```bash
conda install -c astra-toolbox -c nvidia astra-toolbox
```

When ASTRA is importable, projection (`project_xray`, `project_neutron`) and the
iterative reconstruction algorithms (`SIRT`, `SART`, `CGLS`, `EM`, `OSSART`,
`TV_MIN`, `NESTEROV_SIRT`) run on the GPU. You select the backend per call with
`use_astra=True` (the default).

### Graceful fallback

The package never hard-fails on a missing GPU:

- If you request `use_astra=True` but ASTRA is not installed, a warning is emitted and
  projection falls back to NumPy.
- If you request an iterative algorithm without ASTRA, reconstruction falls back to
  scikit-image FBP (with a warning).
- The ASTRA projection path currently assumes **square** 2-D slices. For non-cubic
  phantoms the projector automatically uses the NumPy path even when ASTRA is present.

## Importing the package

The importable package is `neutron_xray_sim`. If you installed by cloning (rather than
packaging), make sure the repository root is on `sys.path`. The bundled example
scripts do this explicitly:

```python
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from neutron_xray_sim import DualModalitySimulation, ArtifactConfig
```

A quick sanity check:

```python
import neutron_xray_sim as nxs
print(nxs.__version__)          # 1.1.0
print(nxs.AVAILABLE_ALGORITHMS) # ['FBP', 'GRIDREC', 'SIRT', ...]
```

## The X-ray data files

The material database derives X-ray attenuation from **NIST XCOM** mass-attenuation
tables stored as plain text under `neutron_xray_sim/lib/xray_data/`, one file per
element (e.g. `Fe.txt`, `Ca.txt`). These ship with the repository.

If a file for a needed element is missing, `materials.py` emits a clear warning at
import time and that element becomes unavailable until you add its file. To add an
element, export its mass-attenuation table from the NIST XCOM database and save it as
`lib/xray_data/<Symbol>.txt` in the same column format as the existing files (the
parser reads the total-with-coherent column; units are cm²/g). The elements currently
supported include H, Li, C, N, O, F, Mg, Al, Si, P, S, Cl, K, Ca, Fe, Co, Ni, Mn, In.

## Verifying GPU vs CPU at runtime

Most top-level functions print which backend they used, e.g.
`[projector] Projecting 120 angles (ASTRA GPU) …` or `(NumPy CPU)`. Pass
`verbose=True` (the default) to `DualModalitySimulation` to see the stage-by-stage log.
