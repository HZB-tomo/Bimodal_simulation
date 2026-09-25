# Contributing to DIANA

DIANA is a shared research instrument. The goal of this guide is that a new
collaborator can add a material, a phantom, an artifact, or a metric **without
reading the whole codebase and without breaking anyone else's results**.

---

## Setting up

```bash
git clone https://github.com/OriolSansPlanell/DIANA.git
cd DIANA
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest                      # ~30 s on CPU, no GPU needed
```

ASTRA (GPU projection and the iterative reconstructors) is optional. Everything
falls back to NumPy/scikit-image on CPU, and the test suite is written to pass
either way — tests that genuinely need a GPU are marked `@pytest.mark.gpu` and
skip themselves.

```bash
conda install -c astra-toolbox -c nvidia astra-toolbox    # optional
```

Before opening a pull request:

```bash
ruff check neutron_xray_sim tests
pytest
```

---

## Where things live

| I want to add… | Edit | Nothing else needs to change because… |
|---|---|---|
| a material everyone will use | `neutron_xray_sim/materials/database.py` | the registry builds it from the spec |
| a material for one project | a JSON/YAML file + `MATERIALS.load_file(...)` | it never enters the package |
| an element | `materials/elements.py` + `lib/xray_data/<El>.txt` | tables are read lazily by symbol |
| a phantom preset | any module, with `@register_phantom(...)` | `make_phantom` dispatches through the registry |
| an artifact | `artifacts.py`: a field on `ArtifactConfig` + an `_apply_*` function | the injector walks the config |
| a reconstruction algorithm | `reconstructor.py`: `AVAILABLE_ALGORITHMS` + a branch in `reconstruct` | callers pass a string |
| a metric | `metrics_table.py` or `histogram.py` | metrics are computed per run |
| a figure | `diana_plots.py` | plots take result objects, not globals |

Each of those is a **registry with one obvious insertion point**. If you find
yourself editing five files to add one thing, that is a design bug — please open
an issue rather than working around it.

---

## Adding a material

Materials are declarative. A `MaterialSpec` says where each channel's numbers
come from; `build_material` is the only thing that constructs a `Material`, so
everything is validated the same way.

**Derived (preferred)** — reproducible from cited inputs:

```python
MaterialSpec(
    key="quartz", name="Quartz", symbol="SiO2",
    density_gcc=2.65, formula="SiO2",
    color="#cccc99", tags=("mineral",),
    reference="Deer, Howie & Zussman (2013), 3rd ed.",
)
```

**Tabulated** — when the package ships no NIST file for one of the elements:

```python
MaterialSpec(
    key="tantalum", name="Tantalum", symbol="Ta", density_gcc=16.65,
    xray_mu_cm=(...13 values at XRAY_E_KEV...),
    neutron_mu_cm=(mu_abs, mu_coh, mu_inc),
    k_edge_keV=67.4,      # declare a real absorption edge in range
    reference="NIST XCOM, retrieved 2026-03; Sears (1992).",
)
```

The two channels are independent — a spec may take X-ray from a table and
neutron from a formula. That is how the built-in metals work.

Rules:

- **Always fill in `reference`.** A number without provenance cannot be
  defended in a paper.
- **Never edit an existing material's numbers in place** without saying so in
  the PR description. Someone's published figure depends on them.
- Run `MATERIALS.audit()` after your change. It flags X-ray tables that rise
  with energy away from a declared edge — nearly always a transcription error.
- Add a test if the material has a property worth pinning (a known contrast
  ratio, an expected cluster position).

Project-specific materials belong in a file next to your analysis, not in the
package:

```python
MATERIALS.load_file("my_beamtime/materials.json")
```

### Adding an element

1. Add entries to `ATOMIC_MASS` and `NEUTRON_XS` in `materials/elements.py`,
   with the Sears (1992) values and a comment if they are unusual.
2. Export the element from [NIST XCOM](https://physics.nist.gov/PhysRefData/Xcom/html/xcom1.html),
   requesting *total attenuation with coherent scattering*, and save the plain
   text as `neutron_xray_sim/lib/xray_data/<Symbol>.txt`. A two-column
   `E[MeV]  mu/rho[cm2/g]` file works too.
3. Check it loaded: `element_data_status()["Si"]` should now be `True`.

---

## Adding a phantom preset

```python
from neutron_xray_sim.phantom import PhantomBuilder, register_phantom, resolve_grid

@register_phantom("sintered_pellet", description="Two-phase sintered pellet, 8 mm")
def make_sintered_pellet_phantom(N=64, voxel_cm=None, Nx=None, Ny=None, Nz=None):
    Nx, Ny, Nz, voxel_cm = resolve_grid(N, Nx, Ny, Nz, voxel_cm, extent_cm=0.8)
    b = PhantomBuilder(Nx=Nx, Ny=Ny, Nz=Nz, voxel_cm=voxel_cm)
    b.add_cylinder("aluminum", radius_cm=0.3, axis="z")
    return b.build("sintered_pellet")
```

The five grid keywords are mandatory — `register_phantom` refuses a factory
without them at import time, so a preset that `make_phantom` could not call
fails during development rather than at the bottom of someone's sweep. Any
extra keywords need defaults, and pass through `make_phantom(..., my_option=…)`.

`tests/test_phantom.py` parametrises over the registry, so your preset is
automatically checked for cubic and non-cubic construction. Add a test of your
own for anything geometry-specific.

---

## Conventions that are not negotiable

These exist because getting them wrong produces plausible-looking wrong physics.

- **Axis order is `(Nz, Nx, Ny)`** for volumes; coordinates are `(z, x, y)`.
  Sinograms are `(n_angles, n_slices, n_detector)`.
- **Units are CGS-ish and explicit in the name**: `_cm`, `_gcc`, `_keV`,
  `_meV`, `_voxels`, `_pixels`. Attenuation is always cm⁻¹.
- **Neutron attenuation is three components** (`abs`, `coh`, `inc`) that sum to
  `mu_n`. To override it, set the three — there is no backing field for the total.
- **Randomness goes through an injected `np.random.Generator`**, never the
  global `np.random`. Every artifact must be reproducible from a seed.
- **No physics in plotting code**, and no plotting in physics code.

---

## Tests

Tests are cheap here and they have already paid for themselves: the alias table
bug and the ring-removal bug in `v1.2.0` were both found by a test, not by
reading.

- Put behaviour tests in `tests/`, use grids of 8–16 voxels, and keep the suite
  under a minute.
- Name the regression: if you fix a bug, the test's docstring should say what
  used to happen (`"""Regression: … raised TypeError on this path."""`).
- Mark GPU-only tests with `@pytest.mark.gpu`; they skip cleanly without ASTRA.
- Mutating the global `MATERIALS` in a test leaks into every later test — use
  the `fresh_registry` fixture.

---

## Notebooks

Notebooks in `notebooks/` are documentation and worked examples, not library
code. Anything reused belongs in the package.

Strip outputs before committing, or the repository grows by megabytes per save
and every diff is unreadable:

```bash
pip install nbstripout && nbstripout --install
```

---

## Pull requests

Keep them focused — one material set, one artifact, one bug. In the description
say what changed **numerically**, if anything: a PR that shifts a material's
attenuation or an artifact's amplitude changes everyone's results, and that
needs to be visible in the log rather than discovered later.

Note that `v1.2.0` moved `materials.py` to a `materials/` package. All previous
imports still resolve (`from neutron_xray_sim.materials import MATERIALS`), so
existing scripts keep working.
