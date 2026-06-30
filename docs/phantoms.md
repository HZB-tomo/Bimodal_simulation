# Phantoms

A phantom is the synthetic, ground-truth sample: a labelled 3-D volume plus the
material attached to each label. From the labels the package builds the neutron and
X-ray attenuation volumes that the projector needs.

## `PhantomData`

The core container. You rarely build it by hand — the presets, the builder, and the
volume importer all return one — but it is useful to know its shape.

| Attribute | Meaning |
|---|---|
| `Nz`, `Nx`, `Ny` | grid dimensions |
| `voxel_cm` | voxel side length (cm) |
| `label_vol` | `(Nz, Nx, Ny)` integer label volume; value *i* → `materials[i]` |
| `materials` | ordered list of `Material`; index 0 = air by convention |
| `mu_n_vol`, `mu_n_abs_vol`, `mu_n_coh_vol`, `mu_n_inc_vol` | neutron attenuation volumes (cm⁻¹) |
| `mu_x_vols` | X-ray attenuation, `(13, Nz, Nx, Ny)` over `XRAY_E_KEV` (cm⁻¹) |

Helpful properties and methods: `shape`, `physical_size_cm`, `N` (cube size,
backward-compatible), `material_name(label)`, and `mu_x_at_energy(keV)` (interpolated
X-ray volume at an arbitrary energy).

> **Storage order is (Nz, Nx, Ny)**, with coordinates ordered (z, x, y). The
> attenuation volumes are rebuilt automatically whenever the phantom is constructed.

## Loading a preset

```python
from neutron_xray_sim import make_phantom

phantom = make_phantom("composite", N=64)          # cubic
phantom = make_phantom("battery", Nx=96, Ny=64, Nz=128)  # non-cubic
```

`make_phantom(preset, N=...)` builds a cubic phantom; pass all of `Nx, Ny, Nz` for a
non-cubic one. Voxel size scales automatically so the physical geometry is preserved
unless you pass `voxel_cm=` explicitly. The preset names are the keys of
`PHANTOM_PRESETS`.

## The preset phantoms

| Preset | Diameter | What it is | Why it is interesting |
|---|---|---|---|
| `composite` | ~1 cm | HDPE matrix cylinder with water, iron, titanium inclusions | Solid, multi-material; HDPE is bright to neutrons, dim to X-rays — the canonical hydrogen-contrast demonstration |
| `battery` | ~1.4 cm | Alkaline AAA cell cross-section | KOH electrolyte and HDPE separator are visible only with neutrons (after LaManna et al., NIST NeXT) |
| `bone_implant` | ~1 cm | Cortical bone with a titanium implant | Neutrons resolve the bone–metal interface where X-rays suffer photon starvation (after Törnquist et al. 2021) |
| `industrial` | ~1 cm | Multi-material part with tungsten and iron inserts | Tungsten causes severe X-ray photon starvation but is well resolved by neutrons — an NDE complementarity case |
| `jellyroll_battery` | cylindrical | Wound cylindrical cell | Layered electrode/separator structure |
| `HDPE_composite` | block | HDPE block with a steel rod, an Al cube, an Fe cube nested inside, and air bubbles (some water-filled) | Mixed-density inclusions and voids in a hydrogen-rich matrix |

Each preset's docstring lists the materials it contains and their expected positions
in the bimodal histogram. For example, the `composite` phantom's clusters sit at
roughly:

| Material | μₓ (cm⁻¹) | μₙ (cm⁻¹) | Character |
|---|---|---|---|
| Air | ~0 | ~0 | background |
| HDPE | ~0.17 | ~2.18 | low X-ray, high neutron (H-rich) |
| Al | ~0.28 | ~0.10 | medium X-ray, low neutron |
| Water | ~0.18 | ~1.38 | high neutron, low X-ray |
| Fe | ~4.12 | ~1.16 | high X-ray, medium neutron |
| Ti | ~1.48 | ~0.64 | high X-ray, low neutron |

(μₓ quoted near 80 keV; exact values come from the material database.)

## Building a custom phantom

`PhantomBuilder` lets you compose a phantom from geometric primitives on a labelled
grid, then finalise it into a `PhantomData`. Use it when none of the presets matches
your geometry but you still want synthetic, exactly-known materials.

```python
from neutron_xray_sim.phantom import PhantomBuilder
from neutron_xray_sim import MATERIALS

builder = PhantomBuilder(Nz=64, Nx=64, Ny=64, voxel_cm=0.02)
# add materials, then primitive shapes (cylinders, spheres, boxes) by label,
# and finalise to a PhantomData.
```

The exact primitive methods are listed in [the API reference](api-reference.md#phantom);
the preset factory functions in `phantom.py` are the best worked examples to copy from.

## Importing a real segmented volume

If you already have an experimentally segmented volume (a labelled TIFF stack, `.npy`,
or `.npz`), you can turn it into a `PhantomData` and run the same pipeline on it. See
the dedicated page: [Importing real segmented data](importing-data.md).

## Choosing a grid size

`N = 64` is a good default for exploration and is comfortable on CPU. Increase to
`N = 128` for publication-quality figures (much slower without a GPU). The physical
sample size is fixed by the preset, so increasing `N` reduces the voxel size rather
than enlarging the object.
