# Importing real segmented data

Besides the synthetic presets, the package can build a `PhantomData` from a real
**segmented** volume — a labelled reconstruction where each voxel has been assigned to a
material class. This lets you run the full bimodal pipeline (projection, artifacts,
reconstruction, histogram analysis) on experimental geometry while still keeping exact,
known material properties. The code lives in `volume_importer.py`.

## What you need

1. **A labelled volume** — integer labels, one per voxel, in one of:
   - `.npy` or `.npz` (NumPy arrays)
   - a single multi-page `.tif` / `.tiff`
   - a folder of single-slice `.tif` / `.tiff` files
2. **Metadata** that says what the labels mean and how big a voxel is.

## The metadata

`SegmentationMetadata` describes the mapping from your labels to materials:

| Field | Meaning |
|---|---|
| `name` | a human-readable phantom name |
| `voxel_cm` | isotropic voxel side length, cm |
| `class_map` | dict mapping each integer label → a material name in `MATERIALS`, e.g. `{0: "air", 1: "aluminum", 2: "hdpe"}` |
| `axis_order` | input axis convention; currently `"zxy"` (matching the `(Nz, Nx, Ny)` storage order) |

You can provide the metadata as a Python dict, a JSON file, or a `SegmentationMetadata`
object.

## The one-call constructor

```python
from neutron_xray_sim.volume_importer import phantom_from_segmented_volume

phantom = phantom_from_segmented_volume(
    "my_segmentation.tif",                  # path, folder, or a NumPy array
    metadata={
        "name": "my_sample",
        "voxel_cm": 0.02,
        "class_map": {0: "air", 1: "hdpe", 2: "iron", 3: "water"},
    },
)
```

This loads and normalises the volume, validates that every label in the volume appears
in the `class_map`, remaps the labels to material indices, and builds a `PhantomData`
with the neutron and X-ray attenuation volumes generated from the mapped materials.
`phantom_from_files` is a friendly alias for the same function.

You can then drop the phantom straight into the simulation:

```python
from neutron_xray_sim import DualModalitySimulation, ArtifactConfig
sim = DualModalitySimulation(phantom=phantom, n_angles=180)
result = sim.run(ArtifactConfig.realistic(), tag="my_sample")
```

## Lower-level helpers

If you need more control, the module also exposes the individual steps:

| Function | Purpose |
|---|---|
| `load_segmented_array(source)` | load + normalise a labelled volume from any supported source |
| `load_segmentation_metadata(path)` | read metadata from a JSON file |
| `metadata_from_dict(dict)` | build `SegmentationMetadata` from a dict |
| `resolve_segmentation_metadata(meta)` | accept a dict / path / object and return a `SegmentationMetadata` |
| `remap_labels_to_material_indices(...)` | remap raw labels to material-index order |
| `phantom_from_array(seg, metadata)` | build the phantom from an in-memory array |

## Tips

- **Every label must be in the `class_map`.** Importing validates this and raises a
  clear error for any label in the volume that has no material assigned.
- **Material names must exist in `MATERIALS`.** If your sample contains a material not
  in the database, add it first with [`material_from_formula` or
  `make_composite_material`](materials.md) and register it in `MATERIALS`.
- **Axis order.** The volume must be in `(Nz, Nx, Ny)` order (`axis_order="zxy"`).
  Transpose your array before importing if it uses a different convention.
- **Air is label 0 by convention** so that background voxels can be masked out of the
  histogram with `mask=(label_vol != 0)`.
