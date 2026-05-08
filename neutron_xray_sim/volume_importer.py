"""
neutron_xray_sim/volume_importer.py
───────────────────────────────────
Import externally segmented image/volume data and convert it into PhantomData.

Expected workflows
------------------
A phantom can be imported in two ways:

1. From files
   - A labelled volume file or TIFF folder.
   - A JSON metadata file containing voxel size and material mapping.

2. From memory
   - A user-provided NumPy array.
   - A metadata dictionary or SegmentationMetadata object.

Supported labelled volume inputs
--------------------------------
.npy
    NumPy array file.
.npz
    NumPy archive. If multiple arrays exist, one must be named one of:
    label_vol, segmentation, labels, volume.
.tif / .tiff
    A 2-D TIFF image or a 3-D/multipage TIFF volume.
folder of TIFF files
    A directory containing ordered 2-D TIFF slices. The slices are stacked along
    z to produce a volume with shape (Nz, Nx, Ny).

Example metadata JSON
---------------------
{
  "name": "imported_sample",
  "voxel_cm": 0.01,
  "axis_order": "zxy",
  "class_map": {
    "0": "air",
    "1": "aluminum",
    "2": "hdpe",
    "3": "copper"
  }
}

The final output is a normal PhantomData object and can be used directly by the
simulation pipeline.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Tuple, Union

import numpy as np

from .materials import MATERIALS, Material
from .phantom import PhantomData


PathLike = Union[str, Path]
MetadataLike = Union["SegmentationMetadata", Mapping[str, object], PathLike]


@dataclass(frozen=True)
class SegmentationMetadata:
    """
    Metadata required to convert a labelled segmentation into PhantomData.

    Parameters
    ----------
    name:
        Human-readable phantom name.
    voxel_cm:
        Isotropic voxel side length in cm.
    class_map:
        Mapping from external segmentation labels to material names in MATERIALS.
        Example: {0: "air", 1: "aluminum", 2: "hdpe"}
    axis_order:
        Axis convention of the input volume. For now only "zxy" is supported,
        matching PhantomData storage order: (Nz, Nx, Ny).
    """

    name: str
    voxel_cm: float
    class_map: Dict[int, str]
    axis_order: str = "zxy"


# ──────────────────────────────────────────────────────────────────────────────
# Metadata loading and validation
# ──────────────────────────────────────────────────────────────────────────────


def metadata_from_dict(metadata: Mapping[str, object]) -> SegmentationMetadata:
    """
    Parse and validate segmentation metadata from a Python dictionary.

    This is the in-memory equivalent of ``load_segmentation_metadata`` and is
    useful when users already loaded or constructed their metadata themselves.

    Example
    -------
    >>> metadata = {
    ...     "name": "sample",
    ...     "voxel_cm": 0.01,
    ...     "class_map": {0: "air", 1: "aluminum"},
    ... }
    >>> meta = metadata_from_dict(metadata)
    """
    if not isinstance(metadata, Mapping):
        raise TypeError("metadata must be a mapping/dictionary-like object.")

    name = str(metadata.get("name", "imported_phantom"))

    if "voxel_cm" not in metadata:
        raise ValueError("Metadata is missing required field: 'voxel_cm'.")
    voxel_cm = float(metadata["voxel_cm"])
    if voxel_cm <= 0:
        raise ValueError("'voxel_cm' must be positive.")

    axis_order = str(metadata.get("axis_order", "zxy")).lower()
    if axis_order != "zxy":
        raise NotImplementedError(
            "Only axis_order='zxy' is supported for now, matching PhantomData "
            "storage order (Nz, Nx, Ny)."
        )

    if "class_map" not in metadata:
        raise ValueError("Metadata is missing required field: 'class_map'.")
    class_map = _parse_class_map(metadata["class_map"])
    _validate_class_map(class_map)

    return SegmentationMetadata(
        name=name,
        voxel_cm=voxel_cm,
        class_map=class_map,
        axis_order=axis_order,
    )


def load_segmentation_metadata(metadata_path: PathLike) -> SegmentationMetadata:
    """
    Load and validate segmentation metadata from a JSON file.

    Parameters
    ----------
    metadata_path:
        Path to a JSON file containing at least ``voxel_cm`` and ``class_map``.

    Returns
    -------
    SegmentationMetadata
        Validated metadata object.
    """
    path = Path(metadata_path)
    if not path.exists():
        raise FileNotFoundError(f"Metadata file does not exist: {path}")
    if path.suffix.lower() != ".json":
        raise ValueError(f"Metadata file must be a .json file, got: {path.suffix}")

    with path.open("r", encoding="utf-8") as f:
        raw = json.load(f)

    if not isinstance(raw, Mapping):
        raise ValueError("Metadata JSON must contain a JSON object at the top level.")

    if "name" not in raw:
        raw = dict(raw)
        raw["name"] = path.stem

    return metadata_from_dict(raw)


def resolve_segmentation_metadata(metadata: MetadataLike) -> SegmentationMetadata:
    """
    Resolve metadata from a SegmentationMetadata object, dictionary, or JSON path.
    """
    if isinstance(metadata, SegmentationMetadata):
        _validate_class_map(metadata.class_map)
        return metadata

    if isinstance(metadata, Mapping):
        return metadata_from_dict(metadata)

    return load_segmentation_metadata(metadata)


def _parse_class_map(raw_class_map: object) -> Dict[int, str]:
    """
    Parse a class map into Dict[int, str].

    JSON object keys are always strings, so labels such as "0" and "1" are
    converted to integers. Python dictionaries may use either int or str keys.
    """
    if not isinstance(raw_class_map, Mapping):
        raise ValueError("'class_map' must be a mapping, e.g. {'0': 'air'}.")

    parsed: Dict[int, str] = {}
    for raw_label, raw_material_name in raw_class_map.items():
        try:
            label = int(raw_label)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Class-map label keys must be integers or integer strings; "
                f"got {raw_label!r}."
            ) from exc

        if label < 0:
            raise ValueError(f"Class-map labels must be non-negative; got {label}.")

        material_name = str(raw_material_name).lower()
        parsed[label] = material_name

    if not parsed:
        raise ValueError("'class_map' must not be empty.")

    return parsed


def _validate_class_map(class_map: Mapping[int, str]) -> None:
    """Validate that every mapped material exists in the package material DB."""
    missing_materials = sorted(
        {mat_name for mat_name in class_map.values() if mat_name not in MATERIALS}
    )
    if missing_materials:
        available = ", ".join(sorted(MATERIALS.keys()))
        raise ValueError(
            "Unknown material name(s) in class_map: "
            f"{missing_materials}. Available materials are: {available}"
        )


# ──────────────────────────────────────────────────────────────────────────────
# Volume loading and validation
# ──────────────────────────────────────────────────────────────────────────────


def load_segmented_array(volume_source: Union[PathLike, np.ndarray]) -> np.ndarray:
    """
    Load or normalise a labelled segmentation array.

    Parameters
    ----------
    volume_source:
        Either a path-like input or an already-loaded NumPy array.

        Supported path-like inputs:
        - .npy
        - .npz
        - .tif / .tiff
        - directory containing .tif/.tiff slices

    Returns
    -------
    np.ndarray
        Integer labelled array with shape (Nz, Nx, Ny). A 2-D input is promoted
        to (1, Nx, Ny).
    """
    if isinstance(volume_source, np.ndarray):
        return _normalise_segmented_array(volume_source)

    path = Path(volume_source)
    if not path.exists():
        raise FileNotFoundError(f"Volume source does not exist: {path}")

    if path.is_dir():
        seg = _load_tiff_folder(path)
        return _normalise_segmented_array(seg)

    suffix = path.suffix.lower()

    if suffix == ".npy":
        seg = np.load(path)
    elif suffix == ".npz":
        seg = _load_npz_array(path)
    elif suffix in {".tif", ".tiff"}:
        seg = _load_tiff_file(path)
    else:
        raise ValueError(
            f"Unsupported segmented volume format: {suffix}. "
            "Currently supported inputs are: .npy, .npz, .tif, .tiff, "
            "or a folder of TIFF slices."
        )

    return _normalise_segmented_array(seg)


def _load_npz_array(path: Path) -> np.ndarray:
    """Load a labelled array from a .npz archive."""
    with np.load(path) as archive:
        keys = list(archive.keys())

        if len(keys) == 1:
            return archive[keys[0]]

        preferred_keys = ("label_vol", "segmentation", "labels", "volume")
        for key in preferred_keys:
            if key in archive:
                return archive[key]

        raise ValueError(
            f"NPZ file contains multiple arrays {keys}, but none are named one "
            f"of {preferred_keys}."
        )


def _load_tiff_file(path: Path) -> np.ndarray:
    """
    Load a 2-D, 3-D, or multipage TIFF file.

    Requires the optional dependency ``tifffile``.
    """
    try:
        import tifffile
    except ImportError as exc:
        raise ImportError(
            "Loading TIFF files requires the optional dependency 'tifffile'. "
            "Install it with: pip install tifffile"
        ) from exc

    return tifffile.imread(path)


def _load_tiff_folder(folder: Path) -> np.ndarray:
    """
    Load a directory of 2-D TIFF slices and stack them along z.

    Files are sorted lexicographically by filename. Users should therefore name
    slices with zero padding, for example: slice_000.tif, slice_001.tif, ...
    """
    tiff_paths = sorted(
        [
            p for p in folder.iterdir()
            if p.is_file() and p.suffix.lower() in {".tif", ".tiff"}
        ]
    )

    if not tiff_paths:
        raise ValueError(f"No .tif/.tiff files found in folder: {folder}")

    slices = [_load_tiff_file(path) for path in tiff_paths]

    first_shape = slices[0].shape
    if len(first_shape) != 2:
        raise ValueError(
            "A folder of TIFF files must contain 2-D slice images. "
            f"First file {tiff_paths[0].name} has shape {first_shape}."
        )

    for path, arr in zip(tiff_paths, slices):
        if arr.ndim != 2:
            raise ValueError(
                "A folder of TIFF files must contain only 2-D slice images. "
                f"File {path.name} has shape {arr.shape}."
            )
        if arr.shape != first_shape:
            raise ValueError(
                "All TIFF slices in a folder must have the same shape. "
                f"Expected {first_shape}, but {path.name} has shape {arr.shape}."
            )

    return np.stack(slices, axis=0)


def _normalise_segmented_array(seg: np.ndarray) -> np.ndarray:
    """
    Validate and normalise a loaded segmentation array.

    - 2-D arrays are promoted to one z-slice.
    - Only integer or integer-like arrays are accepted.
    - The returned dtype is np.uint8, np.uint16, or np.uint32 depending on the
      maximum label value.
    """
    if not isinstance(seg, np.ndarray):
        raise TypeError(f"Loaded volume must be a numpy array, got {type(seg)}.")

    if seg.size == 0:
        raise ValueError("Segmented volume must not be empty.")

    if seg.ndim == 2:
        seg = seg[np.newaxis, :, :]
    elif seg.ndim != 3:
        raise ValueError(
            "Segmented volume must be 2-D or 3-D. "
            f"Got array with shape {seg.shape} and ndim={seg.ndim}."
        )

    if not _is_integer_array(seg):
        raise ValueError(
            "Segmented volume must contain integer class labels. "
            f"Got dtype {seg.dtype}."
        )

    if np.any(seg < 0):
        raise ValueError("Segmented volume contains negative labels, which are not supported.")

    max_label = int(np.max(seg))
    if max_label <= np.iinfo(np.uint8).max:
        return seg.astype(np.uint8, copy=False)
    if max_label <= np.iinfo(np.uint16).max:
        return seg.astype(np.uint16, copy=False)
    return seg.astype(np.uint32, copy=False)


def _is_integer_array(arr: np.ndarray) -> bool:
    """Return True if an array stores integer values safely."""
    if np.issubdtype(arr.dtype, np.integer):
        return True

    if np.issubdtype(arr.dtype, np.floating):
        finite = np.isfinite(arr)
        return bool(finite.all() and np.all(arr == np.floor(arr)))

    return False


# ──────────────────────────────────────────────────────────────────────────────
# Label remapping
# ──────────────────────────────────────────────────────────────────────────────


def remap_labels_to_material_indices(
    seg: np.ndarray,
    class_map: Mapping[int, str],
) -> Tuple[np.ndarray, List[Material]]:
    """
    Remap external segmentation labels to PhantomData material indices.

    PhantomData expects label values to index directly into its ``materials``
    list. External segmentations may use arbitrary labels such as 0, 10, 42.
    This function converts them into compact internal labels 0, 1, 2, ... and
    builds the matching material list.

    Parameters
    ----------
    seg:
        Integer labelled volume with shape (Nz, Nx, Ny).
    class_map:
        Mapping from external labels to material names.

    Returns
    -------
    label_vol:
        Remapped integer volume compatible with PhantomData.
    materials:
        Ordered list of Material objects. ``label_vol == i`` corresponds to
        ``materials[i]``.
    """
    _validate_volume_labels_against_class_map(seg, class_map)

    # Make air index 0 when air is present. This keeps imported phantoms aligned
    # with the rest of the package convention: index 0 = air.
    external_labels = sorted(class_map.keys())
    air_labels = [label for label in external_labels if class_map[label] == "air"]

    ordered_external_labels: List[int] = []
    if air_labels:
        ordered_external_labels.append(air_labels[0])
        ordered_external_labels.extend(
            label for label in external_labels if label != air_labels[0]
        )
    else:
        ordered_external_labels = external_labels

    materials: List[Material] = [MATERIALS[class_map[label]] for label in ordered_external_labels]

    if len(materials) > np.iinfo(np.uint8).max + 1:
        label_dtype = np.uint16
    else:
        label_dtype = np.uint8

    label_vol = np.zeros(seg.shape, dtype=label_dtype)
    for internal_label, external_label in enumerate(ordered_external_labels):
        label_vol[seg == external_label] = internal_label

    return label_vol, materials


def _validate_volume_labels_against_class_map(
    seg: np.ndarray,
    class_map: Mapping[int, str],
) -> None:
    """Check that the volume and class_map describe each other consistently."""
    volume_labels = set(int(v) for v in np.unique(seg))
    mapped_labels = set(int(v) for v in class_map.keys())

    missing_from_map = sorted(volume_labels - mapped_labels)
    if missing_from_map:
        raise ValueError(
            "Segmented volume contains label(s) missing from class_map: "
            f"{missing_from_map}."
        )

    unused_map_labels = sorted(mapped_labels - volume_labels)
    if unused_map_labels:
        # Unused labels are not fatal. Users may keep one standard mapping file
        # for several related segmentations.
        pass


# ──────────────────────────────────────────────────────────────────────────────
# Public constructors
# ──────────────────────────────────────────────────────────────────────────────


def phantom_from_array(
    volume: np.ndarray,
    metadata: Union[SegmentationMetadata, Mapping[str, object]],
    *,
    name: Optional[str] = None,
) -> PhantomData:
    """
    Build a PhantomData object from an already-loaded NumPy segmentation array.

    Use this when the user loaded the volume through any external library or
    custom preprocessing pipeline and already has a NumPy array in memory.

    Parameters
    ----------
    volume:
        2-D or 3-D NumPy array containing integer class labels.
    metadata:
        Either a SegmentationMetadata object or a dictionary containing
        ``voxel_cm`` and ``class_map``.
    name:
        Optional name override. If omitted, metadata.name is used.

    Returns
    -------
    PhantomData
        Fully built phantom with neutron and X-ray attenuation volumes generated
        from the mapped materials.
    """
    metadata_obj = resolve_segmentation_metadata(metadata)
    seg = load_segmented_array(volume)

    label_vol, materials = remap_labels_to_material_indices(
        seg=seg,
        class_map=metadata_obj.class_map,
    )

    Nz, Nx, Ny = label_vol.shape

    return PhantomData(
        Nz=Nz,
        Nx=Nx,
        Ny=Ny,
        voxel_cm=metadata_obj.voxel_cm,
        label_vol=label_vol,
        materials=materials,
        name=name or metadata_obj.name,
    )


def phantom_from_segmented_volume(
    volume_source: Union[PathLike, np.ndarray],
    metadata: MetadataLike,
    *,
    name: Optional[str] = None,
) -> PhantomData:
    """
    Build a PhantomData object from segmented volume data and metadata.

    This is the main public constructor. It accepts either file-based or
    in-memory inputs.

    Parameters
    ----------
    volume_source:
        Either a path-like input or an already-loaded NumPy array.

        Supported path-like inputs:
        - .npy
        - .npz
        - .tif / .tiff
        - directory containing .tif/.tiff slices

    metadata:
        Either:
        - path to a JSON metadata file,
        - Python dictionary,
        - SegmentationMetadata object.
    name:
        Optional name override. If omitted, metadata.name is used.

    Returns
    -------
    PhantomData
        Fully built phantom with neutron and X-ray attenuation volumes generated
        from the mapped materials.
    """
    metadata_obj = resolve_segmentation_metadata(metadata)
    seg = load_segmented_array(volume_source)

    return phantom_from_array(seg, metadata_obj, name=name)


# Friendly alias for users who prefer explicit naming.
phantom_from_files = phantom_from_segmented_volume


__all__ = [
    "SegmentationMetadata",
    "metadata_from_dict",
    "load_segmentation_metadata",
    "resolve_segmentation_metadata",
    "load_segmented_array",
    "remap_labels_to_material_indices",
    "phantom_from_array",
    "phantom_from_segmented_volume",
    "phantom_from_files",
]
