"""
Phantoms: the voxel ground truth of a simulation.

* :mod:`.base`          — ``PhantomData`` container and ``PhantomBuilder``
* :mod:`.presets`       — ready-made phantoms and ``make_phantom``
* :mod:`.importer`      — build phantoms from real segmented volumes
* :mod:`.nmc`           — NMC particle-slab phantoms (needs pandas)
* :mod:`.density_sweep` — solid-fraction sweeps for NMC slabs (needs pandas)
"""

from .base import PhantomBuilder, PhantomData
from .importer import (
    SegmentationMetadata,
    load_segmentation_metadata,
    phantom_from_array,
    phantom_from_segmented_volume,
)
from .presets import (
    PHANTOM_PRESETS,
    make_battery_phantom,
    make_bone_implant_phantom,
    make_composite_phantom,
    make_custom_cylindrical_battery_phantom,
    make_hdpe_composite_phantom,
    make_industrial_phantom,
    make_li_ion_battery_phantom,
    make_phantom,
    register_preset,
    resolve_grid,
)

__all__ = [
    "PhantomBuilder", "PhantomData",
    "SegmentationMetadata", "load_segmentation_metadata",
    "phantom_from_array", "phantom_from_segmented_volume",
    "PHANTOM_PRESETS", "make_phantom", "register_preset", "resolve_grid",
    "make_battery_phantom", "make_bone_implant_phantom", "make_composite_phantom",
    "make_custom_cylindrical_battery_phantom", "make_hdpe_composite_phantom",
    "make_industrial_phantom", "make_li_ion_battery_phantom",
]
