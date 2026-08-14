"""
neutron_xray_sim.materials
──────────────────────────
Material database for dual-modality neutron / X-ray tomography.

Thermal-neutron cross sections are at 25.3 meV (λ = 1.798 Å, 293 K); X-ray
linear attenuation is tabulated on the 13-point :data:`XRAY_E_KEV` grid.  All
linear attenuation values are in cm⁻¹.

Layout
──────
``elements``   Per-element constants and the NIST table reader.
``core``       :class:`Material`, :class:`MaterialSpec`, and the single builder.
``registry``   :class:`MaterialRegistry` and the package-wide ``MATERIALS``.
``database``   The built-in catalogue, as declarative specs. **Edit this file
               to add a material to the package.**

Everything below is re-exported here, so ``from neutron_xray_sim.materials
import MATERIALS, material_from_formula`` keeps working exactly as before the
module became a package.

Quick tour
──────────
::

    from neutron_xray_sim import MATERIALS

    MATERIALS["hdpe"]                  # a Material
    MATERIALS.names(tag="battery")     # catalogue by tag
    print(MATERIALS.table())           # readable overview
    MATERIALS.audit()                  # data-quality report

    # Add your own, from code …
    from neutron_xray_sim.materials import MaterialSpec
    MATERIALS.register(MaterialSpec(
        key="quartz", name="Quartz", symbol="SiO2",
        density_gcc=2.65, formula="SiO2", tags=("mineral",),
    ))

    # … or from a file kept next to your analysis
    MATERIALS.load_file("my_project_materials.json")
"""

from __future__ import annotations

from .core import (
    Component,
    Material,
    MaterialSpec,
    SpecValidationError,
    build_material,
    make_composite_material,
    material_from_formula,
    neutron_components_from_formula,
    validate_spec,
    xray_spectrum,
)
from .elements import (
    ATOMIC_MASS,
    AVOGADRO,
    BARN_CM2,
    NEUTRON_XS,
    XRAY_DATA_DIR,
    XRAY_E_KEV,
    MissingElementDataError,
    available_elements,
    element_data_status,
    element_mu_over_rho,
    parse_formula,
)
from .registry import MATERIALS, DuplicateMaterialError, MaterialRegistry

__all__ = [
    # Value objects and builders
    "Material", "MaterialSpec", "Component",
    "build_material", "material_from_formula", "make_composite_material",
    "neutron_components_from_formula", "validate_spec",
    # Registry
    "MATERIALS", "MaterialRegistry",
    # Elements and constants
    "XRAY_E_KEV", "ATOMIC_MASS", "NEUTRON_XS", "AVOGADRO", "BARN_CM2",
    "XRAY_DATA_DIR", "parse_formula", "element_mu_over_rho",
    "available_elements", "element_data_status",
    # Spectrum
    "xray_spectrum",
    # Errors
    "MissingElementDataError", "SpecValidationError", "DuplicateMaterialError",
    # Convenience aliases
    "AIR", "WATER", "ALUMINUM", "HDPE", "IRON", "TITANIUM", "COPPER", "LEAD",
    "BONE", "TUNGSTEN", "ZINC", "LITHIUM", "STEEL", "GRAPHITE", "LFP",
    "NMC811", "NMC532", "NMC622", "LCO", "SEPARATOR_PE", "SEPARATOR_PP",
    "ELECTROLYTE_LIPF6_1M",
]


def __getattr__(name: str) -> Material:
    """Resolve the upper-case convenience aliases (``AIR``, ``HDPE``, …).

    Done lazily through the module ``__getattr__`` hook so that importing the
    package never forces every material to be built — a material whose NIST
    element file is missing should only fail when someone actually uses it.
    """
    key = name.lower()
    if name.isupper() and key in MATERIALS:
        return MATERIALS[key]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(set(__all__) | set(globals()))
