"""
Physics: element data, the material database, X-ray tube spectra and neutron
beam spectra.

* :mod:`.elements`        — atomic masses, neutron cross sections, NIST XCOM data
* :mod:`.materials`       — ``Material``, ``MATERIALS`` and material builders
* :mod:`.neutron_spectra` — thermal / cold / polychromatic neutron beams
* :mod:`.ncrystal_bragg`  — Bragg-edge cross sections via NCrystal (optional)
"""

from .elements import ATOMIC_MASS, NEUTRON_XS, XRAY_E_KEV, xray_mass_atten
from .materials import (
    MATERIALS,
    Material,
    make_composite_material,
    material_from_formula,
    register_material,
    xray_spectrum,
)
from .neutron_spectra import (
    NEUTRON_MODES,
    NeutronBeam,
    cold_mono_beam,
    cold_poly_beam,
    ill_next_beam,
    mu_n_for_beam,
    mu_n_lut_for_beam,
    mu_n_spectrum_lut,
    plot_spectra,
    thermal_beam,
)

__all__ = [
    "ATOMIC_MASS", "NEUTRON_XS", "XRAY_E_KEV", "xray_mass_atten",
    "MATERIALS", "Material", "make_composite_material", "material_from_formula",
    "register_material", "xray_spectrum",
    "NEUTRON_MODES", "NeutronBeam", "cold_mono_beam", "cold_poly_beam",
    "ill_next_beam", "mu_n_for_beam", "mu_n_lut_for_beam", "mu_n_spectrum_lut",
    "plot_spectra", "thermal_beam",
]
