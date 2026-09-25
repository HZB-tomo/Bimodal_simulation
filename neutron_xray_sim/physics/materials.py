"""
neutron_xray_sim.physics.materials
──────────────────────────────────
Material database for dual-modality neutron / X-ray tomography simulation.

Thermal-neutron cross-sections are at 25.3 meV (λ ≈ 1.798 Å, 293 K).
X-ray linear attenuation coefficients are from NIST XCOM, tabulated on the
13-point energy grid ``XRAY_E_KEV`` that brackets the K-edges of common heavy
elements.  All linear attenuation values are in cm⁻¹.

Per-element data (atomic masses, neutron cross sections, XCOM tables) lives in
:mod:`neutron_xray_sim.physics.elements`.  This module turns it into
:class:`Material` objects:

* ``material_from_formula``   — single chemical formula + density
* ``make_composite_material`` — mixture of phases by weight fraction
* ``MATERIALS``               — the built-in database (dict keyed by id)
* ``xray_spectrum``           — Kramers' law polychromatic tube spectrum
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

from .elements import (
    ATOMIC_MASS,
    AVOGADRO,
    BARN_CM2,
    NEUTRON_XS,
    XRAY_DATA_DIR,
    XRAY_E_KEV,
    xray_mass_atten,
)

__all__ = [
    "Material",
    "MATERIALS",
    "XRAY_E_KEV",
    "XRAY_DATA_DIR",
    "ATOMIC_MASS",
    "NEUTRON_XS",
    "parse_formula",
    "material_from_formula",
    "neutron_components_from_formula",
    "make_composite_material",
    "register_material",
    "xray_spectrum",
]


# ──────────────────────────────────────────────────────────────────────────────
# Formula parsing  
# ──────────────────────────────────────────────────────────────────────────────

_FORMULA_RE = re.compile(r"([A-Z][a-z]?)([0-9]*\.?[0-9]*)")

def parse_formula(formula: str) -> Dict[str, float]:
    comp: Dict[str, float] = {}
    for elem, count_str in _FORMULA_RE.findall(formula):
        count = float(count_str) if count_str else 1.0
        comp[elem] = comp.get(elem, 0.0) + count
    if not comp:
        raise ValueError(f"Could not parse formula '{formula}'")
    return comp


def _check_elements(comp: Dict[str, float], formula: str) -> None:
    """Raise a helpful error if *formula* uses an element with no data."""
    unknown = [el for el in comp if el not in ATOMIC_MASS or el not in NEUTRON_XS]
    if unknown:
        raise KeyError(
            f"Formula '{formula}' uses element(s) {unknown} that are not in "
            "neutron_xray_sim.physics.elements (ATOMIC_MASS / NEUTRON_XS). "
            "See the 'Adding an element' section of that module."
        )


# ──────────────────────────────────────────────────────────────────────────────
# Material dataclass  
# ──────────────────────────────────────────────────────────────────────────────

@dataclass(eq=False)
class Material:
    """
    Physical properties of one material for dual-modality CT simulation.

    Attributes
    ----------
    name            : human-readable name
    symbol          : short label used in plots / legends
    density_gcc     : density  [g / cm³]
    mu_n_abs        : thermal-neutron absorption linear attenuation  [cm⁻¹]
    mu_n_coh        : thermal-neutron coherent-scatter linear attenuation [cm⁻¹]
    mu_n_inc        : thermal-neutron incoherent-scatter l.a.  [cm⁻¹]
    _mu_x_table     : X-ray l.a. at XRAY_E_KEV  [cm⁻¹],  shape (13,)
    color           : matplotlib colour string for visualisation

    Equality is by identity: two ``Material`` objects are the same material
    only if they are the same object (their tables are numpy arrays).
    """

    name: str
    symbol: str
    density_gcc: float

    mu_n_abs: float
    mu_n_coh: float
    mu_n_inc: float

    _mu_x_table: np.ndarray   # shape (13,)

    color: str = "#888888"

    @property
    def mu_n(self) -> float:
        """Total thermal-neutron linear attenuation  [cm⁻¹]."""
        return self.mu_n_abs + self.mu_n_coh + self.mu_n_inc

    @property
    def mu_n_scatter(self) -> float:
        """Total neutron scatter coefficient  [cm⁻¹]."""
        return self.mu_n_coh + self.mu_n_inc

    @property
    def rho(self) -> float:
        """Mass density  [g / cm³]. Alias for ``density_gcc``."""
        return self.density_gcc

    def mu_x_at(self, energy_keV: float) -> float:
        """Interpolate X-ray l.a. at a single energy  [cm⁻¹]."""
        return float(np.interp(energy_keV, XRAY_E_KEV, self._mu_x_table))

    def mu_x_array(self, energies_keV: np.ndarray) -> np.ndarray:
        """Interpolate X-ray l.a. at multiple energies  [cm⁻¹]."""
        return np.interp(energies_keV, XRAY_E_KEV, self._mu_x_table)

    def __repr__(self) -> str:
        return (
            f"Material({self.name}: ρ={self.density_gcc:.2f} g/cm³  "
            f"μₙ={self.mu_n:.3f} cm⁻¹  μₓ(80 keV)={self.mu_x_at(80):.3f} cm⁻¹)"
        )


# ──────────────────────────────────────────────────────────────────────────────
# Constructors
# ──────────────────────────────────────────────────────────────────────────────

def _mat(name, symbol, rho, mu_abs, mu_coh, mu_inc, mu_x_list, color="#888888"):
    """Internal shorthand for the built-in material database."""
    return Material(
        name=name, symbol=symbol, density_gcc=rho,
        mu_n_abs=mu_abs, mu_n_coh=mu_coh, mu_n_inc=mu_inc,
        _mu_x_table=np.asarray(mu_x_list, dtype=float),
        color=color,
    )


def material_from_formula(
    name: str,
    symbol: str,
    formula: str,
    density_gcc: float,
    color: str = "#888888",
    incoherent_scale: float = 1.0,
) -> Material:
    """
    Build a Material from chemical formula + density.

    Parameters
    ----------
    incoherent_scale
        Optional scale factor for the incoherent neutron term.
        Leave at 1.0 for dense inorganic solids.
    """
    comp = parse_formula(formula)
    _check_elements(comp, formula)
    molar_mass = sum(ATOMIC_MASS[el] * n for el, n in comp.items())
    mass_fracs = {el: (ATOMIC_MASS[el] * n) / molar_mass for el, n in comp.items()}

    mu_over_rho = np.zeros_like(XRAY_E_KEV, dtype=float)
    for el, w in mass_fracs.items():
        mu_over_rho += w * xray_mass_atten(el)
    mu_x = density_gcc * mu_over_rho

    sigma_abs = sum(comp[el] * NEUTRON_XS[el]["abs"] for el in comp)
    sigma_coh = sum(comp[el] * NEUTRON_XS[el]["coh"] for el in comp)
    sigma_inc = sum(comp[el] * NEUTRON_XS[el]["inc"] for el in comp)
    n_formula = density_gcc * AVOGADRO / molar_mass

    return Material(
        name=name, symbol=symbol, density_gcc=density_gcc,
        mu_n_abs=n_formula * sigma_abs * BARN_CM2,
        mu_n_coh=n_formula * sigma_coh * BARN_CM2,
        mu_n_inc=n_formula * sigma_inc * BARN_CM2 * incoherent_scale,
        _mu_x_table=np.asarray(mu_x, dtype=float),
        color=color,
    )


def neutron_components_from_formula(
    formula: str,
    density_gcc: float,
    incoherent_scale: float = 1.0,
) -> Tuple[float, float, float]:
    """
    Return (mu_n_abs, mu_n_coh, mu_n_inc) in cm⁻¹ for a formula + density.

    Helper used internally by make_composite_material().
    """
    comp = parse_formula(formula)
    _check_elements(comp, formula)
    molar_mass = sum(ATOMIC_MASS[el] * n for el, n in comp.items())
    N = density_gcc * AVOGADRO / molar_mass
    return (
        N * sum(comp[el] * NEUTRON_XS[el]["abs"] for el in comp) * BARN_CM2,
        N * sum(comp[el] * NEUTRON_XS[el]["coh"] for el in comp) * BARN_CM2,
        N * sum(comp[el] * NEUTRON_XS[el]["inc"] for el in comp) * BARN_CM2 * incoherent_scale,
    )


def make_composite_material(
    name: str,
    symbol: str,
    bulk_density_gcc: float,
    components: List[Tuple[str, float, float]],
    color: str = "#888888",
    incoherent_scale: float = 1.0,
) -> Material:
    """
    Build a Material from a mixture of mineral phases.

    Use this when the material cannot be represented by a single chemical
    formula — e.g. fossilised bone (fluorapatite + organics + calcite) or
    a sedimentary matrix (calcite + dolomite + feldspar + quartz + ...).

    Parameters
    ----------
    name, symbol      : identifiers for the resulting Material
    bulk_density_gcc  : effective bulk density of the mixture (g/cm³)
    components        : list of 3-tuples::

        (formula_string, weight_fraction, end_member_density_gcc)

        formula_string        : standard chemical formula e.g. 'Ca5P3O12F'
        weight_fraction       : 0–1 mass fraction; all must sum to 1 ± 0.02
        end_member_density_gcc: density of the pure phase (g/cm³).
                                Informational only — does not affect the
                                calculation, which uses bulk_density_gcc × wf.

    incoherent_scale  : applied uniformly to all incoherent neutron terms
    color             : matplotlib colour string

    Returns
    -------
    Material

    Example (needs the Ca, N and Si XCOM files in ``physics/data/xray``)
    -------
    >>> bone = make_composite_material(
    ...     name='Fossilised bone',
    ...     symbol='Bone',
    ...     bulk_density_gcc=2.00,
    ...     components=[
    ...         ('Ca5P3O12F',  0.70, 3.20),   # fluorapatite
    ...         ('C10H13NO4',  0.15, 1.35),   # dry collagen proxy
    ...         ('CaCO3',      0.12, 2.71),   # calcite
    ...         ('SiO2',       0.03, 2.65),   # clay silicate proxy
    ...     ],
    ... )
    """
    wfs = np.array([c[1] for c in components], dtype=float)
    if not np.isclose(wfs.sum(), 1.0, atol=0.02):
        raise ValueError(
            f"Component weight fractions sum to {wfs.sum():.4f}; must be 1.0 ± 0.02"
        )

    # ── X-ray: mass-fraction weighted μ/ρ → μ (cm⁻¹) ────────────────────────
    mu_over_rho_mix = np.zeros_like(XRAY_E_KEV, dtype=float)
    for (formula, wf, _rho_end) in components:
        comp = parse_formula(formula)
        _check_elements(comp, formula)
        molar_mass = sum(ATOMIC_MASS[el] * n for el, n in comp.items())
        for el, n in comp.items():
            w_el = (ATOMIC_MASS[el] * n) / molar_mass
            mu_over_rho_mix += wf * w_el * xray_mass_atten(el)

    mu_x = bulk_density_gcc * mu_over_rho_mix

    # ── Neutron: additive macroscopic cross sections ──────────────────────────
    # Each component occupies mass fraction wf of bulk_density_gcc, so its
    # effective partial density is bulk_density_gcc * wf.
    mu_abs_total = mu_coh_total = mu_inc_total = 0.0
    for (formula, wf, _rho_end) in components:
        partial_density = bulk_density_gcc * wf
        a, c, i = neutron_components_from_formula(
            formula, partial_density, incoherent_scale
        )
        mu_abs_total += a
        mu_coh_total += c
        mu_inc_total += i

    return Material(
        name=name, symbol=symbol, density_gcc=bulk_density_gcc,
        mu_n_abs=mu_abs_total,
        mu_n_coh=mu_coh_total,
        mu_n_inc=mu_inc_total,
        _mu_x_table=np.asarray(mu_x, dtype=float),
        color=color,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Material database  
# ──────────────────────────────────────────────────────────────────────────────

MATERIALS: Dict[str, Material] = {

    "air": _mat(
        "Air", "Air", 1.205e-3,
        mu_abs=0.0, mu_coh=0.0, mu_inc=0.0,
        mu_x_list=[1e-4]*13,
        color="#FFFFFF",
    ),

    "water": _mat(
        "Water", "H₂O", 1.00,
        mu_abs=0.022, mu_coh=0.259, mu_inc=1.099,
        mu_x_list=[0.811, 0.380, 0.268, 0.228, 0.206, 0.196,
                   0.184, 0.178, 0.171, 0.163, 0.151, 0.137, 0.119],
        color="#4488CC",
    ),

    "aluminum": _mat(
        "Aluminum", "Al", 2.70,
        mu_abs=0.014, mu_coh=0.083, mu_inc=0.001,
        mu_x_list=[9.29, 2.47, 1.06, 0.616, 0.413, 0.293,
                   0.278, 0.258, 0.239, 0.219, 0.202, 0.193, 0.187],
        color="#C8C8C8",
    ),

    "hdpe": _mat(
        "HDPE", "HDPE", 0.95,
        mu_abs=0.027, mu_coh=0.369, mu_inc=1.784,
        mu_x_list=[0.288, 0.181, 0.159, 0.155, 0.153, 0.157,
                   0.167, 0.165, 0.160, 0.152, 0.141, 0.128, 0.111],
        color="#88CC88",
    ),

    "iron": _mat(
        "Iron", "Fe", 7.87,
        mu_abs=0.217, mu_coh=0.910, mu_inc=0.033,
        mu_x_list=[294., 80.3, 34.2, 17.5, 10.1, 6.41,
                   4.12, 3.05, 2.38, 1.68, 1.26, 1.05, 0.882],
        color="#666666",
    ),

    "titanium": _mat(
        "Titanium", "Ti", 4.51,
        mu_abs=0.345, mu_coh=0.175, mu_inc=0.120,
        mu_x_list=[92.3, 24.8, 9.97, 5.28, 3.11, 2.07,
                   1.48, 1.14, 0.943, 0.770, 0.699, 0.636, 0.568],
        color="#AA9999",
    ),

    "copper": _mat(
        "Copper", "Cu", 8.96,
        mu_abs=0.313, mu_coh=0.760, mu_inc=0.047,
        mu_x_list=[375., 110., 45.4, 22.5, 12.5, 7.87,
                   4.92, 3.70, 2.97, 2.10, 1.55, 1.26, 1.03],
        color="#DD8833",
    ),

    "lead": _mat(
        "Lead", "Pb", 11.35,
        mu_abs=0.006, mu_coh=0.366, mu_inc=0.001,
        mu_x_list=[340., 124., 62.4, 34.4, 20.6, 13.2,
                   22.8, 63.0, 50.7, 27.5, 16.4, 11.1, 7.58],
        color="#556677",
    ),

    "bone": _mat(
        "Bone (HAp)", "HAp", 1.92,
        mu_abs=0.038, mu_coh=0.130, mu_inc=0.392,
        mu_x_list=[6.37, 1.94, 0.980, 0.666, 0.557, 0.481,
                   0.469, 0.421, 0.438, 0.421, 0.400, 0.373, 0.322],
        color="#F0DDB0",
    ),

    "tungsten": _mat(
        "Tungsten", "W", 19.3,
        mu_abs=1.157, mu_coh=0.301, mu_inc=0.102,
        mu_x_list=[1035., 313., 140., 74.4, 45.0, 88.5,
                   88.0, 64.0, 50.0, 30.5, 18.0, 11.0, 6.50],
        color="#222244",
    ),

    "zinc": _mat(
        "Zinc", "Zn", 7.13,
        mu_abs=0.073, mu_coh=0.272, mu_inc=0.005,
        mu_x_list=[215., 66.2, 28.2, 14.1, 7.97, 5.09,
                   3.25, 2.43, 1.97, 1.41, 1.05, 0.862, 0.713],
        color="#BBBB44",
    ),
}


MATERIALS.update({

    "lithium": material_from_formula(
        name="Lithium Metal", symbol="Li",
        formula="Li", density_gcc=0.534, color="#B0B0B0",
    ),

    "electrolyte_lipf6_1m": material_from_formula(
        name="1 M LiPF6 Organic Electrolyte", symbol="LiPF6-sol",
        formula="Li1P1F6C5H10O3", density_gcc=1.20,
        color="#66CCFF", incoherent_scale=0.5,
    ),

    "steel": material_from_formula(
        name="Steel (Fe-Ni)", symbol="Steel",
        formula="Fe0.98Ni0.02", density_gcc=7.85, color="#777777",
    ),

    "graphite": material_from_formula(
        name="Graphite", symbol="Graphite",
        formula="C", density_gcc=2.26, color="#444444",
    ),

    "lfp": material_from_formula(
        name="Lithium Iron Phosphate", symbol="LFP",
        formula="LiFePO4", density_gcc=3.60, color="#6B8E23",
    ),

    "nmc811": material_from_formula(
        name="NMC811", symbol="NMC811",
        formula="LiNi0.8Mn0.1Co0.1O2", density_gcc=4.80, color="#CC6677",
    ),

    "nmc532": material_from_formula(
        name="NMC532", symbol="NMC532",
        formula="LiNi0.5Mn0.3Co0.2O2", density_gcc=4.70, color="#DD8899",
    ),

    "nmc622": material_from_formula(
        name="NMC622", symbol="NMC622",
        formula="LiNi0.6Mn0.2Co0.2O2", density_gcc=4.75, color="#BB5577",
    ),

    "lco": material_from_formula(
        name="Lithium Cobalt Oxide", symbol="LCO",
        formula="LiCoO2", density_gcc=5.05, color="#3366AA",
    ),

    "separator_pe": material_from_formula(
        name="PE Separator", symbol="PE-sep",
        formula="C2H4", density_gcc=0.94,
        color="#EEEEAA", incoherent_scale=0.35,
    ),

    "separator_pp": material_from_formula(
        name="PP Separator", symbol="PP-sep",
        formula="C3H6", density_gcc=0.90,
        color="#FFDDAA", incoherent_scale=0.35,
    ),

    "separator_pe_electrolyte": material_from_formula(
        name="PE Separator + LiPF6 Electrolyte",
        symbol="separator + electrolyte",
        formula="C7H14O3Li1P1F6", density_gcc=1.05,
        color="#BFEFFF", incoherent_scale=0.5,
    ),
})


def register_material(key: str, material: Material, overwrite: bool = False) -> Material:
    """
    Add a material to the global ``MATERIALS`` database.

    Once registered, the material can be referred to by *key* everywhere a
    material name is accepted (``PhantomBuilder.add_sphere("my_key", ...)``,
    ``phantom_from_segmented_volume`` class maps, …).

    >>> register_material("pmma", material_from_formula(
    ...     "PMMA", "PMMA", "C5H8O2", density_gcc=1.18, color="#AADDFF"))
    """
    if key in MATERIALS and not overwrite:
        raise KeyError(
            f"Material '{key}' already exists; pass overwrite=True to replace it."
        )
    MATERIALS[key] = material
    return material


# ──────────────────────────────────────────────────────────────────────────────
# X-ray spectrum generation  
# ──────────────────────────────────────────────────────────────────────────────

def xray_spectrum(
    kVp: float = 120.0,
    filter_mm_Al: float = 2.0,
    filter_mm_Cu: float = 0.0,
    n_bins: int = 12,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate a simplified bremsstrahlung X-ray spectrum (Kramers' law).

    Returns
    -------
    energies  : energy bin centres [keV]  shape (n_bins,)
    weights   : normalised photon fluence weights  shape (n_bins,), sum = 1
    """
    E_min = max(10.0, 0.05 * kVp)
    energies = np.linspace(E_min, kVp, n_bins + 1)
    energies = 0.5 * (energies[:-1] + energies[1:])

    Z_W = 74
    spectrum = Z_W * energies * (kVp - energies)
    spectrum = np.clip(spectrum, 0, None)

    if filter_mm_Al > 0:
        al = MATERIALS["aluminum"]
        spectrum *= np.exp(-al.mu_x_array(energies) * filter_mm_Al * 0.1)

    if filter_mm_Cu > 0:
        cu = MATERIALS["copper"]
        spectrum *= np.exp(-cu.mu_x_array(energies) * filter_mm_Cu * 0.1)

    total = spectrum.sum()
    if total == 0:
        raise ValueError("X-ray spectrum collapsed to zero — check kVp / filter settings.")
    spectrum /= total

    return energies, spectrum


# ──────────────────────────────────────────────────────────────────────────────
# Convenience aliases  
# ──────────────────────────────────────────────────────────────────────────────
AIR       = MATERIALS["air"]
WATER     = MATERIALS["water"]
ALUMINUM  = MATERIALS["aluminum"]
HDPE      = MATERIALS["hdpe"]
IRON      = MATERIALS["iron"]
TITANIUM  = MATERIALS["titanium"]
COPPER    = MATERIALS["copper"]
LEAD      = MATERIALS["lead"]
BONE      = MATERIALS["bone"]
TUNGSTEN  = MATERIALS["tungsten"]
ZINC      = MATERIALS["zinc"]
GRAPHITE     = MATERIALS["graphite"]
LFP          = MATERIALS["lfp"]
NMC811       = MATERIALS["nmc811"]
NMC532       = MATERIALS["nmc532"]
NMC622       = MATERIALS["nmc622"]
LCO          = MATERIALS["lco"]
SEPARATOR_PE = MATERIALS["separator_pe"]
SEPARATOR_PP = MATERIALS["separator_pp"]
