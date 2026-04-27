"""
neutron_xray_sim/materials.py
──────────────────────────────
Material database for dual-modality neutron / X-ray tomography simulation.

Thermal-neutron cross-sections are at 25 meV  (λ ≈ 1.798 Å, 293 K).
X-ray linear attenuation coefficients are from NIST XCOM, tabulated on a
13-point energy grid that brackets the K-edges of common heavy elements.

All linear attenuation values are in cm⁻¹.
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import Dict
import re
from pathlib import Path


XRAY_DATA_DIR = Path(__file__).resolve().parent / "lib" / "xray_data"



# ──────────────────────────────────────────────────────────────────────────────
# Standard X-ray energy grid (keV).  The two "extra" points at 70 and 90 keV
# straddle the W K-edge (69.5 keV) and the Pb K-edge (88.0 keV).
# ──────────────────────────────────────────────────────────────────────────────
XRAY_E_KEV = np.array(
    [20, 30, 40, 50, 60, 70, 80, 90, 100, 120, 150, 200, 300], dtype=float
)


AVOGADRO = 6.02214076e23
BARN_CM2 = 1e-24

ATOMIC_MASS = {
    "H": 1.00794,
    "Li": 6.941,
    "C": 12.0107,
    "O": 15.999,
    "F": 18.998403163,
    "P": 30.973761998,
    "Fe": 55.845,
    "Co": 58.933194,
    "Ni": 58.6934,
    "Mn": 54.938044,
}

# Microscopic thermal-neutron cross sections [barn]
# These should be the elemental values you choose to adopt consistently.
NEUTRON_XS = {
    "H":  {"abs": 0.3326, "coh": 1.7568, "inc": 80.27},
    "Li": {"abs": 70.5,   "coh": 0.454,  "inc": 0.92},
    "C":  {"abs": 0.0035, "coh": 5.551,  "inc": 0.001},
    "O":  {"abs": 0.00019,"coh": 4.232,  "inc": 0.0008},
     "F": {"abs": 0.0096, "coh": 4.018, "inc": 0.0008},
    "P":  {"abs": 0.172,  "coh": 3.307,  "inc": 0.005},
    "Fe": {"abs": 2.56,   "coh": 11.22,  "inc": 0.40},
    "Co": {"abs": 37.18,  "coh": 0.779,  "inc": 4.8},
    "Ni": {"abs": 4.49,   "coh": 13.3,   "inc": 5.2},
    "Mn": {"abs": 13.3,   "coh": 2.15,   "inc": 0.40},
}

# Elemental mass attenuation coefficients [cm^2/g] at XRAY_E_KEV.
# You must fill these arrays from your chosen XCOM export or source file.
#[20, 30, 40, 50, 60, 70, 80, 90, 100, 120, 150, 200, 300]


'''
To create this programatically using NIST dataset .

XRAY_MASS_ATTEN = {
    "H":  np.array([ ... ], dtype=float),
    "Li": np.array([ ... ], dtype=float),
    "C":  np.array([ ... ], dtype=float),
    "O":  np.array([ ... ], dtype=float),
    "P":  np.array([ ... ], dtype=float),
    "Fe": np.array([ ... ], dtype=float),
    "Co": np.array([ ... ], dtype=float),
    "Ni": np.array([ ... ], dtype=float),
    "Mn": np.array([ ... ], dtype=float),
}

'''

def _parse_xcom_txt(filepath: Path):
    """
    Parse NIST XCOM text file.

    Returns
    -------
    energies_keV : np.ndarray
    mu_over_rho  : np.ndarray   (Total attenuation WITH coherent scattering)
    """
    energies = []
    mu_vals = []

    with open(filepath, "r") as f:
        for line in f:
            parts = line.strip().split()

            # skip header / malformed lines
            if len(parts) < 8:
                continue

            try:
                energy_mev = float(parts[0])
                mu_total_with_coh = float(parts[6])  # <-- THIS COLUMN
            except ValueError:
                continue

            energies.append(energy_mev * 1000.0)  # MeV → keV
            mu_vals.append(mu_total_with_coh)

    return np.array(energies), np.array(mu_vals)


def _element_mu_over_rho(element: str) -> np.ndarray:
    """
    Return μ/ρ for one element on XRAY_E_KEV grid.
    """
    filepath = XRAY_DATA_DIR / f"{element}.txt"

    if not filepath.exists():
        raise FileNotFoundError(f"Missing X-ray data file: {filepath}")

    energies, mu = _parse_xcom_txt(filepath)

    # interpolate onto simulation grid
    return np.interp(XRAY_E_KEV, energies, mu)


def build_xray_mass_atten():
    elements = ["H", "Li", "C", "O", "F", "P", "Fe", "Co", "Ni", "Mn"]

    data = {}
    for el in elements:
        data[el] = _element_mu_over_rho(el)

    return data

XRAY_MASS_ATTEN = build_xray_mass_atten()



_FORMULA_RE = re.compile(r"([A-Z][a-z]?)([0-9]*\.?[0-9]*)")

def parse_formula(formula: str) -> Dict[str, float]:
    comp: Dict[str, float] = {}
    for elem, count_str in _FORMULA_RE.findall(formula):
        count = float(count_str) if count_str else 1.0
        comp[elem] = comp.get(elem, 0.0) + count
    if not comp:
        raise ValueError(f"Could not parse formula '{formula}'")
    return comp

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
        For H-rich imaging-effective materials, you may reduce or tune it.
    """
    comp = parse_formula(formula)

    # Molar mass [g/mol]
    molar_mass = sum(ATOMIC_MASS[el] * n for el, n in comp.items())

    # Mass fractions for X-ray mixture rule
    mass_fracs = {
        el: (ATOMIC_MASS[el] * n) / molar_mass
        for el, n in comp.items()
    }

    # X-ray: mu/rho mix rule, then mu = rho * mu/rho
    mu_over_rho = np.zeros_like(XRAY_E_KEV, dtype=float)
    for el, w in mass_fracs.items():
        mu_over_rho += w * XRAY_MASS_ATTEN[el]
    mu_x = density_gcc * mu_over_rho

    # Neutron microscopic cross sections per formula unit [barn]
    sigma_abs = sum(comp[el] * NEUTRON_XS[el]["abs"] for el in comp)
    sigma_coh = sum(comp[el] * NEUTRON_XS[el]["coh"] for el in comp)
    sigma_inc = sum(comp[el] * NEUTRON_XS[el]["inc"] for el in comp)

    # Number density of formula units [1/cm^3]
    n_formula = density_gcc * AVOGADRO / molar_mass

    # Macroscopic cross sections [cm^-1]
    mu_abs = n_formula * sigma_abs * BARN_CM2
    mu_coh = n_formula * sigma_coh * BARN_CM2
    mu_inc = n_formula * sigma_inc * BARN_CM2 * incoherent_scale

    return Material(
        name=name,
        symbol=symbol,
        density_gcc=density_gcc,
        mu_n_abs=mu_abs,
        mu_n_coh=mu_coh,
        mu_n_inc=mu_inc,
        _mu_x_table=np.asarray(mu_x, dtype=float),
        color=color,
    )


@dataclass
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
                      (dominant for H-rich materials: σ_inc(H) ≈ 80 barn/atom)
    _mu_x_table     : X-ray l.a. at XRAY_E_KEV  [cm⁻¹],  shape (13,)
    color           : matplotlib colour string for visualisation
    """

    name: str
    symbol: str
    density_gcc: float

    # Neutron components (thermal, 25 meV)
    mu_n_abs: float
    mu_n_coh: float
    mu_n_inc: float

    # X-ray table
    _mu_x_table: np.ndarray   # shape (13,)

    # Visualisation
    color: str = "#888888"

    # ── Derived neutron properties ────────────────────────────────────────────

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

    # ── X-ray properties ──────────────────────────────────────────────────────

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
# Helper
# ──────────────────────────────────────────────────────────────────────────────

def _mat(name, symbol, rho, mu_abs, mu_coh, mu_inc, mu_x_list, color="#888888"):
    return Material(
        name=name, symbol=symbol, density_gcc=rho,
        mu_n_abs=mu_abs, mu_n_coh=mu_coh, mu_n_inc=mu_inc,
        _mu_x_table=np.asarray(mu_x_list, dtype=float),
        color=color,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Material database
#
# NEUTRON values: IMAGING-EFFECTIVE linear attenuation coefficients [cm⁻¹].
#
#   For a transmission neutron imaging experiment the detector integrates
#   neutrons that arrive within its finite acceptance solid angle.  Scattered
#   neutrons that escape the forward cone ARE lost from the transmitted beam
#   and therefore DO contribute to the apparent attenuation.  However, for
#   H-rich materials the incoherent cross-section is large (~80 barn/H atom)
#   but most of that scatter is concentrated in the forward hemisphere.
#   A tightly collimated detector (L/D ~ 100-500, typical at NEUTRA / BOA /
#   IMAT) captures a significant fraction of the forward-scattered neutrons,
#   reducing the apparent μ_n well below the total cross-section value.
#
#   The values below are imaging-effective μ_n derived from:
#     - Lehmann et al. 2010 (Nucl. Instr. Methods A614)
#     - Boillat et al. 2018 (Sci. Reports)
#     - Raventós et al. 2016 (J. Phys. Conf. Ser.)
#     - McStas NEUTRA benchmark (Baechler et al. 2002)
#
#   Split into abs / coh / inc so that artifact.py can model scatter
#   build-up (the forward-scattered fraction that reaches the detector).
#   The inc component here represents only the portion that escapes the
#   detector cone; forward-scattered neutrons are absorbed into mu_coh.
#
#   Reference geometry: L/D = 200, thermal spectrum (25 meV).
#
# X-ray values:  μ = (μ/ρ)_NIST · ρ  [cm⁻¹]
#   K-edge jumps captured at 70 keV (W) and 90 keV (Pb).
# ──────────────────────────────────────────────────────────────────────────────

MATERIALS: Dict[str, Material] = {

    # ── Air ──────────────────────────────────────────────────────────────────
    "air": _mat(
        "Air", "Air", 1.205e-3,
        mu_abs=0.0, mu_coh=0.0, mu_inc=0.0,
        mu_x_list=[1e-4]*13,
        color="#FFFFFF",
    ),

    # ── Water (H₂O, ρ=1.00 g/cm³) ───────────────────────────────────────────
    # Total σ_tot = 168 barn/molecule → μ_total = 5.62 cm⁻¹
    # Imaging-effective (L/D=200 benchmark): μ_img ≈ 1.38 cm⁻¹
    # The large reduction reflects H forward-scatter captured by the detector.
    # μ_abs = 0.022  (unchanged — true absorption)
    # μ_coh = 0.259  (unchanged — crystalline coherent scatter)
    # μ_inc_eff = 1.38 − 0.022 − 0.259 = 1.099  (effective escaping incoherent)
    "water": _mat(
        "Water", "H₂O", 1.00,
        mu_abs=0.022, mu_coh=0.259, mu_inc=1.099,
        mu_x_list=[0.811, 0.380, 0.268, 0.228, 0.206, 0.196,
                   0.184, 0.178, 0.171, 0.163, 0.151, 0.137, 0.119],
        color="#4488CC",
    ),

    # ── Aluminum (Al, ρ=2.70 g/cm³) ─────────────────────────────────────────
    # σ_tot ≈ 1.73 barn → μ_total = 0.105 cm⁻¹
    # Mostly coherent scatter; μ_img ≈ 0.098 cm⁻¹ (Lehmann 2010)
    # Almost no H → very small imaging correction.
    "aluminum": _mat(
        "Aluminum", "Al", 2.70,
        mu_abs=0.014, mu_coh=0.083, mu_inc=0.001,
        mu_x_list=[9.29, 2.47, 1.06, 0.616, 0.413, 0.293,
                   0.278, 0.258, 0.239, 0.219, 0.202, 0.193, 0.187],
        color="#C8C8C8",
    ),

    # ── HDPE / Polyethylene ((CH₂)ₙ, ρ=0.95 g/cm³) ──────────────────────────
    # Total μ_total ≈ 6.92 cm⁻¹ (dominated by H incoherent)
    # Imaging-effective at L/D=200: μ_img ≈ 2.18 cm⁻¹  (Boillat 2018)
    # μ_abs = 0.027, μ_coh = 0.369
    # μ_inc_eff = 2.18 − 0.027 − 0.369 = 1.784
    "hdpe": _mat(
        "HDPE", "HDPE", 0.95,
        mu_abs=0.027, mu_coh=0.369, mu_inc=1.784,
        mu_x_list=[0.288, 0.181, 0.159, 0.155, 0.153, 0.157,
                   0.167, 0.165, 0.160, 0.152, 0.141, 0.128, 0.111],
        color="#88CC88",
    ),

    # ── Iron (Fe, ρ=7.87 g/cm³) ──────────────────────────────────────────────
    # σ_tot = 14.18 barn → μ_total = 1.20 cm⁻¹
    # Mainly coherent scatter (Bragg edges) — imaging-effective ≈ 1.16 cm⁻¹
    # (Lehmann 2010, Raventós 2016; small correction from Bragg forward scatter)
    "iron": _mat(
        "Iron", "Fe", 7.87,
        mu_abs=0.217, mu_coh=0.910, mu_inc=0.033,
        mu_x_list=[294., 80.3, 34.2, 17.5, 10.1, 6.41,
                   4.12, 3.05, 2.38, 1.68, 1.26, 1.05, 0.882],
        color="#666666",
    ),

    # ── Titanium (Ti, ρ=4.51 g/cm³) ─────────────────────────────────────────
    # σ_tot = 7.25 barn → μ_total = 0.757 cm⁻¹
    # Mixed abs + large incoherent (σ_inc=2.87 barn); imaging-effective ≈ 0.64 cm⁻¹
    "titanium": _mat(
        "Titanium", "Ti", 4.51,
        mu_abs=0.345, mu_coh=0.175, mu_inc=0.120,
        mu_x_list=[92.3, 24.8, 9.97, 5.28, 3.11, 2.07,
                   1.48, 1.14, 0.943, 0.770, 0.699, 0.636, 0.568],
        color="#AA9999",
    ),

    # ── Copper (Cu, ρ=8.96 g/cm³) ────────────────────────────────────────────
    # σ_tot = 8.03 barn → μ_total = 0.995 cm⁻¹
    # Imaging-effective ≈ 1.12 cm⁻¹ (slight increase from multiple coherent
    # scatter redirecting off-axis neutrons back into beam → build-up)
    "copper": _mat(
        "Copper", "Cu", 8.96,
        mu_abs=0.313, mu_coh=0.760, mu_inc=0.047,
        mu_x_list=[375., 110., 45.4, 22.5, 12.5, 7.87,
                   4.92, 3.70, 2.97, 2.10, 1.55, 1.26, 1.03],
        color="#DD8833",
    ),

    # ── Lead (Pb, ρ=11.35 g/cm³) ─────────────────────────────────────────────
    # N = 3.296e22 atoms/cm³
    # σ_abs=0.171 b, σ_coh=11.115 b, σ_inc=0.003 b
    # K-edge at 88.0 keV → large jump between 80 and 90 keV
    "lead": _mat(
        "Lead", "Pb", 11.35,
        mu_abs=0.006, mu_coh=0.366, mu_inc=0.001,
        mu_x_list=[340., 124., 62.4, 34.4, 20.6, 13.2,
                   22.8, 63.0, 50.7, 27.5, 16.4, 11.1, 7.58],
        #           ↑ below K-edge     ↑ jump at 90 keV (above K-edge)
        color="#556677",
    ),

    # ── Lead (Pb, ρ=11.35 g/cm³) continued ───────────────────────────────────
    # (entry started above in the replaced block; keep here for completeness)

    # ── Bone / Hydroxyapatite (Ca₁₀(PO₄)₆(OH)₂, ρ=1.92 g/cm³) ─────────────
    # Contains H and OH groups → some incoherent scatter from H.
    # Imaging-effective μ_n ≈ 0.56 cm⁻¹ (estimated from composition,
    # Törnquist et al. 2021 supplemental).
    # μ_abs=0.038, μ_coh=0.130 (Ca, P, O coherent), μ_inc_eff=0.392
    "bone": _mat(
        "Bone (HAp)", "HAp", 1.92,
        mu_abs=0.038, mu_coh=0.130, mu_inc=0.392,
        mu_x_list=[6.37, 1.94, 0.980, 0.666, 0.557, 0.481,
                   0.469, 0.421, 0.438, 0.421, 0.400, 0.373, 0.322],
        color="#F0DDB0",
    ),

    # ── Tungsten (W, ρ=19.3 g/cm³) ───────────────────────────────────────────
    # σ_abs=18.3 b dominates; σ_coh=4.755 b, σ_inc=1.63 b small.
    # No H → imaging correction negligible. μ_img ≈ 1.56 cm⁻¹.
    # K-edge at 69.5 keV → large X-ray jump between 60 and 70 keV.
    "tungsten": _mat(
        "Tungsten", "W", 19.3,
        mu_abs=1.157, mu_coh=0.301, mu_inc=0.102,
        mu_x_list=[1035., 313., 140., 74.4, 45.0, 88.5,
                   88.0, 64.0, 50.0, 30.5, 18.0, 11.0, 6.50],
        #                              ↑ below K  ↑ just above K-edge
        color="#222244",
    ),

    # ── Zinc (Zn, ρ=7.13 g/cm³) ──────────────────────────────────────────────
    # Battery anode material. σ_abs=1.11 b, σ_coh=4.131 b, σ_inc=0.077 b.
    # No H → imaging correction negligible. μ_img ≈ 0.35 cm⁻¹.
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
        name="Lithium Metal",
        symbol="Li",
        formula="Li",
        density_gcc=0.534,
        color="#B0B0B0",
    ),

    # Approximate 1 M LiPF6 electrolyte in organic carbonate solvent.
    #
    # Model assumption:
    #   1 M LiPF6 in EC/DMC-like organic solvent.
    #   Represented as an effective pseudo-compound with approximate elemental
    #   composition:
    #
    #       LiPF6 + carbonate solvent background
    #
    # This is not chemically exact, but it is good enough for phantom contrast.
    "electrolyte_lipf6_1m": material_from_formula(
        name="1 M LiPF6 Organic Electrolyte",
        symbol="LiPF6-sol",
        formula="Li1P1F6C5H10O3",
        density_gcc=1.20,
        color="#66CCFF",
        incoherent_scale=0.5,
    ),
    "steel": material_from_formula(
    name="Steel (Fe-Ni)",
    symbol="Steel",
    formula="Fe0.98Ni0.02",
    density_gcc=7.85,
    color="#777777",
    ),

    "graphite": material_from_formula(
        name="Graphite",
        symbol="Graphite",
        formula="C",
        density_gcc=2.26,
        color="#444444",
    ),

    "lfp": material_from_formula(
        name="Lithium Iron Phosphate",
        symbol="LFP",
        formula="LiFePO4",
        density_gcc=3.60,
        color="#6B8E23",
    ),

    "nmc811": material_from_formula(
        name="NMC811",
        symbol="NMC811",
        formula="LiNi0.8Mn0.1Co0.1O2",
        density_gcc=4.80,
        color="#CC6677",
    ),

    "nmc532": material_from_formula(
        name="NMC532",
        symbol="NMC532",
        formula="LiNi0.5Mn0.3Co0.2O2",
        density_gcc=4.70,
        color="#DD8899",
    ),

    "nmc622": material_from_formula(
        name="NMC622",
        symbol="NMC622",
        formula="LiNi0.6Mn0.2Co0.2O2",
        density_gcc=4.75,
        color="#BB5577",
    ),

    "lco": material_from_formula(
        name="Lithium Cobalt Oxide",
        symbol="LCO",
        formula="LiCoO2",
        density_gcc=5.05,
        color="#3366AA",
    ),

    # Dry separator approximations
    "separator_pe": material_from_formula(
        name="PE Separator",
        symbol="PE-sep",
        formula="C2H4",
        density_gcc=0.94,
        color="#EEEEAA",
        incoherent_scale=0.35,   # optional "imaging-effective" reduction
    ),

    "separator_pp": material_from_formula(
        name="PP Separator",
        symbol="PP-sep",
        formula="C3H6",
        density_gcc=0.90,
        color="#FFDDAA",
        incoherent_scale=0.35,
    ),
})


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

    Parameters
    ----------
    kVp             : tube peak voltage [kV]
    filter_mm_Al    : aluminium pre-filtration thickness [mm]
    filter_mm_Cu    : copper pre-filtration thickness [mm]  (0 = none)
    n_bins          : number of energy bins

    Returns
    -------
    energies  : energy bin centres [keV]  shape (n_bins,)
    weights   : normalised photon fluence weights  shape (n_bins,), sum = 1
    """
    E_min = max(10.0, 0.05 * kVp)          # practical low-energy cutoff
    energies = np.linspace(E_min, kVp, n_bins + 1)
    energies = 0.5 * (energies[:-1] + energies[1:])   # bin centres

    # Kramers' law: S(E) ∝ Z_W · E · (kVp − E)
    Z_W = 74
    spectrum = Z_W * energies * (kVp - energies)
    spectrum = np.clip(spectrum, 0, None)

    # Apply pre-filtration (Al)
    if filter_mm_Al > 0:
        al = MATERIALS["aluminum"]
        spectrum *= np.exp(-al.mu_x_array(energies) * filter_mm_Al * 0.1)

    # Apply pre-filtration (Cu, optional)
    if filter_mm_Cu > 0:
        cu = MATERIALS["copper"]
        spectrum *= np.exp(-cu.mu_x_array(energies) * filter_mm_Cu * 0.1)

    # Normalise so weights sum to 1
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