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

# ──────────────────────────────────────────────────────────────────────────────
# Standard X-ray energy grid (keV).  The two "extra" points at 70 and 90 keV
# straddle the W K-edge (69.5 keV) and the Pb K-edge (88.0 keV).
# ──────────────────────────────────────────────────────────────────────────────
XRAY_E_KEV = np.array(
    [20, 30, 40, 50, 60, 70, 80, 90, 100, 120, 150, 200, 300], dtype=float
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
