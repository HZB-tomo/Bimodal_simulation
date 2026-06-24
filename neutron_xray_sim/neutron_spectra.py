"""
neutron_xray_sim.neutron_spectra
═════════════════════════════════
Energy-dependent neutron cross sections and beam spectra for cold and
polychromatic neutron tomography simulations.

The original package uses a single thermal-neutron attenuation value
mu_n = N * Σ_atoms(n_i * σ_bound_i) evaluated once at 25.3 meV.  This
module adds the energy dependence needed to simulate:

  1. Monochromatic cold neutrons at any energy E (meV)
  2. Polychromatic cold beams with a Maxwell-Boltzmann-like spectrum
  3. The ILL-NeXT instrument spectrum specifically

Physics
───────
Total macroscopic attenuation at neutron energy E:

  Σ(E) = N/Mw · Σ_atoms [ n_i · σ_total_i(E) ]

where the energy dependence of each partial cross section is:

  σ_abs(E)   = σ_abs(E₀) · √(E₀/E)          [1/v law, exact for non-resonance]
  σ_coh(E)   ≈ σ_coh_bound                    [energy-independent below Bragg cutoff
                                                 for fine-grained crystalline / amorphous
                                                 materials; Bragg edges are NOT modelled
                                                 here — add them via sigma_coh_with_bragg
                                                 if needed]
  σ_inc(E)   ≈ σ_inc_bound                    [slowly varying; Debye-Waller correction
                                                 is < 5% for T > 100 K and E < 25 meV]

E₀ = 25.3 meV  (reference thermal energy, 2 200 m/s)

For polychromatic beams the effective attenuation is computed as a
flux-weighted average over the spectrum:

  Σ_eff = ∫ φ(E) · Σ(E) dE  /  ∫ φ(E) dE

This is valid in the thin-sample (single-scattering) limit.  For thick
samples or high-scattering materials the proper treatment is
transmission-weighted:  Σ_eff = -ln(∫ φ(E) exp(-Σ(E)·t) dE) / t  —
but that requires knowing t, which is not available at the projection
stage.  The flux-weighted average is used throughout here and the
approximation is noted in the simulation.

ILL-NeXT spectrum
─────────────────
NeXT (Neutron and X-ray Tomography) at ILL H18 cold guide:
  Moderator: liquid-deuterium cold source
  Effective moderator temperature: ~35 K (kT ≈ 3.0 meV)
  Useful wavelength range: 1–12 Å (0.6–80 meV)
  Flux peak: ~3–5 meV
  Reference: Tengattini et al. (2020) Rev. Sci. Instrum. 91, 045103

The spectrum is approximated as:
  φ(E) = E · exp(-E / kT_eff) + ε · E⁻¹
          ───────────────────   ─────────
          Maxwell-Boltzmann     thermal tail (Be-filtered leakage)
          cold wing

with T_eff = 35 K and ε = 0.08 (8% thermal tail contribution).

Public API
──────────
  NEUTRON_MODES            — dict of pre-built NeutronBeam objects
  NeutronBeam              — dataclass: name, E_grid, phi, is_mono
  thermal_beam()           — monochromatic at 25.3 meV (baseline)
  cold_mono_beam(E_meV)    — monochromatic cold at E meV
  ill_next_beam()          — ILL-NeXT polychromatic cold spectrum
  cold_poly_beam(T_K, eps) — generic cold spectrum at moderator temperature T_K

  mu_n_at_energy(formula, density_gcc, E_meV)
      Σ(E) for one formula + density at a single energy

  mu_n_for_beam(formula, density_gcc, beam)
      Flux-weighted effective Σ for a NeutronBeam

  mu_n_lut_for_beam(materials, beam)
      Precompute a 1-D LUT: mu_n[label_index] for fast projection
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np

# ── NumPy 1.x / 2.x compatibility ────────────────────────────────────────────
# np.trapezoid introduced in 2.0; np.trapz removed in 2.0.
# This shim makes the code work on both versions.
if not hasattr(np, 'trapezoid'):
    np.trapezoid = np.trapz          # NumPy < 2.0  # type: ignore[attr-defined]

# ── Physical constants ────────────────────────────────────────────────────────
_E0_MEV  = 25.3          # thermal reference energy (meV)
_KB_EV_K = 8.617333e-5   # Boltzmann constant (eV/K)
_NA      = 6.02214076e23  # Avogadro (mol⁻¹)
_BARN    = 1e-24          # 1 barn in cm²
_HBAR    = 1.054571817e-34  # J·s
_MN      = 1.674927471e-27  # neutron mass (kg)
_EV2J    = 1.602176634e-19  # eV → J


# ── Bound-atom cross sections (barn) at E₀ = 25.3 meV ────────────────────────
# Source: Sears (1992) Neutron News 3(3):26-37; ENDF/B-VIII.0
# These are the same values as in materials.py NEUTRON_XS, reproduced here
# for the energy-dependent calculation.
SIGMA_BOUND: Dict[str, Dict[str, float]] = {
    'H':  {'abs': 0.3326,  'coh': 1.7568,  'inc': 80.27},
    'Li': {'abs': 70.5,    'coh': 0.454,   'inc': 0.92},
    'C':  {'abs': 0.0035,  'coh': 5.551,   'inc': 0.001},
    'N':  {'abs': 1.90,    'coh': 11.01,   'inc': 0.50},
    'O':  {'abs': 0.00019, 'coh': 4.232,   'inc': 0.0008},
    'F':  {'abs': 0.0096,  'coh': 4.017,   'inc': 0.0008},
    'Na': {'abs': 0.530,   'coh': 1.66,    'inc': 1.62},
    'Mg': {'abs': 0.063,   'coh': 3.631,   'inc': 0.08},
    'Al': {'abs': 0.231,   'coh': 1.495,   'inc': 0.0082},
    'Si': {'abs': 0.171,   'coh': 2.163,   'inc': 0.004},
    'P':  {'abs': 0.172,   'coh': 3.307,   'inc': 0.005},
    'S':  {'abs': 0.53,    'coh': 1.026,   'inc': 0.007},
    'K':  {'abs': 2.1,     'coh': 1.69,    'inc': 0.27},
    'Ca': {'abs': 0.43,    'coh': 2.830,   'inc': 0.05},
    'Fe': {'abs': 2.56,    'coh': 11.22,   'inc': 0.40},
    'Co': {'abs': 37.18,   'coh': 0.779,   'inc': 4.8},
    'Ni': {'abs': 4.49,    'coh': 13.3,    'inc': 5.2},
    'Cu': {'abs': 3.78,    'coh': 7.485,   'inc': 0.55},
    'Mn': {'abs': 13.3,    'coh': 2.15,    'inc': 0.40},
    'Pb': {'abs': 0.171,   'coh': 11.115,  'inc': 0.003},
    'Ti': {'abs': 6.09,    'coh': 1.485,   'inc': 2.87},
    'Zn': {'abs': 1.11,    'coh': 4.054,   'inc': 0.077},
    'W':  {'abs': 18.3,    'coh': 2.97,    'inc': 1.63},
}

# Atomic masses (g/mol) — IUPAC 2021
ATOMIC_MASS: Dict[str, float] = {
    'H':   1.008,  'Li': 6.941,   'C':  12.011,  'N':  14.007,
    'O':  15.999,  'F': 18.998,   'Na': 22.990,  'Mg': 24.305,
    'Al': 26.982,  'Si': 28.086,  'P':  30.974,  'S':  32.06,
    'K':  39.098,  'Ca': 40.078,  'Ti': 47.867,  'Fe': 55.845,
    'Co': 58.933,  'Ni': 58.693,  'Cu': 63.546,  'Zn': 65.38,
    'Mn': 54.938,  'Pb': 207.2,   'W': 183.84,
}


# ──────────────────────────────────────────────────────────────────────────────
#  NeutronBeam dataclass
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class NeutronBeam:
    """
    Description of a neutron beam for simulation.

    Attributes
    ----------
    name        : human-readable label
    E_grid_meV  : energy grid (meV), 1-D array, monotonically increasing
    phi         : normalised flux weights on E_grid (sums to 1 over E_grid)
    is_mono     : True if this is effectively monochromatic (single-energy)
    E_mean_meV  : flux-weighted mean energy (computed on construction)
    lambda_mean_angstrom : flux-weighted mean wavelength (Å)
    description : one-line note on the beam source / parameters
    """
    name:        str
    E_grid_meV:  np.ndarray
    phi:         np.ndarray
    is_mono:     bool = False
    description: str  = ""
    E_mean_meV:  float = field(init=False)
    lambda_mean_angstrom: float = field(init=False)

    def __post_init__(self):
        if self.E_grid_meV.shape != self.phi.shape:
            raise ValueError("E_grid_meV and phi must have the same shape")
        # Re-normalise in place to absorb floating-point drift from the
        # calling constructor (trapezoid vs sum normalisation mismatch).
        phi_sum = float(self.phi.sum())
        if phi_sum <= 0:
            raise ValueError("phi must contain positive values")
        self.phi = self.phi / phi_sum

        if len(self.E_grid_meV) == 1:
            # Monochromatic: single-point trapezoid is undefined; use value directly.
            self.E_mean_meV = float(self.E_grid_meV[0])
            self.lambda_mean_angstrom = float(
                _energy_meV_to_lambda_angstrom(self.E_grid_meV)[0]
            )
        else:
            denom = float(np.trapezoid(self.phi, self.E_grid_meV))
            self.E_mean_meV = float(
                np.trapezoid(self.phi * self.E_grid_meV, self.E_grid_meV) / denom
            )
            lam = _energy_meV_to_lambda_angstrom(self.E_grid_meV)
            self.lambda_mean_angstrom = float(
                np.trapezoid(self.phi * lam, self.E_grid_meV) / denom
            )

    def __repr__(self) -> str:
        kind = "mono" if self.is_mono else "poly"
        return (f"NeutronBeam({self.name!r}, {kind}, "
                f"<E>={self.E_mean_meV:.2f} meV, "
                f"<λ>={self.lambda_mean_angstrom:.2f} Å)")


# ──────────────────────────────────────────────────────────────────────────────
#  Conversion helpers
# ──────────────────────────────────────────────────────────────────────────────

def _energy_meV_to_lambda_angstrom(E_meV: np.ndarray) -> np.ndarray:
    """Convert neutron kinetic energy (meV) to de Broglie wavelength (Å)."""
    E_J = np.asarray(E_meV, dtype=float) * 1e-3 * _EV2J
    v   = np.sqrt(2.0 * E_J / _MN)
    return (_HBAR * 2.0 * np.pi) / (_MN * v) * 1e10  # Å


# ──────────────────────────────────────────────────────────────────────────────
#  Beam constructors
# ──────────────────────────────────────────────────────────────────────────────

def thermal_beam() -> NeutronBeam:
    """
    Monochromatic thermal neutron beam at 25.3 meV (v = 2 200 m/s, λ = 1.798 Å).

    This matches the baseline used throughout the package (mu_n stored in
    Material objects).  Use this beam to reproduce the existing simulation.
    """
    E = np.array([_E0_MEV])
    return NeutronBeam(
        name="thermal_mono",
        E_grid_meV=E,
        phi=np.array([1.0]),
        is_mono=True,
        description="Monochromatic thermal neutrons at 25.3 meV (2200 m/s standard)",
    )


def cold_mono_beam(E_meV: float = 5.0) -> NeutronBeam:
    """
    Monochromatic cold neutron beam at a user-specified energy.

    Parameters
    ----------
    E_meV : float, default 5.0
        Beam energy in meV.  Cold neutrons are typically 0.5–10 meV.
        Common choices:
          2 meV  → λ = 6.4 Å  (very cold, long wavelength)
          5 meV  → λ = 4.0 Å  (cold source peak, e.g. ILL H18 guide)
         10 meV  → λ = 2.9 Å  (near thermal-cold boundary)
    """
    if E_meV <= 0:
        raise ValueError(f"E_meV must be > 0, got {E_meV}")
    lam = float(_energy_meV_to_lambda_angstrom(np.array([E_meV]))[0])
    return NeutronBeam(
        name=f"cold_mono_{E_meV:.1f}meV",
        E_grid_meV=np.array([float(E_meV)]),
        phi=np.array([1.0]),
        is_mono=True,
        description=f"Monochromatic cold neutrons at {E_meV:.1f} meV (λ = {lam:.2f} Å)",
    )


def ill_next_beam(
    n_bins:       int   = 200,
    E_min_meV:    float = 0.5,
    E_max_meV:    float = 25.0,
) -> NeutronBeam:
    """
    Approximate ILL-NeXT polychromatic cold spectrum.

    NeXT (Neutron and X-ray Tomography beamline) at ILL uses the H18 cold
    neutron guide fed by the liquid-deuterium cold source.

    Spectrum model
    ──────────────
    φ(E) = E · exp(-E / kT_eff)  +  ε · E⁻¹
             ──────────────────     ─────────
             Maxwell-Boltzmann      thermal tail
             cold wing

    Parameters: T_eff = 35 K (kT ≈ 3.0 meV), ε = 0.08
    These reproduce the published flux distribution shape from:
      Tengattini et al. (2020) Rev. Sci. Instrum. 91, 045103

    Note: the absolute flux calibration is irrelevant for CT simulation
    (only the normalised spectral shape φ(E)/∫φ matters).

    Parameters
    ----------
    n_bins       : number of energy bins in the integration grid
    E_min_meV    : lower integration limit (meV); default 0.5
    E_max_meV    : upper integration limit (meV); default 25.0
    """
    return cold_poly_beam(
        T_K=35.0, epsilon=0.08,
        n_bins=n_bins, E_min_meV=E_min_meV, E_max_meV=E_max_meV,
        name="ILL-NeXT",
        description=(
            "ILL-NeXT cold guide H18: T_eff=35 K, kT=3.0 meV, 8% thermal tail. "
            "Ref: Tengattini et al. (2020) RSI 91, 045103."
        ),
    )


def cold_poly_beam(
    T_K:        float  = 35.0,
    epsilon:    float  = 0.08,
    n_bins:     int    = 200,
    E_min_meV:  float  = 0.5,
    E_max_meV:  float  = 25.0,
    name:       Optional[str] = None,
    description: str  = "",
) -> NeutronBeam:
    """
    Generic polychromatic cold neutron spectrum.

    φ(E) ∝ E · exp(-E / kT)  +  ε · E⁻¹

    Parameters
    ----------
    T_K        : effective moderator temperature in Kelvin
    epsilon    : fractional thermal tail contribution (0 = pure cold M-B)
    n_bins     : number of energy points on the integration grid
    E_min_meV  : lower energy bound (meV)
    E_max_meV  : upper energy bound (meV)
    name       : beam identifier (auto-generated if None)
    description: one-line source description
    """
    kT = _KB_EV_K * T_K * 1e3  # meV
    E  = np.linspace(E_min_meV, E_max_meV, n_bins)

    phi_cold    = E * np.exp(-E / kT)
    phi_thermal = epsilon / E
    phi         = phi_cold + phi_thermal
    phi        /= np.trapezoid(phi, E)  # normalise to 1

    if name is None:
        name = f"cold_poly_T{T_K:.0f}K"
    if not description:
        description = (
            f"Maxwell-Boltzmann cold spectrum T={T_K:.0f} K "
            f"(kT={kT:.2f} meV) + {epsilon*100:.0f}% thermal tail"
        )

    return NeutronBeam(
        name=name,
        E_grid_meV=E,
        phi=phi,
        is_mono=False,
        description=description,
    )


# ──────────────────────────────────────────────────────────────────────────────
#  Cross-section calculations
# ──────────────────────────────────────────────────────────────────────────────

def mu_n_at_energy(
    formula:     Dict[str, float],
    density_gcc: float,
    E_meV:       float,
) -> float:
    """
    Macroscopic total neutron attenuation Σ (cm⁻¹) at energy E_meV.

    Parameters
    ----------
    formula     : {element_symbol: count_per_formula_unit}
    density_gcc : bulk density (g/cm³)
    E_meV       : neutron energy (meV)

    Returns
    -------
    Σ (cm⁻¹)

    Physics
    -------
    Σ(E) = (ρ·Nₐ/Mw) · Σ_atoms [ nᵢ · (σ_abs,i·√(E₀/E) + σ_coh,i + σ_inc,i) ]

    The 1/v law for absorption is exact for s-wave non-resonance capture.
    Coherent and incoherent scattering cross sections are taken as their
    bound-atom values (energy-independent below the Debye-Waller regime).
    """
    Mw    = sum(n * ATOMIC_MASS[el] for el, n in formula.items())
    N_mol = density_gcc * _NA / Mw  # formula units/cm³
    scale = np.sqrt(_E0_MEV / E_meV)

    sigma = 0.0
    for el, n in formula.items():
        if el not in SIGMA_BOUND:
            raise KeyError(
                f"Element '{el}' not in SIGMA_BOUND table. "
                f"Add it to neutron_spectra.SIGMA_BOUND."
            )
        s = SIGMA_BOUND[el]
        sigma += n * (s['abs'] * scale + s['coh'] + s['inc'])

    return N_mol * sigma * _BARN


def mu_n_for_beam(
    formula:     Dict[str, float],
    density_gcc: float,
    beam:        NeutronBeam,
) -> float:
    """
    Flux-weighted effective macroscopic attenuation for a NeutronBeam.

    Σ_eff = ∫ φ(E) · Σ(E) dE  /  ∫ φ(E) dE

    For monochromatic beams (is_mono=True) this reduces to Σ(E_beam).

    Parameters
    ----------
    formula     : {element_symbol: count_per_formula_unit}
    density_gcc : bulk density (g/cm³)
    beam        : NeutronBeam object

    Returns
    -------
    Σ_eff (cm⁻¹)
    """
    if beam.is_mono:
        return mu_n_at_energy(formula, density_gcc, float(beam.E_grid_meV[0]))

    mu_arr = np.array([
        mu_n_at_energy(formula, density_gcc, E)
        for E in beam.E_grid_meV
    ])
    return float(
        np.trapezoid(beam.phi * mu_arr, beam.E_grid_meV)
        / np.trapezoid(beam.phi, beam.E_grid_meV)
    )


def mu_n_lut_for_beam(
    materials: Sequence,
    beam:      NeutronBeam,
    label_to_mat: Optional[Dict[int, object]] = None,
) -> np.ndarray:
    """
    Build a 1-D LUT  mu_n_lut[label_index]  for fast projection.

    Parameters
    ----------
    materials    : list of Material objects (index = label value), OR
    label_to_mat : dict {label_int → Material} (takes priority if provided)
    beam         : NeutronBeam to evaluate at

    Returns
    -------
    np.ndarray, shape (n_labels,), dtype float32

    Usage
    -----
    >>> lut = mu_n_lut_for_beam([mat_air, mat_matrix, mat_bone], ill_next_beam())
    >>> mu_slice = lut[label_vol[iz]]   # shape (ny, nx), dtype float32
    """
    if label_to_mat is not None:
        n = max(label_to_mat.keys()) + 1
        lut = np.zeros(n, dtype=np.float32)
        for idx, mat in label_to_mat.items():
            lut[idx] = _mu_n_material_beam(mat, beam)
        return lut

    lut = np.zeros(len(materials), dtype=np.float32)
    for i, mat in enumerate(materials):
        lut[i] = _mu_n_material_beam(mat, beam)
    return lut



def mu_n_spectrum_lut(materials, beam: NeutronBeam, n_bins=None):
    """
    Per-energy neutron attenuation LUT for a *polychromatic* forward projector.

    This mirrors the X-ray (xray_spectrum + Material.mu_x_at) pattern on the
    neutron side, so a projector can model neutron spectral self-shielding (the
    neutron analogue of X-ray beam hardening):

        T(L)        = Σ_k w_k · exp(-mu_n(E_k) · L)
        mu_apparent = -ln(T(L)) / L

    A strong 1/v absorber (e.g. indium) preferentially removes the coldest
    neutrons, hardening the transmitted spectrum, so the apparent mu_n through a
    thick region is LOWER than the flux-weighted value returned by
    mu_n_lut_for_beam (which is the thin-sample / no-self-shielding limit).

    Parameters
    ----------
    materials : sequence of Material objects (index == label value)
    beam      : NeutronBeam
    n_bins    : if given and the spectrum has more points, rebin to n_bins groups
                (projector efficiency; None = use the full native grid)

    Returns
    -------
    E_grid_meV : np.ndarray (n_E,)   beam energy grid (meV)
    weights    : np.ndarray (n_E,)   DISCRETE flux weights, sum == 1, trapezoidal
                 over E_grid so that  Σ_k w_k·mu_n(E_k)  exactly reproduces
                 mu_n_lut_for_beam / _mu_n_material_beam in the thin-sample limit.
    mu_lut     : np.ndarray (n_mat, n_E) float32   mu_n(E_k) per material [cm^-1],
                 mu_n(E) = mu_n_abs·sqrt(E0/E) + mu_n_coh + mu_n_inc.

    For monochromatic beams (n_E == 1) this returns weights = [1.0] and the
    projector reduces exactly to the existing single-mu neutron projection.
    """
    E = np.asarray(beam.E_grid_meV, dtype=float)
    n_E = E.size
    if n_E == 1:
        w = np.array([1.0], dtype=float)
    else:
        phi = np.asarray(beam.phi, dtype=float)
        omega = np.empty(n_E)                       # trapezoidal node weights
        omega[0]    = 0.5 * (E[1] - E[0])
        omega[-1]   = 0.5 * (E[-1] - E[-2])
        omega[1:-1] = 0.5 * (E[2:] - E[:-2])
        w = phi * omega
        w = w / w.sum()

    # Optional rebinning to n_bins contiguous groups for projector efficiency
    # (mirrors xray_spectrum's n_bins). Effective energy per group = flux-weighted
    # mean; group weight = summed flux. Conserves total flux; thin-limit consistency
    # is then approximate (as for any finite-bin polychromatic projector).
    if n_bins is not None and E.size > n_bins:
        groups = np.array_split(np.arange(E.size), n_bins)
        E_r, w_r = [], []
        for g in groups:
            wg = w[g].sum()
            Eg = float(np.average(E[g], weights=w[g])) if wg > 0 else float(E[g].mean())
            E_r.append(Eg); w_r.append(wg)
        E = np.asarray(E_r, dtype=float)
        w = np.asarray(w_r, dtype=float); w = w / w.sum()

    scale = np.sqrt(_E0_MEV / E)                     # 1/v absorption factor / energy
    mu_lut = np.zeros((len(materials), E.size), dtype=np.float32)
    for i, m in enumerate(materials):
        mu_lut[i, :] = m.mu_n_abs * scale + m.mu_n_coh + m.mu_n_inc
    return E, w.astype(np.float32), mu_lut


def _mu_n_material_beam(mat, beam: NeutronBeam) -> float:
    """
    Compute mu_n for one Material object at beam energy/spectrum.

    The Material dataclass stores mu_n_abs, mu_n_coh, mu_n_inc at the thermal
    reference energy (25.3 meV).  For a polychromatic or cold-mono beam we
    need to recompute from first principles.

    If the Material has a `_formula_components` attribute (set by
    make_composite_material when beam-awareness is enabled), we use the
    component formulas directly.  Otherwise we fall back to the 1/v
    scaling of the absorption term plus the energy-independent scatter terms.
    """
    if beam.is_mono and np.isclose(float(beam.E_grid_meV[0]), _E0_MEV, rtol=1e-3):
        # Thermal reference — return stored value directly
        return float(mat.mu_n)

    # Fallback: reconstruct from the three stored macroscopic components.
    # mu_n_abs = N * Σ n_i * σ_abs_i  (at E0)
    # → at energy E: mu_n_abs(E) = mu_n_abs(E0) * √(E0/E)
    # mu_n_coh, mu_n_inc are energy-independent (bound-atom values)
    def _scale_at_E(E_meV):
        return (mat.mu_n_abs * np.sqrt(_E0_MEV / E_meV)
                + mat.mu_n_coh
                + mat.mu_n_inc)

    if beam.is_mono:
        return float(_scale_at_E(float(beam.E_grid_meV[0])))

    # Polychromatic: integrate
    mu_arr = np.array([_scale_at_E(E) for E in beam.E_grid_meV])
    return float(
        np.trapezoid(beam.phi * mu_arr, beam.E_grid_meV)
        / np.trapezoid(beam.phi, beam.E_grid_meV)
    )


# ──────────────────────────────────────────────────────────────────────────────
#  Pre-built mode catalogue
# ──────────────────────────────────────────────────────────────────────────────

NEUTRON_MODES: Dict[str, NeutronBeam] = {
    "thermal":    thermal_beam(),
    "cold_5meV":  cold_mono_beam(5.0),
    "cold_2meV":  cold_mono_beam(2.0),
    "ill_next":   ill_next_beam(),
}
"""
Pre-built NeutronBeam objects for common simulation scenarios.

Keys
────
"thermal"   : monochromatic 25.3 meV — matches existing package baseline
"cold_5meV" : monochromatic 5 meV    — cold source peak, λ = 4.0 Å
"cold_2meV" : monochromatic 2 meV    — very cold, λ = 6.4 Å
"ill_next"  : ILL-NeXT polychromatic cold spectrum

Add custom beams with cold_mono_beam() or cold_poly_beam().
"""


# ──────────────────────────────────────────────────────────────────────────────
#  Plotting helper
# ──────────────────────────────────────────────────────────────────────────────

def plot_spectra(
    beams:   Optional[Dict[str, NeutronBeam]] = None,
    figsize: tuple = (8, 4),
) -> "plt.Figure":  # noqa: F821
    """
    Plot flux vs energy for one or more NeutronBeam objects.

    Parameters
    ----------
    beams : dict {label: NeutronBeam}; defaults to NEUTRON_MODES
    """
    import matplotlib.pyplot as plt

    if beams is None:
        beams = NEUTRON_MODES

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    colors = plt.cm.tab10(np.linspace(0, 0.9, len(beams)))

    for (label, beam), col in zip(beams.items(), colors):
        if beam.is_mono:
            ax1.axvline(beam.E_grid_meV[0], color=col, lw=2,
                        label=f"{label} ({beam.E_grid_meV[0]:.1f} meV)")
            ax2.axvline(beam.lambda_mean_angstrom, color=col, lw=2,
                        label=f"{label} ({beam.lambda_mean_angstrom:.2f} Å)")
        else:
            ax1.plot(beam.E_grid_meV, beam.phi, color=col, lw=2, label=label)
            lam = _energy_meV_to_lambda_angstrom(beam.E_grid_meV)
            # φ(λ) = φ(E) · |dE/dλ|; plot normalised shape
            dphi_dlam = np.abs(np.gradient(beam.phi, lam))
            ax2.plot(lam, dphi_dlam / max(dphi_dlam.max(), 1e-30),
                     color=col, lw=2, label=label)

    ax1.set_xlabel("Energy (meV)", fontsize=11)
    ax1.set_ylabel("Normalised flux φ(E)", fontsize=10)
    ax1.set_title("Neutron spectra — energy domain")
    ax1.legend(fontsize=8)
    ax1.set_xlim(0, 30)
    ax1.grid(True, alpha=0.3)

    ax2.set_xlabel("Wavelength (Å)", fontsize=11)
    ax2.set_ylabel("Normalised flux φ(λ)  [a.u.]", fontsize=10)
    ax2.set_title("Neutron spectra — wavelength domain")
    ax2.legend(fontsize=8)
    ax2.set_xlim(0, 14)
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    return fig
