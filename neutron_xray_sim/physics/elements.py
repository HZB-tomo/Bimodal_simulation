"""
neutron_xray_sim.physics.elements
─────────────────────────────────
Single source of truth for per-element physical data.

* ``ATOMIC_MASS``     — standard atomic weights  [g / mol]
* ``NEUTRON_XS``      — bound-atom thermal-neutron cross sections at 25.3 meV
                        (``abs`` / ``coh`` / ``inc``)  [barn]
* ``xray_mass_atten`` — NIST XCOM mass attenuation μ/ρ on ``XRAY_E_KEV``
                        [cm² / g], loaded lazily from ``data/xray/<El>.txt``

Every other module (``materials``, ``neutron_spectra``, …) reads from here, so a
corrected value only ever needs to be changed in one place.

Adding an element
─────────────────
1. Add its atomic mass to ``ATOMIC_MASS``.
2. Add its bound cross sections to ``NEUTRON_XS`` (Sears 1992 is the reference).
3. Export the NIST XCOM table for the element (energies in MeV, total
   attenuation *with* coherent scattering in column 7) and save it as
   ``neutron_xray_sim/physics/data/xray/<El>.txt``.

Sources
───────
* Sears, V. F. (1992) *Neutron News* 3(3): 26-37; ENDF/B-VIII.0 at 25.3 meV.
* NIST XCOM photon cross-section database (Berger et al.).
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Dict

import numpy as np

__all__ = [
    "XRAY_E_KEV",
    "XRAY_DATA_DIR",
    "ATOMIC_MASS",
    "NEUTRON_XS",
    "AVOGADRO",
    "BARN_CM2",
    "E0_THERMAL_MEV",
    "xray_mass_atten",
    "available_xray_elements",
]

#: Standard X-ray energy grid [keV] on which every material stores μ_x.
XRAY_E_KEV = np.array(
    [20, 30, 40, 50, 60, 70, 80, 90, 100, 120, 150, 200, 300], dtype=float
)

#: Directory holding the NIST XCOM exports (one ``<El>.txt`` per element).
XRAY_DATA_DIR = Path(__file__).resolve().parent / "data" / "xray"

AVOGADRO = 6.02214076e23      # mol⁻¹
BARN_CM2 = 1e-24              # 1 barn in cm²
E0_THERMAL_MEV = 25.3         # thermal reference energy (2200 m/s)

# ── Atomic masses (g/mol) ────────────────────────────────────────────────────
ATOMIC_MASS: Dict[str, float] = {
    "H":  1.00794,
    "Li": 6.941,
    "C":  12.0107,
    "N":  14.0067,
    "O":  15.999,
    "F":  18.998403163,
    "Na": 22.990,
    "Mg": 24.305,
    "Al": 26.9815,
    "Si": 28.0855,
    "P":  30.973761998,
    "S":  32.06,
    "Cl": 35.45,
    "K":  39.0983,
    "Ca": 40.078,
    "Ti": 47.867,
    "Mn": 54.938044,
    "Fe": 55.845,
    "Co": 58.933194,
    "Ni": 58.6934,
    "Cu": 63.546,
    "Zn": 65.38,
    "In": 114.818,
    "W":  183.84,
    "Pb": 207.2,
}

# ── Bound-atom thermal-neutron cross sections (barn) at 25.3 meV ─────────────
# abs = absorption, coh = bound coherent scatter, inc = bound incoherent scatter
NEUTRON_XS: Dict[str, Dict[str, float]] = {
    "H":  {"abs": 0.3326,  "coh": 1.7568, "inc": 80.27},
    "Li": {"abs": 70.5,    "coh": 0.454,  "inc": 0.92},
    "C":  {"abs": 0.0035,  "coh": 5.551,  "inc": 0.001},
    "N":  {"abs": 1.90,    "coh": 11.01,  "inc": 0.50},
    "O":  {"abs": 0.00019, "coh": 4.232,  "inc": 0.0008},
    "F":  {"abs": 0.0096,  "coh": 4.017,  "inc": 0.0008},
    "Na": {"abs": 0.530,   "coh": 1.66,   "inc": 1.62},
    "Mg": {"abs": 0.063,   "coh": 3.631,  "inc": 0.08},
    "Al": {"abs": 0.231,   "coh": 1.495,  "inc": 0.0082},
    "Si": {"abs": 0.171,   "coh": 2.163,  "inc": 0.004},
    "P":  {"abs": 0.172,   "coh": 3.307,  "inc": 0.005},
    "S":  {"abs": 0.53,    "coh": 1.026,  "inc": 0.007},
    "Cl": {"abs": 33.5,    "coh": 11.528, "inc": 5.3},
    "K":  {"abs": 2.1,     "coh": 1.69,   "inc": 0.27},
    "Ca": {"abs": 0.43,    "coh": 2.830,  "inc": 0.05},
    "Ti": {"abs": 6.09,    "coh": 1.485,  "inc": 2.87},
    "Mn": {"abs": 13.3,    "coh": 1.75,   "inc": 0.40},
    "Fe": {"abs": 2.56,    "coh": 11.22,  "inc": 0.40},
    "Co": {"abs": 37.18,   "coh": 0.779,  "inc": 4.8},
    "Ni": {"abs": 4.49,    "coh": 13.3,   "inc": 5.2},
    "Cu": {"abs": 3.78,    "coh": 7.485,  "inc": 0.55},
    "Zn": {"abs": 1.11,    "coh": 4.054,  "inc": 0.077},
    "In": {"abs": 193.8,   "coh": 2.08,   "inc": 0.54},
    "W":  {"abs": 18.3,    "coh": 2.97,   "inc": 1.63},
    "Pb": {"abs": 0.171,   "coh": 11.115, "inc": 0.003},
}


# ──────────────────────────────────────────────────────────────────────────────
# NIST XCOM loader
# ──────────────────────────────────────────────────────────────────────────────

def _parse_xcom_txt(filepath: Path):
    """
    Parse a NIST XCOM text export.

    Returns
    -------
    energies_keV : np.ndarray
    mu_over_rho  : np.ndarray   total attenuation *with* coherent scattering
    """
    energies, mu_vals = [], []
    with open(filepath, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 8:
                continue
            try:
                energy_mev = float(parts[0])
                mu_total_with_coh = float(parts[6])
            except ValueError:
                continue
            energies.append(energy_mev * 1000.0)  # MeV → keV
            mu_vals.append(mu_total_with_coh)
    return np.array(energies), np.array(mu_vals)


def available_xray_elements() -> list:
    """Elements for which an XCOM data file is present."""
    return sorted(p.stem for p in XRAY_DATA_DIR.glob("*.txt"))


@lru_cache(maxsize=None)
def xray_mass_atten(element: str) -> np.ndarray:
    """
    Mass attenuation coefficient μ/ρ of *element* on ``XRAY_E_KEV`` [cm²/g].

    Loaded lazily from ``data/xray/<element>.txt`` and cached.

    Raises
    ------
    FileNotFoundError
        If the NIST XCOM export for the element has not been added yet.
    """
    filepath = XRAY_DATA_DIR / f"{element}.txt"
    if not filepath.exists():
        raise FileNotFoundError(
            f"No X-ray attenuation data for element '{element}'.\n"
            f"Export the NIST XCOM table for '{element}' and save it as\n"
            f"    {filepath}\n"
            f"Elements currently available: {available_xray_elements()}"
        )
    energies, mu = _parse_xcom_txt(filepath)
    return np.interp(XRAY_E_KEV, energies, mu)
