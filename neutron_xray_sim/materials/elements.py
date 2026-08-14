"""
neutron_xray_sim.materials.elements
───────────────────────────────────
Per-element physical constants and the NIST X-ray attenuation tables.

This module is the **single source of elemental data** for the whole package.
Everything above it (materials, phantoms, projectors) derives its numbers from
here, so adding support for a new element means editing exactly two things:

1. an entry in :data:`ATOMIC_MASS` and :data:`NEUTRON_XS` below;
2. a NIST attenuation file at ``lib/xray_data/<Symbol>.txt``.

See ``docs/materials.md`` for the download procedure.

Data provenance
───────────────
Atomic masses
    IUPAC 2021 standard atomic weights (conventional values).
Thermal-neutron cross sections
    Sears, *Neutron News* **3**(3):26-37 (1992), tabulated at 25.3 meV
    (λ = 1.798 Å).  ``abs`` is the 2200 m/s absorption cross section;
    ``coh`` and ``inc`` are the bound-atom scattering cross sections.
X-ray mass attenuation
    NIST XCOM / *X-Ray Mass Attenuation Coefficients* (Hubbell & Seltzer,
    NISTIR 5632).  Column 7 of the XCOM export ("Tot. w/ Coherent") is used.
"""

from __future__ import annotations

from functools import cache
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

__all__ = [
    "XRAY_E_KEV",
    "AVOGADRO",
    "BARN_CM2",
    "ATOMIC_MASS",
    "NEUTRON_XS",
    "XRAY_DATA_DIR",
    "MissingElementDataError",
    "element_mu_over_rho",
    "available_elements",
    "element_data_status",
    "parse_formula",
]

XRAY_DATA_DIR = Path(__file__).resolve().parent.parent / "lib" / "xray_data"

# ──────────────────────────────────────────────────────────────────────────────
# Physical constants
# ──────────────────────────────────────────────────────────────────────────────

#: Energy grid (keV) on which every material's X-ray attenuation is tabulated.
#: Chosen to bracket the K-edges of the common heavy elements (W: 69.5 keV,
#: Pb: 88.0 keV) so that edge behaviour is representable.
XRAY_E_KEV: np.ndarray = np.array(
    [20, 30, 40, 50, 60, 70, 80, 90, 100, 120, 150, 200, 300], dtype=float
)

AVOGADRO = 6.02214076e23    # mol⁻¹  (SI 2019 exact)
BARN_CM2 = 1e-24            # cm² per barn


# ── Atomic masses (g/mol) ─────────────────────────────────────────────────────
ATOMIC_MASS: Dict[str, float] = {
    "H":  1.00794,
    "Li": 6.941,
    "C":  12.0107,
    "N":  14.0067,
    "O":  15.999,
    "F":  18.998403163,
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

# ── Microscopic thermal-neutron cross sections (barn) ─────────────────────────
# abs = absorption at 2200 m/s, coh = bound coherent, inc = bound incoherent.
NEUTRON_XS: Dict[str, Dict[str, float]] = {
    "H":  {"abs": 0.3326,  "coh": 1.7568, "inc": 80.27},
    "Li": {"abs": 70.5,    "coh": 0.454,  "inc": 0.92},
    "C":  {"abs": 0.0035,  "coh": 5.551,  "inc": 0.001},
    "N":  {"abs": 1.90,    "coh": 11.01,  "inc": 0.50},
    "O":  {"abs": 0.00019, "coh": 4.232,  "inc": 0.0008},
    "F":  {"abs": 0.0096,  "coh": 4.017,  "inc": 0.0008},
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


class MissingElementDataError(KeyError):
    """Raised when an element has no X-ray attenuation table on disk.

    Carries an actionable message: which file is missing, where to get it,
    and which elements *are* currently available.  Subclasses ``KeyError`` so
    that code written against the old ``XRAY_MASS_ATTEN[el]`` lookup still
    catches it, but prints the message plainly instead of ``repr``-ing it.
    """

    def __str__(self) -> str:
        return self.args[0] if self.args else ""


# ──────────────────────────────────────────────────────────────────────────────
# Formula parsing
# ──────────────────────────────────────────────────────────────────────────────

import re  # noqa: E402  (kept next to its only consumer for readability)

_FORMULA_RE = re.compile(r"([A-Z][a-z]?)([0-9]*\.?[0-9]*)")


def parse_formula(formula: str) -> Dict[str, float]:
    """Parse a flat chemical formula into ``{element: stoichiometric count}``.

    Supports decimal subscripts, so non-stoichiometric compounds work:
    ``'SiO2'``, ``'Fe0.98Ni0.02'``, ``'LiNi0.8Mn0.1Co0.1O2'``.

    Bracketed or hydrated notation is **not** supported — expand it by hand
    (hydroxyapatite as ``'Ca10P6O26H2'``, gypsum as ``'CaSO6H4'``).

    Raises
    ------
    ValueError
        If nothing parses, or if an element symbol is unknown to the package.
    """
    comp: Dict[str, float] = {}
    matched_len = 0
    for elem, count_str in _FORMULA_RE.findall(formula):
        count = float(count_str) if count_str else 1.0
        comp[elem] = comp.get(elem, 0.0) + count
        matched_len += len(elem) + len(count_str)

    if not comp:
        raise ValueError(f"Could not parse chemical formula {formula!r}.")

    if matched_len != len(formula.replace(" ", "")):
        raise ValueError(
            f"Formula {formula!r} contains characters this parser does not "
            "understand. Use flat notation with optional decimal subscripts, "
            "e.g. 'Ca10P6O26H2' rather than 'Ca10(PO4)6(OH)2'."
        )

    unknown = sorted(set(comp) - set(ATOMIC_MASS))
    if unknown:
        raise ValueError(
            f"Formula {formula!r} references unknown element(s) {unknown}. "
            f"Known elements: {sorted(ATOMIC_MASS)}. "
            "Add the element to ATOMIC_MASS and NEUTRON_XS in "
            "neutron_xray_sim/materials/elements.py to extend the database."
        )
    return comp


# ──────────────────────────────────────────────────────────────────────────────
# NIST table parsing
# ──────────────────────────────────────────────────────────────────────────────

def _parse_attenuation_file(filepath: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Read an attenuation table and return ``(energies_keV, mu_over_rho)``.

    Two layouts are accepted so contributors are not forced through the XCOM
    web form:

    * **XCOM export** (8 numeric columns) — column 0 is energy in MeV and
      column 6 is the total attenuation *with* coherent scattering.
    * **Simple two-column** — energy in MeV followed by μ/ρ in cm²/g, which is
      the layout of the NIST *X-Ray Mass Attenuation Coefficients* tables.

    Non-numeric header lines are skipped.  Absorption edges appear in NIST
    tables as duplicated energies; they are preserved and handled by the
    interpolator.
    """
    energies: List[float] = []
    mu_vals: List[float] = []

    with open(filepath) as fh:
        for line in fh:
            parts = line.split()
            if len(parts) < 2:
                continue
            try:
                energy_mev = float(parts[0])
                mu = float(parts[6]) if len(parts) >= 8 else float(parts[1])
            except (ValueError, IndexError):
                continue
            energies.append(energy_mev * 1000.0)   # MeV → keV
            mu_vals.append(mu)

    if len(energies) < 2:
        raise ValueError(
            f"{filepath} contains fewer than two usable data rows. Expected a "
            "NIST XCOM export (8 columns) or a two-column 'E[MeV]  mu/rho' table."
        )
    return np.asarray(energies, dtype=float), np.asarray(mu_vals, dtype=float)


def _loglog_interp(
    x_new: np.ndarray, x: np.ndarray, y: np.ndarray
) -> np.ndarray:
    """Interpolate μ/ρ in log-log space, the convention NIST recommends.

    Photoelectric attenuation follows an approximate power law
    (μ/ρ ∝ E⁻³ away from edges), so a straight line in log-log space is a far
    better local model than one in linear space. On the shipped tables this
    changes off-node grid points by up to ~9 % for medium-Z elements.

    Duplicated energies at absorption edges are nudged apart by one part in
    10⁶ so that ``np.interp`` stays monotone and returns the *above-edge*
    value exactly at the edge energy.
    """
    order = np.argsort(x, kind="stable")
    xs, ys = x[order], y[order]

    positive = (xs > 0) & (ys > 0)
    xs, ys = xs[positive], ys[positive]

    # Break ties at absorption edges (identical energies, different μ/ρ).
    dup = np.flatnonzero(np.diff(xs) == 0)
    for i in dup:
        xs[i + 1] = np.nextafter(xs[i + 1], np.inf) * (1.0 + 1e-6)

    return np.exp(np.interp(np.log(x_new), np.log(xs), np.log(ys)))


@cache
def element_mu_over_rho(element: str) -> np.ndarray:
    """Return μ/ρ [cm²/g] for one element, resampled onto :data:`XRAY_E_KEV`.

    Results are cached, and the file is only read the first time an element is
    actually used — importing the package no longer touches the disk.

    Raises
    ------
    MissingElementDataError
        If ``lib/xray_data/<element>.txt`` is absent, with instructions for
        adding it.
    """
    filepath = XRAY_DATA_DIR / f"{element}.txt"
    if not filepath.exists():
        raise MissingElementDataError(
            f"No X-ray attenuation data for element {element!r}.\n"
            f"  Expected file : {filepath}\n"
            f"  Get it from   : https://physics.nist.gov/PhysRefData/Xcom/html/xcom1.html\n"
            f"                  (select the element, request 'Total attenuation with\n"
            f"                   coherent scattering', save the plain-text export)\n"
            f"  Available now : {', '.join(available_elements()) or '(none)'}\n"
            f"  A two-column 'E[MeV]  mu/rho[cm2/g]' text file also works."
        )
    energies, mu = _parse_attenuation_file(filepath)
    # ``.copy()`` because the lru_cache hands the same object to every caller
    # and NumPy arrays are mutable.
    return _loglog_interp(XRAY_E_KEV, energies, mu).copy()


def available_elements() -> List[str]:
    """Elements with an X-ray table on disk, sorted by atomic mass."""
    if not XRAY_DATA_DIR.is_dir():
        return []
    present = {p.stem for p in XRAY_DATA_DIR.glob("*.txt")}
    return sorted(present & set(ATOMIC_MASS), key=lambda e: ATOMIC_MASS[e])


def element_data_status() -> Dict[str, bool]:
    """Map every known element to whether its X-ray table is available.

    Useful as a first diagnostic when a formula-built material fails::

        >>> from neutron_xray_sim.materials import element_data_status
        >>> {k: v for k, v in element_data_status().items() if not v}
    """
    have = set(available_elements())
    return {el: (el in have) for el in sorted(ATOMIC_MASS, key=lambda e: ATOMIC_MASS[e])}
