"""
neutron_xray_sim.materials.core
───────────────────────────────
The :class:`Material` value object, the declarative :class:`MaterialSpec` that
describes how to build one, and the single builder that turns the second into
the first.

Why a spec layer
────────────────
Before v1.2 there were two unrelated ways to obtain a ``Material``: a
hand-written table of 13 X-ray numbers per material, and a formula-driven
constructor.  They shared no validation, so a typo in a table was silent while
the same typo in a formula raised a bare ``KeyError``.

Now every material — built-in, contributed, or created at runtime — is a
:class:`MaterialSpec` that goes through :func:`build_material`.  A spec declares
*where each channel's numbers come from*, and the two channels are independent:

============  ================================================================
Channel       Sources (pick one)
============  ================================================================
X-ray         ``formula`` / ``components``  →  derived from NIST element tables
              ``xray_mu_cm``                →  explicit 13-point table [cm⁻¹]
Neutron       ``formula`` / ``components``  →  derived from Sears cross sections
              ``neutron_mu_cm``             →  explicit ``(abs, coh, inc)`` [cm⁻¹]
============  ================================================================

The split matters in practice: several built-in metals have measured X-ray
tables but no NIST element file shipped with the package, so they take the
tabulated X-ray path while still deriving their neutron attenuation from the
formula.  One code path, one validator, no duplicated numbers.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, replace
from typing import Dict, Iterable, Mapping, Optional, Sequence, Tuple

import numpy as np

from .elements import (
    ATOMIC_MASS,
    AVOGADRO,
    BARN_CM2,
    NEUTRON_XS,
    XRAY_E_KEV,
    element_mu_over_rho,
    parse_formula,
)

__all__ = [
    "Material",
    "MaterialSpec",
    "Component",
    "build_material",
    "material_from_formula",
    "make_composite_material",
    "neutron_components_from_formula",
    "xray_spectrum",
    "validate_spec",
]


# ──────────────────────────────────────────────────────────────────────────────
# The value object
# ──────────────────────────────────────────────────────────────────────────────

@dataclass(eq=False)
class Material:
    """Physical properties of one material for dual-modality CT simulation.

    Instances are compared and hashed **by identity** (``eq=False``).  The old
    dataclass-generated ``__eq__`` compared a NumPy field, which raised
    "truth value of an array is ambiguous" as soon as two materials shared a
    name — a latent crash in every ``material in list`` check.

    Attributes
    ----------
    name, symbol
        Human-readable name and the short label used in plots and legends.
    density_gcc
        Mass density [g/cm³].  Also available as :attr:`rho`.
    mu_n_abs, mu_n_coh, mu_n_inc
        Thermal-neutron linear attenuation components [cm⁻¹]:
        absorption, bound-coherent scatter, bound-incoherent scatter.
    mu_x_table
        X-ray linear attenuation [cm⁻¹] at each energy in ``XRAY_E_KEV``.
    color
        Matplotlib colour used consistently across all package plots.
    key
        Registry key, if the material came from the database.
    reference
        Literature or database citation for the numbers.
    tags
        Free-form labels for grouping (``'metal'``, ``'battery'``, …).
    """

    name: str
    symbol: str
    density_gcc: float

    mu_n_abs: float
    mu_n_coh: float
    mu_n_inc: float

    mu_x_table: np.ndarray          # shape == XRAY_E_KEV.shape

    color: str = "#888888"
    key: str = ""
    reference: str = ""
    tags: Tuple[str, ...] = ()

    # ── Derived quantities ───────────────────────────────────────────────────

    @property
    def mu_n(self) -> float:
        """Total thermal-neutron linear attenuation [cm⁻¹]."""
        return self.mu_n_abs + self.mu_n_coh + self.mu_n_inc

    @property
    def mu_n_scatter(self) -> float:
        """Total neutron scatter coefficient (coherent + incoherent) [cm⁻¹]."""
        return self.mu_n_coh + self.mu_n_inc

    @property
    def rho(self) -> float:
        """Mass density [g/cm³]. Alias for :attr:`density_gcc`."""
        return self.density_gcc

    @property
    def _mu_x_table(self) -> np.ndarray:
        """Deprecated alias for :attr:`mu_x_table`, kept for older scripts."""
        return self.mu_x_table

    def mu_x_at(self, energy_keV: float) -> float:
        """X-ray linear attenuation at a single energy [cm⁻¹].

        Interpolated log-log in energy, matching how the underlying NIST
        tables are resampled onto ``XRAY_E_KEV``.
        """
        return float(self.mu_x_array(np.asarray([energy_keV]))[0])

    def mu_x_array(self, energies_keV: np.ndarray) -> np.ndarray:
        """X-ray linear attenuation at several energies [cm⁻¹]."""
        e = np.asarray(energies_keV, dtype=float)
        table = np.asarray(self.mu_x_table, dtype=float)

        # Log-log interpolation needs strictly positive values; materials such
        # as air are effectively zero, so fall back to linear for those.
        if np.all(table > 0):
            out = np.exp(
                np.interp(np.log(np.clip(e, 1e-12, None)),
                          np.log(XRAY_E_KEV), np.log(table))
            )
        else:
            out = np.interp(e, XRAY_E_KEV, table)
        return out if out.shape else float(out)

    def as_dict(self) -> Dict[str, object]:
        """Plain-Python view, suitable for JSON export or a metadata sidecar."""
        return {
            "key": self.key,
            "name": self.name,
            "symbol": self.symbol,
            "density_gcc": float(self.density_gcc),
            "mu_n_abs": float(self.mu_n_abs),
            "mu_n_coh": float(self.mu_n_coh),
            "mu_n_inc": float(self.mu_n_inc),
            "mu_n_total": float(self.mu_n),
            "mu_x_cm": [float(v) for v in self.mu_x_table],
            "color": self.color,
            "reference": self.reference,
            "tags": list(self.tags),
        }

    def __repr__(self) -> str:
        return (
            f"Material({self.name}: ρ={self.density_gcc:.3g} g/cm³  "
            f"μₙ={self.mu_n:.3f} cm⁻¹  μₓ(80 keV)={self.mu_x_at(80):.3f} cm⁻¹)"
        )


# ──────────────────────────────────────────────────────────────────────────────
# The declarative spec
# ──────────────────────────────────────────────────────────────────────────────

#: One phase of a composite: ``(formula, weight_fraction, end_member_density)``.
#: The density is informational — the calculation uses ``bulk_density × wf``.
Component = Tuple[str, float, float]


@dataclass(frozen=True)
class MaterialSpec:
    """Declarative recipe for one material.

    A spec is data, not code: it can be written in Python, loaded from JSON or
    YAML, round-tripped, diffed in a pull request, and validated before any
    physics runs.  See the module docstring for the channel/source table.

    Parameters
    ----------
    key
        Registry key, lower-case with underscores (``'separator_pe'``).
    name, symbol
        Display name and short plot label.
    density_gcc
        Bulk mass density [g/cm³]. Must be > 0.
    formula
        Flat chemical formula, e.g. ``'LiNi0.8Mn0.1Co0.1O2'``. Mutually
        exclusive with *components*.
    components
        Mixture of phases as ``(formula, weight_fraction, end_member_density)``
        tuples; the weight fractions must sum to 1 ± 0.02.
    xray_mu_cm
        Explicit X-ray linear attenuation [cm⁻¹] at each ``XRAY_E_KEV`` energy.
        Overrides the derived X-ray channel when given.
    neutron_mu_cm
        Explicit ``(mu_abs, mu_coh, mu_inc)`` [cm⁻¹]. Overrides the derived
        neutron channel when given.
    incoherent_scale
        Multiplier on the incoherent neutron term. The bound-atom cross section
        over-predicts scattering for liquids and soft polymers where molecular
        motion is not frozen; values of 0.3–0.6 are typical there, 1.0 for
        dense inorganic solids.
    k_edge_keV
        Energy of an absorption edge that legitimately falls inside the
        tabulated range. Declaring it suppresses the non-monotonic warning from
        :func:`validate_spec` at that energy.
    color, reference, tags
        Plot colour, literature citation, and grouping labels.
    """

    key: str
    name: str
    symbol: str
    density_gcc: float

    formula: Optional[str] = None
    components: Optional[Tuple[Component, ...]] = None

    xray_mu_cm: Optional[Tuple[float, ...]] = None
    neutron_mu_cm: Optional[Tuple[float, float, float]] = None

    incoherent_scale: float = 1.0
    k_edge_keV: Optional[float] = None

    color: str = "#888888"
    reference: str = ""
    tags: Tuple[str, ...] = ()

    def replace(self, **changes) -> MaterialSpec:
        """Return a copy with *changes* applied (e.g. a different density)."""
        return replace(self, **changes)


class SpecValidationError(ValueError):
    """Raised when a :class:`MaterialSpec` is internally inconsistent."""


def validate_spec(spec: MaterialSpec, *, warn_physics: bool = True) -> None:
    """Check a spec for internal consistency, raising on hard errors.

    Hard errors (raise :class:`SpecValidationError`)
        no source for a channel; both ``formula`` and ``components``; a
        non-positive density; a wrong-length X-ray table; weight fractions that
        do not sum to 1 ± 0.02.

    Soft warnings (only when *warn_physics*)
        an X-ray table that increases with energy away from a declared
        absorption edge, which almost always means transposed or
        mis-transcribed numbers.
    """
    where = f"material spec {spec.key!r}"

    if spec.formula is not None and spec.components is not None:
        raise SpecValidationError(
            f"{where}: give either 'formula' or 'components', not both."
        )

    derived = spec.formula is not None or spec.components is not None
    if not derived:
        if spec.xray_mu_cm is None or spec.neutron_mu_cm is None:
            raise SpecValidationError(
                f"{where}: without 'formula' or 'components' you must supply "
                "both 'xray_mu_cm' and 'neutron_mu_cm'."
            )

    if not spec.density_gcc > 0:
        raise SpecValidationError(
            f"{where}: density_gcc must be > 0, got {spec.density_gcc!r}."
        )

    if spec.xray_mu_cm is not None and len(spec.xray_mu_cm) != len(XRAY_E_KEV):
        raise SpecValidationError(
            f"{where}: xray_mu_cm has {len(spec.xray_mu_cm)} entries but "
            f"XRAY_E_KEV has {len(XRAY_E_KEV)}. The table must give one linear "
            f"attenuation per energy in {list(XRAY_E_KEV)}."
        )

    if spec.neutron_mu_cm is not None and len(spec.neutron_mu_cm) != 3:
        raise SpecValidationError(
            f"{where}: neutron_mu_cm must be (mu_abs, mu_coh, mu_inc)."
        )

    if spec.components is not None:
        if not spec.components:
            raise SpecValidationError(f"{where}: 'components' is empty.")
        total = sum(float(c[1]) for c in spec.components)
        if not np.isclose(total, 1.0, atol=0.02):
            raise SpecValidationError(
                f"{where}: component weight fractions sum to {total:.4f}; "
                "they must sum to 1.0 ± 0.02."
            )

    if warn_physics and spec.xray_mu_cm is not None:
        _warn_if_table_implausible(spec)


def _warn_if_table_implausible(spec: MaterialSpec) -> None:
    """Flag X-ray tables that rise with energy away from a declared edge."""
    table = np.asarray(spec.xray_mu_cm, dtype=float)
    rises = np.flatnonzero(np.diff(table) > 0)
    if rises.size == 0:
        return

    suspect = []
    for i in rises:
        lo, hi = XRAY_E_KEV[i], XRAY_E_KEV[i + 1]
        if spec.k_edge_keV is not None and lo <= spec.k_edge_keV <= hi:
            continue    # a declared absorption edge — expected
        suspect.append(f"{lo:.0f}→{hi:.0f} keV ({table[i]:.3g}→{table[i+1]:.3g})")

    if suspect:
        warnings.warn(
            f"X-ray table for material {spec.key!r} increases with energy at "
            + "; ".join(suspect)
            + ". Attenuation should fall with energy except at an absorption "
              "edge. Declare 'k_edge_keV' if the rise is a real edge, otherwise "
              "re-check the transcribed NIST values.",
            stacklevel=4,
        )


# ──────────────────────────────────────────────────────────────────────────────
# Channel builders
# ──────────────────────────────────────────────────────────────────────────────

def _mass_fractions(formula: str) -> Tuple[Dict[str, float], float]:
    """Return ``({element: mass fraction}, molar mass [g/mol])``."""
    comp = parse_formula(formula)
    molar_mass = sum(ATOMIC_MASS[el] * n for el, n in comp.items())
    return ({el: (ATOMIC_MASS[el] * n) / molar_mass for el, n in comp.items()},
            molar_mass)


def _xray_mu_from_phases(
    phases: Iterable[Tuple[str, float]], bulk_density_gcc: float
) -> np.ndarray:
    """μ_x [cm⁻¹] for a set of ``(formula, weight_fraction)`` phases."""
    mu_over_rho = np.zeros_like(XRAY_E_KEV, dtype=float)
    for formula, weight in phases:
        mass_fracs, _ = _mass_fractions(formula)
        for el, w_el in mass_fracs.items():
            mu_over_rho += weight * w_el * element_mu_over_rho(el)
    return bulk_density_gcc * mu_over_rho


def neutron_components_from_formula(
    formula: str,
    density_gcc: float,
    incoherent_scale: float = 1.0,
) -> Tuple[float, float, float]:
    """Return ``(mu_n_abs, mu_n_coh, mu_n_inc)`` [cm⁻¹] for a formula + density.

    The macroscopic cross section is ``N · σ`` with ``N`` the number density of
    formula units, so the three components scale linearly with density.
    """
    comp = parse_formula(formula)
    molar_mass = sum(ATOMIC_MASS[el] * n for el, n in comp.items())
    n_formula = density_gcc * AVOGADRO / molar_mass
    return (
        n_formula * sum(comp[el] * NEUTRON_XS[el]["abs"] for el in comp) * BARN_CM2,
        n_formula * sum(comp[el] * NEUTRON_XS[el]["coh"] for el in comp) * BARN_CM2,
        n_formula * sum(comp[el] * NEUTRON_XS[el]["inc"] for el in comp) * BARN_CM2
        * incoherent_scale,
    )


def _neutron_mu_from_phases(
    phases: Iterable[Tuple[str, float]],
    bulk_density_gcc: float,
    incoherent_scale: float,
) -> Tuple[float, float, float]:
    """Macroscopic neutron cross sections add, so sum over partial densities."""
    mu_abs = mu_coh = mu_inc = 0.0
    for formula, weight in phases:
        a, c, i = neutron_components_from_formula(
            formula, bulk_density_gcc * weight, incoherent_scale
        )
        mu_abs += a
        mu_coh += c
        mu_inc += i
    return mu_abs, mu_coh, mu_inc


# ──────────────────────────────────────────────────────────────────────────────
# The single builder
# ──────────────────────────────────────────────────────────────────────────────

def build_material(spec: MaterialSpec, *, validate: bool = True) -> Material:
    """Turn a :class:`MaterialSpec` into a :class:`Material`.

    This is the only place in the package where a ``Material`` is constructed,
    so every material — built-in, contributed, or ad hoc — is validated and
    computed identically.

    Raises
    ------
    SpecValidationError
        If the spec is internally inconsistent.
    MissingElementDataError
        If a derived X-ray channel needs an element whose NIST table is not
        installed.
    """
    if validate:
        validate_spec(spec)

    if spec.components is not None:
        phases = [(str(c[0]), float(c[1])) for c in spec.components]
    elif spec.formula is not None:
        phases = [(spec.formula, 1.0)]
    else:
        phases = []

    # ── X-ray channel ────────────────────────────────────────────────────────
    if spec.xray_mu_cm is not None:
        mu_x = np.asarray(spec.xray_mu_cm, dtype=float)
    else:
        mu_x = _xray_mu_from_phases(phases, spec.density_gcc)

    # ── Neutron channel ──────────────────────────────────────────────────────
    if spec.neutron_mu_cm is not None:
        mu_abs, mu_coh, mu_inc = (float(v) for v in spec.neutron_mu_cm)
    else:
        mu_abs, mu_coh, mu_inc = _neutron_mu_from_phases(
            phases, spec.density_gcc, spec.incoherent_scale
        )

    return Material(
        name=spec.name,
        symbol=spec.symbol,
        density_gcc=float(spec.density_gcc),
        mu_n_abs=mu_abs,
        mu_n_coh=mu_coh,
        mu_n_inc=mu_inc,
        mu_x_table=mu_x,
        color=spec.color,
        key=spec.key,
        reference=spec.reference,
        tags=tuple(spec.tags),
    )


# ──────────────────────────────────────────────────────────────────────────────
# Backwards-compatible convenience constructors
# ──────────────────────────────────────────────────────────────────────────────

def material_from_formula(
    name: str,
    symbol: str,
    formula: str,
    density_gcc: float,
    color: str = "#888888",
    incoherent_scale: float = 1.0,
    *,
    key: str = "",
    reference: str = "",
    tags: Sequence[str] = (),
) -> Material:
    """Build a :class:`Material` from a chemical formula and a density.

    Thin wrapper over :func:`build_material`; kept because it reads naturally
    in notebooks::

        quartz = material_from_formula("Quartz", "SiO2", "SiO2", 2.65)
    """
    return build_material(MaterialSpec(
        key=key or symbol.lower(),
        name=name, symbol=symbol, density_gcc=density_gcc,
        formula=formula, incoherent_scale=incoherent_scale,
        color=color, reference=reference, tags=tuple(tags),
    ))


def make_composite_material(
    name: str,
    symbol: str,
    bulk_density_gcc: float,
    components: Sequence[Component],
    color: str = "#888888",
    incoherent_scale: float = 1.0,
    *,
    key: str = "",
    reference: str = "",
    tags: Sequence[str] = (),
) -> Material:
    """Build a :class:`Material` from a mixture of mineral or polymer phases.

    Use this when one chemical formula cannot represent the sample — fossilised
    bone (fluorapatite + collagen + calcite), a sedimentary matrix, an
    electrode with binder and conductive additive.

    Parameters
    ----------
    components
        ``(formula, weight_fraction, end_member_density)`` triples. Weight
        fractions must sum to 1 ± 0.02. The end-member density is recorded for
        provenance but does not enter the calculation, which uses
        ``bulk_density_gcc × weight_fraction`` as each phase's partial density.

    Example
    -------
    >>> bone = make_composite_material(
    ...     "Fossilised bone", "Bone", 2.00,
    ...     [("Ca5P3O12F", 0.70, 3.20),    # fluorapatite
    ...      ("C10H13NO4", 0.15, 1.35),    # dry collagen proxy
    ...      ("CaCO3",     0.12, 2.71),    # calcite
    ...      ("SiO2",      0.03, 2.65)],   # clay silicate proxy
    ... )
    """
    return build_material(MaterialSpec(
        key=key or symbol.lower(),
        name=name, symbol=symbol, density_gcc=bulk_density_gcc,
        components=tuple((str(f), float(w), float(d)) for f, w, d in components),
        incoherent_scale=incoherent_scale,
        color=color, reference=reference, tags=tuple(tags),
    ))


# ──────────────────────────────────────────────────────────────────────────────
# X-ray source spectrum
# ──────────────────────────────────────────────────────────────────────────────

def xray_spectrum(
    kVp: float = 120.0,
    filter_mm_Al: float = 2.0,
    filter_mm_Cu: float = 0.0,
    n_bins: int = 12,
    *,
    materials: Optional[Mapping[str, Material]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate a simplified bremsstrahlung X-ray spectrum (Kramers' law).

    Parameters
    ----------
    kVp
        Tube accelerating voltage [kV]; also the spectrum end-point energy.
    filter_mm_Al, filter_mm_Cu
        Pre-filter thicknesses [mm], applied as Beer-Lambert attenuation.
    n_bins
        Number of energy bins.
    materials
        Material lookup for the filters. Defaults to the package registry;
        injectable so the function stays testable without global state.

    Returns
    -------
    energies : (n_bins,) bin-centre energies [keV]
    weights  : (n_bins,) normalised photon fluence weights, summing to 1
    """
    if kVp <= 0:
        raise ValueError(f"kVp must be positive, got {kVp!r}.")
    if n_bins < 1:
        raise ValueError(f"n_bins must be at least 1, got {n_bins!r}.")

    if materials is None:
        from .registry import MATERIALS
        materials = MATERIALS

    E_min = max(10.0, 0.05 * kVp)
    if E_min >= kVp:
        raise ValueError(
            f"kVp={kVp} kV is below the {E_min:.0f} keV low-energy cut-off; "
            "no spectrum can be formed."
        )

    edges = np.linspace(E_min, kVp, n_bins + 1)
    energies = 0.5 * (edges[:-1] + edges[1:])

    Z_TUNGSTEN = 74
    spectrum = np.clip(Z_TUNGSTEN * energies * (kVp - energies), 0, None)

    if filter_mm_Al > 0:
        spectrum *= np.exp(-materials["aluminum"].mu_x_array(energies) * filter_mm_Al * 0.1)
    if filter_mm_Cu > 0:
        spectrum *= np.exp(-materials["copper"].mu_x_array(energies) * filter_mm_Cu * 0.1)

    total = spectrum.sum()
    if total <= 0:
        raise ValueError(
            "X-ray spectrum collapsed to zero — check kVp and filter settings."
        )
    return energies, spectrum / total
