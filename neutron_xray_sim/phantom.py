"""
neutron_xray_sim/phantom.py
────────────────────────────
Voxelised 3-D phantom builder.

A Phantom stores a label volume (integer material indices) plus the
corresponding pair of attenuation-coefficient arrays (one for neutrons,
one per energy bin for X-rays).  The helper methods add geometric
primitives, and several preset phantoms are provided.
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from .materials import Material, MATERIALS, XRAY_E_KEV, xray_spectrum


# ──────────────────────────────────────────────────────────────────────────────
# Data containers
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class PhantomData:
    """
    Container for a fully specified 3-D phantom.

    Attributes
    ----------
    N              : voxel dimension (cubic: N × N × N)
    voxel_cm       : voxel side length  [cm]
    label_vol      : integer label volume,  shape (N, N, N)
    materials      : ordered list of Material objects;  index 0 = air
    name           : descriptive name
    mu_n_vol       : total thermal-neutron l.a.  [cm⁻¹],  shape (N, N, N)
    mu_n_abs_vol   : absorption component only
    mu_n_coh_vol   : coherent-scatter component only
    mu_n_inc_vol   : incoherent-scatter component only
    mu_x_vols      : X-ray l.a. at XRAY_E_KEV  [cm⁻¹], shape (13, N, N, N)
    """

    N: int
    voxel_cm: float
    label_vol: np.ndarray               # (N, N, N)  uint8
    materials: List[Material]
    name: str = "phantom"

    # Derived attenuation volumes (filled lazily)
    mu_n_vol:     Optional[np.ndarray] = field(default=None, repr=False)
    mu_n_abs_vol: Optional[np.ndarray] = field(default=None, repr=False)
    mu_n_coh_vol: Optional[np.ndarray] = field(default=None, repr=False)
    mu_n_inc_vol: Optional[np.ndarray] = field(default=None, repr=False)
    mu_x_vols:    Optional[np.ndarray] = field(default=None, repr=False)

    def __post_init__(self):
        self._build_mu_vols()

    # ── Internal builders ────────────────────────────────────────────────────

    def _build_mu_vols(self):
        """Build attenuation-coefficient arrays from label_vol + materials."""
        N = self.N
        n_E = len(XRAY_E_KEV)

        mu_n     = np.zeros((N, N, N), dtype=np.float32)
        mu_n_abs = np.zeros_like(mu_n)
        mu_n_coh = np.zeros_like(mu_n)
        mu_n_inc = np.zeros_like(mu_n)
        mu_x     = np.zeros((n_E, N, N, N), dtype=np.float32)

        for idx, mat in enumerate(self.materials):
            mask = self.label_vol == idx
            if not mask.any():
                continue
            mu_n    [mask] = mat.mu_n
            mu_n_abs[mask] = mat.mu_n_abs
            mu_n_coh[mask] = mat.mu_n_coh
            mu_n_inc[mask] = mat.mu_n_inc
            for e, _ in enumerate(XRAY_E_KEV):
                mu_x[e][mask] = mat._mu_x_table[e]

        self.mu_n_vol     = mu_n
        self.mu_n_abs_vol = mu_n_abs
        self.mu_n_coh_vol = mu_n_coh
        self.mu_n_inc_vol = mu_n_inc
        self.mu_x_vols    = mu_x

    # ── Public helpers ────────────────────────────────────────────────────────

    @property
    def physical_size_cm(self) -> float:
        return self.N * self.voxel_cm

    def material_name(self, label: int) -> str:
        return self.materials[label].name if label < len(self.materials) else "unknown"

    def mu_x_at_energy(self, energy_keV: float) -> np.ndarray:
        """Interpolate X-ray attenuation volume at arbitrary energy  [cm⁻¹]."""
        result = np.zeros((self.N, self.N, self.N), dtype=np.float32)
        for idx, mat in enumerate(self.materials):
            mask = self.label_vol == idx
            if mask.any():
                result[mask] = mat.mu_x_at(energy_keV)
        return result

    def __repr__(self):
        mats = ", ".join(m.symbol for m in self.materials)
        return (
            f"PhantomData('{self.name}', {self.N}³, "
            f"{self.voxel_cm} cm/voxel, materials=[{mats}])"
        )


# ──────────────────────────────────────────────────────────────────────────────
# Builder class
# ──────────────────────────────────────────────────────────────────────────────

class PhantomBuilder:
    """
    Builds a cubic voxel phantom by compositing geometric primitives.

    Usage
    -----
    >>> b = PhantomBuilder(N=64, voxel_cm=0.15)
    >>> b.add_cylinder(material='aluminum', radius_cm=4.5, height_cm=9.0)
    >>> b.add_sphere(material='water', center_cm=(0,1,0), radius_cm=1.5)
    >>> b.add_box(material='hdpe', center_cm=(2,0,-2), half_extents_cm=(1,1,1))
    >>> phantom = b.build(name='my_phantom')
    """

    def __init__(self, N: int = 64, voxel_cm: float = 0.15):
        self.N = N
        self.voxel_cm = voxel_cm
        self._label_vol = np.zeros((N, N, N), dtype=np.uint8)
        self._materials: List[Material] = [MATERIALS["air"]]  # index 0 = air

        # Coordinate arrays for geometry tests  (physical coords, cm)
        L = N * voxel_cm / 2.0
        lin = np.linspace(-L + voxel_cm / 2, L - voxel_cm / 2, N)
        self._Y, self._X, self._Z = np.meshgrid(lin, lin, lin, indexing="ij")
        # Convention:  axis 0 = y (vertical), axis 1 = x, axis 2 = z (beam)

    # ── Material registry ─────────────────────────────────────────────────────

    def _mat_index(self, material) -> int:
        if isinstance(material, str):
            material = MATERIALS[material]
        if material not in self._materials:
            self._materials.append(material)
        return self._materials.index(material)

    # ── Primitive operations ──────────────────────────────────────────────────

    def fill(self, material):
        """Fill the entire volume with one material."""
        idx = self._mat_index(material)
        self._label_vol[:] = idx
        return self

    def add_sphere(
        self,
        material,
        center_cm: Tuple[float, float, float] = (0, 0, 0),
        radius_cm: float = 1.0,
    ):
        """Add a solid sphere."""
        cy, cx, cz = center_cm
        r2 = (self._Y - cy) ** 2 + (self._X - cx) ** 2 + (self._Z - cz) ** 2
        mask = r2 <= radius_cm ** 2
        self._label_vol[mask] = self._mat_index(material)
        return self

    def add_ellipsoid(
        self,
        material,
        center_cm: Tuple[float, float, float] = (0, 0, 0),
        semi_axes_cm: Tuple[float, float, float] = (1, 1, 1),
    ):
        """Add a solid axis-aligned ellipsoid."""
        cy, cx, cz = center_cm
        ay, ax, az = semi_axes_cm
        inside = (
            ((self._Y - cy) / ay) ** 2
            + ((self._X - cx) / ax) ** 2
            + ((self._Z - cz) / az) ** 2
        ) <= 1.0
        self._label_vol[inside] = self._mat_index(material)
        return self

    def add_cylinder(
        self,
        material,
        center_cm: Tuple[float, float, float] = (0, 0, 0),
        radius_cm: float = 1.0,
        height_cm: Optional[float] = None,
        axis: str = "y",
    ):
        """
        Add a solid cylinder.  axis can be 'x', 'y', or 'z'.
        If height_cm is None the cylinder spans the full phantom.
        """
        cy, cx, cz = center_cm
        L = self.N * self.voxel_cm          # full phantom side length

        if axis == "y":
            r2 = (self._X - cx) ** 2 + (self._Z - cz) ** 2
            h_half = (height_cm / 2) if height_cm else L
            in_cyl = (r2 <= radius_cm ** 2) & (np.abs(self._Y - cy) <= h_half)
        elif axis == "x":
            r2 = (self._Y - cy) ** 2 + (self._Z - cz) ** 2
            h_half = (height_cm / 2) if height_cm else L
            in_cyl = (r2 <= radius_cm ** 2) & (np.abs(self._X - cx) <= h_half)
        elif axis == "z":
            r2 = (self._Y - cy) ** 2 + (self._X - cx) ** 2
            h_half = (height_cm / 2) if height_cm else L
            in_cyl = (r2 <= radius_cm ** 2) & (np.abs(self._Z - cz) <= h_half)
        else:
            raise ValueError("axis must be 'x', 'y', or 'z'")

        self._label_vol[in_cyl] = self._mat_index(material)
        return self

    def add_hollow_cylinder(
        self,
        material,
        center_cm: Tuple[float, float, float] = (0, 0, 0),
        inner_radius_cm: float = 1.0,
        outer_radius_cm: float = 1.2,
        height_cm: Optional[float] = None,
        axis: str = "y",
    ):
        """Add a hollow cylindrical shell."""
        cy, cx, cz = center_cm
        L = self.N * self.voxel_cm

        if axis == "y":
            r2 = (self._X - cx) ** 2 + (self._Z - cz) ** 2
            h_half = (height_cm / 2) if height_cm else L
            in_shell = (
                (r2 >= inner_radius_cm ** 2)
                & (r2 <= outer_radius_cm ** 2)
                & (np.abs(self._Y - cy) <= h_half)
            )
        else:
            raise NotImplementedError("Only axis='y' supported for hollow cylinder")

        self._label_vol[in_shell] = self._mat_index(material)
        return self

    def add_box(
        self,
        material,
        center_cm: Tuple[float, float, float] = (0, 0, 0),
        half_extents_cm: Tuple[float, float, float] = (1, 1, 1),
    ):
        """Add a rectangular cuboid."""
        cy, cx, cz = center_cm
        hy, hx, hz = half_extents_cm
        mask = (
            (np.abs(self._Y - cy) <= hy)
            & (np.abs(self._X - cx) <= hx)
            & (np.abs(self._Z - cz) <= hz)
        )
        self._label_vol[mask] = self._mat_index(material)
        return self

    def add_layer(
        self,
        material,
        position_cm: float,
        thickness_cm: float,
        axis: str = "y",
    ):
        """Add an infinite planar slab perpendicular to one axis."""
        lo = position_cm - thickness_cm / 2
        hi = position_cm + thickness_cm / 2
        if axis == "y":
            mask = (self._Y >= lo) & (self._Y < hi)
        elif axis == "x":
            mask = (self._X >= lo) & (self._X < hi)
        elif axis == "z":
            mask = (self._Z >= lo) & (self._Z < hi)
        else:
            raise ValueError("axis must be 'x', 'y', or 'z'")
        self._label_vol[mask] = self._mat_index(material)
        return self

    def add_rod(
        self,
        material,
        center_cm: Tuple[float, float] = (0, 0),
        radius_cm: float = 0.3,
        axis: str = "y",
    ):
        """Add a thin rod (infinite cylinder) along one axis."""
        if axis == "y":
            cx, cz = center_cm
            r2 = (self._X - cx) ** 2 + (self._Z - cz) ** 2
        elif axis == "x":
            cy, cz = center_cm
            r2 = (self._Y - cy) ** 2 + (self._Z - cz) ** 2
        elif axis == "z":
            cy, cx = center_cm
            r2 = (self._Y - cy) ** 2 + (self._X - cx) ** 2
        else:
            raise ValueError("axis must be 'x', 'y', or 'z'")
        self._label_vol[r2 <= radius_cm ** 2] = self._mat_index(material)
        return self

    # ── Finaliser ─────────────────────────────────────────────────────────────

    def build(self, name: str = "phantom") -> PhantomData:
        """Return the finished PhantomData object."""
        return PhantomData(
            N=self.N,
            voxel_cm=self.voxel_cm,
            label_vol=self._label_vol.copy(),
            materials=list(self._materials),
            name=name,
        )


# ──────────────────────────────────────────────────────────────────────────────
# Pre-built phantoms
# ──────────────────────────────────────────────────────────────────────────────

def make_composite_phantom(N: int = 64, voxel_cm: float = None) -> PhantomData:
    """
    HDPE-matrix composite phantom — 1 cm diameter.

    A solid multi-material cylinder filled with HDPE (H-rich polymer matrix)
    containing inclusions of water, iron, and titanium.  This is realistic for
    neutron tomography: real samples are solid objects, not mostly hollow.

    Expected bimodal histogram clusters:
      Air   : (mu_x ~ 0,    mu_n ~ 0)     -- exterior background
      HDPE  : (mu_x ~ 0.17, mu_n ~ 2.18)  -- LOW X-ray, HIGH neutron (H-rich)
      Al    : (mu_x ~ 0.28, mu_n ~ 0.10)  -- medium both
      Water : (mu_x ~ 0.18, mu_n ~ 1.38)  -- similar to HDPE in neutron, lower
      Fe    : (mu_x ~ 4.12, mu_n ~ 1.16)  -- HIGH X-ray, medium neutron
      Ti    : (mu_x ~ 1.48, mu_n ~ 0.64)  -- high X-ray, low neutron

    Optical depths at 1 cm traverse (imaging-effective):
      HDPE : OD_n = 2.18  T_n = 0.11   OD_x(80) = 0.17  T_x = 0.85
      Water: OD_n = 1.38  T_n = 0.25   OD_x(80) = 0.18  T_x = 0.84
      Fe   : OD_n = 1.16  T_n = 0.31   OD_x(80) = 4.12  T_x = 0.016
      Al   : OD_n = 0.10  T_n = 0.91   OD_x(80) = 0.28  T_x = 0.76
    """
    if voxel_cm is None:
        voxel_cm = 1.0 / N

    b = PhantomBuilder(N, voxel_cm)
    L = N * voxel_cm / 2     # half-width in cm (= 0.5 cm at N=64)

    r_outer = 0.82 * L
    wall    = max(2 * voxel_cm, 0.02 * L)

    # Aluminium outer shell
    b.add_hollow_cylinder("aluminum", outer_radius_cm=r_outer,
                          inner_radius_cm=r_outer - wall)

    # HDPE fills the interior (base matrix — H-rich, invisible to X-rays)
    b.add_cylinder("hdpe", radius_cm=r_outer - wall)

    # Water inclusion (sphere) — same low mu_x as HDPE but lower mu_n
    b.add_sphere("water", center_cm=(0, 0.20*L, 0), radius_cm=0.18*L)

    # Iron rod — high mu_x, visible only in X-ray channel
    # Radius = 12% of L → 3.8 px at N=64, 7.7 px at N=128
    b.add_rod("iron", center_cm=(0.30*L, -0.12*L), radius_cm=0.12*L, axis="y")

    # Titanium sphere — high mu_x, lower mu_n than HDPE
    # Radius = 10% of L → 3.2 px at N=64, 6.4 px at N=128
    b.add_sphere("titanium", center_cm=(0, 0.28*L, 0.24*L), radius_cm=0.10*L)

    # Air void (simulates a crack or pore in the matrix)
    b.add_sphere("air", center_cm=(0, -0.18*L, -0.22*L), radius_cm=0.07*L)

    return b.build("composite")


def make_battery_phantom(N: int = 64, voxel_cm: float = None) -> PhantomData:
    """
    Alkaline AAA battery cross-section phantom — 1.4 cm diameter.

    Matches a real AAA cell (diameter ≈ 10.5 mm) scaled to simulation.
    Demonstrates H-sensitivity of neutron imaging (HDPE separator,
    water-based KOH electrolyte visible only with neutrons).

    After LaManna et al. (NIST NeXT simultaneous neutron + X-ray).
    """
    if voxel_cm is None:
        voxel_cm = 1.4 / N   # 1.4 cm object (realistic AAA diameter)

    b = PhantomBuilder(N, voxel_cm)
    L = N * voxel_cm / 2
    r_can = 0.70 * L
    wall  = max(2 * voxel_cm, 0.025 * L)

    # Steel can (iron approximation)
    b.add_hollow_cylinder("iron", outer_radius_cm=r_can,
                          inner_radius_cm=r_can - wall)

    # KOH electrolyte (approximated as water)
    b.add_cylinder("water", radius_cm=r_can - wall)

    # HDPE separator ring
    b.add_hollow_cylinder("hdpe", outer_radius_cm=0.55 * L,
                          inner_radius_cm=0.45 * L)

    # Zinc anode rod
    b.add_cylinder("zinc", radius_cm=0.44 * L)

    # Central air void (current collector channel)
    b.add_cylinder("air", radius_cm=0.06 * L)

    return b.build("battery")


def make_bone_implant_phantom(N: int = 64, voxel_cm: float = None) -> PhantomData:
    """
    Cortical bone + titanium implant phantom — 1 cm diameter.

    After Törnquist et al. 2021 (Phys. Med. Biol. 66, 13).
    Demonstrates that neutrons resolve the bone–metal interface
    where X-rays suffer photon starvation next to the Ti implant.
    """
    if voxel_cm is None:
        voxel_cm = 1.0 / N

    b = PhantomBuilder(N, voxel_cm)
    L = N * voxel_cm / 2
    wall = max(2 * voxel_cm, 0.02 * L)

    # Cortical bone outer shell
    b.add_hollow_cylinder("bone", outer_radius_cm=0.75*L,
                          inner_radius_cm=0.55*L)

    # Water-based marrow
    b.add_cylinder("water", radius_cm=0.55*L)

    # Titanium screw (runs along the y-axis through the sample)
    # Diameter ≈ 2×0.09L ≈ 0.09 cm → ~0.9 mm, realistic for a 1cm sample
    b.add_rod("titanium", center_cm=(0.22*L, 0.0), radius_cm=0.09*L, axis="y")

    # Peri-implant bone (thin ring around screw)
    b.add_hollow_cylinder("bone", center_cm=(0, 0, 0),
                          inner_radius_cm=0.09*L,
                          outer_radius_cm=0.17*L)

    return b.build("bone_implant")


def make_industrial_phantom(N: int = 64, voxel_cm: float = None) -> PhantomData:
    """
    Industrial multi-material phantom — 1 cm diameter.

    Contains tungsten and iron inserts to showcase beam hardening
    and neutron complementarity in NDE applications.
    W screws: μ_x(80keV)=88 cm⁻¹ → photon starvation even at ~0.5mm.
    W screws: μ_n=1.56 cm⁻¹ → well-resolved by neutrons.
    """
    if voxel_cm is None:
        voxel_cm = 1.0 / N

    b = PhantomBuilder(N, voxel_cm)
    L = N * voxel_cm / 2
    wall = max(2 * voxel_cm, 0.02 * L)

    # Aluminium housing
    b.add_hollow_cylinder("aluminum", outer_radius_cm=0.80*L,
                          inner_radius_cm=0.80*L - wall)

    # HDPE matrix (H-rich filler, strongly scattering for neutrons)
    b.add_cylinder("hdpe", radius_cm=0.80*L - wall)

    # Tungsten rods — 4 at cardinal positions (screws / pins)
    # Radius = 3% of L ≈ 0.015 cm → ~0.3 mm diameter
    for ang in [0, 90, 180, 270]:
        rad = np.radians(ang)
        cx = 0.40*L * np.cos(rad)
        cz = 0.40*L * np.sin(rad)
        b.add_rod("tungsten", center_cm=(cx, cz), radius_cm=0.03*L, axis="y")

    # Iron support bar (thin, central)
    b.add_box("iron", center_cm=(0, 0.0, 0.0),
              half_extents_cm=(0.05*L, 0.55*L, 0.05*L))

    # Water pocket (coolant or defect)
    b.add_cylinder("water", center_cm=(0, 0.22*L, 0), radius_cm=0.10*L)

    # Air voids (defects / porosity)
    b.add_sphere("air", center_cm=(0, -0.28*L, 0.18*L), radius_cm=0.05*L)
    b.add_sphere("air", center_cm=(0,  0.10*L, -0.28*L), radius_cm=0.04*L)

    return b.build("industrial")


# ── Registry ──────────────────────────────────────────────────────────────────

PHANTOM_PRESETS: Dict[str, callable] = {
    "composite":     make_composite_phantom,
    "battery":       make_battery_phantom,
    "bone_implant":  make_bone_implant_phantom,
    "industrial":    make_industrial_phantom,
}


def make_phantom(preset: str = "composite", N: int = 64) -> PhantomData:
    """
    Load a named preset phantom at size N³.

    Sample sizes are physically realistic for neutron tomography:
      composite / bone_implant / industrial  →  1.0 cm diameter
      battery                                →  1.4 cm diameter (AAA cell)

    Voxel size scales automatically with N so geometry is preserved.
    Use N ≥ 128 for publication-quality reconstructions.
    """
    if preset not in PHANTOM_PRESETS:
        raise ValueError(f"Unknown preset '{preset}'. Choose from: {list(PHANTOM_PRESETS)}")
    return PHANTOM_PRESETS[preset](N=N)
