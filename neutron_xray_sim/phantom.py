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
    Nz             : voxel dimension along z
    Nx             : voxel dimension along x
    Ny             : voxel dimension along y
    voxel_cm       : voxel side length [cm]
    label_vol      : integer label volume, shape (Nz, Nx, Ny)
    materials      : ordered list of Material objects; index 0 = air
    name           : descriptive name
    mu_n_vol       : total thermal-neutron l.a. [cm⁻¹], shape (Nz, Nx, Ny)
    mu_n_abs_vol   : absorption component only
    mu_n_coh_vol   : coherent-scatter component only
    mu_n_inc_vol   : incoherent-scatter component only
    mu_x_vols      : X-ray l.a. at XRAY_E_KEV [cm⁻¹], shape (13, Nz, Nx, Ny)
    """

    Nz: int
    Nx: int
    Ny: int
    voxel_cm: float
    label_vol: np.ndarray               # (Nz, Nx, Ny) uint8
    materials: List[Material]
    name: str = "phantom"

    # Derived attenuation volumes (filled lazily)
    mu_n_vol: Optional[np.ndarray] = field(default=None, repr=False)
    mu_n_abs_vol: Optional[np.ndarray] = field(default=None, repr=False)
    mu_n_coh_vol: Optional[np.ndarray] = field(default=None, repr=False)
    mu_n_inc_vol: Optional[np.ndarray] = field(default=None, repr=False)
    mu_x_vols: Optional[np.ndarray] = field(default=None, repr=False)

    def __post_init__(self):
        self._validate_shape()
        self._build_mu_vols()

    # ── Backward-compatible aliases ──────────────────────────────────────────

    @property
    def N(self) -> int:
        """
        Backward-compatible cube dimension.

        For cubic phantoms, this returns the common dimension.  For non-cubic
        phantoms, it returns Nz because the old single-N assumption is no longer
        well-defined.
        """
        return self.Nz

    @property
    def shape(self) -> Tuple[int, int, int]:
        """Volume shape in storage order: (Nz, Nx, Ny)."""
        return self.Nz, self.Nx, self.Ny

    @property
    def physical_size_cm(self) -> Tuple[float, float, float]:
        """Physical size in storage/order convention: (z_cm, x_cm, y_cm)."""
        return (
            self.Nz * self.voxel_cm,
            self.Nx * self.voxel_cm,
            self.Ny * self.voxel_cm,
        )

    # ── Internal builders ────────────────────────────────────────────────────

    def _validate_shape(self):
        expected = (self.Nz, self.Nx, self.Ny)
        if self.label_vol.shape != expected:
            raise ValueError(
                f"label_vol shape must be {expected} for (Nz, Nx, Ny), "
                f"got {self.label_vol.shape}"
            )

    def _build_mu_vols(self):
        """Build attenuation-coefficient arrays from label_vol + materials."""
        n_E = len(XRAY_E_KEV)
        shape = (self.Nz, self.Nx, self.Ny)

        mu_n = np.zeros(shape, dtype=np.float32)
        mu_n_abs = np.zeros_like(mu_n)
        mu_n_coh = np.zeros_like(mu_n)
        mu_n_inc = np.zeros_like(mu_n)
        mu_x = np.zeros((n_E, *shape), dtype=np.float32)

        for idx, mat in enumerate(self.materials):
            mask = self.label_vol == idx
            if not mask.any():
                continue

            mu_n[mask] = mat.mu_n
            mu_n_abs[mask] = mat.mu_n_abs
            mu_n_coh[mask] = mat.mu_n_coh
            mu_n_inc[mask] = mat.mu_n_inc

            for e, _ in enumerate(XRAY_E_KEV):
                mu_x[e][mask] = mat._mu_x_table[e]

        self.mu_n_vol = mu_n
        self.mu_n_abs_vol = mu_n_abs
        self.mu_n_coh_vol = mu_n_coh
        self.mu_n_inc_vol = mu_n_inc
        self.mu_x_vols = mu_x

    # ── Public helpers ────────────────────────────────────────────────────────

    def material_name(self, label: int) -> str:
        return self.materials[label].name if label < len(self.materials) else "unknown"

    def mu_x_at_energy(self, energy_keV: float) -> np.ndarray:
        """Interpolate X-ray attenuation volume at arbitrary energy [cm⁻¹]."""
        result = np.zeros((self.Nz, self.Nx, self.Ny), dtype=np.float32)
        for idx, mat in enumerate(self.materials):
            mask = self.label_vol == idx
            if mask.any():
                result[mask] = mat.mu_x_at(energy_keV)
        return result

    def __repr__(self):
        mats = ", ".join(m.symbol for m in self.materials)
        return (
            f"PhantomData('{self.name}', shape=(Nz={self.Nz}, Nx={self.Nx}, Ny={self.Ny}), "
            f"{self.voxel_cm} cm/voxel, materials=[{mats}])"
        )


# ──────────────────────────────────────────────────────────────────────────────
# Builder class
# ──────────────────────────────────────────────────────────────────────────────

class PhantomBuilder:
    """
    Builds a voxel phantom by compositing geometric primitives.

    Storage / coordinate convention
    -------------------------------
    label_vol shape is (Nz, Nx, Ny):
        axis 0 = z
        axis 1 = x
        axis 2 = y

    Coordinates and centers use the same semantic order: (z, x, y).

    Usage
    -----
    Cubic, backward-compatible:
        >>> b = PhantomBuilder(N=64, voxel_cm=0.15)

    Non-cubic:
        >>> b = PhantomBuilder(Nx=96, Ny=64, Nz=128, voxel_cm=0.15)

    Geometry:
        >>> b.add_cylinder(material='aluminum', radius_cm=4.5, height_cm=9.0)
        >>> b.add_sphere(material='water', center_cm=(0, 0, 1), radius_cm=1.5)
        >>> b.add_box(material='hdpe', center_cm=(-2, 2, 0), half_extents_cm=(1, 1, 1))
        >>> phantom = b.build(name='my_phantom')
    """

    def __init__(
        self,
        N: Optional[int] = 64,
        voxel_cm: float = 0.15,
        Nx: Optional[int] = None,
        Ny: Optional[int] = None,
        Nz: Optional[int] = None,
    ):
        """
        Parameters
        ----------
        N:
            Backward-compatible cubic dimension. If Nx, Ny, and Nz are not
            supplied, the phantom shape is (N, N, N).
        voxel_cm:
            Voxel side length [cm].
        Nx, Ny, Nz:
            Optional non-cubic dimensions. If any of these are supplied, all
            three must be supplied. Storage shape will be (Nz, Nx, Ny).
        """
        if any(v is not None for v in (Nx, Ny, Nz)):
            if not all(v is not None for v in (Nx, Ny, Nz)):
                raise ValueError("Provide either N only, or all of Nx, Ny, and Nz.")
        else:
            if N is None:
                raise ValueError("Provide either N or all of Nx, Ny, and Nz.")
            Nx = Ny = Nz = N

        self.Nx = int(Nx)
        self.Ny = int(Ny)
        self.Nz = int(Nz)
        self.voxel_cm = float(voxel_cm)

        if self.Nx <= 0 or self.Ny <= 0 or self.Nz <= 0:
            raise ValueError("Nx, Ny, and Nz must all be positive integers.")
        if self.voxel_cm <= 0:
            raise ValueError("voxel_cm must be positive.")

        self._label_vol = np.zeros((self.Nz, self.Nx, self.Ny), dtype=np.uint8)
        self._materials: List[Material] = [MATERIALS["air"]]  # index 0 = air

        # Coordinate arrays for geometry tests (physical coords, cm).
        # Storage convention: axis 0 = z, axis 1 = x, axis 2 = y.
        z = self._axis_centres(self.Nz)
        x = self._axis_centres(self.Nx)
        y = self._axis_centres(self.Ny)
        self._Z, self._X, self._Y = np.meshgrid(z, x, y, indexing="ij")

    # ── Coordinate helpers ───────────────────────────────────────────────────

    def _axis_centres(self, n: int) -> np.ndarray:
        L = n * self.voxel_cm / 2.0
        return np.linspace(-L + self.voxel_cm / 2, L - self.voxel_cm / 2, n)

    def _full_length(self, axis: str) -> float:
        if axis == "z":
            return self.Nz * self.voxel_cm
        if axis == "x":
            return self.Nx * self.voxel_cm
        if axis == "y":
            return self.Ny * self.voxel_cm
        raise ValueError("axis must be 'x', 'y', or 'z'")

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
        """Add a solid sphere. center_cm is (z, x, y)."""
        cz, cx, cy = center_cm
        r2 = (
            (self._Z - cz) ** 2
            + (self._X - cx) ** 2
            + (self._Y - cy) ** 2
        )
        mask = r2 <= radius_cm ** 2
        self._label_vol[mask] = self._mat_index(material)
        return self

    def add_ellipsoid(
        self,
        material,
        center_cm: Tuple[float, float, float] = (0, 0, 0),
        semi_axes_cm: Tuple[float, float, float] = (1, 1, 1),
    ):
        """Add a solid axis-aligned ellipsoid. center/semi-axes are (z, x, y)."""
        cz, cx, cy = center_cm
        az, ax, ay = semi_axes_cm

        if az <= 0 or ax <= 0 or ay <= 0:
            raise ValueError("semi_axes_cm values must all be positive.")

        inside = (
            ((self._Z - cz) / az) ** 2
            + ((self._X - cx) / ax) ** 2
            + ((self._Y - cy) / ay) ** 2
        ) <= 1.0
        self._label_vol[inside] = self._mat_index(material)
        return self

    def add_disk(
        self,
        material,
        center_cm: Tuple[float, float, float] = (0, 0, 0),
        radius_cm: float = 1.0,
        thickness_cm: float = 0.1,
        axis: str = "z",
    ):
        """
        Add a circular disk, i.e. a very short filled cylinder.

        axis is the disk normal. Default is 'z'.
        """
        return self.add_cylinder(
            material=material,
            center_cm=center_cm,
            radius_cm=radius_cm,
            height_cm=thickness_cm,
            axis=axis,
        )

    def add_cylinder(
        self,
        material,
        center_cm: Tuple[float, float, float] = (0, 0, 0),
        radius_cm: float = 1.0,
        height_cm: Optional[float] = None,
        axis: str = "z",
    ):
        """
        Add a solid cylinder. axis can be 'x', 'y', or 'z'.

        center_cm is (z, x, y). If height_cm is None, the cylinder spans the
        full phantom along the cylinder axis. Default axis is 'z'.
        """
        cz, cx, cy = center_cm
        h_half = (height_cm / 2) if height_cm is not None else self._full_length(axis)

        if axis == "z":
            r2 = (self._X - cx) ** 2 + (self._Y - cy) ** 2
            in_cyl = (r2 <= radius_cm ** 2) & (np.abs(self._Z - cz) <= h_half)
        elif axis == "x":
            r2 = (self._Z - cz) ** 2 + (self._Y - cy) ** 2
            in_cyl = (r2 <= radius_cm ** 2) & (np.abs(self._X - cx) <= h_half)
        elif axis == "y":
            r2 = (self._Z - cz) ** 2 + (self._X - cx) ** 2
            in_cyl = (r2 <= radius_cm ** 2) & (np.abs(self._Y - cy) <= h_half)
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
        axis: str = "z",
    ):
        """
        Add a hollow cylindrical shell.

        center_cm is (z, x, y). Default axis is 'z'.
        """
        if inner_radius_cm < 0:
            raise ValueError("inner_radius_cm must be non-negative.")
        if outer_radius_cm <= inner_radius_cm:
            raise ValueError("outer_radius_cm must be larger than inner_radius_cm.")

        cz, cx, cy = center_cm
        h_half = (height_cm / 2) if height_cm is not None else self._full_length(axis)

        if axis == "z":
            r2 = (self._X - cx) ** 2 + (self._Y - cy) ** 2
            in_shell = (
                (r2 >= inner_radius_cm ** 2)
                & (r2 <= outer_radius_cm ** 2)
                & (np.abs(self._Z - cz) <= h_half)
            )
        elif axis == "x":
            r2 = (self._Z - cz) ** 2 + (self._Y - cy) ** 2
            in_shell = (
                (r2 >= inner_radius_cm ** 2)
                & (r2 <= outer_radius_cm ** 2)
                & (np.abs(self._X - cx) <= h_half)
            )
        elif axis == "y":
            r2 = (self._Z - cz) ** 2 + (self._X - cx) ** 2
            in_shell = (
                (r2 >= inner_radius_cm ** 2)
                & (r2 <= outer_radius_cm ** 2)
                & (np.abs(self._Y - cy) <= h_half)
            )
        else:
            raise ValueError("axis must be 'x', 'y', or 'z'")

        self._label_vol[in_shell] = self._mat_index(material)
        return self

    def add_box(
        self,
        material,
        center_cm: Tuple[float, float, float] = (0, 0, 0),
        half_extents_cm: Tuple[float, float, float] = (1, 1, 1),
    ):
        """Add a rectangular cuboid. center/half-extents are (z, x, y)."""
        cz, cx, cy = center_cm
        hz, hx, hy = half_extents_cm
        mask = (
            (np.abs(self._Z - cz) <= hz)
            & (np.abs(self._X - cx) <= hx)
            & (np.abs(self._Y - cy) <= hy)
        )
        self._label_vol[mask] = self._mat_index(material)
        return self

    def add_layer(
        self,
        material,
        position_cm: float,
        thickness_cm: float,
        axis: str = "z",
    ):
        """Add an infinite planar slab perpendicular to one axis. Default axis is 'z'."""
        lo = position_cm - thickness_cm / 2
        hi = position_cm + thickness_cm / 2

        if axis == "z":
            mask = (self._Z >= lo) & (self._Z < hi)
        elif axis == "x":
            mask = (self._X >= lo) & (self._X < hi)
        elif axis == "y":
            mask = (self._Y >= lo) & (self._Y < hi)
        else:
            raise ValueError("axis must be 'x', 'y', or 'z'")

        self._label_vol[mask] = self._mat_index(material)
        return self

    def add_rod(
        self,
        material,
        center_cm: Tuple[float, float] = (0, 0),
        radius_cm: float = 0.3,
        axis: str = "z",
    ):
        """
        Add a thin rod, i.e. an infinite cylinder along one axis.

        Default axis is 'z'. The 2-D center uses the two coordinates
        perpendicular to the rod axis:
            axis='z' -> center_cm = (x, y)
            axis='x' -> center_cm = (z, y)
            axis='y' -> center_cm = (z, x)
        """
        if axis == "z":
            cx, cy = center_cm
            r2 = (self._X - cx) ** 2 + (self._Y - cy) ** 2
        elif axis == "x":
            cz, cy = center_cm
            r2 = (self._Z - cz) ** 2 + (self._Y - cy) ** 2
        elif axis == "y":
            cz, cx = center_cm
            r2 = (self._Z - cz) ** 2 + (self._X - cx) ** 2
        else:
            raise ValueError("axis must be 'x', 'y', or 'z'")

        self._label_vol[r2 <= radius_cm ** 2] = self._mat_index(material)
        return self

    # ── Finaliser ─────────────────────────────────────────────────────────────

    def build(self, name: str = "phantom") -> PhantomData:
        """Return the finished PhantomData object."""
        return PhantomData(
            Nz=self.Nz,
            Nx=self.Nx,
            Ny=self.Ny,
            voxel_cm=self.voxel_cm,
            label_vol=self._label_vol.copy(),
            materials=list(self._materials),
            name=name,
        )


def make_composite_phantom(
    N: Optional[int] = 64,
    voxel_cm: Optional[float] = None,
    Nx: Optional[int] = None,
    Ny: Optional[int] = None,
    Nz: Optional[int] = None,
) -> PhantomData:
    """
    HDPE-matrix composite phantom — approximately 1 cm diameter by default.

    Backward-compatible use:
        make_composite_phantom(N=64)

    Non-cubic use:
        make_composite_phantom(Nx=96, Ny=64, Nz=128)

    Storage/order convention is (Nz, Nx, Ny). Coordinates are (z, x, y).

    A solid multi-material cylinder filled with HDPE (H-rich polymer matrix)
    containing inclusions of water, iron, and titanium. This is realistic for
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
    if any(v is not None for v in (Nx, Ny, Nz)):
        if not all(v is not None for v in (Nx, Ny, Nz)):
            raise ValueError("Provide either N only, or all of Nx, Ny, and Nz.")
        dims = (int(Nx), int(Ny), int(Nz))
    else:
        if N is None:
            raise ValueError("Provide either N or all of Nx, Ny, and Nz.")
        dims = (int(N), int(N), int(N))
        Nx = Ny = Nz = int(N)

    if voxel_cm is None:
        # Preserve old behavior for cubic phantoms: total side length = 1 cm.
        # For non-cubic phantoms, use the largest dimension so the largest side
        # is approximately 1 cm and geometry remains inside the volume.
        voxel_cm = 1.0 / max(dims)

    b = PhantomBuilder(N=None, Nx=Nx, Ny=Ny, Nz=Nz, voxel_cm=voxel_cm)

    # Use the smallest half-width so the circular cross-section fits inside
    # non-cubic x/y dimensions. The cylinder axis is now z by default.
    Lx = b.Nx * b.voxel_cm / 2
    Ly = b.Ny * b.voxel_cm / 2
    Lz = b.Nz * b.voxel_cm / 2
    L = min(Lx, Ly)

    r_outer = 0.82 * L
    wall = max(2 * voxel_cm, 0.02 * L)

    # Aluminium outer shell, cylinder axis along z.
    b.add_hollow_cylinder(
        "aluminum",
        outer_radius_cm=r_outer,
        inner_radius_cm=r_outer - wall,
        axis="z",
    )

    # HDPE fills the interior (base matrix — H-rich, invisible to X-rays).
    b.add_cylinder("hdpe", radius_cm=r_outer - wall, axis="z")

    # Water inclusion (sphere) — same low mu_x as HDPE but lower mu_n.
    b.add_sphere("water", center_cm=(0, 0.20 * L, 0), radius_cm=0.18 * L)

    # Iron rod — high mu_x, visible only in X-ray channel.
    # axis='z' means center is (x, y).
    b.add_rod("iron", center_cm=(0.30 * L, -0.12 * L), radius_cm=0.12 * L, axis="z")

    # Titanium sphere — high mu_x, lower mu_n than HDPE.
    b.add_sphere("titanium", center_cm=(0, 0.28 * L, 0.24 * L), radius_cm=0.10 * L)

    # Air void (simulates a crack or pore in the matrix).
    b.add_sphere("air", center_cm=(0, -0.18 * L, -0.22 * L), radius_cm=0.07 * L)

    return b.build("composite")



def make_battery_phantom(
    N: Optional[int] = 64,
    voxel_cm: Optional[float] = None,
    Nx: Optional[int] = None,
    Ny: Optional[int] = None,
    Nz: Optional[int] = None,
) -> PhantomData:
    """
    Alkaline AAA battery cross-section phantom — 1.4 cm diameter by default.

    Backward-compatible use:
        make_battery_phantom(N=64)

    Non-cubic use:
        make_battery_phantom(Nx=96, Ny=64, Nz=128)

    Storage/order convention is (Nz, Nx, Ny). Coordinates are (z, x, y).

    Matches a real AAA cell (diameter ≈ 10.5 mm) scaled to simulation.
    Demonstrates H-sensitivity of neutron imaging (HDPE separator,
    water-based KOH electrolyte visible only with neutrons).

    After LaManna et al. (NIST NeXT simultaneous neutron + X-ray).
    """
    if any(v is not None for v in (Nx, Ny, Nz)):
        if not all(v is not None for v in (Nx, Ny, Nz)):
            raise ValueError("Provide either N only, or all of Nx, Ny, and Nz.")
        dims = (int(Nx), int(Ny), int(Nz))
    else:
        if N is None:
            raise ValueError("Provide either N or all of Nx, Ny, and Nz.")
        Nx = Ny = Nz = int(N)
        dims = (Nx, Ny, Nz)

    if voxel_cm is None:
        # Preserve old cubic behavior: largest transverse side is 1.4 cm.
        voxel_cm = 1.4 / max(dims)

    b = PhantomBuilder(N=None, Nx=Nx, Ny=Ny, Nz=Nz, voxel_cm=voxel_cm)

    # Circular cross-section lies in the x-y plane; cylinder axis is z.
    Lx = b.Nx * b.voxel_cm / 2
    Ly = b.Ny * b.voxel_cm / 2
    L = min(Lx, Ly)

    r_can = 0.70 * L
    wall = max(2 * voxel_cm, 0.025 * L)

    # Steel can (iron approximation)
    b.add_hollow_cylinder(
        "iron",
        outer_radius_cm=r_can,
        inner_radius_cm=r_can - wall,
        axis="z",
    )

    # KOH electrolyte (approximated as water)
    b.add_cylinder("water", radius_cm=r_can - wall, axis="z")

    # HDPE separator ring
    b.add_hollow_cylinder(
        "hdpe",
        outer_radius_cm=0.55 * L,
        inner_radius_cm=0.45 * L,
        axis="z",
    )

    # Zinc anode rod
    b.add_cylinder("zinc", radius_cm=0.44 * L, axis="z")

    # Central air void (current collector channel)
    b.add_cylinder("air", radius_cm=0.06 * L, axis="z")

    return b.build("battery")



def make_bone_implant_phantom(
    N: Optional[int] = 64,
    voxel_cm: Optional[float] = None,
    Nx: Optional[int] = None,
    Ny: Optional[int] = None,
    Nz: Optional[int] = None,
) -> PhantomData:
    """
    Cortical bone + titanium implant phantom — 1 cm diameter by default.

    Backward-compatible use:
        make_bone_implant_phantom(N=64)

    Non-cubic use:
        make_bone_implant_phantom(Nx=96, Ny=64, Nz=128)

    Storage/order convention is (Nz, Nx, Ny). Coordinates are (z, x, y).

    After Törnquist et al. 2021 (Phys. Med. Biol. 66, 13).
    Demonstrates that neutrons resolve the bone–metal interface where X-rays
    suffer photon starvation next to the Ti implant.
    """
    if any(v is not None for v in (Nx, Ny, Nz)):
        if not all(v is not None for v in (Nx, Ny, Nz)):
            raise ValueError("Provide either N only, or all of Nx, Ny, and Nz.")
        dims = (int(Nx), int(Ny), int(Nz))
    else:
        if N is None:
            raise ValueError("Provide either N or all of Nx, Ny, and Nz.")
        Nx = Ny = Nz = int(N)
        dims = (Nx, Ny, Nz)

    if voxel_cm is None:
        # Preserve old cubic behavior: largest side is 1.0 cm.
        voxel_cm = 1.0 / max(dims)

    b = PhantomBuilder(N=None, Nx=Nx, Ny=Ny, Nz=Nz, voxel_cm=voxel_cm)

    # Main sample cross-section lies in x-y; cylinder axis is z.
    Lx = b.Nx * b.voxel_cm / 2
    Ly = b.Ny * b.voxel_cm / 2
    L = min(Lx, Ly)

    wall = max(2 * voxel_cm, 0.02 * L)

    # Cortical bone outer shell
    b.add_hollow_cylinder(
        "bone",
        outer_radius_cm=0.75 * L,
        inner_radius_cm=0.55 * L,
        axis="z",
    )

    # Water-based marrow
    b.add_cylinder("water", radius_cm=0.55 * L, axis="z")

    # Titanium screw, now running along the z-axis by default.
    # center_cm for axis='z' is (x, y).
    b.add_rod(
        "titanium",
        center_cm=(0.22 * L, 0.0),
        radius_cm=0.09 * L,
        axis="z",
    )

    # Peri-implant bone, thin ring around screw, also along z.
    b.add_hollow_cylinder(
        "bone",
        center_cm=(0, 0.22 * L, 0.0),
        inner_radius_cm=0.09 * L,
        outer_radius_cm=0.17 * L,
        axis="z",
    )

    return b.build("bone_implant")



def make_industrial_phantom(
    N: Optional[int] = 64,
    voxel_cm: Optional[float] = None,
    Nx: Optional[int] = None,
    Ny: Optional[int] = None,
    Nz: Optional[int] = None,
) -> PhantomData:
    """
    Industrial multi-material phantom — 1 cm diameter by default.

    Backward-compatible use:
        make_industrial_phantom(N=64)

    Non-cubic use:
        make_industrial_phantom(Nx=96, Ny=64, Nz=128)

    Storage/order convention is (Nz, Nx, Ny). Coordinates are (z, x, y).

    Contains tungsten and iron inserts to showcase beam hardening and neutron
    complementarity in NDE applications.
    W screws: μ_x(80keV)=88 cm⁻¹ → photon starvation even at ~0.5mm.
    W screws: μ_n=1.56 cm⁻¹ → well-resolved by neutrons.
    """
    if any(v is not None for v in (Nx, Ny, Nz)):
        if not all(v is not None for v in (Nx, Ny, Nz)):
            raise ValueError("Provide either N only, or all of Nx, Ny, and Nz.")
        dims = (int(Nx), int(Ny), int(Nz))
    else:
        if N is None:
            raise ValueError("Provide either N or all of Nx, Ny, and Nz.")
        Nx = Ny = Nz = int(N)
        dims = (Nx, Ny, Nz)

    if voxel_cm is None:
        # Preserve old cubic behavior: largest side is 1.0 cm.
        voxel_cm = 1.0 / max(dims)

    b = PhantomBuilder(N=None, Nx=Nx, Ny=Ny, Nz=Nz, voxel_cm=voxel_cm)

    # Circular cross-section lies in x-y; cylinder axis is z.
    Lx = b.Nx * b.voxel_cm / 2
    Ly = b.Ny * b.voxel_cm / 2
    L = min(Lx, Ly)

    wall = max(2 * voxel_cm, 0.02 * L)

    # Aluminium housing
    b.add_hollow_cylinder(
        "aluminum",
        outer_radius_cm=0.80 * L,
        inner_radius_cm=0.80 * L - wall,
        axis="z",
    )

    # HDPE matrix (H-rich filler, strongly scattering for neutrons)
    b.add_cylinder("hdpe", radius_cm=0.80 * L - wall, axis="z")

    # Tungsten rods — 4 at cardinal positions in the x-y plane.
    # axis='z' means center is (x, y).
    for ang in [0, 90, 180, 270]:
        rad = np.radians(ang)
        cx = 0.40 * L * np.cos(rad)
        cy = 0.40 * L * np.sin(rad)
        b.add_rod("tungsten", center_cm=(cx, cy), radius_cm=0.03 * L, axis="z")

    # Iron support bar, thin central bar.
    # center and half-extents are (z, x, y).
    b.add_box(
        "iron",
        center_cm=(0, 0.0, 0.0),
        half_extents_cm=(0.05 * L, 0.55 * L, 0.05 * L),
    )

    # Water pocket (coolant or defect), cylinder along z.
    b.add_cylinder(
        "water",
        center_cm=(0, 0.22 * L, 0),
        radius_cm=0.10 * L,
        axis="z",
    )

    # Air voids (defects / porosity). center is (z, x, y).
    b.add_sphere("air", center_cm=(0, -0.28 * L, 0.18 * L), radius_cm=0.05 * L)
    b.add_sphere("air", center_cm=(0, 0.10 * L, -0.28 * L), radius_cm=0.04 * L)

    return b.build("industrial")



def make_custom_cylindrical_battery_phantom(
    N: int = 256,
    length_cm: float = 2.0,
    voxel_cm: Optional[float] = None,
    *,
    diameter_cm: float = 1.0,
    n_jellyroll_turns: int = 2,

    shell_material: str = "Iron",
    cathode_collector_material: str = "aluminum",
    cathode_active_material: str = "nmc811",
    anode_collector_material: str = "copper",
    anode_active_material: str = "graphite",
    separator_material: str = "separator_pe",
    lithium_material: str = "Li",
    central_collector_material: str = "copper",

    can_thickness_cm: float = 0.03,
    cap_thickness_cm: float = 0.03,
    jellyroll_gap_top_cm: float = 0.05,
    jellyroll_gap_bottom_cm: float = 0.05,
    central_collector_inner_radius_cm: float = 0.04,
    central_collector_outer_radius_cm: float = 0.07,
    gap_to_first_jellyroll_cm: float = 0.05,

    separator_t_cm: float = 0.015,
    lithium_t_cm: float = 0.005,
    al_t_cm: float = 0.01,
    cathode_t_cm: float = 0.025,
    anode_t_cm: float = 0.025,
    copper_t_cm: float = 0.01,

    center_cm: Tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> PhantomData:
    """
    Build a parameterized cylindrical battery phantom.

    Coordinate convention:
        center_cm = (z, x, y)
        storage shape = (Nz, Nx, Ny)

    Default geometry:
        diameter = 1 cm
        length   = 2 cm
        axis     = z

    Phantom grid:
        Nx = N
        Ny = N
        Nz chosen from length_cm / voxel_cm

    Jellyroll:
        separator + lithium
        aluminum cathode collector
        cathode active material
        separator + lithium
        anode active material
        copper anode collector
        separator + lithium
    """

    required_materials = {
        "shell_material": shell_material,
        "cathode_collector_material": cathode_collector_material,
        "cathode_active_material": cathode_active_material,
        "anode_collector_material": anode_collector_material,
        "anode_active_material": anode_active_material,
        "separator_material": separator_material,
        "lithium_material": lithium_material,
        "central_collector_material": central_collector_material,
    }

    missing = [k for k, v in required_materials.items() if v not in MATERIALS]
    if missing:
        raise ValueError(
            "Unknown material(s): "
            + ", ".join(f"{k}='{required_materials[k]}'" for k in missing)
            + f". Available materials: {list(MATERIALS.keys())}"
        )

    if voxel_cm is None:
        voxel_cm = diameter_cm / N

    Nx = N
    Ny = N
    Nz = int(np.ceil(length_cm / voxel_cm))

    b = PhantomBuilder(
        Nx=Nx,
        Ny=Ny,
        Nz=Nz,
        voxel_cm=voxel_cm,
    )

    cz0, cx0, cy0 = center_cm

    outer_radius_cm = diameter_cm / 2
    outer_height_cm = length_cm

    inner_radius_cm = outer_radius_cm - can_thickness_cm
    if inner_radius_cm <= 0:
        raise ValueError("can_thickness_cm is too large.")

    inner_height_cm = outer_height_cm - 2 * cap_thickness_cm
    jellyroll_height_cm = (
        inner_height_cm
        - jellyroll_gap_top_cm
        - jellyroll_gap_bottom_cm
    )

    if jellyroll_height_cm <= 0:
        raise ValueError(
            "Jellyroll height became non-positive. "
            "Reduce cap_thickness_cm or jellyroll axial gaps."
        )

    if central_collector_outer_radius_cm >= inner_radius_cm:
        raise ValueError("Central collector is too large for the cell.")

    # Outer cylindrical can
    b.add_hollow_cylinder(
        shell_material,
        center_cm=center_cm,
        inner_radius_cm=inner_radius_cm,
        outer_radius_cm=outer_radius_cm,
        height_cm=outer_height_cm,
        axis="z",
    )

    # Bottom and top caps
    bottom_cap_z = cz0 - outer_height_cm / 2 + cap_thickness_cm / 2
    top_cap_z = cz0 + outer_height_cm / 2 - cap_thickness_cm / 2

    b.add_disk(
        shell_material,
        center_cm=(bottom_cap_z, cx0, cy0),
        radius_cm=outer_radius_cm,
        thickness_cm=cap_thickness_cm,
        axis="z",
    )

    b.add_disk(
        shell_material,
        center_cm=(top_cap_z, cx0, cy0),
        radius_cm=outer_radius_cm,
        thickness_cm=cap_thickness_cm,
        axis="z",
    )

    # Central hollow copper current collector
    b.add_hollow_cylinder(
        central_collector_material,
        center_cm=center_cm,
        inner_radius_cm=central_collector_inner_radius_cm,
        outer_radius_cm=central_collector_outer_radius_cm,
        height_cm=jellyroll_height_cm,
        axis="z",
    )

    # Jellyroll layers
    r_inner = central_collector_outer_radius_cm + gap_to_first_jellyroll_cm

    layer_sequence = [
        (separator_material, separator_t_cm),
        (lithium_material, lithium_t_cm),

        (cathode_collector_material, al_t_cm),
        (cathode_active_material, cathode_t_cm),

        (separator_material, separator_t_cm),
        (lithium_material, lithium_t_cm),

        (anode_active_material, anode_t_cm),
        (anode_collector_material, copper_t_cm),

        (separator_material, separator_t_cm),
        (lithium_material, lithium_t_cm),
    ]

    for turn_idx in range(n_jellyroll_turns):
        for mat, t_cm in layer_sequence:
            r_outer = r_inner + t_cm

            if r_outer >= inner_radius_cm:
                raise ValueError(
                    f"Jellyroll exceeds inner cell radius during turn {turn_idx + 1}. "
                    "Reduce n_jellyroll_turns or layer thicknesses."
                )

            b.add_hollow_cylinder(
                mat,
                center_cm=center_cm,
                inner_radius_cm=r_inner,
                outer_radius_cm=r_outer,
                height_cm=jellyroll_height_cm,
                axis="z",
            )

            r_inner = r_outer

    name = (
        f"custom_cylindrical_battery_"
        f"D{diameter_cm:g}cm_L{length_cm:g}cm_"
        f"{n_jellyroll_turns}_turns"
    )

    return b.build(name)




# ── Registry ──────────────────────────────────────────────────────────────────

PHANTOM_PRESETS: Dict[str, callable] = {
    "composite":     make_composite_phantom,
    "battery":       make_battery_phantom,
    "bone_implant":  make_bone_implant_phantom,
    "industrial":    make_industrial_phantom,
    "jellyroll_battery" : make_custom_cylindrical_battery_phantom
    


}

def make_phantom(
    preset: str = "composite",
    N: Optional[int] = 64,
    Nx: Optional[int] = None,
    Ny: Optional[int] = None,
    Nz: Optional[int] = None,
    voxel_cm: Optional[float] = None,
) -> PhantomData:
    """
    Load a named preset phantom.

    Backward-compatible cubic use:
        make_phantom("composite", N=64)

    Non-cubic use:
        make_phantom("composite", Nx=96, Ny=64, Nz=128)

    Storage convention is (Nz, Nx, Ny), with coordinates ordered as (z, x, y).

    Sample sizes are physically realistic for neutron tomography:
      composite / bone_implant / industrial  →  1.0 cm diameter
      battery                                →  1.4 cm diameter (AAA cell)

    Voxel size scales automatically with N or max(Nx, Ny, Nz) so geometry is
    preserved unless voxel_cm is explicitly supplied.
    """
    if preset not in PHANTOM_PRESETS:
        raise ValueError(
            f"Unknown preset '{preset}'. Choose from: {list(PHANTOM_PRESETS)}"
        )

    if any(v is not None for v in (Nx, Ny, Nz)):
        if not all(v is not None for v in (Nx, Ny, Nz)):
            raise ValueError("Provide either N only, or all of Nx, Ny, and Nz.")

        return PHANTOM_PRESETS[preset](
            N=None,
            Nx=Nx,
            Ny=Ny,
            Nz=Nz,
            voxel_cm=voxel_cm,
        )

    if N is None:
        raise ValueError("Provide either N or all of Nx, Ny, and Nz.")

    return PHANTOM_PRESETS[preset](
        N=N,
        voxel_cm=voxel_cm,
    )