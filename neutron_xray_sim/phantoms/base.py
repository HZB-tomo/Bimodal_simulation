"""
neutron_xray_sim.phantoms.base
──────────────────────────────
Voxelised 3-D phantom container and builder.

A :class:`PhantomData` stores a label volume (integer material indices) plus
the derived attenuation-coefficient volumes (one per neutron cross-section
component, one per energy bin for X-rays).  :class:`PhantomBuilder` composites
geometric primitives into a label volume.

Preset phantoms live in :mod:`neutron_xray_sim.phantoms.presets`.

Conventions
───────────
* Storage order is ``(Nz, Nx, Ny)``; physical coordinates are ``(z, x, y)`` in
  cm, centred on the volume centre.
* Label 0 is always air.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Union

import numpy as np

from ..physics.materials import MATERIALS, XRAY_E_KEV, Material

__all__ = ["PhantomData", "PhantomBuilder", "MaterialLike"]

#: Anything accepted where a material is expected: a ``Material`` or a
#: ``MATERIALS`` key.
MaterialLike = Union[str, Material]


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

    def _mat_index(self, material: MaterialLike) -> int:
        if isinstance(material, str):
            if material not in MATERIALS:
                raise KeyError(
                    f"Unknown material '{material}'. Available: {sorted(MATERIALS)}"
                )
            material = MATERIALS[material]
        for idx, known in enumerate(self._materials):
            if known is material:
                return idx
        if len(self._materials) >= 256:
            raise ValueError("A phantom can hold at most 256 materials (uint8 labels).")
        self._materials.append(material)
        return len(self._materials) - 1

    def paint(self, material: MaterialLike, mask: np.ndarray):
        """
        Assign *material* to every voxel where boolean *mask* is True.

        Use this for custom geometry that the primitives below cannot express;
        coordinate grids are available as ``builder.Z``, ``builder.X``,
        ``builder.Y`` (cm, shape ``(Nz, Nx, Ny)``).
        """
        mask = np.asarray(mask, dtype=bool)
        if mask.shape != self._label_vol.shape:
            raise ValueError(
                f"mask shape {mask.shape} does not match volume {self._label_vol.shape}"
            )
        self._label_vol[mask] = self._mat_index(material)
        return self

    # Public read-only access to the coordinate grids (cm).
    @property
    def Z(self) -> np.ndarray:
        return self._Z

    @property
    def X(self) -> np.ndarray:
        return self._X

    @property
    def Y(self) -> np.ndarray:
        return self._Y

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
