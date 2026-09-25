"""
neutron_xray_sim/phantom.py
────────────────────────────
Voxelised 3-D phantom builder.

A phantom stores a label volume (integer material indices) plus the material
list it indexes into.  Everything else — the neutron attenuation volumes and
the per-energy X-ray volumes — is *derived* from those two, and is built lazily
on first access (see :class:`PhantomData`).

Adding a preset
───────────────
Decorate a factory with :func:`register_phantom` and it becomes available
through :func:`make_phantom`, the GUI, and every example script::

    @register_phantom("my_sample", description="Two-phase sintered pellet")
    def make_my_sample_phantom(N=64, voxel_cm=None, Nx=None, Ny=None, Nz=None):
        Nx, Ny, Nz, voxel_cm = resolve_grid(N, Nx, Ny, Nz, voxel_cm, extent_cm=1.0)
        b = PhantomBuilder(Nx=Nx, Ny=Ny, Nz=Nz, voxel_cm=voxel_cm)
        ...
        return b.build("my_sample")

Every preset takes the same five keyword arguments, so ``make_phantom`` can
call any of them the same way.  :func:`resolve_grid` implements the shared
"either N, or all of Nx/Ny/Nz" rule that used to be copy-pasted into each
factory.
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass, field
from math import ceil
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from .materials import MATERIALS, XRAY_E_KEV, Material

__all__ = [
    "PhantomData", "PhantomBuilder", "make_phantom", "resolve_grid",
    "register_phantom", "PHANTOM_PRESETS", "PHANTOM_DESCRIPTIONS",
    "make_composite_phantom", "make_battery_phantom",
    "make_bone_implant_phantom", "make_industrial_phantom",
    "make_hdpe_composite_phantom", "make_custom_cylindrical_battery_phantom",
    "make_li_ion_battery_phantom",
]


# ──────────────────────────────────────────────────────────────────────────────
# Preset registry
# ──────────────────────────────────────────────────────────────────────────────

#: ``preset name → factory``. Populated by :func:`register_phantom`; never edit
#: it directly, so that the signature check stays enforced.
PHANTOM_PRESETS: Dict[str, Callable[..., PhantomData]] = {}

#: ``preset name → one-line description``, for menus, docs, and the GUI.
PHANTOM_DESCRIPTIONS: Dict[str, str] = {}

#: Every preset factory must accept exactly these keyword arguments so that
#: :func:`make_phantom` can call any of them identically.
_PRESET_SIGNATURE = ("N", "voxel_cm", "Nx", "Ny", "Nz")


def register_phantom(name: str, *, description: str = ""):
    """Register a phantom factory under *name*.

    The factory must accept the five standard keyword arguments
    ``N, voxel_cm, Nx, Ny, Nz`` — extra keyword arguments with defaults are
    fine.  The check happens at import time, so a preset that ``make_phantom``
    could not call fails loudly during development rather than silently at the
    bottom of someone's sweep.

    Raises
    ------
    ValueError
        If *name* is already registered, or the signature is incompatible.
    """
    def decorator(func: Callable[..., PhantomData]):
        if name in PHANTOM_PRESETS:
            raise ValueError(
                f"Phantom preset {name!r} is already registered by "
                f"{PHANTOM_PRESETS[name].__module__}.{PHANTOM_PRESETS[name].__name__}."
            )
        params = inspect.signature(func).parameters
        missing = [p for p in _PRESET_SIGNATURE if p not in params]
        if missing:
            raise ValueError(
                f"Phantom preset {name!r} ({func.__name__}) is missing required "
                f"keyword argument(s) {missing}. Every preset must accept "
                f"{list(_PRESET_SIGNATURE)} so make_phantom() can call it."
            )
        PHANTOM_PRESETS[name] = func
        PHANTOM_DESCRIPTIONS[name] = description or (func.__doc__ or "").strip().split("\n")[0]
        return func
    return decorator


def resolve_grid(
    N: Optional[int] = 64,
    Nx: Optional[int] = None,
    Ny: Optional[int] = None,
    Nz: Optional[int] = None,
    voxel_cm: Optional[float] = None,
    *,
    extent_cm: float = 1.0,
) -> Tuple[int, int, int, float]:
    """Resolve the "either N, or all of Nx/Ny/Nz" convention into a grid.

    Every preset factory starts by calling this, which is why they all accept
    the same arguments and reject the same mistakes.  Previously each factory
    carried its own copy of the logic, and they had drifted apart.

    Parameters
    ----------
    N
        Cubic grid size. Used when *Nx*, *Ny*, *Nz* are all omitted.
    Nx, Ny, Nz
        Non-cubic grid. All three must be given together.
    voxel_cm
        Voxel side length [cm]. When ``None``, it is chosen so the phantom's
        largest side spans *extent_cm*.
    extent_cm
        Physical size of the largest side [cm] used for the automatic voxel
        size — 1.0 for most samples, 1.4 for the AAA-cell battery.

    Returns
    -------
    (Nx, Ny, Nz, voxel_cm)
    """
    given = [v for v in (Nx, Ny, Nz) if v is not None]
    if given:
        if len(given) != 3:
            raise ValueError(
                "Provide either N alone, or all three of Nx, Ny and Nz "
                f"(got Nx={Nx!r}, Ny={Ny!r}, Nz={Nz!r})."
            )
        Nx, Ny, Nz = int(Nx), int(Ny), int(Nz)
    else:
        if N is None:
            raise ValueError("Provide either N, or all three of Nx, Ny and Nz.")
        Nx = Ny = Nz = int(N)

    if min(Nx, Ny, Nz) <= 0:
        raise ValueError(
            f"Grid dimensions must be positive, got Nx={Nx}, Ny={Ny}, Nz={Nz}."
        )

    if voxel_cm is None:
        voxel_cm = extent_cm / max(Nx, Ny, Nz)
    voxel_cm = float(voxel_cm)
    if voxel_cm <= 0:
        raise ValueError(f"voxel_cm must be positive, got {voxel_cm}.")

    return Nx, Ny, Nz, voxel_cm


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

    # Derived attenuation volumes. These are *outputs*, not inputs: they are
    # computed from label_vol + materials on first access and cached here.
    # Passing them to the constructor is supported (NCrystal overrides the
    # neutron channel that way) but is not the normal path.
    mu_n_vol: Optional[np.ndarray] = field(default=None, repr=False)
    mu_n_abs_vol: Optional[np.ndarray] = field(default=None, repr=False)
    mu_n_coh_vol: Optional[np.ndarray] = field(default=None, repr=False)
    mu_n_inc_vol: Optional[np.ndarray] = field(default=None, repr=False)

    def __post_init__(self):
        self._validate_shape()
        self._mu_x_cache: Optional[np.ndarray] = None
        if self.mu_n_vol is None:
            self._build_neutron_vols()

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

    def _build_neutron_vols(self):
        """Fill the four neutron attenuation volumes from labels + materials.

        A single pass over the label volume builds a per-material lookup table
        and indexes with it, which is both faster and clearer than one boolean
        mask per material per array.
        """
        labels = self.label_vol
        n_mat = len(self.materials)

        lut = np.zeros((4, max(n_mat, int(labels.max()) + 1)), dtype=np.float32)
        for idx, mat in enumerate(self.materials):
            lut[:, idx] = (mat.mu_n, mat.mu_n_abs, mat.mu_n_coh, mat.mu_n_inc)

        self.mu_n_vol = lut[0][labels]
        self.mu_n_abs_vol = lut[1][labels]
        self.mu_n_coh_vol = lut[2][labels]
        self.mu_n_inc_vol = lut[3][labels]

    # ── Derived X-ray volumes (lazy) ─────────────────────────────────────────

    @property
    def mu_x_vols(self) -> np.ndarray:
        """X-ray attenuation at every ``XRAY_E_KEV`` energy, ``(13, Nz, Nx, Ny)``.

        Built on first access and cached.  Note the size: this array is 13×
        the label volume in float32, so a 512³ phantom needs ~7 GB.  Prefer
        :meth:`mu_x_at_index` or :meth:`mu_x_at_energy`, which materialise one
        energy at a time — the projector and histogram code both do.
        """
        if self._mu_x_cache is None:
            self._mu_x_cache = np.stack(
                [self.mu_x_at_index(e) for e in range(len(XRAY_E_KEV))]
            )
        return self._mu_x_cache

    @mu_x_vols.setter
    def mu_x_vols(self, value: Optional[np.ndarray]) -> None:
        self._mu_x_cache = value

    def _map_over_labels(self, values: np.ndarray) -> np.ndarray:
        """Expand a per-material value array into a full volume."""
        lut = np.zeros(max(len(self.materials), int(self.label_vol.max()) + 1),
                       dtype=np.float32)
        lut[:len(values)] = values
        return lut[self.label_vol]

    def mu_x_at_index(self, energy_idx: int) -> np.ndarray:
        """X-ray attenuation volume at ``XRAY_E_KEV[energy_idx]`` [cm⁻¹]."""
        if not 0 <= energy_idx < len(XRAY_E_KEV):
            raise IndexError(
                f"energy_idx {energy_idx} is out of range for the "
                f"{len(XRAY_E_KEV)}-point grid {list(XRAY_E_KEV)}."
            )
        return self._map_over_labels(
            np.array([m.mu_x_table[energy_idx] for m in self.materials], dtype=np.float32)
        )

    def mu_x_at_energy(self, energy_keV: float) -> np.ndarray:
        """X-ray attenuation volume interpolated at an arbitrary energy [cm⁻¹]."""
        return self._map_over_labels(
            np.array([m.mu_x_at(energy_keV) for m in self.materials], dtype=np.float32)
        )

    # ── Public helpers ────────────────────────────────────────────────────────

    def material_name(self, label: int) -> str:
        return self.materials[label].name if label < len(self.materials) else "unknown"

    def voxel_counts(self) -> np.ndarray:
        """Number of voxels assigned to each material, indexed like `materials`."""
        return np.bincount(self.label_vol.ravel(), minlength=len(self.materials))

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
        self._material_index: Dict[int, int] = {id(self._materials[0]): 0}

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
        """Return the label index for *material*, appending it if new.

        Indexed by ``id()`` rather than by value: ``Material`` holds a NumPy
        array, so the old ``material in self._materials`` scan compared arrays
        elementwise and raised "truth value of an array is ambiguous" whenever
        two materials shared a name.  Identity is also the right semantics —
        two distinct ``Material`` objects are two phases even if their numbers
        happen to coincide.
        """
        if isinstance(material, str):
            try:
                material = MATERIALS[material]
            except KeyError as exc:
                raise KeyError(
                    f"Unknown material {material!r}. "
                    f"Available: {', '.join(sorted(MATERIALS))}. "
                    "Register new materials with MATERIALS.register(...)."
                ) from exc

        key = id(material)
        if key not in self._material_index:
            self._material_index[key] = len(self._materials)
            self._materials.append(material)
            if len(self._materials) > 255:
                raise ValueError(
                    "A phantom may hold at most 255 materials plus air, because "
                    "the label volume is uint8."
                )
        return self._material_index[key]

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


@register_phantom("composite", description="HDPE matrix with water / Fe / Ti inclusions (~1 cm)")
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
    Nx, Ny, Nz, voxel_cm = resolve_grid(
        N, Nx, Ny, Nz, voxel_cm, extent_cm=1.0
    )
    b = PhantomBuilder(Nx=Nx, Ny=Ny, Nz=Nz, voxel_cm=voxel_cm)

    # Use the smallest half-width so the circular cross-section fits inside
    # non-cubic x/y dimensions. The cylinder axis is now z by default.
    Lx = b.Nx * b.voxel_cm / 2
    Ly = b.Ny * b.voxel_cm / 2
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



@register_phantom("battery", description="Alkaline AAA cell cross-section (~1.4 cm)")
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
    Nx, Ny, Nz, voxel_cm = resolve_grid(
        N, Nx, Ny, Nz, voxel_cm, extent_cm=1.4
    )
    b = PhantomBuilder(Nx=Nx, Ny=Ny, Nz=Nz, voxel_cm=voxel_cm)

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



@register_phantom("bone_implant", description="Cortical bone with a titanium screw (~1 cm)")
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
    Nx, Ny, Nz, voxel_cm = resolve_grid(
        N, Nx, Ny, Nz, voxel_cm, extent_cm=1.0
    )
    b = PhantomBuilder(Nx=Nx, Ny=Ny, Nz=Nz, voxel_cm=voxel_cm)

    # Main sample cross-section lies in x-y; cylinder axis is z.
    Lx = b.Nx * b.voxel_cm / 2
    Ly = b.Ny * b.voxel_cm / 2
    L = min(Lx, Ly)

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



@register_phantom("industrial", description="Multi-material NDE part with W and Fe inserts (~1 cm)")
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
    Nx, Ny, Nz, voxel_cm = resolve_grid(
        N, Nx, Ny, Nz, voxel_cm, extent_cm=1.0
    )
    b = PhantomBuilder(Nx=Nx, Ny=Ny, Nz=Nz, voxel_cm=voxel_cm)

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

@register_phantom("HDPE_composite", description="HDPE block with steel rod, Al/Fe cubes and voids")
def make_hdpe_composite_phantom(
    N: Optional[int] = 64,
    voxel_cm: Optional[float] = None,
    Nx: Optional[int] = None,
    Ny: Optional[int] = None,
    Nz: Optional[int] = None,
) -> PhantomData:
    """
    HDPE block composite phantom.

    Geometry:
      - HDPE rectangular block as matrix
      - Steel rod along z direction in one corner
      - Aluminum cube in the center
      - Iron cube inside the aluminum cube
      - Air bubbles dispersed in HDPE
      - Some bubbles filled with water

    Storage/order convention is (Nz, Nx, Ny).
    Coordinates are (z, x, y).
    """

    Nx, Ny, Nz, voxel_cm = resolve_grid(
        N, Nx, Ny, Nz, voxel_cm, extent_cm=1.0
    )
    b = PhantomBuilder(Nx=Nx, Ny=Ny, Nz=Nz, voxel_cm=voxel_cm)

    Lx = b.Nx * b.voxel_cm / 2
    Ly = b.Ny * b.voxel_cm / 2
    Lz = b.Nz * b.voxel_cm / 2
    L = min(Lx, Ly, Lz)

    # ------------------------------------------------------------------
    # 1. HDPE base block
    # ------------------------------------------------------------------
    block_hz = 0.78 * Lz
    block_hx = 0.78 * Lx
    block_hy = 0.78 * Ly

    b.add_box(
        "hdpe",
        center_cm=(0.0, 0.0, 0.0),
        half_extents_cm=(block_hz, block_hx, block_hy),
    )

    # ------------------------------------------------------------------
    # 2. Steel rod along z direction, close to one corner
    # ------------------------------------------------------------------
    rod_radius = 0.08 * L
    rod_x = -0.50 * Lx
    rod_y = -0.50 * Ly

    b.add_rod(
        "steel",
        center_cm=(rod_x, rod_y),
        radius_cm=rod_radius,
        axis="z",
    )

    # ------------------------------------------------------------------
    # 3. Central aluminum cube
    # ------------------------------------------------------------------
    al_half = 0.22 * L

    b.add_box(
        "aluminum",
        center_cm=(0.0, 0.0, 0.0),
        half_extents_cm=(al_half, al_half, al_half),
    )

    # ------------------------------------------------------------------
    # 4. Iron cube inside aluminum cube
    # ------------------------------------------------------------------
    fe_half = 0.10 * L

    b.add_box(
        "iron",
        center_cm=(0.0, 0.0, 0.0),
        half_extents_cm=(fe_half, fe_half, fe_half),
    )

    # ------------------------------------------------------------------
    # 5. Air bubbles, placed away from rod and central cube
    # ------------------------------------------------------------------
    air_bubbles = [
        ((-0.45 * Lz,  0.45 * Lx, -0.35 * Ly), 0.060 * L),
        (( 0.42 * Lz, -0.35 * Lx,  0.42 * Ly), 0.055 * L),
        ((-0.25 * Lz,  0.55 * Lx,  0.35 * Ly), 0.050 * L),
        (( 0.50 * Lz,  0.35 * Lx, -0.45 * Ly), 0.055 * L),
    ]

    for center, radius in air_bubbles:
        b.add_sphere(
            "air",
            center_cm=center,
            radius_cm=radius,
        )

    # ------------------------------------------------------------------
    # 6. Water-filled bubbles
    # ------------------------------------------------------------------
    water_bubbles = [
        (( 0.35 * Lz, -0.55 * Lx, -0.20 * Ly), 0.060 * L),
        ((-0.55 * Lz,  0.25 * Lx,  0.45 * Ly), 0.055 * L),
    ]

    for center, radius in water_bubbles:
        b.add_sphere(
            "water",
            center_cm=center,
            radius_cm=radius,
        )

    return b.build("hdpe_composite")




@register_phantom("spiral_battery", description="Cylindrical Li-ion cell with an Archimedean-spiral jellyroll")
def make_li_ion_battery_phantom(
    N: Optional[int] = 512,
    voxel_cm: Optional[float] = None,
    Nx: Optional[int] = None,
    Ny: Optional[int] = None,
    Nz: Optional[int] = None,
    *,
    diameter_cm: float = 1.0,
    length_cm: float = 2.0,
    cathode_material: str = "nmc811",
    can_material: str = "steel",
    separator_material: str = "separator_pe_electrolyte",
    name: str = "li_ion_battery_spiral",
) -> PhantomData:
    """
    Cylindrical lithium-ion battery phantom with:
    - steel can and end disks
    - central hollow/solid copper collector rod
    - Archimedean spiral jellyroll
    - layer sequence:
      separator / cathode / Al / cathode / separator /
      graphite / Cu / graphite / separator

    Storage order: (Nz, Nx, Ny)
    Coordinate order: (z, x, y)
    """

    # -------------------------
    # Unit conversions
    # -------------------------
    mm = 0.1       # cm
    um = 1e-4      # cm

    can_thickness = 0.25 * mm          # 0.025 cm
    cathode_coating = 62.5 * um        # 0.00625 cm
    anode_coating = 70.0 * um          # 0.007 cm
    separator = 20.0 * um              # 0.002 cm
    al_cc = 15.0 * um                  # 0.0015 cm
    cu_cc = 10.0 * um                  # 0.001 cm

    # End caps
    bottom_disk_thickness = can_thickness
    top_disk_thickness = 2.0 * can_thickness

    # Central copper rod
    rod_outer_diameter_cm = 1.5 * mm   # 1.5 mm total diameter
    rod_wall_thickness_cm = 0.2 * mm   # as requested

    rod_outer_radius = rod_outer_diameter_cm / 2
    rod_inner_radius = rod_outer_radius - rod_wall_thickness_cm

    # If wall thickness is larger than radius, make it solid.
    rod_is_solid = rod_inner_radius <= 0
    rod_inner_radius = max(0.0, rod_inner_radius)

    rod_height_cm = length_cm - length_cm/8
    rod_bottom_z = -length_cm / 2 + bottom_disk_thickness + length_cm/20
    rod_center_z = rod_bottom_z + rod_height_cm / 2 \
        - length_cm /80 # clearance from the bootom

    # Jellyroll geometry
    outer_radius = diameter_cm / 2
    inner_can_radius = outer_radius - can_thickness

    jellyroll_inner_gap = 0.1 * mm
    jellyroll_inner_radius = rod_outer_radius + jellyroll_inner_gap
    jellyroll_outer_radius = inner_can_radius - 0.02  # small clearance
    # Jellyroll starts exactly where rod starts
    jellyroll_z_min = rod_bottom_z

    # It fills upward but still respects top clearance
    jellyroll_z_max = min(
    rod_bottom_z + rod_height_cm,   # optional: same height as rod
    length_cm / 2 - top_disk_thickness
    ) - length_cm/40 # clearance from the bottom


    layer_stack = [
        ("separator", separator_material, separator),
        ("cathode_1", cathode_material, cathode_coating),
        ("al_cc", "aluminum", al_cc),
        ("cathode_2", cathode_material, cathode_coating),
        ("separator_2", separator_material, separator),
        ("graphite_1", "graphite", anode_coating),
        ("cu_cc", "copper", cu_cc),
        ("graphite_2", "graphite", anode_coating),
        ("separator_3", separator_material, separator),
    ]

    pitch_cm = sum(t for _, _, t in layer_stack)

    # -------------------------
    # Grid
    # -------------------------
    # The cell is not cubic: N sets the transverse sampling across the
    # diameter, and the axial extent follows from length_cm.
    if Nx is None and Ny is None and Nz is None:
        if voxel_cm is None:
            if not N:
                raise ValueError("Provide N, voxel_cm, or all of Nx, Ny and Nz.")
            voxel_cm = diameter_cm / int(N)
        Nx = Ny = int(round(diameter_cm / voxel_cm))
        Nz = int(ceil(length_cm / voxel_cm))

    Nx, Ny, Nz, voxel_cm = resolve_grid(
        None, Nx, Ny, Nz, voxel_cm, extent_cm=max(diameter_cm, length_cm)
    )
    b = PhantomBuilder(Nx=Nx, Ny=Ny, Nz=Nz, voxel_cm=voxel_cm)

    # -------------------------
    # Can wall and end disks
    # -------------------------
    b.add_hollow_cylinder(
        can_material,
        inner_radius_cm=inner_can_radius,
        outer_radius_cm=outer_radius,
        height_cm=length_cm,
        axis="z",
    )

    b.add_disk(
        can_material,
        center_cm=(-length_cm / 2 + bottom_disk_thickness / 2, 0, 0),
        radius_cm=outer_radius,
        thickness_cm=bottom_disk_thickness,
        axis="z",
    )

    b.add_disk(
        can_material,
        center_cm=(length_cm / 2 - top_disk_thickness / 2, 0, 0),
        radius_cm=outer_radius,
        thickness_cm=top_disk_thickness,
        axis="z",
    )

    # -------------------------
    # Central copper collector rod
    # -------------------------
    if rod_is_solid:
        b.add_cylinder(
            "copper",
            center_cm=(rod_center_z, 0, 0),
            radius_cm=rod_outer_radius,
            height_cm=rod_height_cm,
            axis="z",
        )
    else:
        b.add_hollow_cylinder(
            "copper",
            center_cm=(rod_center_z, 0, 0),
            inner_radius_cm=rod_inner_radius,
            outer_radius_cm=rod_outer_radius,
            height_cm=rod_height_cm,
            axis="z",
        )

    # -------------------------
    # Spiral jellyroll
    # -------------------------
    X = b._X
    Y = b._Y
    Z = b._Z

    r = np.sqrt(X**2 + Y**2)
    theta = np.arctan2(Y, X)
    theta = np.mod(theta, 2 * np.pi)

    # Archimedean spiral coordinate.
    # This makes the material bands wind around the center instead of
    # forming purely concentric rings.
    spiral_coord = r - jellyroll_inner_radius - pitch_cm * theta / (2 * np.pi)

    jellyroll_mask = (
        (r >= jellyroll_inner_radius)
        & (r <= jellyroll_outer_radius)
        & (Z >= jellyroll_z_min)
        & (Z <= jellyroll_z_max)
        & (spiral_coord >= 0)
    )

    layer_pos = np.mod(spiral_coord, pitch_cm)

    cumulative = 0.0
    for _, material, thickness in layer_stack:
        lo = cumulative
        hi = cumulative + thickness

        layer_mask = jellyroll_mask & (layer_pos >= lo) & (layer_pos < hi)
        b._label_vol[layer_mask] = b._mat_index(material)

        cumulative = hi

    phantom = b.build(name)

    # Useful metadata as dynamic attributes
    phantom.pitch_cm = pitch_cm
    phantom.pitch_um = pitch_cm / um
    phantom.voxel_um = voxel_cm / um
    phantom.jellyroll_inner_radius_cm = jellyroll_inner_radius
    phantom.jellyroll_outer_radius_cm = jellyroll_outer_radius
    phantom.rod_is_solid = rod_is_solid

    return phantom


@register_phantom("jellyroll_battery", description="Parameterised cylindrical Li-ion cell with concentric jellyroll turns")
def make_custom_cylindrical_battery_phantom(
    N: Optional[int] = 256,
    voxel_cm: Optional[float] = None,
    Nx: Optional[int] = None,
    Ny: Optional[int] = None,
    Nz: Optional[int] = None,
    *,
    length_cm: float = 2.0,
    diameter_cm: float = 1.0,
    n_jellyroll_turns: int = 2,

    shell_material: str = "steel",
    cathode_collector_material: str = "aluminum",
    cathode_active_material: str = "nmc811",
    anode_collector_material: str = "copper",
    anode_active_material: str = "graphite",
    separator_material: str = "separator_pe",
    electrolyte_material: str = "electrolyte_lipf6_1m",
    central_collector_material: str = "copper",

    can_thickness_cm: float = 0.03,
    cap_thickness_cm: float = 0.03,
    jellyroll_gap_top_cm: float = 0.05,
    jellyroll_gap_bottom_cm: float = 0.05,
    central_collector_inner_radius_cm: float = 0.04,
    central_collector_outer_radius_cm: float = 0.07,
    gap_to_first_jellyroll_cm: float = 0.05,

    separator_t_cm: float = 0.015,
    electrolyte_t_cm: float = 0.005,
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
        "electrolyte_material": electrolyte_material,
        "central_collector_material": central_collector_material,
    }

    missing = [k for k, v in required_materials.items() if v not in MATERIALS]
    if missing:
        raise ValueError(
            "Unknown material(s): "
            + ", ".join(f"{k}='{required_materials[k]}'" for k in missing)
            + f". Available materials: {list(MATERIALS.keys())}"
        )

    # As for the spiral cell: N samples the diameter, the axial extent follows
    # from length_cm, so the grid is deliberately non-cubic.
    if Nx is None and Ny is None and Nz is None:
        if voxel_cm is None:
            if not N:
                raise ValueError("Provide N, voxel_cm, or all of Nx, Ny and Nz.")
            voxel_cm = diameter_cm / int(N)
        Nx = Ny = int(round(diameter_cm / voxel_cm))
        Nz = int(ceil(length_cm / voxel_cm))

    Nx, Ny, Nz, voxel_cm = resolve_grid(
        None, Nx, Ny, Nz, voxel_cm, extent_cm=max(diameter_cm, length_cm)
    )
    b = PhantomBuilder(Nx=Nx, Ny=Ny, Nz=Nz, voxel_cm=voxel_cm)

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
        (electrolyte_material, electrolyte_t_cm),

        (cathode_collector_material, al_t_cm),
        (cathode_active_material, cathode_t_cm),

        (separator_material, separator_t_cm),
        (electrolyte_material, electrolyte_t_cm),

        (anode_active_material, anode_t_cm),
        (anode_collector_material, copper_t_cm),

        (separator_material, separator_t_cm),
        (electrolyte_material, electrolyte_t_cm),
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




# ── Preset dispatch ───────────────────────────────────────────────────────────

def make_phantom(
    preset: str = "composite",
    N: Optional[int] = 64,
    Nx: Optional[int] = None,
    Ny: Optional[int] = None,
    Nz: Optional[int] = None,
    voxel_cm: Optional[float] = None,
    **preset_kwargs,
) -> PhantomData:
    """
    Build a registered preset phantom.

    Cubic::

        make_phantom("composite", N=64)

    Non-cubic::

        make_phantom("composite", Nx=96, Ny=64, Nz=128)

    Preset-specific options pass straight through::

        make_phantom("jellyroll_battery", N=256, n_jellyroll_turns=3)

    Storage convention is ``(Nz, Nx, Ny)``, with coordinates ordered ``(z, x, y)``.
    Voxel size scales with ``N`` (or ``max(Nx, Ny, Nz)``) so the physical sample
    size stays fixed unless *voxel_cm* is given explicitly.

    Every preset accepts the same five grid arguments — enforced by
    :func:`register_phantom` — so a preset that works here works everywhere.

    See Also
    --------
    PHANTOM_PRESETS, PHANTOM_DESCRIPTIONS, register_phantom
    """
    if preset not in PHANTOM_PRESETS:
        available = "\n".join(
            f"    {name:<20} {PHANTOM_DESCRIPTIONS.get(name, '')}"
            for name in sorted(PHANTOM_PRESETS)
        )
        raise ValueError(f"Unknown phantom preset {preset!r}. Available:\n{available}")

    return PHANTOM_PRESETS[preset](
        N=N, Nx=Nx, Ny=Ny, Nz=Nz, voxel_cm=voxel_cm, **preset_kwargs
    )
