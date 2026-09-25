"""
neutron_xray_sim.phantoms.presets
─────────────────────────────────
Ready-made phantoms and the preset registry.

Every preset factory accepts either a cubic size ``N`` or explicit
``Nx, Ny, Nz`` plus an optional ``voxel_cm``.  When ``voxel_cm`` is omitted the
voxel size is chosen so the sample keeps its physical size, i.e. changing the
resolution does not change the geometry.

Adding a preset
───────────────
Write a ``make_<name>_phantom(N=64, voxel_cm=None, Nx=None, Ny=None, Nz=None)``
function that uses :func:`resolve_grid` and a :class:`PhantomBuilder`, then add
it to ``PHANTOM_PRESETS`` (or call :func:`register_preset`).
"""

from __future__ import annotations

import inspect
from math import ceil
from typing import Callable, Dict, Optional, Tuple

import numpy as np

from ..physics.materials import MATERIALS
from .base import PhantomBuilder, PhantomData

__all__ = [
    "resolve_grid",
    "make_composite_phantom",
    "make_battery_phantom",
    "make_bone_implant_phantom",
    "make_industrial_phantom",
    "make_hdpe_composite_phantom",
    "make_li_ion_battery_phantom",
    "make_custom_cylindrical_battery_phantom",
    "PHANTOM_PRESETS",
    "register_preset",
    "make_phantom",
]


def resolve_grid(
    N: Optional[int],
    Nx: Optional[int],
    Ny: Optional[int],
    Nz: Optional[int],
    voxel_cm: Optional[float],
    sample_size_cm: float,
) -> Tuple[int, int, int, float]:
    """
    Validate grid arguments shared by all preset factories.

    Returns ``(Nx, Ny, Nz, voxel_cm)``.  Either ``N`` alone (cubic) or all of
    ``Nx, Ny, Nz`` must be given.  If *voxel_cm* is None it is chosen so the
    largest dimension spans *sample_size_cm*.
    """
    if any(v is not None for v in (Nx, Ny, Nz)):
        if not all(v is not None for v in (Nx, Ny, Nz)):
            raise ValueError("Provide either N only, or all of Nx, Ny, and Nz.")
        Nx, Ny, Nz = int(Nx), int(Ny), int(Nz)
    else:
        if N is None:
            raise ValueError("Provide either N or all of Nx, Ny, and Nz.")
        Nx = Ny = Nz = int(N)
    if voxel_cm is None:
        voxel_cm = sample_size_cm / max(Nx, Ny, Nz)
    return Nx, Ny, Nz, float(voxel_cm)


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
    Nx, Ny, Nz, voxel_cm = resolve_grid(N, Nx, Ny, Nz, voxel_cm, sample_size_cm=1.0)

    b = PhantomBuilder(N=None, Nx=Nx, Ny=Ny, Nz=Nz, voxel_cm=voxel_cm)

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
    Nx, Ny, Nz, voxel_cm = resolve_grid(N, Nx, Ny, Nz, voxel_cm, sample_size_cm=1.4)

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
    Nx, Ny, Nz, voxel_cm = resolve_grid(N, Nx, Ny, Nz, voxel_cm, sample_size_cm=1.0)

    b = PhantomBuilder(N=None, Nx=Nx, Ny=Ny, Nz=Nz, voxel_cm=voxel_cm)

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
    Nx, Ny, Nz, voxel_cm = resolve_grid(N, Nx, Ny, Nz, voxel_cm, sample_size_cm=1.0)

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

    Nx, Ny, Nz, voxel_cm = resolve_grid(N, Nx, Ny, Nz, voxel_cm, sample_size_cm=1.0)

    b = PhantomBuilder(N=None, Nx=Nx, Ny=Ny, Nz=Nz, voxel_cm=voxel_cm)

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




def make_li_ion_battery_phantom(
    diameter_cm: float = 1.0,
    length_cm: float = 2.0,
    N: int = 512,
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
    voxel_cm = diameter_cm / N
    Nx = Ny = int(N)
    Nz = int(ceil(length_cm / voxel_cm))

    b = PhantomBuilder(N=None, Nx=Nx, Ny=Ny, Nz=Nz, voxel_cm=voxel_cm)

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


def make_custom_cylindrical_battery_phantom(
    N: int = 256,
    length_cm: float = 2.0,
    voxel_cm: Optional[float] = None,
    *,
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




# ── Registry ──────────────────────────────────────────────────────────────────

PHANTOM_PRESETS: Dict[str, Callable[..., PhantomData]] = {
    "composite":         make_composite_phantom,
    "battery":           make_battery_phantom,
    "bone_implant":      make_bone_implant_phantom,
    "industrial":        make_industrial_phantom,
    "jellyroll_battery": make_custom_cylindrical_battery_phantom,
    "HDPE_composite":    make_hdpe_composite_phantom,
    "li_ion_spiral":     make_li_ion_battery_phantom,
}


def register_preset(name: str, factory: Callable[..., PhantomData],
                    overwrite: bool = False) -> None:
    """Register *factory* so ``make_phantom(name, ...)`` and
    ``DualModalitySimulation(preset=name)`` can build it."""
    if name in PHANTOM_PRESETS and not overwrite:
        raise KeyError(f"Preset '{name}' already exists; pass overwrite=True.")
    PHANTOM_PRESETS[name] = factory


def make_phantom(
    preset: str = "composite",
    N: Optional[int] = 64,
    Nx: Optional[int] = None,
    Ny: Optional[int] = None,
    Nz: Optional[int] = None,
    voxel_cm: Optional[float] = None,
    **kwargs,
) -> PhantomData:
    """
    Build a named preset phantom.

    Cubic use:        ``make_phantom("composite", N=64)``
    Non-cubic use:    ``make_phantom("composite", Nx=96, Ny=64, Nz=128)``

    Extra keyword arguments are forwarded to the preset factory (for example
    ``n_jellyroll_turns`` for ``"jellyroll_battery"``).

    Sample sizes are physically realistic for neutron tomography:
      composite / bone_implant / industrial / HDPE_composite → 1.0 cm
      battery                                              → 1.4 cm (AAA cell)
      jellyroll_battery / li_ion_spiral                    → 1.0 cm × 2.0 cm

    Voxel size scales automatically with N (or max(Nx, Ny, Nz)) so geometry
    is preserved unless *voxel_cm* is given explicitly.
    """
    if preset not in PHANTOM_PRESETS:
        raise ValueError(
            f"Unknown preset '{preset}'. Choose from: {list(PHANTOM_PRESETS)}"
        )
    factory = PHANTOM_PRESETS[preset]
    params = inspect.signature(factory).parameters

    call = dict(kwargs)
    if voxel_cm is not None:
        call["voxel_cm"] = voxel_cm
    if any(v is not None for v in (Nx, Ny, Nz)):
        if not all(v is not None for v in (Nx, Ny, Nz)):
            raise ValueError("Provide either N only, or all of Nx, Ny, and Nz.")
        if not {"Nx", "Ny", "Nz"} <= set(params):
            raise ValueError(
                f"Preset '{preset}' only supports a cubic/transverse size N; "
                "it does not accept Nx/Ny/Nz."
            )
        call.update(N=None, Nx=Nx, Ny=Ny, Nz=Nz)
    else:
        if N is None:
            raise ValueError("Provide either N or all of Nx, Ny, and Nz.")
        call["N"] = N
    return factory(**call)
