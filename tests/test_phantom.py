"""Tests for phantom construction, the preset registry, and derived volumes."""

from __future__ import annotations

import numpy as np
import pytest

from neutron_xray_sim.materials import MATERIALS, XRAY_E_KEV, material_from_formula
from neutron_xray_sim.phantom import (
    PHANTOM_DESCRIPTIONS,
    PHANTOM_PRESETS,
    PhantomBuilder,
    make_phantom,
    register_phantom,
    resolve_grid,
)

N = 16


# ── The shared grid resolver ─────────────────────────────────────────────────

def test_resolve_grid_cubic():
    assert resolve_grid(N=32, extent_cm=1.0) == (32, 32, 32, 1.0 / 32)


def test_resolve_grid_non_cubic_uses_the_largest_side_for_the_extent():
    Nx, Ny, Nz, voxel = resolve_grid(None, 8, 16, 32, None, extent_cm=1.0)
    assert (Nx, Ny, Nz) == (8, 16, 32)
    assert voxel == pytest.approx(1.0 / 32)


def test_resolve_grid_requires_all_three_or_none():
    with pytest.raises(ValueError, match="all three"):
        resolve_grid(None, 8, 16, None, None)


def test_resolve_grid_rejects_nonsense_dimensions():
    with pytest.raises(ValueError, match="positive"):
        resolve_grid(None, 8, 0, 8, None)
    with pytest.raises(ValueError, match="voxel_cm must be positive"):
        resolve_grid(N=8, voxel_cm=-1.0)


def test_explicit_voxel_size_wins():
    assert resolve_grid(N=8, voxel_cm=0.25)[3] == 0.25


# ── The preset registry ──────────────────────────────────────────────────────

@pytest.mark.parametrize("preset", sorted(PHANTOM_PRESETS))
def test_every_preset_builds_cubic(preset):
    phantom = make_phantom(preset, N=N)
    assert phantom.label_vol.ndim == 3
    assert phantom.label_vol.dtype == np.uint8
    assert len(phantom.materials) >= 2
    assert phantom.voxel_cm > 0


@pytest.mark.parametrize("preset", sorted(PHANTOM_PRESETS))
def test_every_preset_accepts_a_non_cubic_grid(preset):
    """Regression: `jellyroll_battery` raised TypeError on this path, because
    its factory did not take Nx/Ny/Nz like the others."""
    phantom = make_phantom(preset, Nx=12, Ny=12, Nz=16)
    assert phantom.shape == (16, 12, 12)


@pytest.mark.parametrize("preset", sorted(PHANTOM_PRESETS))
def test_every_preset_is_described(preset):
    assert PHANTOM_DESCRIPTIONS.get(preset), f"{preset} has no description"


def test_unknown_preset_lists_the_available_ones():
    with pytest.raises(ValueError, match="composite"):
        make_phantom("not_a_preset")


def test_preset_kwargs_reach_the_factory():
    two = make_phantom("jellyroll_battery", N=N, n_jellyroll_turns=1)
    three = make_phantom("jellyroll_battery", N=N, n_jellyroll_turns=2)
    assert len(three.materials) >= len(two.materials)


def test_registering_a_preset_with_a_bad_signature_is_refused():
    with pytest.raises(ValueError, match="missing required keyword"):
        @register_phantom("broken_preset")
        def _factory(size):        # no N / voxel_cm / Nx / Ny / Nz
            ...


def test_registering_a_duplicate_preset_name_is_refused():
    with pytest.raises(ValueError, match="already registered"):
        @register_phantom("composite")
        def _factory(N=64, voxel_cm=None, Nx=None, Ny=None, Nz=None):
            ...


# ── Builder primitives ───────────────────────────────────────────────────────

def test_builder_starts_full_of_air():
    phantom = PhantomBuilder(N=8, voxel_cm=0.1).build("empty")
    assert phantom.materials == [MATERIALS["air"]]
    assert (phantom.label_vol == 0).all()


def test_material_index_is_stable_and_by_identity():
    """Regression: the old value-based lookup crashed on same-named materials."""
    b = PhantomBuilder(N=8, voxel_cm=0.1)
    twin_a = material_from_formula("Twin", "T", "C", 2.0)
    twin_b = material_from_formula("Twin", "T", "C", 2.0)

    b.add_box(twin_a, half_extents_cm=(0.1, 0.1, 0.1))
    b.add_sphere(twin_b, radius_cm=0.05)          # must not raise
    phantom = b.build("twins")
    assert len(phantom.materials) == 3            # air + two distinct phases


def test_adding_the_same_material_twice_reuses_its_label():
    b = PhantomBuilder(N=8, voxel_cm=0.1)
    b.add_box("iron", center_cm=(0, 0, 0), half_extents_cm=(0.1, 0.1, 0.1))
    b.add_box("iron", center_cm=(0.2, 0, 0), half_extents_cm=(0.1, 0.1, 0.1))
    assert len(b.build("x").materials) == 2       # air + iron


def test_unknown_material_name_names_the_alternatives():
    with pytest.raises(KeyError, match="Available"):
        PhantomBuilder(N=8, voxel_cm=0.1).fill("adamantium")


def test_hollow_cylinder_rejects_inverted_radii():
    b = PhantomBuilder(N=8, voxel_cm=0.1)
    with pytest.raises(ValueError, match="outer_radius_cm"):
        b.add_hollow_cylinder("iron", inner_radius_cm=0.3, outer_radius_cm=0.2)


def test_primitives_are_chainable():
    phantom = (
        PhantomBuilder(N=8, voxel_cm=0.1)
        .fill("hdpe")
        .add_sphere("water", radius_cm=0.2)
        .add_rod("iron", center_cm=(0.2, 0.2), radius_cm=0.05)
        .build("chained")
    )
    assert len(phantom.materials) == 4


# ── Derived attenuation volumes ──────────────────────────────────────────────

def test_neutron_volumes_match_the_material_values(small_phantom):
    for idx, mat in enumerate(small_phantom.materials):
        mask = small_phantom.label_vol == idx
        if not mask.any():
            continue
        assert small_phantom.mu_n_vol[mask] == pytest.approx(mat.mu_n, rel=1e-5)
        assert small_phantom.mu_n_abs_vol[mask] == pytest.approx(mat.mu_n_abs, rel=1e-5)


def test_neutron_components_sum_to_the_total(small_phantom):
    total = (small_phantom.mu_n_abs_vol
             + small_phantom.mu_n_coh_vol
             + small_phantom.mu_n_inc_vol)
    assert np.allclose(total, small_phantom.mu_n_vol, rtol=1e-5)


def test_mu_x_vols_agrees_with_the_per_energy_accessor(small_phantom):
    stacked = small_phantom.mu_x_vols
    assert stacked.shape == (len(XRAY_E_KEV), *small_phantom.shape)
    for idx in (0, 6, len(XRAY_E_KEV) - 1):
        assert np.array_equal(stacked[idx], small_phantom.mu_x_at_index(idx))


def test_mu_x_at_energy_matches_the_grid_accessor(small_phantom):
    idx = 6
    assert np.allclose(
        small_phantom.mu_x_at_energy(float(XRAY_E_KEV[idx])),
        small_phantom.mu_x_at_index(idx),
        rtol=1e-5,
    )


def test_mu_x_at_index_rejects_out_of_range(small_phantom):
    with pytest.raises(IndexError, match="out of range"):
        small_phantom.mu_x_at_index(len(XRAY_E_KEV))


def test_x_ray_volumes_are_built_lazily(small_phantom):
    """A phantom must not pay for the 13-energy stack it may never use."""
    from neutron_xray_sim.phantom import make_phantom as _make

    fresh = _make("composite", N=8)
    assert fresh._mu_x_cache is None
    assert fresh.mu_x_vols is not None
    assert fresh._mu_x_cache is not None


def test_voxel_counts_covers_every_material(small_phantom):
    counts = small_phantom.voxel_counts()
    assert len(counts) >= len(small_phantom.materials)
    assert counts.sum() == small_phantom.label_vol.size


def test_shape_validation_rejects_mismatched_label_volume():
    from neutron_xray_sim.phantom import PhantomData

    with pytest.raises(ValueError, match="label_vol shape"):
        PhantomData(Nz=4, Nx=4, Ny=4,
                    voxel_cm=0.1,
                    label_vol=np.zeros((4, 4, 5), dtype=np.uint8),
                    materials=[MATERIALS["air"]])
