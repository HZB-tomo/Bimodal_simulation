"""Tests for the material registry, specs, and element data."""

from __future__ import annotations

import json

import numpy as np
import pytest

from neutron_xray_sim.materials import (
    MATERIALS,
    XRAY_E_KEV,
    Material,
    MaterialSpec,
    build_material,
    element_data_status,
    make_composite_material,
    material_from_formula,
    parse_formula,
)
from neutron_xray_sim.materials.core import SpecValidationError
from neutron_xray_sim.materials.elements import MissingElementDataError
from neutron_xray_sim.materials.registry import DuplicateMaterialError

# ── Registry behaves like the dict it replaced ───────────────────────────────

def test_registry_supports_the_mapping_protocol():
    assert "water" in MATERIALS
    assert isinstance(MATERIALS["water"], Material)
    assert "water" in list(MATERIALS.keys())
    assert len(MATERIALS) == len(list(MATERIALS))
    with pytest.raises(KeyError):
        MATERIALS["definitely_not_a_material"]


def test_unknown_material_error_lists_the_alternatives():
    with pytest.raises(KeyError, match="Available"):
        MATERIALS["unobtainium"]


def test_every_builtin_material_builds():
    """A material whose spec is broken should fail here, not mid-simulation."""
    unavailable = {}
    for key in MATERIALS.names():
        try:
            MATERIALS[key]
        except Exception as exc:              # pragma: no cover - failure path
            unavailable[key] = str(exc)
    assert not unavailable, f"materials failed to build: {unavailable}"


def test_tags_and_search():
    assert "nmc811" in MATERIALS.names(tag="cathode")
    assert "battery" in MATERIALS.tags()
    assert "separator_pe" in MATERIALS.search("separator")


# ── The Material value object ────────────────────────────────────────────────

def test_material_equality_is_identity_not_array_comparison():
    """Regression: dataclass __eq__ over a NumPy field raised ValueError.

    Two materials that agree on every leading field used to reach the array
    comparison and blow up inside `in`/`.index()`.
    """
    a = material_from_formula("Twin", "T", "C", 2.0)
    b = material_from_formula("Twin", "T", "C", 2.0)
    assert a != b                      # distinct objects are distinct phases
    assert a == a
    assert a in [b, a]                 # must not raise
    assert len({a, b}) == 2            # hashable


def test_mu_x_table_has_one_entry_per_grid_energy():
    for key in ("water", "iron", "nmc811"):
        assert MATERIALS[key].mu_x_table.shape == XRAY_E_KEV.shape


def test_mu_x_at_matches_the_table_on_grid_points():
    fe = MATERIALS["iron"]
    for idx, energy in enumerate(XRAY_E_KEV):
        assert fe.mu_x_at(float(energy)) == pytest.approx(fe.mu_x_table[idx], rel=1e-9)


def test_legacy_mu_x_table_alias_still_resolves():
    """Scripts and notebooks reach for the old private name."""
    assert np.array_equal(MATERIALS["water"]._mu_x_table, MATERIALS["water"].mu_x_table)


def test_neutron_total_is_the_sum_of_its_components():
    m = MATERIALS["hdpe"]
    assert m.mu_n == pytest.approx(m.mu_n_abs + m.mu_n_coh + m.mu_n_inc)
    assert m.mu_n_scatter == pytest.approx(m.mu_n_coh + m.mu_n_inc)


# ── Formula parsing ──────────────────────────────────────────────────────────

def test_parse_formula_handles_decimal_subscripts():
    comp = parse_formula("LiNi0.8Mn0.1Co0.1O2")
    assert comp == pytest.approx({"Li": 1.0, "Ni": 0.8, "Mn": 0.1, "Co": 0.1, "O": 2.0})


def test_parse_formula_rejects_bracket_notation_with_guidance():
    with pytest.raises(ValueError, match="flat notation"):
        parse_formula("Ca10(PO4)6(OH)2")


def test_parse_formula_rejects_unknown_elements():
    with pytest.raises(ValueError, match="unknown element"):
        parse_formula("XyO2")


def test_missing_element_data_error_is_actionable():
    """Silicon has no shipped NIST table; the error must say what to do."""
    if element_data_status().get("Si"):
        pytest.skip("Si data is now installed")
    with pytest.raises(MissingElementDataError) as excinfo:
        material_from_formula("Quartz", "SiO2", "SiO2", 2.65)
    message = str(excinfo.value)
    assert "Si.txt" in message
    assert "nist.gov" in message
    assert "Available now" in message


# ── Derived materials ────────────────────────────────────────────────────────

def test_neutron_attenuation_scales_linearly_with_density():
    thin = material_from_formula("C-thin", "C", "C", 1.0)
    thick = material_from_formula("C-thick", "C", "C", 2.0)
    assert thick.mu_n == pytest.approx(2.0 * thin.mu_n, rel=1e-12)


def test_incoherent_scale_only_touches_the_incoherent_term():
    full = material_from_formula("PE", "PE", "C2H4", 0.94)
    damped = material_from_formula("PE", "PE", "C2H4", 0.94, incoherent_scale=0.5)
    assert damped.mu_n_inc == pytest.approx(0.5 * full.mu_n_inc)
    assert damped.mu_n_abs == pytest.approx(full.mu_n_abs)
    assert damped.mu_n_coh == pytest.approx(full.mu_n_coh)


def test_composite_equals_its_single_phase_limit():
    single = material_from_formula("Graphite", "C", "C", 2.0)
    composite = make_composite_material("Graphite", "C", 2.0, [("C", 1.0, 2.0)])
    assert composite.mu_n == pytest.approx(single.mu_n, rel=1e-12)
    assert composite.mu_x_table == pytest.approx(single.mu_x_table, rel=1e-12)


def test_composite_weight_fractions_must_sum_to_one():
    with pytest.raises(SpecValidationError, match="sum to 1"):
        make_composite_material("Bad", "X", 2.0, [("C", 0.5, 2.0), ("O2", 0.2, 1.4)])


# ── Spec validation ──────────────────────────────────────────────────────────

def test_spec_needs_at_least_one_source_per_channel():
    with pytest.raises(SpecValidationError, match="xray_mu_cm"):
        build_material(MaterialSpec(key="k", name="n", symbol="s", density_gcc=1.0))


def test_spec_rejects_both_formula_and_components():
    with pytest.raises(SpecValidationError, match="not both"):
        build_material(MaterialSpec(
            key="k", name="n", symbol="s", density_gcc=1.0,
            formula="C", components=(("C", 1.0, 2.0),),
        ))


def test_spec_rejects_a_wrong_length_xray_table():
    with pytest.raises(SpecValidationError, match="one linear attenuation per energy"):
        build_material(MaterialSpec(
            key="k", name="n", symbol="s", density_gcc=1.0,
            xray_mu_cm=(1.0, 2.0), neutron_mu_cm=(0.1, 0.1, 0.1),
        ))


def test_spec_rejects_a_non_positive_density():
    with pytest.raises(SpecValidationError, match="density_gcc"):
        build_material(MaterialSpec(
            key="k", name="n", symbol="s", density_gcc=0.0, formula="C",
        ))


def test_a_rising_xray_table_warns_unless_an_edge_is_declared():
    rising = tuple(float(i + 1) for i in range(len(XRAY_E_KEV)))
    spec = MaterialSpec(key="k", name="n", symbol="s", density_gcc=1.0,
                        xray_mu_cm=rising, neutron_mu_cm=(0.1, 0.1, 0.1))
    with pytest.warns(UserWarning, match="increases with energy"):
        build_material(spec)


def test_mixed_channel_sources_are_allowed():
    """X-ray from a table, neutron derived from the formula."""
    table = tuple(10.0 / (i + 1) for i in range(len(XRAY_E_KEV)))
    mat = build_material(MaterialSpec(
        key="mixed", name="Mixed", symbol="M", density_gcc=2.0,
        formula="C", xray_mu_cm=table,
    ))
    reference = material_from_formula("C", "C", "C", 2.0)
    assert mat.mu_x_table == pytest.approx(np.asarray(table))
    assert mat.mu_n == pytest.approx(reference.mu_n)


# ── Contribution paths ───────────────────────────────────────────────────────

def test_register_then_use(fresh_registry):
    fresh_registry.register(MaterialSpec(
        key="graphite_dense", name="Dense graphite", symbol="C*",
        density_gcc=2.26, formula="C", tags=("custom",),
    ))
    assert "graphite_dense" in fresh_registry
    assert fresh_registry["graphite_dense"].density_gcc == 2.26
    assert "graphite_dense" in fresh_registry.names(tag="custom")


def test_registering_a_duplicate_key_is_refused(fresh_registry):
    spec = MaterialSpec(key="water", name="Other water", symbol="W",
                        density_gcc=1.1, formula="H2O")
    with pytest.raises(DuplicateMaterialError, match="already registered"):
        fresh_registry.register(spec)
    fresh_registry.register(spec, overwrite=True)
    assert fresh_registry["water"].density_gcc == 1.1


def test_load_materials_from_a_json_file(tmp_path, fresh_registry):
    path = tmp_path / "project_materials.json"
    path.write_text(json.dumps([{
        "key": "petg", "name": "PETG", "symbol": "PETG",
        "density_gcc": 1.27, "formula": "C10H8O4", "tags": ["polymer"],
        "reference": "manufacturer datasheet",
    }]))
    assert fresh_registry.load_file(path) == ["petg"]
    assert fresh_registry["petg"].symbol == "PETG"
    assert fresh_registry.spec("petg").reference == "manufacturer datasheet"


def test_a_json_file_with_an_unknown_field_names_the_offender(tmp_path, fresh_registry):
    path = tmp_path / "bad.json"
    path.write_text(json.dumps([{
        "key": "x", "name": "X", "symbol": "X", "density_gcc": 1.0,
        "formula": "C", "densitiy_gcc": 1.0,     # typo
    }]))
    with pytest.raises(ValueError, match="densitiy_gcc"):
        fresh_registry.load_file(path)


def test_specs_round_trip_through_a_file(tmp_path, fresh_registry):
    path = tmp_path / "export.json"
    fresh_registry.save_file(path, keys=["water", "nmc811"])
    reloaded = type(fresh_registry)()
    assert sorted(reloaded.load_file(path)) == ["nmc811", "water"]
    for key in ("water", "nmc811"):
        assert reloaded[key].mu_n == pytest.approx(fresh_registry[key].mu_n)
        assert reloaded[key].mu_x_table == pytest.approx(fresh_registry[key].mu_x_table)


def test_audit_reports_the_known_bad_tables():
    """The three transcription errors in the shipped tables stay visible.

    If someone replaces one with verified NIST data, this test tells them to
    drop it from the list rather than silently losing the guard.
    """
    findings = MATERIALS.audit()
    assert set(findings) == {"bone", "hdpe", "lead"}, (
        "the set of known-bad X-ray tables changed; update database.py's "
        "'Known data issues' section and this test together"
    )


def test_table_renders_without_raising():
    text = MATERIALS.table()
    assert "water" in text and "mu_n" in text
