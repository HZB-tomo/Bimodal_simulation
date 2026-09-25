import numpy as np
import pytest

import neutron_xray_sim as nxs
from neutron_xray_sim.physics import elements


def test_water_from_formula():
    w = nxs.material_from_formula("Water", "H2O", "H2O", density_gcc=1.0)
    # X-ray: NIST XCOM mixture rule reproduces the tabulated water value.
    assert w.mu_x_at(80) == pytest.approx(nxs.MATERIALS["water"].mu_x_at(80), rel=0.1)
    # Neutron: bound-atom cross sections (H incoherent dominates, ~5.6 cm⁻¹);
    # the database entry instead stores an imaging-effective value.
    assert w.mu_n == pytest.approx(5.65, rel=0.02)
    assert w.mu_n == pytest.approx(w.mu_n_abs + w.mu_n_coh + w.mu_n_inc)
    half = nxs.material_from_formula("W", "W", "H2O", 1.0, incoherent_scale=0.5)
    assert half.mu_n_inc == pytest.approx(0.5 * w.mu_n_inc)


def test_unknown_element_gives_clear_error():
    with pytest.raises(KeyError, match="elements"):
        nxs.material_from_formula("x", "x", "Xx2O", density_gcc=1.0)


def test_missing_xcom_file_is_lazy_and_explicit():
    # Importing the package must not require every XCOM file …
    missing = set(elements.ATOMIC_MASS) - set(elements.available_xray_elements())
    if not missing:
        pytest.skip("all XCOM files present")
    el = sorted(missing)[0]
    # … but using one without data raises an actionable error.
    with pytest.raises(FileNotFoundError, match=el):
        elements.xray_mass_atten(el)


def test_composite_weight_fractions_validated():
    with pytest.raises(ValueError, match="sum"):
        nxs.make_composite_material("bad", "b", 1.0, [("H2O", 0.5, 1.0)])


def test_register_material_roundtrip():
    m = nxs.material_from_formula("Test PE", "tPE", "C2H4", density_gcc=0.95)
    nxs.register_material("_test_pe", m)
    try:
        assert nxs.MATERIALS["_test_pe"] is m
        with pytest.raises(KeyError):
            nxs.register_material("_test_pe", m)
    finally:
        del nxs.MATERIALS["_test_pe"]


def test_single_element_table_is_shared():
    from neutron_xray_sim.physics import neutron_spectra
    assert neutron_spectra.SIGMA_BOUND is elements.NEUTRON_XS
    assert elements.NEUTRON_XS["Mn"]["coh"] == pytest.approx(1.75)


def test_xray_spectrum_normalised():
    e, w = nxs.xray_spectrum(120, 2.0)
    assert np.isclose(w.sum(), 1.0) and np.all(e > 0)
