"""End-to-end tests for projection, artifacts, reconstruction, and caching.

These run on the NumPy CPU path with tiny grids.  They assert *behaviour*
(shapes, invariants, that an artifact actually changes the result) rather than
image quality, which needs a GPU and far more voxels.
"""

from __future__ import annotations

import numpy as np
import pytest

from neutron_xray_sim import (
    ArtifactConfig,
    DualModalitySimulation,
    SimCache,
    compute_bimodal_histogram,
    inject_sinogram_artifacts,
    make_phantom,
    make_sinogram_pair,
    reconstruct,
)

N_ANGLES = 16


@pytest.fixture(scope="module")
def sinogram_pair():
    phantom = make_phantom("composite", N=16)
    return make_sinogram_pair(phantom, n_angles=N_ANGLES, use_astra=False)


# ── Projection ───────────────────────────────────────────────────────────────

def test_sinogram_shapes_and_keys(sinogram_pair):
    xray, neutron = sinogram_pair
    for sino in (xray, neutron):
        assert sino["sino_lam"].shape == (N_ANGLES, 16, 16)
        assert sino["voxel_cm"] > 0
        assert len(sino["angles_deg"]) == N_ANGLES


def test_transmission_stays_physical(sinogram_pair):
    for sino in sinogram_pair:
        assert sino["sino_trans"].min() >= 0.0
        assert sino["sino_trans"].max() <= 1.0 + 1e-6
        assert (sino["sino_lam"] >= 0).all()


def test_neutron_channels_multiply_to_the_total(sinogram_pair):
    _, neutron = sinogram_pair
    assert np.allclose(
        neutron["sino_lam"],
        neutron["sino_abs_lam"] + neutron["sino_scatter_lam"],
        atol=1e-4,
    )


def test_monochromatic_projection_supports_non_cubic_phantoms():
    """Regression: this path used `phantom.N` for all three axes and crashed."""
    from neutron_xray_sim.projector import project_xray_monochromatic

    phantom = make_phantom("composite", Nx=12, Ny=16, Nz=8)
    out = project_xray_monochromatic(
        phantom, np.linspace(0, 180, 8, endpoint=False),
        energy_keV=80.0, use_astra=False,
    )
    assert out["sino_lam"].shape == (8, 8, 16)


def test_unknown_xray_mode_is_rejected():
    phantom = make_phantom("composite", N=8)
    with pytest.raises(ValueError, match="polychromatic"):
        make_sinogram_pair(phantom, n_angles=4, xray_mode="quantum", use_astra=False)


# ── Artifact injection ───────────────────────────────────────────────────────

def test_clean_config_leaves_the_sinogram_untouched(sinogram_pair):
    xray, neutron = sinogram_pair
    x_out, n_out = inject_sinogram_artifacts(xray, neutron, ArtifactConfig.clean())
    assert np.allclose(x_out["sino_lam"], xray["sino_lam"])
    assert np.allclose(n_out["sino_lam"], neutron["sino_lam"])


def test_clean_config_reports_itself_as_clean():
    assert ArtifactConfig.clean().is_clean()
    assert ArtifactConfig.clean().summary() == "clean (no artifacts)"
    assert not ArtifactConfig(photon_noise=True).is_clean()


def test_beam_hardening_correction_survives_a_later_scatter_step(sinogram_pair):
    """Regression: the scatter models rebuilt their output from the *raw*
    transmission, discarding any correction applied before them."""
    xray, neutron = sinogram_pair
    cfg_scatter = ArtifactConfig(xray_scatter=True, xray_scatter_fraction=0.04)
    cfg_both = ArtifactConfig(xray_scatter=True, xray_scatter_fraction=0.04,
                              apply_bh_correction=True)

    scatter_only, _ = inject_sinogram_artifacts(xray, neutron, cfg_scatter)
    both, _ = inject_sinogram_artifacts(xray, neutron, cfg_both)

    assert not np.allclose(scatter_only["sino_lam"], both["sino_lam"]), (
        "applying BHC before scatter made no difference — it is being dropped"
    )


def test_each_artifact_changes_the_sinogram(sinogram_pair):
    xray, neutron = sinogram_pair
    configs = {
        "noise": ArtifactConfig(photon_noise=True, I0_xray=1e3, I0_neutron=1e3),
        "bhc": ArtifactConfig(apply_bh_correction=True),
        "n_scatter": ArtifactConfig(neutron_scatter=True, scatter_fraction=0.1),
        "x_scatter": ArtifactConfig(xray_scatter=True, xray_scatter_fraction=0.1),
        "psf": ArtifactConfig(detector_psf=True, psf_sigma_xray_pixels=1.5,
                              psf_sigma_neutron_pixels=1.5),
        "rings": ArtifactConfig(ring_artifacts=True, n_bad_columns=2,
                                ring_amplitude=0.1),
    }
    for name, cfg in configs.items():
        x_out, n_out = inject_sinogram_artifacts(xray, neutron, cfg)
        changed = (not np.allclose(x_out["sino_lam"], xray["sino_lam"])
                   or not np.allclose(n_out["sino_lam"], neutron["sino_lam"]))
        assert changed, f"artifact {name!r} had no effect"


def test_injection_does_not_mutate_its_input(sinogram_pair):
    xray, neutron = sinogram_pair
    before = xray["sino_lam"].copy()
    inject_sinogram_artifacts(xray, neutron, ArtifactConfig.realistic())
    assert np.array_equal(xray["sino_lam"], before)


def test_noise_is_reproducible_for_a_fixed_seed(sinogram_pair):
    xray, neutron = sinogram_pair
    cfg = ArtifactConfig(photon_noise=True, I0_xray=1e3, I0_neutron=1e3)
    a, _ = inject_sinogram_artifacts(xray, neutron, cfg,
                                     rng=np.random.default_rng(7))
    b, _ = inject_sinogram_artifacts(xray, neutron, cfg,
                                     rng=np.random.default_rng(7))
    assert np.array_equal(a["sino_lam"], b["sino_lam"])


def test_more_dose_means_less_noise(sinogram_pair):
    xray, neutron = sinogram_pair
    scatter = {}
    for I0 in (1e3, 1e6):
        out, _ = inject_sinogram_artifacts(
            xray, neutron,
            ArtifactConfig(photon_noise=True, I0_xray=I0, I0_neutron=I0),
            rng=np.random.default_rng(0),
        )
        scatter[I0] = float(np.std(out["sino_lam"] - xray["sino_lam"]))
    assert scatter[1e6] < scatter[1e3]


def test_ring_artifacts_cannot_exceed_the_detector_width(sinogram_pair):
    xray, neutron = sinogram_pair
    cfg = ArtifactConfig(ring_artifacts=True, n_bad_columns=9999)
    out, _ = inject_sinogram_artifacts(xray, neutron, cfg)   # must not raise
    assert out["sino_lam"].shape == xray["sino_lam"].shape


# ── Reconstruction ───────────────────────────────────────────────────────────

def test_fbp_recovers_the_right_shape_and_sign(sinogram_pair):
    xray, _ = sinogram_pair
    vol = reconstruct(xray, algorithm="FBP", use_astra=False)
    assert vol.shape == (16, 16, 16)
    assert vol.min() >= 0.0             # clip_negative default


def test_iterative_algorithms_degrade_to_fbp_without_astra(sinogram_pair):
    pytest.importorskip  # noqa: B018 - documented behaviour, no GPU here
    xray, _ = sinogram_pair
    from neutron_xray_sim.reconstructor import _astra_ok

    if _astra_ok():
        pytest.skip("ASTRA present — the fallback path is not exercised")
    with pytest.warns(UserWarning, match="falling back"):
        vol = reconstruct(xray, algorithm="SIRT", use_astra=True)
    assert vol.shape == (16, 16, 16)


def test_unknown_algorithm_is_rejected(sinogram_pair):
    xray, _ = sinogram_pair
    with pytest.raises(ValueError, match="Unknown algorithm"):
        reconstruct(xray, algorithm="magic")


def test_algorithm_aliases_resolve():
    from neutron_xray_sim.reconstructor import _resolve_algorithm

    assert _resolve_algorithm("fbp") == "FBP"
    assert _resolve_algorithm("ram-lak") == "FBP"
    assert _resolve_algorithm("os-sart") == "OSSART"
    assert _resolve_algorithm("nesterov") == "NESTEROV_SIRT"


def test_ring_removal_and_ring_injection_are_not_both_applied():
    """Regression: ring removal ran unconditionally and cancelled the artifact."""
    sim = DualModalitySimulation(preset="composite", N=12, n_angles=12,
                                 verbose=False, use_astra=False)
    clean = sim.run(ArtifactConfig.clean(), tag="clean")
    rings = sim.run(
        ArtifactConfig(ring_artifacts=True, n_bad_columns=3, ring_amplitude=0.1),
        tag="rings",
    )
    delta = float(np.abs(rings.vol_xray - clean.vol_xray).max())
    assert delta > 1e-3, "the injected ring artifact was removed before analysis"


# ── Histogram ────────────────────────────────────────────────────────────────

def test_bimodal_histogram_shape_and_mass(sinogram_pair):
    xray, neutron = sinogram_pair
    vol_x = reconstruct(xray, use_astra=False)
    vol_n = reconstruct(neutron, use_astra=False)
    hist = compute_bimodal_histogram(vol_x, vol_n, bins=32)
    assert hist.H.shape == (32, 32)
    assert hist.H.sum() > 0
    assert len(hist.x_edges) == 33


# ── Orchestrator ─────────────────────────────────────────────────────────────

def test_run_default_config_is_not_shared_between_calls():
    """Regression: `cfg` defaulted to a single module-level ArtifactConfig."""
    sim = DualModalitySimulation(preset="composite", N=12, n_angles=8,
                                 verbose=False, use_astra=False)
    first = sim.run(tag="a")
    first.cfg.photon_noise = True          # mutate the config we were handed
    second = sim.run(tag="b")
    assert second.cfg.photon_noise is False


def test_simulation_tracks_a_supplied_phantom():
    """Regression: self.N kept the constructor default and could overrun."""
    phantom = make_phantom("composite", Nx=10, Ny=10, Nz=20)
    sim = DualModalitySimulation(N=64, phantom=phantom, n_angles=8,
                                 verbose=False, use_astra=False)
    assert sim.N == phantom.Nz
    result = sim.run(tag="supplied")
    sim.comparison_slices([result])        # indexes with sim.N — must not raise


def test_results_are_indexed_by_tag():
    sim = DualModalitySimulation(preset="composite", N=12, n_angles=8,
                                 verbose=False, use_astra=False)
    sim.run(ArtifactConfig.clean(), tag="one")
    sim.run(ArtifactConfig.noise_only(), tag="two")
    assert set(sim.results) == {"one", "two"}
    assert sim.signature_table().count("\n") >= 4


def test_cache_round_trips_a_whole_run(tmp_path):
    sim = DualModalitySimulation(preset="composite", N=12, n_angles=8,
                                 verbose=False, use_astra=False,
                                 cache_dir=str(tmp_path), overwrite_cache=True)
    result = sim.run(ArtifactConfig.clean(), tag="cached")

    cache = SimCache(tmp_path)
    assert np.allclose(cache.load_run_volume("cached", "xray"), result.vol_xray)
    assert cache.load_phantom().shape == sim.phantom.shape
    assert "cached" in cache.list_run_tags()


def test_saved_phantom_does_not_write_the_derived_volumes(tmp_path):
    """They are reproducible from labels + materials, and 16× the size."""
    cache = SimCache(tmp_path)
    cache.save_phantom(make_phantom("composite", N=12))
    written = {p.name for p in cache.phantom_dir.glob("*.npy")}
    assert written == {"label_vol.npy"}


def test_load_phantom_without_a_saved_one_is_explicit(tmp_path):
    with pytest.raises(FileNotFoundError, match="No phantom in this cache"):
        SimCache(tmp_path).load_phantom()
