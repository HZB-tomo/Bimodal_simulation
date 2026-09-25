import matplotlib.pyplot as plt
import numpy as np
import pytest

import neutron_xray_sim as nxs


def test_simulation_run(small_phantom):
    sim = nxs.DualModalitySimulation(phantom=small_phantom, n_angles=40,
                                     use_astra=False, verbose=False,
                                     histogram_bins=64)
    clean = sim.run(tag="clean")
    real = sim.run(nxs.ArtifactConfig.realistic(), tag="real", ref_result=clean,
                   n_gmm_components=4)
    assert clean.vol_xray.shape == small_phantom.shape
    assert real.gmm is not None and real.signatures is not None
    assert "real" in sim.signature_table()
    for fig in (sim.comparison_grid(), sim.comparison_slices(), clean.plot_slices(),
                clean.plot_histogram(),
                nxs.plot_ground_truth_comparison(small_phantom, clean.histogram)):
        assert isinstance(fig, plt.Figure)
    plt.close("all")


def test_default_run_config_is_not_shared():
    sim = nxs.DualModalitySimulation(preset="composite", N=12, n_angles=8,
                                     use_astra=False, verbose=False)
    r1 = sim.run(tag="a")
    r1.cfg.photon_noise = True
    assert sim.run(tag="b").cfg.photon_noise is False


@pytest.mark.slow
def test_artifact_survey():
    results, fig, metrics = nxs.run_artifact_survey(
        N=20, n_angles=30, use_astra=False, verbose=False, include_combinations=False)
    assert len(results) == 10 and len(metrics) == 10
    assert isinstance(fig, plt.Figure)
    plt.close("all")


@pytest.mark.slow
def test_cross_algorithm_grid(small_phantom):
    algs = ["FBP"]
    xs, ns = nxs.make_cross_algorithm_sinos(small_phantom, algs, n_angles=30,
                                            use_astra=False)
    fig, hists = nxs.plot_cross_algorithm_grid(small_phantom, xs, ns,
                                               [("FBP", "FBP")], use_astra=False)
    assert ("FBP", "FBP") in hists
    plt.close("all")


def test_bragg_scan_restores_phantom(small_phantom, monkeypatch):
    from neutron_xray_sim.physics import ncrystal_bragg as nb
    n_lab = len(small_phantom.materials)
    monkeypatch.setattr(nb, "mu_n_bragg_lut",
                        lambda phases, wl, temperature_K=0: np.ones((n_lab, len(wl)),
                                                                     np.float32))
    before = small_phantom.mu_n_vol.copy()
    out = nb.run_bragg_edge_scan(small_phantom, [None] * n_lab, np.array([3.0, 4.0]),
                                 np.linspace(0, 180, 10, endpoint=False),
                                 use_astra=False, verbose=False)
    assert out["vol_n"].shape[0] == 2
    np.testing.assert_array_equal(small_phantom.mu_n_vol, before)
