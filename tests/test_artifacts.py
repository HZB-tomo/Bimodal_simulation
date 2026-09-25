import numpy as np
import pytest

import neutron_xray_sim as nxs
from neutron_xray_sim.acquisition.artifacts import _blur_projections


def test_clean_config_is_identity(clean_sinos):
    x, n = clean_sinos
    x2, n2 = nxs.inject_sinogram_artifacts(x, n, nxs.ArtifactConfig.clean())
    np.testing.assert_array_equal(x2["sino_lam"], x["sino_lam"])
    np.testing.assert_array_equal(n2["sino_lam"], n["sino_lam"])
    assert x2["sino_lam"] is not x["sino_lam"]


def test_psf_blurs_whole_projection_image():
    # Regression: the PSF blurred only along detector rows (1-D).
    sino = np.zeros((2, 9, 9), dtype=np.float32)
    sino[:, 4, 4] = 1.0
    out = _blur_projections(sino, 1.0)
    assert out[0, 3, 4] > 0 and out[0, 4, 3] > 0      # both image axes
    np.testing.assert_allclose(out[0], out[1])         # angles independent


def test_noise_statistics(clean_sinos):
    x, n = clean_sinos
    I0 = 1e4
    cfg = nxs.ArtifactConfig(photon_noise=True, I0_xray=I0, I0_neutron=I0)
    _, n2 = nxs.inject_sinogram_artifacts(x, n, cfg, rng=np.random.default_rng(3))
    air = n["sino_lam"] < 1e-6
    measured = np.std(n2["sino_lam"][air])
    assert measured == pytest.approx(nxs.predicted_sigma_lambda(0.0, I0), rel=0.15)


def test_noise_is_applied_after_blur(clean_sinos):
    # Detector blur must not smooth away the counting noise.
    x, n = clean_sinos
    I0 = 1e4
    base = dict(photon_noise=True, I0_xray=I0, I0_neutron=I0)
    _, noisy = nxs.inject_sinogram_artifacts(x, n, nxs.ArtifactConfig(**base))
    _, blurred = nxs.inject_sinogram_artifacts(
        x, n, nxs.ArtifactConfig(**base, detector_psf=True, psf_sigma_neutron_pixels=2.0))
    air = n["sino_lam"] < 1e-6
    assert np.std(blurred["sino_lam"][air]) > 0.7 * np.std(noisy["sino_lam"][air])


def test_scatter_lowers_optical_depth(clean_sinos):
    x, n = clean_sinos
    cfg = nxs.ArtifactConfig(neutron_scatter=True, scatter_fraction=0.1)
    _, n2 = nxs.inject_sinogram_artifacts(x, n, cfg)
    thick = n["sino_lam"] > 1.0
    assert np.all(n2["sino_lam"][thick] < n["sino_lam"][thick])


def test_rings_and_volume_artifacts(clean_sinos):
    x, n = clean_sinos
    cfg = nxs.ArtifactConfig(ring_artifacts=True, misalignment=True,
                             translation_voxels=(0, 2, 0), salt_pepper=True)
    x2, _ = nxs.inject_sinogram_artifacts(x, n, cfg)
    assert not np.allclose(x2["sino_lam"], x["sino_lam"])
    vol = np.random.default_rng(0).random((8, 8, 8)).astype(np.float32)
    vx, vn = nxs.inject_volume_artifacts(vol, vol, cfg)
    assert vx.shape == vn.shape == vol.shape
    assert not np.allclose(vn, vol)


def test_realistic_summary():
    assert "noise" in nxs.ArtifactConfig.realistic().summary()
