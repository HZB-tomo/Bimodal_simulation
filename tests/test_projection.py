import numpy as np
import pytest
from skimage.transform import radon

import neutron_xray_sim as nxs
from neutron_xray_sim.acquisition.projector import line_integrals


def test_cpu_projector_matches_skimage_radon():
    rng = np.random.default_rng(1)
    img = np.zeros((32, 32))
    img[8:14, 18:22] = 1.0
    img[20:25, 5:9] = 2.0
    img += 0.1 * rng.random(img.shape) * (np.hypot(*np.mgrid[-16:16, -16:16]) < 14)
    angles = np.linspace(0, 180, 45, endpoint=False)
    ours = line_integrals(img[None], angles, use_astra=False)[:, 0, :]
    ref = radon(img, theta=angles, circle=True).T
    np.testing.assert_allclose(ours, ref, atol=1e-5)


def test_cpu_projection_reconstruction_is_not_flipped(asymmetric_phantom):
    # Regression: the CPU projector used the opposite rotation sense to the
    # CPU FBP, so reconstructions came out mirrored relative to the phantom.
    ph = asymmetric_phantom
    _, n = nxs.make_sinogram_pair(ph, n_angles=90, use_astra=False, verbose=False)
    vol = nxs.reconstruct(n, remove_rings=False)
    s = ph.shape[0] // 2
    corr = np.corrcoef(vol[s].ravel(), ph.mu_n_vol[s].ravel())[0, 1]
    assert corr > 0.95


def test_sinogram_shapes_non_cubic():
    ph = nxs.make_phantom("composite", Nx=20, Ny=24, Nz=12)
    x, n = nxs.make_sinogram_pair(ph, n_angles=10, use_astra=False, verbose=False)
    assert x["sino_lam"].shape == (10, 12, 24)
    assert n["sino_lam"].shape == (10, 12, 24)


def test_monochromatic_non_cubic():
    # Regression: the monochromatic projector assumed a cubic phantom.
    ph = nxs.make_phantom("composite", Nx=20, Ny=24, Nz=12)
    x, _ = nxs.make_sinogram_pair(ph, n_angles=8, xray_mode="monochromatic",
                                  xray_energy_keV=80, use_astra=False, verbose=False)
    assert x["sino_lam"].shape == (8, 12, 24)


def test_neutron_components_sum(clean_sinos):
    _, n = clean_sinos
    np.testing.assert_allclose(n["sino_abs_lam"] + n["sino_scatter_lam"],
                               n["sino_lam"], rtol=1e-4, atol=1e-4)


def test_polychromatic_beam_hardening():
    # A thick iron slab: the polychromatic optical depth grows sub-linearly.
    b = nxs.PhantomBuilder(N=24, voxel_cm=0.01)
    b.add_box("iron", half_extents_cm=(1, 0.1, 1))
    x, _ = nxs.make_sinogram_pair(b.build(), n_angles=1, use_astra=False, verbose=False)
    lam = x["sino_lam"][0, 12, 12]
    mono = nxs.IRON.mu_x_at(x["spectrum"]["energies_keV"][0]) * 0.2
    assert 0 < lam < mono


def test_reconstruct_unknown_algorithm():
    with pytest.raises(ValueError, match="Unknown algorithm"):
        nxs.reconstruct({"sino_lam": np.zeros((2, 1, 4)), "angles_deg": np.zeros(2)},
                        algorithm="magic")
