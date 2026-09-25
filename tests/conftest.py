"""Shared fixtures.  Everything runs on CPU (NumPy / scikit-image backends)."""

import matplotlib

matplotlib.use("Agg")

import numpy as np  # noqa: E402
import pytest  # noqa: E402

import neutron_xray_sim as nxs  # noqa: E402


@pytest.fixture
def small_phantom():
    """A 32³ composite phantom — big enough to have every material."""
    return nxs.make_phantom("composite", N=32)


@pytest.fixture
def asymmetric_phantom():
    """Off-centre, asymmetric inclusions: sensitive to flips / rotations."""
    b = nxs.PhantomBuilder(N=40, voxel_cm=0.02)
    b.add_box("iron", center_cm=(0, 0.15, 0.08), half_extents_cm=(0.4, 0.06, 0.12))
    b.add_sphere("water", center_cm=(0, -0.15, -0.12), radius_cm=0.09)
    return b.build("asymmetric")


@pytest.fixture
def clean_sinos(small_phantom):
    return nxs.make_sinogram_pair(small_phantom, n_angles=60, use_astra=False,
                                  verbose=False)


@pytest.fixture
def rng():
    return np.random.default_rng(0)
