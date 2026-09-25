import numpy as np
import pytest

import neutron_xray_sim as nxs
from neutron_xray_sim.analysis.quality import (
    davies_bouldin, ground_truth_labels_for, match_components, pairwise_overlap,
)


def _gt_volumes(phantom, noise=0.02, seed=0):
    rng = np.random.default_rng(seed)
    vx = phantom.mu_x_vols[nxs.DEFAULT_GT_ENERGY_IDX]
    vn = phantom.mu_n_vol
    return (vx + noise * rng.standard_normal(vx.shape),
            vn + noise * rng.standard_normal(vn.shape))


def test_histogram_mask_is_recorded(small_phantom):
    vx, vn = _gt_volumes(small_phantom)
    mask = small_phantom.label_vol > 0
    h = nxs.compute_bimodal_histogram(vx, vn, bins=32, mask=mask)
    assert h.total_voxels == mask.sum()
    labels = ground_truth_labels_for(h, small_phantom)
    assert labels.size == h.vol_x_flat.size and labels.min() > 0


def test_gmm_fit_and_segment(small_phantom):
    # Regression: segment_by_gmm crashed on current scikit-learn.
    vx, vn = _gt_volumes(small_phantom)
    h = nxs.compute_bimodal_histogram(vx, vn, bins=64)
    gmm = nxs.fit_gmm(h, n_components=len(small_phantom.materials), n_init=1)
    assert gmm.covariances.shape == (gmm.n_components, 2, 2)
    seg = nxs.segment_by_gmm(vx, vn, gmm)
    assert seg.shape == vx.shape
    np.testing.assert_array_equal(seg.ravel(), gmm.labels_flat)


@pytest.mark.parametrize("cov", ["diag", "spherical", "tied"])
def test_gmm_covariance_types_are_full(small_phantom, cov):
    vx, vn = _gt_volumes(small_phantom)
    h = nxs.compute_bimodal_histogram(vx, vn, bins=32)
    gmm = nxs.fit_gmm(h, n_components=3, covariance_type=cov, n_init=1)
    assert gmm.covariances.shape == (3, 2, 2)


def test_evaluate_quality_near_ground_truth(small_phantom):
    vx, vn = _gt_volumes(small_phantom, noise=0.01)
    h = nxs.compute_bimodal_histogram(vx, vn, bins=64)
    m = nxs.evaluate_histogram_quality(h, small_phantom, gmm_n_init=2)
    assert m.n_matched == len(small_phantom.materials) - 1
    assert m.mean_centroid_error < 0.05


def test_evaluate_quality_with_masked_histogram(small_phantom):
    # Regression: masked histograms silently produced no overlap fractions.
    vx, vn = _gt_volumes(small_phantom, noise=0.3)
    h = nxs.compute_bimodal_histogram(vx, vn, bins=64, mask=small_phantom.label_vol > 0)
    m = nxs.evaluate_histogram_quality(h, small_phantom, gmm_n_init=1)
    assert m.overlap_fractions


def test_metrics_tables(small_phantom):
    vx, vn = _gt_volumes(small_phantom)
    h = nxs.compute_bimodal_histogram(vx, vn, bins=64)
    t = nxs.compute_histogram_metrics(small_phantom, h, gt_seeded_gmm=True)
    assert t.scalars["CE"] < 0.1
    t2, regions = nxs.compute_histogram_metrics_morphology_aware(
        small_phantom, h, vx, vn, mode="morphology_explore")
    assert t2.scalars["CE"] < 0.1 and regions
    assert len(t.to_records()) > 0


def test_primitives():
    assert match_components(np.array([[0, 0], [5, 5]]),
                            np.array([[5.1, 5], [0.1, 0], [9, 9]])) == {0: 1, 1: 0}
    assert np.isnan(davies_bouldin(np.zeros((1, 2)), [1.0]))
    assert davies_bouldin(np.array([[0, 0], [10, 0]]), [1, 1]) == pytest.approx(0.2)
    true = np.array([1, 1, 2, 2])
    assert pairwise_overlap(true, np.array([1, 2, 2, 2]), [(1, 2)]) == {(1, 2): 0.25}


def test_signatures_reference_shift(small_phantom):
    vx, vn = _gt_volumes(small_phantom)
    h = nxs.compute_bimodal_histogram(vx, vn, bins=32)
    h2 = nxs.compute_bimodal_histogram(vx, vn + 0.1, bins=32)
    s = nxs.detect_artifact_signatures(h2, ref_hist=h)
    assert s.marginal_shift_n == pytest.approx(0.1, rel=1e-3)
