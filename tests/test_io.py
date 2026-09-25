import numpy as np
import pytest

import neutron_xray_sim as nxs


def test_phantom_roundtrip(tmp_path, small_phantom):
    cache = nxs.SimCache(tmp_path)
    cache.save_phantom(small_phantom)
    ph = cache.load_phantom()
    np.testing.assert_array_equal(ph.label_vol, small_phantom.label_vol)
    np.testing.assert_allclose(ph.mu_n_vol, small_phantom.mu_n_vol)
    assert [m.name for m in ph.materials] == [m.name for m in small_phantom.materials]


def test_run_roundtrip_and_stale_cache(tmp_path):
    kw = dict(preset="composite", N=16, n_angles=12, use_astra=False, verbose=False,
              cache_dir=tmp_path)
    sim = nxs.DualModalitySimulation(**kw)
    r = sim.run(tag="clean run")
    cache = nxs.SimCache(tmp_path)
    assert cache.list_run_tags() == ["clean run"]
    np.testing.assert_allclose(cache.load_run_volume("clean run", "xray"), r.vol_xray)
    assert cache.load_raw_xray_sino()["sino_lam"].shape[0] == 12

    # Same settings: reuses the cache.
    nxs.DualModalitySimulation(**kw)
    # Different projection settings: must not silently reuse stale sinograms.
    with pytest.warns(UserWarning, match="different parameters"):
        with pytest.raises(ValueError, match="overwrite_cache"):
            nxs.DualModalitySimulation(**{**kw, "n_angles": 20})
    # Different phantom in the same directory is refused.
    with pytest.raises(ValueError, match="holds phantom"):
        nxs.DualModalitySimulation(**{**kw, "N": 20})


def test_reads_do_not_create_directories(tmp_path):
    cache = nxs.SimCache(tmp_path)
    assert not cache.has_run("nope")
    assert not (tmp_path / "runs" / "nope").exists()


def test_tag_to_slug():
    assert nxs.tag_to_slug("Clean (reference)") == "clean_reference"
