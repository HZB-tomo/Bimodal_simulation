import os
import subprocess
import sys

import pytest

import neutron_xray_sim as nxs


def test_import_has_no_side_effects(tmp_path):
    # Only warnings raised by DIANA itself count: third-party libraries (e.g.
    # older matplotlib + new pyparsing) may emit deprecation warnings on import.
    code = (
        "import os, warnings\n"
        "import matplotlib as m\n"
        "before = dict(m.rcParams)\n"
        "with warnings.catch_warnings(record=True) as caught:\n"
        "    warnings.simplefilter('always')\n"
        "    import neutron_xray_sim\n"
        "ours = [str(w.message) for w in caught if 'neutron_xray_sim' in w.filename]\n"
        "assert not ours, ours\n"
        "assert not os.path.exists('outputs_disc_sweep')\n"
        "assert dict(m.rcParams) == before, 'rcParams changed'\n"
    )
    res = subprocess.run([sys.executable, "-c", code],
                         cwd=tmp_path, capture_output=True, text=True,
                         env={**os.environ, "MPLBACKEND": "Agg"})
    assert res.returncode == 0, res.stderr


def test_top_level_plot_bimodal_histogram_is_the_documented_one():
    # Regression: the publication variant silently shadowed it.
    assert nxs.plot_bimodal_histogram.__module__ == "neutron_xray_sim.plotting.histograms"


@pytest.mark.parametrize("old, name", [
    ("histogram", "fit_gmm"),
    ("histogram", "plot_cross_algorithm_grid"),
    ("artifacts", "ArtifactConfig"),
    ("phantom", "make_composite_phantom"),
    ("materials", "MATERIALS"),
    ("io", "SimCache"),
    ("volume_importer", "phantom_from_array"),
    ("metrics_table_morphology", "compute_histogram_metrics_morphology_aware"),
    ("diana_plots", "FONT"),
])
def test_legacy_module_paths(old, name):
    import importlib
    mod = importlib.import_module(f"neutron_xray_sim.{old}")
    assert hasattr(mod, name)


def test_legacy_top_level_publication_plots():
    assert callable(nxs.plot_CE_vs_nprojections)
    with pytest.raises(AttributeError):
        nxs.does_not_exist


def test_all_is_importable():
    for name in nxs.__all__:
        assert hasattr(nxs, name), name
