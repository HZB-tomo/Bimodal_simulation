"""Shared pytest fixtures.

Every test here runs on CPU in seconds.  Anything needing a GPU is marked
``@pytest.mark.gpu`` and skipped automatically when ASTRA is unavailable, so
``pytest`` gives the same green result on a laptop and on a compute node.
"""

from __future__ import annotations

import matplotlib
import pytest

matplotlib.use("Agg")   # no display on CI runners

from neutron_xray_sim.phantom import make_phantom  # noqa: E402

#: Deliberately tiny. These tests check behaviour, not image quality; a 16³
#: phantom keeps the whole suite under a few seconds on CPU.
TEST_N = 16


@pytest.fixture(scope="session")
def small_phantom():
    """A 16³ composite phantom — six materials, all four contrast regimes."""
    return make_phantom("composite", N=TEST_N)


@pytest.fixture
def fresh_registry():
    """A ``MaterialRegistry`` seeded with the built-ins, safe to mutate.

    Tests that register materials must use this rather than the global
    ``MATERIALS``, or they leak state into whatever runs next.
    """
    from neutron_xray_sim.materials.database import BUILTIN_SPECS
    from neutron_xray_sim.materials.registry import MaterialRegistry

    return MaterialRegistry(BUILTIN_SPECS)


def pytest_runtest_setup(item):
    """Skip GPU-marked tests when ASTRA is not importable."""
    if "gpu" in item.keywords:
        pytest.importorskip("astra", reason="ASTRA toolbox not installed")
