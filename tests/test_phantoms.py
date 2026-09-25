import numpy as np
import pytest

import neutron_xray_sim as nxs


@pytest.mark.parametrize("preset", ["composite", "battery", "bone_implant",
                                    "industrial", "HDPE_composite"])
def test_presets_build(preset):
    ph = nxs.make_phantom(preset, N=24)
    assert ph.shape == (24, 24, 24)
    assert ph.materials[0].name == "Air"
    assert len(np.unique(ph.label_vol)) >= 3
    assert ph.mu_x_vols.shape == (len(nxs.XRAY_E_KEV), 24, 24, 24)


def test_non_cubic_preset():
    ph = nxs.make_phantom("composite", Nx=20, Ny=24, Nz=16)
    assert ph.shape == (16, 20, 24)


def test_preset_without_nxnynz_raises_value_error():
    # Regression: this used to raise TypeError from inside the factory.
    with pytest.raises(ValueError, match="Nx/Ny/Nz"):
        nxs.make_phantom("jellyroll_battery", Nx=16, Ny=16, Nz=16)


def test_preset_kwargs_forwarded():
    ph = nxs.make_phantom("jellyroll_battery", N=32, n_jellyroll_turns=1)
    assert "1_turns" in ph.name


def test_voxel_size_preserves_geometry():
    a = nxs.make_phantom("composite", N=16)
    b = nxs.make_phantom("composite", N=32)
    assert a.voxel_cm * 16 == pytest.approx(b.voxel_cm * 32)


def test_builder_material_identity_and_paint():
    b = nxs.PhantomBuilder(N=10, voxel_cm=0.1)
    b.add_sphere("water", radius_cm=0.3).add_sphere(nxs.WATER, radius_cm=0.2)
    b.paint("iron", b.X > 0.4)
    ph = b.build()
    assert [m.name for m in ph.materials] == ["Air", "Water", "Iron"]
    with pytest.raises(KeyError, match="Unknown material"):
        b.add_sphere("unobtainium")


def test_register_preset():
    nxs.register_preset("_tiny", lambda N=8, voxel_cm=None, **_: nxs.PhantomBuilder(
        N=N, voxel_cm=0.1).add_sphere("water", radius_cm=0.2).build("tiny"))
    try:
        assert nxs.make_phantom("_tiny", N=8).name == "tiny"
    finally:
        del nxs.PHANTOM_PRESETS["_tiny"]


def test_phantom_from_array():
    seg = np.zeros((4, 6, 6), dtype=np.uint8)
    seg[:, 2:4, 2:4] = 7
    ph = nxs.phantom_from_array(seg, {"voxel_cm": 0.1, "class_map": {0: "air", 7: "iron"}})
    assert ph.shape == (4, 6, 6)
    assert ph.materials[1].name == "Iron"
    assert np.all(ph.label_vol[seg == 7] == 1)
