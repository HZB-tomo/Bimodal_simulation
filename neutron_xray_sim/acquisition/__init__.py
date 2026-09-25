"""
Acquisition: everything between the phantom and the measured sinograms.

* :mod:`.projector`       — polychromatic X-ray and thermal-neutron projection
* :mod:`.artifacts`       — ``ArtifactConfig`` and artifact injection
* :mod:`.noise`           — Poisson counting statistics and dose metrics
* :mod:`.cone3d_geometry` — ASTRA cone-beam / laminography geometries
* :mod:`.laminography`    — 3-D cone-beam projection & reconstruction (ASTRA)
"""

from .artifacts import (
    PRESET_CONFIGS,
    ArtifactConfig,
    inject_sinogram_artifacts,
    inject_volume_artifacts,
)
from .noise import (
    DoseConfig,
    add_poisson_noise,
    add_poisson_noise_streaming,
    cnr,
    d_prime,
    joint_d_prime,
    material_stats,
    measure_sigma_lambda,
    predicted_sigma_lambda,
    rng_for,
    rose_dose_threshold,
)
from .projector import (
    ASTRA_OK,
    line_integrals,
    make_sinogram_pair,
    project_neutron,
    project_xray,
    project_xray_monochromatic,
)

__all__ = [
    "PRESET_CONFIGS", "ArtifactConfig", "inject_sinogram_artifacts",
    "inject_volume_artifacts",
    "DoseConfig", "add_poisson_noise", "add_poisson_noise_streaming", "cnr",
    "d_prime", "joint_d_prime", "material_stats", "measure_sigma_lambda",
    "predicted_sigma_lambda", "rng_for", "rose_dose_threshold",
    "ASTRA_OK", "line_integrals", "make_sinogram_pair", "project_neutron",
    "project_xray", "project_xray_monochromatic",
]
