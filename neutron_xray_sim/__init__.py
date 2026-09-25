"""
neutron_xray_sim  (DIANA)
═════════════════════════
Dual-modality neutron / X-ray tomography simulation and bimodal-histogram
analysis.

Pipeline::

    phantom ─► forward projection ─► artifact injection ─► reconstruction
            ─► bimodal histogram ─► GMM / segmentation ─► quality metrics

Quick start::

    from neutron_xray_sim import DualModalitySimulation, ArtifactConfig

    sim = DualModalitySimulation(preset="composite", N=64, n_angles=120)
    r_clean = sim.run(ArtifactConfig.clean(), tag="clean")
    r_real  = sim.run(ArtifactConfig.realistic(), tag="realistic")
    fig = sim.comparison_grid()

Package layout
──────────────
physics         element data, materials, X-ray and neutron spectra
phantoms        PhantomData / PhantomBuilder, presets, segmented-volume import
acquisition     projectors, artifacts, noise, cone-beam geometry
reconstruction  FBP / iterative reconstruction, Fourier fusion
analysis        histograms, GMM, artifact signatures, quality metrics
plotting        histogram figures and the publication figure set
simulation      DualModalitySimulation orchestrator + artifact survey
io              SimCache: on-disk cache of every pipeline stage

Everything listed in ``__all__`` is the stable public API and can be imported
from the top level.  The pre-2.0 module paths (``neutron_xray_sim.histogram``,
``neutron_xray_sim.artifacts``, …) remain importable — see ``_compat``.
"""

from ._version import __version__

# ── Physics ──────────────────────────────────────────────────────────────────
from .physics.materials import (
    MATERIALS,
    XRAY_E_KEV,
    Material,
    make_composite_material,
    material_from_formula,
    register_material,
    xray_spectrum,
)
from .physics.neutron_spectra import (
    NEUTRON_MODES,
    NeutronBeam,
    cold_mono_beam,
    cold_poly_beam,
    ill_next_beam,
    mu_n_lut_for_beam,
    mu_n_spectrum_lut,
    plot_spectra,
    thermal_beam,
)

AIR = MATERIALS["air"]
WATER = MATERIALS["water"]
ALUMINUM = MATERIALS["aluminum"]
HDPE = MATERIALS["hdpe"]
IRON = MATERIALS["iron"]
TITANIUM = MATERIALS["titanium"]
COPPER = MATERIALS["copper"]
LEAD = MATERIALS["lead"]
BONE = MATERIALS["bone"]
TUNGSTEN = MATERIALS["tungsten"]
ZINC = MATERIALS["zinc"]

# ── Phantoms ─────────────────────────────────────────────────────────────────
from .phantoms.base import PhantomBuilder, PhantomData
from .phantoms.importer import phantom_from_array, phantom_from_segmented_volume
from .phantoms.presets import (
    PHANTOM_PRESETS,
    make_battery_phantom,
    make_bone_implant_phantom,
    make_composite_phantom,
    make_custom_cylindrical_battery_phantom,
    make_hdpe_composite_phantom,
    make_industrial_phantom,
    make_li_ion_battery_phantom,
    make_phantom,
    register_preset,
)

# ── Acquisition ──────────────────────────────────────────────────────────────
from .acquisition.artifacts import (
    PRESET_CONFIGS,
    ArtifactConfig,
    inject_sinogram_artifacts,
    inject_volume_artifacts,
)
from .acquisition.noise import (
    add_poisson_noise,
    add_poisson_noise_streaming,
    cnr,
    d_prime,
    joint_d_prime,
    material_stats,
    measure_sigma_lambda,
    predicted_sigma_lambda,
    rose_dose_threshold,
)
from .acquisition.projector import (
    make_sinogram_pair,
    project_neutron,
    project_xray,
    project_xray_monochromatic,
)

# ── Reconstruction ───────────────────────────────────────────────────────────
from .reconstruction.reconstructor import AVAILABLE_ALGORITHMS, reconstruct, reconstruct_pair

# ── Analysis ─────────────────────────────────────────────────────────────────
from .analysis.cross_algorithm import make_cross_algorithm_sinos
from .analysis.gmm import (
    GMMFitResult,
    auto_fit_gmm,
    fit_gmm,
    predict_gmm,
    segment_by_gmm,
    segment_by_polygon,
)
from .analysis.histogram import (
    DEFAULT_GT_ENERGY_IDX,
    HistogramResult,
    compute_bimodal_histogram,
    compute_ground_truth_histogram,
)
from .analysis.metrics_morphology import compute_histogram_metrics_morphology_aware
from .analysis.metrics_table import HistogramMetricsTable, compute_histogram_metrics
from .analysis.quality import (
    ClusterQualityMetrics,
    compare_algorithms,
    evaluate_histogram_quality,
)
from .analysis.signatures import ArtifactSignatures, detect_artifact_signatures

# ── Plotting ─────────────────────────────────────────────────────────────────
from .plotting.histograms import (
    plot_artifact_survey,
    plot_bimodal_histogram,
    plot_comparison_grid,
    plot_cross_algorithm_grid,
    plot_ground_truth_comparison,
)

# ── Orchestration & I/O ──────────────────────────────────────────────────────
from .io import SimCache, tag_to_slug
from .simulation import DualModalitySimulation, SimulationResult, run_artifact_survey

__all__ = [
    "__version__",
    # Physics
    "Material", "MATERIALS", "XRAY_E_KEV", "xray_spectrum",
    "material_from_formula", "make_composite_material", "register_material",
    "AIR", "WATER", "ALUMINUM", "HDPE", "IRON", "TITANIUM",
    "COPPER", "LEAD", "BONE", "TUNGSTEN", "ZINC",
    "NEUTRON_MODES", "NeutronBeam", "thermal_beam", "cold_mono_beam",
    "cold_poly_beam", "ill_next_beam", "mu_n_lut_for_beam", "mu_n_spectrum_lut",
    "plot_spectra",
    # Phantoms
    "PhantomData", "PhantomBuilder", "make_phantom", "register_preset",
    "PHANTOM_PRESETS", "make_composite_phantom", "make_battery_phantom",
    "make_bone_implant_phantom", "make_industrial_phantom",
    "make_hdpe_composite_phantom", "make_li_ion_battery_phantom",
    "make_custom_cylindrical_battery_phantom",
    "phantom_from_array", "phantom_from_segmented_volume",
    # Acquisition
    "project_xray", "project_xray_monochromatic", "project_neutron",
    "make_sinogram_pair",
    "ArtifactConfig", "PRESET_CONFIGS", "inject_sinogram_artifacts",
    "inject_volume_artifacts",
    "add_poisson_noise", "add_poisson_noise_streaming", "predicted_sigma_lambda",
    "measure_sigma_lambda", "material_stats", "cnr", "d_prime", "joint_d_prime",
    "rose_dose_threshold",
    # Reconstruction
    "reconstruct", "reconstruct_pair", "AVAILABLE_ALGORITHMS",
    # Analysis
    "HistogramResult", "compute_bimodal_histogram", "compute_ground_truth_histogram",
    "DEFAULT_GT_ENERGY_IDX",
    "GMMFitResult", "fit_gmm", "auto_fit_gmm", "predict_gmm",
    "segment_by_gmm", "segment_by_polygon",
    "ArtifactSignatures", "detect_artifact_signatures",
    "ClusterQualityMetrics", "evaluate_histogram_quality", "compare_algorithms",
    "HistogramMetricsTable", "compute_histogram_metrics",
    "compute_histogram_metrics_morphology_aware",
    "make_cross_algorithm_sinos",
    # Plotting
    "plot_bimodal_histogram", "plot_comparison_grid", "plot_ground_truth_comparison",
    "plot_cross_algorithm_grid", "plot_artifact_survey",
    # Orchestration & I/O
    "DualModalitySimulation", "SimulationResult", "run_artifact_survey",
    "SimCache", "tag_to_slug",
]


def __getattr__(name):
    """
    Pre-2.0 compatibility: the publication figure functions (``diana_plots``)
    used to be re-exported at the top level.  They now live in
    ``neutron_xray_sim.plotting.publication`` and are resolved on demand.
    """
    from .plotting import publication

    if hasattr(publication, name) and not name.startswith("__"):
        return getattr(publication, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


from ._compat import install_aliases as _install_aliases  # noqa: E402

_install_aliases(__import__(__name__))
del _install_aliases
