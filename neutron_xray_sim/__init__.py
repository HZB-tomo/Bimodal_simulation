"""
neutron_xray_sim
════════════════
Dual-modality neutron / X-ray tomography simulation package (DIANA).

Complete pipeline::

    PhantomData → forward projection → artifact injection
                → CT reconstruction → bimodal histogram analysis

Quick start
───────────
::

    from neutron_xray_sim import DualModalitySimulation, ArtifactConfig
    import matplotlib.pyplot as plt

    sim = DualModalitySimulation(preset="composite", N=64, n_angles=120)

    r_clean = sim.run(ArtifactConfig.clean(),     tag="clean")
    r_real  = sim.run(ArtifactConfig.realistic(), tag="realistic")

    fig = sim.comparison_grid([r_clean, r_real])
    plt.show()

Modules
───────
======================  ======================================================
``materials``           Material registry, specs, and element data
``phantom``             ``PhantomData``, ``PhantomBuilder``, preset registry
``projector``           Polychromatic X-ray + thermal neutron forward projection
``artifacts``           ``ArtifactConfig`` and all artifact injection
``reconstructor``       FBP / SIRT / SART / CGLS / … reconstruction
``histogram``           2-D bimodal histogram, GMM fitting, segmentation
``simulation``          ``DualModalitySimulation`` orchestrator
``neutron_spectra``     Thermal / cold / ILL-NeXT beam models
``volume_importer``     Phantoms from real segmented volumes
``metrics_table``       Tabulated cluster-quality metrics
``noise``               Dose models and detectability metrics
``io``                  On-disk ``SimCache`` for pipeline stages
``diana_plots``         Publication-figure plotting helpers
======================  ======================================================

Plotting namespaces
───────────────────
``histogram`` and ``diana_plots`` both define ``plot_bimodal_histogram``, with
different signatures and styling.  They are *not* both re-exported flat here —
doing so silently shadowed one with the other.  Import the one you want::

    from neutron_xray_sim.histogram import plot_bimodal_histogram      # analysis
    from neutron_xray_sim import diana_plots                           # figures
    diana_plots.plot_bimodal_histogram(...)
"""

from __future__ import annotations

# Submodules exposed as attributes, so `neutron_xray_sim.diana_plots.…` works
# after a plain `import neutron_xray_sim`.
from . import (
    artifacts,
    diana_plots,
    histogram,
    io,
    materials,
    noise,
    phantom,
    projector,
    reconstructor,
    simulation,
)
from .artifacts import (
    PRESET_CONFIGS,
    ArtifactConfig,
    inject_sinogram_artifacts,
    inject_volume_artifacts,
)
from .histogram import (
    ArtifactSignatures,
    ClusterQualityMetrics,
    GMMFitResult,
    HistogramResult,
    auto_fit_gmm,
    compare_algorithms,
    compute_bimodal_histogram,
    compute_ground_truth_histogram,
    detect_artifact_signatures,
    evaluate_histogram_quality,
    fit_gmm,
    make_cross_algorithm_sinos,
    plot_bimodal_histogram,
    plot_comparison_grid,
    plot_cross_algorithm_grid,
    plot_ground_truth_comparison,
    segment_by_gmm,
    segment_by_polygon,
)
from .io import SimCache, tag_to_slug
from .materials import (
    MATERIALS,
    XRAY_E_KEV,
    Material,
    MaterialRegistry,
    MaterialSpec,
    available_elements,
    build_material,
    element_data_status,
    make_composite_material,
    material_from_formula,
    xray_spectrum,
)
from .metrics_table import HistogramMetricsTable, compute_histogram_metrics
from .metrics_table_morphology import compute_histogram_metrics_morphology_aware
from .neutron_spectra import (
    NEUTRON_MODES,
    cold_mono_beam,
    cold_poly_beam,
    ill_next_beam,
    mu_n_lut_for_beam,
    mu_n_spectrum_lut,
    plot_spectra,
)
from .noise import (
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
from .phantom import (
    PHANTOM_DESCRIPTIONS,
    PHANTOM_PRESETS,
    PhantomBuilder,
    PhantomData,
    make_battery_phantom,
    make_bone_implant_phantom,
    make_composite_phantom,
    make_custom_cylindrical_battery_phantom,
    make_hdpe_composite_phantom,
    make_industrial_phantom,
    make_li_ion_battery_phantom,
    make_phantom,
    register_phantom,
    resolve_grid,
)
from .projector import make_sinogram_pair, project_neutron, project_xray
from .reconstructor import AVAILABLE_ALGORITHMS, reconstruct, reconstruct_pair
from .simulation import DualModalitySimulation, SimulationResult, run_artifact_survey

__version__ = "1.2.0"
__author__ = "DIANA contributors, Helmholtz-Zentrum Berlin"

__all__ = [
    "__version__",
    # Submodules
    "artifacts", "diana_plots", "histogram", "io", "materials", "noise",
    "phantom", "projector", "reconstructor", "simulation",
    # Materials
    "Material", "MaterialSpec", "MaterialRegistry", "MATERIALS", "XRAY_E_KEV",
    "build_material", "material_from_formula", "make_composite_material",
    "xray_spectrum", "available_elements", "element_data_status",
    # Phantom
    "PhantomData", "PhantomBuilder", "make_phantom", "resolve_grid",
    "register_phantom", "PHANTOM_PRESETS", "PHANTOM_DESCRIPTIONS",
    "make_composite_phantom", "make_battery_phantom",
    "make_bone_implant_phantom", "make_industrial_phantom",
    "make_hdpe_composite_phantom", "make_li_ion_battery_phantom",
    "make_custom_cylindrical_battery_phantom",
    # Projection
    "project_xray", "project_neutron", "make_sinogram_pair",
    # Artifacts
    "ArtifactConfig", "inject_sinogram_artifacts", "inject_volume_artifacts",
    "PRESET_CONFIGS",
    # Reconstruction
    "reconstruct", "reconstruct_pair", "AVAILABLE_ALGORITHMS",
    # IO / cache
    "SimCache", "tag_to_slug",
    # Histogram
    "HistogramResult", "GMMFitResult", "ArtifactSignatures",
    "ClusterQualityMetrics", "compute_bimodal_histogram",
    "compute_ground_truth_histogram", "fit_gmm", "auto_fit_gmm",
    "segment_by_gmm", "segment_by_polygon", "detect_artifact_signatures",
    "evaluate_histogram_quality", "compare_algorithms",
    "plot_bimodal_histogram", "plot_ground_truth_comparison",
    "plot_comparison_grid", "plot_cross_algorithm_grid",
    "make_cross_algorithm_sinos",
    # Metrics
    "HistogramMetricsTable", "compute_histogram_metrics",
    "compute_histogram_metrics_morphology_aware",
    # Neutron spectra
    "NEUTRON_MODES", "mu_n_lut_for_beam", "mu_n_spectrum_lut", "plot_spectra",
    "cold_mono_beam", "cold_poly_beam", "ill_next_beam",
    # Noise / dose
    "add_poisson_noise", "add_poisson_noise_streaming",
    "predicted_sigma_lambda", "measure_sigma_lambda", "material_stats",
    "cnr", "d_prime", "joint_d_prime", "rose_dose_threshold",
    # Simulation
    "SimulationResult", "DualModalitySimulation", "run_artifact_survey",
]
