"""
neutron_xray_sim
════════════════
Dual-modality neutron / X-ray tomography simulation package.

Complete pipeline:
    PhantomData → forward projection → artifact injection
    → CT reconstruction → bimodal histogram analysis

Quick start
───────────
::

    from neutron_xray_sim import DualModalitySimulation, ArtifactConfig
    import matplotlib.pyplot as plt

    sim = DualModalitySimulation(preset="composite", N=64, n_angles=120)

    # Clean reference
    r_clean = sim.run(ArtifactConfig.clean(), tag="clean")

    # Realistic artifacts
    r_real  = sim.run(ArtifactConfig.realistic(), tag="realistic")

    # Custom: misalignment only
    r_mis   = sim.run(
        ArtifactConfig(misalignment=True, translation_voxels=(4,0,0)),
        tag="misalignment",
    )

    fig = sim.comparison_grid()
    plt.show()

Modules
───────
materials     : Material database (11 materials, X-ray and neutron coefficients)
phantom       : Voxelised 3-D phantom builder + 4 preset phantoms
projector     : Polychromatic X-ray + thermal neutron forward projection
artifacts     : ArtifactConfig dataclass + all artifact injection functions
reconstructor : FBP / SIRT / CGLS CT reconstruction (ASTRA GPU or NumPy)
histogram     : 2-D bimodal histogram, GMM fitting, segmentation, plotting
simulation    : DualModalitySimulation orchestrator + SimulationResult
"""

from .materials import (
    Material,
    MATERIALS,
    XRAY_E_KEV,
    xray_spectrum,
    AIR, WATER, ALUMINUM, HDPE, IRON, TITANIUM,
    COPPER, LEAD, BONE, TUNGSTEN, ZINC,
    make_composite_material,
    material_from_formula,
)

from .phantom import (
    PhantomData,
    PhantomBuilder,
    make_phantom,
    make_composite_phantom,
    make_battery_phantom,
    make_bone_implant_phantom,
    make_industrial_phantom,
    make_custom_cylindrical_battery_phantom,
    make_hdpe_composite_phantom,
    
    PHANTOM_PRESETS,
)

from .projector import (
    project_xray,
    project_neutron,
    make_sinogram_pair,
)

from .artifacts import (
    ArtifactConfig,
    inject_sinogram_artifacts,
    inject_volume_artifacts,
    PRESET_CONFIGS,
)

from .reconstructor import (
    reconstruct,
    reconstruct_pair,
    AVAILABLE_ALGORITHMS,
)

from .io import (
    SimCache,
    tag_to_slug,
)

from .histogram import (
    HistogramResult,
    GMMFitResult,
    ArtifactSignatures,
    ClusterQualityMetrics,
    compute_bimodal_histogram,
    compute_ground_truth_histogram,
    fit_gmm,
    auto_fit_gmm,
    segment_by_gmm,
    segment_by_polygon,
    detect_artifact_signatures,
    evaluate_histogram_quality,
    compare_algorithms,
    plot_bimodal_histogram,
    plot_ground_truth_comparison,
    plot_comparison_grid,
    plot_cross_algorithm_grid,
    make_cross_algorithm_sinos,
)

from .simulation import (
    SimulationResult,
    DualModalitySimulation,
    run_artifact_survey,
)

from .metrics_table import (
    HistogramMetricsTable,
    compute_histogram_metrics,
)

from .metrics_table_morphology import (
    compute_histogram_metrics_morphology_aware,
)

from .neutron_spectra import (
    NEUTRON_MODES,
    mu_n_lut_for_beam,
    mu_n_spectrum_lut,
    plot_spectra,
    cold_mono_beam,
    cold_poly_beam,
    ill_next_beam,
    
)

from .diana_plots import (
	plot_phantom_label_map,
	plot_bimodal_histogram,
	plot_xray_marginal,
	plot_neutron_marginal,
	plot_reconstruction_slice,
	plot_CE_vs_nprojections,
	plot_DB_vs_nprojections,
	plot_sigma_x_vs_nprojections,
	plot_sigma_n_vs_nprojections,
	plot_CE_all_geometries,
	plot_DB_all_geometries,
	plot_cross_sweep_heatmap,
	plot_xray_attenuation_spectra,
	plot_epsilon_vs_nprojections,
	plot_pairwise_overlap_vs_nprojections,
	plot_artifact_fingerprint,
	plot_metric_by_artifact,
	plot_algorithm_cross_heatmap,
	plot_rare_phase_failure,
	plot_metric_vs_param_multi,
	plot_nstar_vs_margin,
	plot_recovery_heatmap,
	plot_grouped_bars_with_ci,
	_make_dummy_hist,
	_demo,
	
)

__version__  = "1.1.0"
__author__   = "neutron_xray_sim contributors"

__all__ = [
    # Materials
    "Material", "MATERIALS", "XRAY_E_KEV", "xray_spectrum",
    "AIR", "WATER", "ALUMINUM", "HDPE", "IRON", "TITANIUM",
    "COPPER", "LEAD", "BONE", "TUNGSTEN", "ZINC",
    # Phantom
    "PhantomData", "PhantomBuilder", "make_phantom",
    "make_composite_phantom", "make_battery_phantom",
    "make_bone_implant_phantom", "make_industrial_phantom",
    "PHANTOM_PRESETS",
    # Projection
    "project_xray", "project_neutron", "make_sinogram_pair",
    # Artifacts
    "ArtifactConfig", "inject_sinogram_artifacts",
    "inject_volume_artifacts", "PRESET_CONFIGS",
    # Reconstruction
    "reconstruct", "reconstruct_pair", "AVAILABLE_ALGORITHMS",
    # IO / cache
    "SimCache", "tag_to_slug",
    # Histogram
    "HistogramResult", "GMMFitResult", "ArtifactSignatures", "ClusterQualityMetrics",
    "compute_bimodal_histogram", "compute_ground_truth_histogram",
    "fit_gmm", "auto_fit_gmm",
    "segment_by_gmm", "segment_by_polygon",
    "detect_artifact_signatures",
    "evaluate_histogram_quality", "compare_algorithms",
    "plot_bimodal_histogram", "plot_ground_truth_comparison", "plot_comparison_grid",
    "plot_cross_algorithm_grid", "make_cross_algorithm_sinos",
    # Simulation
    "SimulationResult", "DualModalitySimulation", "run_artifact_survey",
    "ClusterQualityMetrics",
]
