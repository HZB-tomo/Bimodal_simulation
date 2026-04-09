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
)

from .phantom import (
    PhantomData,
    PhantomBuilder,
    make_phantom,
    make_composite_phantom,
    make_battery_phantom,
    make_bone_implant_phantom,
    make_industrial_phantom,
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
)

from .histogram import (
    HistogramResult,
    GMMFitResult,
    ArtifactSignatures,
    compute_bimodal_histogram,
    compute_ground_truth_histogram,
    fit_gmm,
    auto_fit_gmm,
    segment_by_gmm,
    segment_by_polygon,
    detect_artifact_signatures,
    plot_bimodal_histogram,
    plot_ground_truth_comparison,
    plot_comparison_grid,
)

from .simulation import (
    SimulationResult,
    DualModalitySimulation,
    run_artifact_survey,
)

__version__  = "1.0.0"
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
    "reconstruct", "reconstruct_pair",
    # Histogram
    "HistogramResult", "GMMFitResult", "ArtifactSignatures",
    "compute_bimodal_histogram", "compute_ground_truth_histogram",
    "fit_gmm", "auto_fit_gmm",
    "segment_by_gmm", "segment_by_polygon",
    "detect_artifact_signatures",
    "plot_bimodal_histogram", "plot_ground_truth_comparison", "plot_comparison_grid",
    # Simulation
    "SimulationResult", "DualModalitySimulation", "run_artifact_survey",
]
