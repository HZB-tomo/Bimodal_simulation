"""
Analysis: bimodal histograms and the metrics computed on them.

* :mod:`.histogram`          — ``HistogramResult`` and histogram computation
* :mod:`.gmm`                — Gaussian-mixture fitting and segmentation
* :mod:`.signatures`         — ground-truth-free artifact-signature scores
* :mod:`.quality`            — ground-truth-anchored cluster-quality metrics
* :mod:`.metrics_table`      — full shape + quality metric table (v2)
* :mod:`.metrics_morphology` — label-anchored / per-region metrics (v3)
* :mod:`.cross_algorithm`    — per-modality algorithm comparisons
* :mod:`.fusion_metrics`     — particle segmentation metrics (needs pandas)
"""

from .cross_algorithm import compute_cross_algorithm_histograms, make_cross_algorithm_sinos
from .gmm import (
    GMMFitResult,
    auto_fit_gmm,
    fit_gmm,
    predict_gmm,
    segment_by_gmm,
    segment_by_polygon,
)
from .histogram import (
    DEFAULT_GT_ENERGY_IDX,
    HistogramResult,
    compute_bimodal_histogram,
    compute_ground_truth_histogram,
)
from .metrics_morphology import RegionSummary, compute_histogram_metrics_morphology_aware
from .metrics_table import HistogramMetricsTable, compute_histogram_metrics
from .quality import (
    ClusterQualityMetrics,
    compare_algorithms,
    davies_bouldin,
    evaluate_histogram_quality,
    ground_truth_positions,
    match_components,
    pairwise_overlap,
)
from .signatures import ArtifactSignatures, detect_artifact_signatures

__all__ = [
    "compute_cross_algorithm_histograms", "make_cross_algorithm_sinos",
    "GMMFitResult", "auto_fit_gmm", "fit_gmm", "predict_gmm", "segment_by_gmm",
    "segment_by_polygon",
    "DEFAULT_GT_ENERGY_IDX", "HistogramResult", "compute_bimodal_histogram",
    "compute_ground_truth_histogram",
    "RegionSummary", "compute_histogram_metrics_morphology_aware",
    "HistogramMetricsTable", "compute_histogram_metrics",
    "ClusterQualityMetrics", "compare_algorithms", "davies_bouldin",
    "evaluate_histogram_quality", "ground_truth_positions", "match_components",
    "pairwise_overlap",
    "ArtifactSignatures", "detect_artifact_signatures",
]
