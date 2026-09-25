"""
Plotting.

* :mod:`.histograms`  — bimodal-histogram figures used throughout the package
* :mod:`.publication` — the paper's figure set (``classic`` serif style, saves
  PDFs; formerly ``diana_plots``)
"""

from .histograms import (
    plot_artifact_survey,
    plot_bimodal_histogram,
    plot_comparison_grid,
    plot_cross_algorithm_grid,
    plot_ground_truth_comparison,
)

__all__ = [
    "plot_artifact_survey", "plot_bimodal_histogram", "plot_comparison_grid",
    "plot_cross_algorithm_grid", "plot_ground_truth_comparison",
]
