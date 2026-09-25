"""
neutron_xray_sim.analysis.cross_algorithm
─────────────────────────────────────────
Compare reconstruction algorithms per modality: reconstruct the X-ray sinogram
with one algorithm and the neutron sinogram with another, then build the
bimodal histogram of every requested (alg_x, alg_n) pair.

The figure is drawn by :func:`neutron_xray_sim.plotting.plot_cross_algorithm_grid`.
"""

from __future__ import annotations

import warnings
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

from .histogram import DEFAULT_GT_ENERGY_IDX, HistogramResult, compute_bimodal_histogram
from .quality import ClusterQualityMetrics, evaluate_histogram_quality

__all__ = [
    "make_cross_algorithm_sinos",
    "compute_cross_algorithm_histograms",
]

AlgPair = Tuple[str, str]


def make_cross_algorithm_sinos(
    phantom,
    algorithms: Iterable[str],
    n_angles: int = 120,
    angle_range_deg: float = 180.0,
    kVp: float = 120.0,
    filter_mm_Al: float = 2.0,
    filter_mm_Cu: float = 0.0,
    n_spectrum_bins: int = 12,
    I0: float = 1e5,
    cfg=None,
    use_astra: bool = True,
):
    """
    Project *phantom* once and map every algorithm name to that sinogram.

    Reconstruction is the only thing that differs between grid panels, so a
    single forward projection is shared by all of them.

    Parameters
    ----------
    phantom    : PhantomData
    algorithms : all algorithm names that appear in any pair
    cfg        : optional ArtifactConfig; sinogram-domain artifacts are
                 injected (seed 0) before returning
    other args : forwarded to ``make_sinogram_pair``

    Returns
    -------
    (xray_sinos, neutron_sinos) : dicts {algorithm → sinogram dict}; all
    values are the *same* dict object.
    """
    from ..acquisition.artifacts import inject_sinogram_artifacts
    from ..acquisition.projector import make_sinogram_pair

    xray_sino, neutron_sino = make_sinogram_pair(
        phantom,
        n_angles=n_angles,
        angle_range_deg=angle_range_deg,
        kVp=kVp,
        filter_mm_Al=filter_mm_Al,
        filter_mm_Cu=filter_mm_Cu,
        n_spectrum_bins=n_spectrum_bins,
        I0_xray=I0,
        I0_neutron=I0,
        use_astra=use_astra,
    )
    if cfg is not None:
        xray_sino, neutron_sino = inject_sinogram_artifacts(
            xray_sino, neutron_sino, cfg, rng=np.random.default_rng(seed=0))

    unique = list(dict.fromkeys(algorithms))
    return ({alg: xray_sino for alg in unique},
            {alg: neutron_sino for alg in unique})


def compute_cross_algorithm_histograms(
    phantom,
    xray_sinos: Dict[str, dict],
    neutron_sinos: Dict[str, dict],
    algorithm_pairs: Iterable[AlgPair],
    bins: int = 128,
    compute_metrics: bool = True,
    energy_idx: int = DEFAULT_GT_ENERGY_IDX,
    use_astra: bool = True,
    recon_kwargs: Optional[dict] = None,
) -> Tuple[List[AlgPair], Dict[AlgPair, HistogramResult],
           Dict[AlgPair, Optional[ClusterQualityMetrics]]]:
    """
    Reconstruct and histogram every (alg_x, alg_n) pair.

    Each (modality, algorithm) volume is reconstructed once and reused.

    Returns
    -------
    pairs      : de-duplicated pairs, in input order
    histograms : {pair → HistogramResult}
    metrics    : {pair → ClusterQualityMetrics or None} (empty if disabled)
    """
    from ..reconstruction.reconstructor import reconstruct

    pairs = list(dict.fromkeys(algorithm_pairs))
    if not pairs:
        raise ValueError("algorithm_pairs is empty.")
    for alg_x, alg_n in pairs:
        if alg_x not in xray_sinos:
            raise KeyError(f"alg_x='{alg_x}' not in xray_sinos: {list(xray_sinos)}")
        if alg_n not in neutron_sinos:
            raise KeyError(f"alg_n='{alg_n}' not in neutron_sinos: {list(neutron_sinos)}")

    kw = dict(filter_name="shepp-logan", remove_rings=True, clip_negative=True,
              use_astra=use_astra)
    kw.update(recon_kwargs or {})
    volumes: Dict[Tuple[str, str], np.ndarray] = {}

    def volume(modality: str, alg: str, sino: dict) -> np.ndarray:
        if (modality, alg) not in volumes:
            volumes[(modality, alg)] = reconstruct(sino, algorithm=alg, **kw)
        return volumes[(modality, alg)]

    histograms = {
        (ax, an): compute_bimodal_histogram(
            volume("x", ax, xray_sinos[ax]), volume("n", an, neutron_sinos[an]),
            bins=bins)
        for ax, an in pairs
    }

    metrics: Dict[AlgPair, Optional[ClusterQualityMetrics]] = {}
    if compute_metrics:
        for pair, hist in histograms.items():
            try:
                metrics[pair] = evaluate_histogram_quality(
                    hist, phantom, n_components=len(phantom.materials),
                    energy_idx=energy_idx, exclude_air=True)
            except Exception as exc:  # noqa: BLE001 - keep the other panels
                warnings.warn(f"Metrics failed for {pair}: {exc}", stacklevel=2)
                metrics[pair] = None
    return pairs, histograms, metrics
