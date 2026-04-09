"""
neutron_xray_sim/simulation.py
────────────────────────────────
Top-level orchestrator for dual-modality neutron / X-ray tomography simulation.

Usage
-----
::

    from neutron_xray_sim import DualModalitySimulation, ArtifactConfig

    sim = DualModalitySimulation(
        preset   = "composite",   # phantom preset
        N        = 64,            # voxel grid size
        n_angles = 180,
    )

    # --- Clean reference run ---
    result_clean = sim.run(ArtifactConfig.clean(), tag="clean")

    # --- With all realistic artifacts ---
    result_real  = sim.run(ArtifactConfig.realistic(), tag="realistic")

    # --- Custom: noise + misalignment only ---
    cfg = ArtifactConfig(
        photon_noise  = True,  I0_xray=2e4, I0_neutron=2e4,
        misalignment  = True,  translation_voxels=(4, 0, 0),
    )
    result_custom = sim.run(cfg, tag="noise+misalign")

    # --- Comparison figure ---
    from neutron_xray_sim.histogram import plot_comparison_grid
    fig = sim.comparison_grid([result_clean, result_real, result_custom])
    fig.savefig("comparison.png", dpi=150, bbox_inches="tight")
"""

from __future__ import annotations

import time
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from .phantom      import PhantomData, make_phantom
from .projector    import make_sinogram_pair
from .artifacts    import ArtifactConfig, inject_sinogram_artifacts, inject_volume_artifacts
from .reconstructor import reconstruct_pair
from .histogram    import (
    HistogramResult,
    GMMFitResult,
    compute_bimodal_histogram,
    fit_gmm,
    auto_fit_gmm,
    plot_bimodal_histogram,
    plot_comparison_grid,
    detect_artifact_signatures,
    ArtifactSignatures,
)

__all__ = ["SimulationResult", "DualModalitySimulation"]


# ──────────────────────────────────────────────────────────────────────────────
# Result container
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class SimulationResult:
    """
    Complete output of one simulation run.

    Attributes
    ----------
    tag              : run label (e.g. 'clean', 'noise_only', …)
    cfg              : ArtifactConfig used
    phantom          : PhantomData (ground-truth)
    vol_xray         : (N, N, N) reconstructed X-ray volume  [cm⁻¹]
    vol_neutron      : (N, N, N) reconstructed neutron volume [cm⁻¹]
    xray_sino        : sinogram dict (after artifact injection)
    neutron_sino     : sinogram dict (after artifact injection)
    histogram        : bimodal HistogramResult
    gmm              : optional GMMFitResult from auto-fitting
    signatures       : optional ArtifactSignatures metrics
    elapsed_s        : wall-clock runtime in seconds
    """
    tag:          str
    cfg:          ArtifactConfig
    phantom:      PhantomData
    vol_xray:     np.ndarray
    vol_neutron:  np.ndarray
    xray_sino:    dict
    neutron_sino: dict
    histogram:    HistogramResult
    gmm:          Optional[GMMFitResult] = None
    signatures:   Optional[ArtifactSignatures] = None
    elapsed_s:    float = 0.0

    def summary(self) -> str:
        lines = [
            f"═══ SimulationResult: '{self.tag}' ═══",
            f"  Phantom   : {self.phantom.name}  ({self.phantom.N}³ voxels)",
            f"  Artifacts : {self.cfg.summary()}",
            f"  Vol shape : {self.vol_xray.shape}",
            f"  μ_x range : [{self.vol_xray.min():.3f}, {self.vol_xray.max():.3f}] cm⁻¹",
            f"  μ_n range : [{self.vol_neutron.min():.3f}, {self.vol_neutron.max():.3f}] cm⁻¹",
            f"  Runtime   : {self.elapsed_s:.1f} s",
        ]
        if self.gmm is not None:
            lines.append(f"  GMM       : {self.gmm.n_components} components  "
                         f"BIC={self.gmm.bic:.0f}")
        if self.signatures is not None:
            s = self.signatures
            lines += [
                f"  Signatures:",
                f"    horiz_streak  = {s.horizontal_streak_score:.3f}",
                f"    vert_streak   = {s.vertical_streak_score:.3f}",
                f"    diag_smear    = {s.diagonal_smear_score:.3f}",
                f"    x_asymmetry   = {s.marginal_asymmetry_x:.4f}",
                f"    n_shift       = {s.marginal_shift_n:.4f} cm⁻¹",
            ]
        return "\n".join(lines)

    def plot_histogram(self, **kwargs) -> plt.Figure:
        """Convenience: plot the bimodal histogram for this result."""
        return plot_bimodal_histogram(
            self.histogram,
            gmm=self.gmm,
            title=f"Bimodal Histogram — {self.tag}",
            **kwargs,
        )

    def plot_slices(self, slice_idx: Optional[int] = None) -> plt.Figure:
        """
        Plot central orthogonal slices of both reconstructed volumes.
        """
        N  = self.vol_xray.shape[0]
        si = slice_idx if slice_idx is not None else N // 2

        fig, axes = plt.subplots(2, 3, figsize=(12, 8))
        titles_x = ["X-ray XY", "X-ray XZ", "X-ray YZ"]
        titles_n = ["Neutron XY", "Neutron XZ", "Neutron YZ"]

        for col, (ax_x, ax_n, ttx, ttn, slc) in enumerate(zip(
            axes[0], axes[1], titles_x, titles_n,
            [
                (self.vol_xray[si], self.vol_neutron[si]),
                (self.vol_xray[:, si, :], self.vol_neutron[:, si, :]),
                (self.vol_xray[:, :, si], self.vol_neutron[:, :, si]),
            ]
        )):
            vx_sl, vn_sl = slc
            vmin_x, vmax_x = np.percentile(self.vol_xray, [1, 99])
            vmin_n, vmax_n = np.percentile(self.vol_neutron, [1, 99])

            ax_x.imshow(vx_sl, cmap="gray", vmin=vmin_x, vmax=vmax_x)
            ax_x.set_title(ttx, fontsize=10)
            ax_x.axis("off")

            ax_n.imshow(vn_sl, cmap="gray", vmin=vmin_n, vmax=vmax_n)
            ax_n.set_title(ttn, fontsize=10)
            ax_n.axis("off")

        fig.suptitle(f"Reconstructed volumes — '{self.tag}'", fontsize=12)
        fig.tight_layout()
        return fig


# ──────────────────────────────────────────────────────────────────────────────
# Main simulation class
# ──────────────────────────────────────────────────────────────────────────────

class DualModalitySimulation:
    """
    End-to-end dual-modality CT simulation.

    Manages the phantom, projection, artifact injection, reconstruction,
    and bimodal histogram analysis.  Results from multiple runs (different
    artifact configurations) are cached and can be compared visually.

    Parameters
    ----------
    preset          : phantom preset name  ('composite', 'battery',
                      'bone_implant', 'industrial')
    N               : voxel grid size (N × N × N)
    n_angles        : number of projection angles
    angle_range_deg : angular range [°] (180 = half-scan)
    kVp             : X-ray tube voltage [kV]
    filter_mm_Al    : Al pre-filter thickness [mm]
    filter_mm_Cu    : Cu pre-filter thickness [mm]
    n_spectrum_bins : polychromatic energy bins
    algorithm       : CT reconstruction algorithm ('FBP', 'SIRT', 'CGLS')
    filter_name     : FBP ramp filter ('ram-lak', 'shepp-logan', …)
    n_iter          : iterations for SIRT / CGLS
    histogram_bins  : bins per axis in the 2-D histogram
    auto_gmm        : auto-fit GMM after each run
    max_gmm_k       : max components to try in BIC selection
    use_astra       : use ASTRA GPU if available
    verbose         : print progress messages
    phantom         : supply a custom PhantomData (overrides preset)
    """

    def __init__(
        self,
        preset: str = "composite",
        N: int = 64,
        n_angles: int = 180,
        angle_range_deg: float = 180.0,
        kVp: float = 120.0,
        filter_mm_Al: float = 2.0,
        filter_mm_Cu: float = 0.0,
        n_spectrum_bins: int = 12,
        algorithm: str = "FBP",
        filter_name: str = "ram-lak",
        n_iter: int = 50,
        histogram_bins: int = 200,
        auto_gmm: bool = False,
        max_gmm_k: int = 7,
        use_astra: bool = True,
        verbose: bool = True,
        phantom: Optional[PhantomData] = None,
    ):
        self.preset          = preset
        self.N               = N
        self.n_angles        = n_angles
        self.angle_range_deg = angle_range_deg
        self.kVp             = kVp
        self.filter_mm_Al    = filter_mm_Al
        self.filter_mm_Cu    = filter_mm_Cu
        self.n_spectrum_bins = n_spectrum_bins
        self.algorithm       = algorithm
        self.filter_name     = filter_name
        self.n_iter          = n_iter
        self.histogram_bins  = histogram_bins
        self.auto_gmm        = auto_gmm
        self.max_gmm_k       = max_gmm_k
        self.use_astra       = use_astra
        self.verbose         = verbose

        # Load or use supplied phantom
        if phantom is not None:
            self.phantom = phantom
        else:
            if verbose:
                print(f"[sim] Loading phantom '{preset}' at N={N} …")
            self.phantom = make_phantom(preset, N)
            if verbose:
                print(f"[sim] {self.phantom}")

        # Cache for raw (clean) sinograms — computed once, reused for all runs
        self._raw_xray_sino    = None
        self._raw_neutron_sino = None

        # Results store
        self.results: Dict[str, SimulationResult] = {}

    # ──────────────────────────────────────────────────────────────────────────
    # Internal
    # ──────────────────────────────────────────────────────────────────────────

    def _ensure_sinograms(self, I0_xray: float = 1e5, I0_neutron: float = 1e5):
        """Project phantom if not already done (results cached)."""
        if self._raw_xray_sino is None:
            if self.verbose:
                print("[sim] Computing raw sinograms (cached for subsequent runs) …")
            self._raw_xray_sino, self._raw_neutron_sino = make_sinogram_pair(
                self.phantom,
                n_angles        = self.n_angles,
                angle_range_deg = self.angle_range_deg,
                kVp             = self.kVp,
                filter_mm_Al    = self.filter_mm_Al,
                filter_mm_Cu    = self.filter_mm_Cu,
                n_spectrum_bins = self.n_spectrum_bins,
                I0_xray         = I0_xray,
                I0_neutron      = I0_neutron,
                use_astra       = self.use_astra,
            )

    # ──────────────────────────────────────────────────────────────────────────
    # Run
    # ──────────────────────────────────────────────────────────────────────────

    def run(
        self,
        cfg: ArtifactConfig = ArtifactConfig.clean(),
        tag: Optional[str] = None,
        rng_seed: int = 0,
        ref_result: Optional[SimulationResult] = None,
        n_gmm_components: Optional[int] = None,
    ) -> SimulationResult:
        """
        Execute one full simulation run with the given artifact configuration.

        Parameters
        ----------
        cfg              : ArtifactConfig describing which artifacts to inject
        tag              : label for this run (default: cfg.summary())
        rng_seed         : random seed for reproducibility
        ref_result       : reference SimulationResult for shift comparison in
                           artifact-signature analysis
        n_gmm_components : if set, fit a GMM with exactly this many components;
                           if None and auto_gmm=True, uses BIC selection

        Returns
        -------
        SimulationResult
        """
        if tag is None:
            tag = cfg.summary()[:60]

        t_start = time.perf_counter()
        rng     = np.random.default_rng(rng_seed)

        if self.verbose:
            print(f"\n{'─'*60}")
            print(f"[sim] Run: '{tag}'")
            print(f"[sim] Artifacts: {cfg.summary()}")

        # ── 1. Project (use cached raw sinograms) ─────────────────────────────
        self._ensure_sinograms(cfg.I0_xray, cfg.I0_neutron)

        xray_sino    = {k: v.copy() if isinstance(v, np.ndarray) else v
                        for k, v in self._raw_xray_sino.items()}
        neutron_sino = {k: v.copy() if isinstance(v, np.ndarray) else v
                        for k, v in self._raw_neutron_sino.items()}

        # ── 2. Sinogram-domain artifacts ─────────────────────────────────────
        if self.verbose:
            print("[sim] Injecting sinogram artifacts …")
        xray_sino, neutron_sino = inject_sinogram_artifacts(
            xray_sino, neutron_sino, cfg, rng=rng
        )

        # ── 3. Reconstruct ────────────────────────────────────────────────────
        if self.verbose:
            print("[sim] Reconstructing …")
        vol_x, vol_n = reconstruct_pair(
            xray_sino, neutron_sino,
            algorithm    = self.algorithm,
            filter_name  = self.filter_name,
            n_iter       = self.n_iter,
            use_astra    = self.use_astra,
            remove_rings = True,
        )

        # ── 4. Volume-domain artifacts ────────────────────────────────────────
        if self.verbose:
            print("[sim] Injecting volume artifacts …")
        vol_x, vol_n = inject_volume_artifacts(vol_x, vol_n, cfg, rng=rng)

        # ── 5. Bimodal histogram ──────────────────────────────────────────────
        if self.verbose:
            print("[sim] Computing bimodal histogram …")
        hist = compute_bimodal_histogram(
            vol_x, vol_n,
            bins=self.histogram_bins,
        )

        # ── 6. GMM fitting ────────────────────────────────────────────────────
        gmm = None
        if self.auto_gmm or n_gmm_components is not None:
            if self.verbose:
                print("[sim] Fitting GMM …")
            if n_gmm_components is not None:
                gmm = fit_gmm(hist, n_components=n_gmm_components)
            else:
                n_mat = len(self.phantom.materials)
                gmm   = auto_fit_gmm(hist, min_k=max(2, n_mat-1),
                                     max_k=min(self.max_gmm_k, n_mat+2))

        # ── 7. Artifact signatures ────────────────────────────────────────────
        ref_hist = ref_result.histogram if ref_result is not None else None
        sigs     = detect_artifact_signatures(hist, gmm=gmm, ref_hist=ref_hist)

        elapsed  = time.perf_counter() - t_start

        result = SimulationResult(
            tag          = tag,
            cfg          = cfg,
            phantom      = self.phantom,
            vol_xray     = vol_x,
            vol_neutron  = vol_n,
            xray_sino    = xray_sino,
            neutron_sino = neutron_sino,
            histogram    = hist,
            gmm          = gmm,
            signatures   = sigs,
            elapsed_s    = elapsed,
        )

        self.results[tag] = result

        if self.verbose:
            print(result.summary())

        return result

    # ──────────────────────────────────────────────────────────────────────────
    # Batch: run multiple configs
    # ──────────────────────────────────────────────────────────────────────────

    def run_batch(
        self,
        configs: List[Tuple[str, ArtifactConfig]],
        ref_tag: Optional[str] = None,
    ) -> Dict[str, SimulationResult]:
        """
        Run a list of (tag, ArtifactConfig) pairs in sequence.

        Parameters
        ----------
        configs  : list of (tag, ArtifactConfig) pairs
        ref_tag  : tag of the reference run for artifact-signature comparison

        Returns
        -------
        dict  {tag → SimulationResult}
        """
        ref_result = self.results.get(ref_tag) if ref_tag else None
        for tag, cfg in configs:
            result = self.run(cfg, tag=tag, ref_result=ref_result)
            if ref_tag is None and ref_result is None:
                ref_result = result     # first run becomes the reference
        return self.results

    # ──────────────────────────────────────────────────────────────────────────
    # Convenience comparison plots
    # ──────────────────────────────────────────────────────────────────────────

    def comparison_grid(
        self,
        results: Optional[List[SimulationResult]] = None,
        ncols: int = 3,
        **kwargs,
    ) -> plt.Figure:
        """
        Plot a grid of bimodal histograms for easy comparison.

        If results is None, all stored results are plotted.
        """
        if results is None:
            results = list(self.results.values())
        pairs = [(r.tag, r.histogram) for r in results]
        gmms  = [r.gmm for r in results]
        return plot_comparison_grid(pairs, gmm_results=gmms, ncols=ncols, **kwargs)

    def comparison_slices(
        self,
        results: Optional[List[SimulationResult]] = None,
        slice_idx: Optional[int] = None,
        figsize_per_col: Tuple[float, float] = (4.0, 4.0),
    ) -> plt.Figure:
        """
        Side-by-side central slice comparison for multiple runs.
        Rows: [X-ray, Neutron].  Columns: one per result.
        """
        if results is None:
            results = list(self.results.values())

        n    = len(results)
        N    = self.N
        si   = slice_idx if slice_idx is not None else N // 2
        fw   = figsize_per_col[0] * n
        fh   = figsize_per_col[1] * 2

        fig, axes = plt.subplots(2, n, figsize=(fw, fh))
        if n == 1:
            axes = axes[:, np.newaxis]

        for col, r in enumerate(results):
            vmin_x = np.percentile(r.vol_xray, 1)
            vmax_x = np.percentile(r.vol_xray, 99)
            vmin_n = np.percentile(r.vol_neutron, 1)
            vmax_n = np.percentile(r.vol_neutron, 99)

            axes[0, col].imshow(r.vol_xray[si], cmap="gray",
                                vmin=vmin_x, vmax=vmax_x)
            axes[0, col].set_title(f"X-ray\n{r.tag}", fontsize=8)
            axes[0, col].axis("off")

            axes[1, col].imshow(r.vol_neutron[si], cmap="gray",
                                vmin=vmin_n, vmax=vmax_n)
            axes[1, col].set_title(f"Neutron\n{r.tag}", fontsize=8)
            axes[1, col].axis("off")

        fig.tight_layout()
        return fig

    def signature_table(
        self,
        results: Optional[List[SimulationResult]] = None,
    ) -> str:
        """Return an ASCII table of artifact-signature metrics for all runs."""
        if results is None:
            results = list(self.results.values())

        hdr = (f"{'Tag':<30} {'H-streak':>9} {'V-streak':>9} "
               f"{'DiagSmear':>10} {'X-asym':>8} {'N-shift':>9}")
        sep = "─" * len(hdr)
        lines = [sep, hdr, sep]

        for r in results:
            s = r.signatures
            if s is None:
                continue
            lines.append(
                f"{r.tag[:30]:<30} {s.horizontal_streak_score:9.3f} "
                f"{s.vertical_streak_score:9.3f} {s.diagonal_smear_score:10.3f} "
                f"{s.marginal_asymmetry_x:8.4f} {s.marginal_shift_n:9.4f}"
            )
        lines.append(sep)
        return "\n".join(lines)


# ──────────────────────────────────────────────────────────────────────────────
# Artifact survey: one alteration per run + combinations
# ──────────────────────────────────────────────────────────────────────────────

from .histogram import (
    compute_ground_truth_histogram,
    plot_ground_truth_comparison,
)


def run_artifact_survey(
    preset: str = "composite",
    N: int = 64,
    n_angles: int = 120,
    kVp: float = 120.0,
    filter_mm_Al: float = 2.0,
    histogram_bins: int = 128,
    algorithm: str = "FBP",
    use_astra: bool = True,
    verbose: bool = True,
    include_combinations: bool = True,
    figsize_per_panel: Tuple[float, float] = (5.5, 5.0),
    cmap: str = "inferno",
    phantom: Optional["PhantomData"] = None,
) -> Tuple[Dict[str, "SimulationResult"], plt.Figure]:
    """
    Run the full simulation pipeline (projection → reconstruction → 2-D histogram)
    once per artifact type, and once per selected combination, with a clean
    reference as the first run.  No segmentation or GMM fitting is performed.

    The function returns:
      - a dict mapping each run tag to its SimulationResult
      - a publication-quality comparison figure: ground-truth scatter on the
        first panel, then one histogram panel per run (shared axis limits)

    Single-artifact runs
    --------------------
    1.  Clean (reference)
    2.  Poisson noise – moderate  (I₀ = 5×10⁴)
    3.  Poisson noise – low dose  (I₀ = 5×10³)
    4.  Beam hardening (polychromatic, no BHC)
    5.  Beam hardening corrected (polynomial BHC applied)
    6.  Neutron scatter build-up  (f = 6 %)
    7.  X-ray scatter             (f = 4 %)
    8.  Detector PSF blur         (σ_x = 0.8 px, σ_n = 1.5 px)
    9.  Ring artifacts            (3 bad columns)
    10. Misalignment              (3 voxel translation, 1° rotation)

    Combination runs (when include_combinations=True)
    --------------------------------------------------
    11. Noise + misalignment
    12. Scatter (both) + PSF
    13. Noise + BH (no correction) + rings
    14. All realistic

    Parameters
    ----------
    preset              : phantom preset name
    N                   : voxel grid size
    n_angles            : number of projection angles
    kVp                 : X-ray tube voltage [kV]
    filter_mm_Al        : Al pre-filter thickness [mm]
    histogram_bins      : bins per axis in 2-D histograms
    algorithm           : CT reconstruction algorithm
    use_astra           : use ASTRA GPU if available
    verbose             : print progress
    include_combinations: also run the combination configurations
    figsize_per_panel   : (width, height) of each histogram panel in the grid
    cmap                : colourmap for 2-D panels
    phantom             : optional custom PhantomData (overrides preset)

    Returns
    -------
    results : dict {tag -> SimulationResult}
    fig     : matplotlib Figure with all histograms in a grid
    """

    # ── Define all single-artifact configurations ─────────────────────────────
    single_configs: List[Tuple[str, "ArtifactConfig"]] = [
        (
            "Clean (reference)",
            ArtifactConfig.clean(),
        ),
        (
            "Noise moderate\n(I₀=5×10⁴)",
            ArtifactConfig(photon_noise=True, I0_xray=5e4, I0_neutron=5e4),
        ),
        (
            "Noise low dose\n(I₀=5×10³)",
            ArtifactConfig(photon_noise=True, I0_xray=5e3, I0_neutron=5e3),
        ),
        (
            "Beam hardening\n(no BHC)",
            ArtifactConfig(apply_bh_correction=False),
        ),
        (
            "Beam hardening\n(BHC corrected)",
            ArtifactConfig(apply_bh_correction=True, bh_correction_order=3),
        ),
        (
            "Neutron scatter\n(f=6%)",
            ArtifactConfig(neutron_scatter=True, scatter_fraction=0.06,
                           scatter_sigma_pixels=9.0),
        ),
        (
            "X-ray scatter\n(f=4%)",
            ArtifactConfig(xray_scatter=True, xray_scatter_fraction=0.04),
        ),
        (
            "Detector PSF\n(σ_x=0.8, σ_n=1.5 px)",
            ArtifactConfig(detector_psf=True,
                           psf_sigma_xray_pixels=0.8,
                           psf_sigma_neutron_pixels=1.5),
        ),
        (
            "Ring artifacts\n(3 bad cols)",
            ArtifactConfig(ring_artifacts=True, n_bad_columns=3,
                           ring_amplitude=0.05),
        ),
        (
            "Misalignment\n(3 vx, 1°)",
            ArtifactConfig(misalignment=True,
                           translation_voxels=(3.0, 0.0, 0.0),
                           rotation_deg=(0.0, 1.0, 0.0)),
        ),
    ]

    # ── Combination configurations ─────────────────────────────────────────────
    combo_configs: List[Tuple[str, "ArtifactConfig"]] = [
        (
            "Noise + misalign",
            ArtifactConfig(
                photon_noise=True,    I0_xray=5e4,   I0_neutron=5e4,
                misalignment=True,    translation_voxels=(3.0, 0.0, 0.0),
            ),
        ),
        (
            "Scatter (n+x) + PSF",
            ArtifactConfig(
                neutron_scatter=True, scatter_fraction=0.06,
                xray_scatter=True,    xray_scatter_fraction=0.04,
                detector_psf=True,    psf_sigma_xray_pixels=0.8,
                                      psf_sigma_neutron_pixels=1.5,
            ),
        ),
        (
            "Noise + BH + rings",
            ArtifactConfig(
                photon_noise=True,    I0_xray=3e4,   I0_neutron=3e4,
                apply_bh_correction=False,
                ring_artifacts=True,  n_bad_columns=3, ring_amplitude=0.05,
            ),
        ),
        (
            "All realistic",
            ArtifactConfig.realistic(),
        ),
    ]

    all_configs = single_configs + (combo_configs if include_combinations else [])
    n_runs = len(all_configs)

    # ── Build simulation instance and run ─────────────────────────────────────
    sim = DualModalitySimulation(
        preset          = preset,
        N               = N,
        n_angles        = n_angles,
        kVp             = kVp,
        filter_mm_Al    = filter_mm_Al,
        algorithm       = algorithm,
        histogram_bins  = histogram_bins,
        auto_gmm        = False,
        use_astra       = use_astra,
        verbose         = verbose,
        phantom         = phantom,
    )

    if verbose:
        print(f"\n{'═'*60}")
        print(f"  Artifact survey: {n_runs} runs on '{preset}' phantom (N={N})")
        print(f"{'═'*60}")

    results: Dict[str, "SimulationResult"] = {}
    ref_result = None

    for tag, cfg in all_configs:
        r = sim.run(cfg, tag=tag, ref_result=ref_result)
        results[tag] = r
        if ref_result is None:
            ref_result = r   # first (clean) run is the reference

    # ── Compute ground-truth histogram ────────────────────────────────────────
    hist_gt = compute_ground_truth_histogram(sim.phantom, bins=histogram_bins)

    # ── Figure layout ─────────────────────────────────────────────────────────
    # Row 0: [GT scatter] [clean] [noise moderate] [noise low]
    # Row 1: [BH no-BHC]  [BH corrected] [n scatter] [x scatter]
    # Row 2: [PSF]  [rings] [misalign] [noise+misalign]   (if combinations)
    # Row 3: [scatter+PSF] [noise+BH+rings] [all realistic] (if combinations)
    # First panel is the GT scatter plot; the remaining panels are histograms.

    # Shared axis limits across ALL runs
    x_max_recon = max(r.histogram.x_edges[-1] for r in results.values())
    n_max_recon = max(r.histogram.n_edges[-1] for r in results.values())
    mu_x_gt = np.array([m._mu_x_table[6] for m in sim.phantom.materials])
    mu_n_gt = np.array([m.mu_n           for m in sim.phantom.materials])
    x_max = max(x_max_recon, float(mu_x_gt.max()) * 1.08)
    n_max = max(n_max_recon, float(mu_n_gt.max()) * 1.08)
    shared_extent = [0.0, x_max, 0.0, n_max]

    # Grid dimensions: GT panel + one per run
    n_panels = 1 + n_runs
    ncols    = 4
    nrows    = int(np.ceil(n_panels / ncols))
    fw       = figsize_per_panel[0] * ncols
    fh       = figsize_per_panel[1] * nrows

    fig = plt.figure(figsize=(fw, fh), constrained_layout=True)
    fig.suptitle(
        f"Artifact survey — '{preset}' phantom  (N={N}, {n_angles} angles)\n"
        f"White ◆ markers show ground-truth material positions",
        fontsize=11,
    )

    # Create a grid of subfigures (one per panel)
    subfigs_flat = fig.subfigures(nrows, ncols, wspace=0.04, hspace=0.06).ravel()

    # Panel 0: ground-truth scatter
    _draw_gt_scatter_panel(
        subfigs_flat[0], sim.phantom,
        title=f"Ground Truth\n(exact positions)",
        shared_extent=shared_extent,
    )

    # Panels 1..n_runs: one histogram per run
    result_list = list(results.values())
    for panel_idx, r in enumerate(result_list, start=1):
        _draw_survey_histogram_panel(
            subfigs_flat[panel_idx], r.histogram, r.tag,
            log_scale=True, cmap=cmap,
            shared_extent=shared_extent,
            gt_mu_x=mu_x_gt, gt_mu_n=mu_n_gt,
            materials=sim.phantom.materials,
        )

    # Hide any unused subfigure slots
    for idx in range(n_panels, len(subfigs_flat)):
        subfigs_flat[idx].set_visible(False)

    return results, fig


# ── Panel drawing helpers ──────────────────────────────────────────────────────

def _draw_gt_scatter_panel(
    subfig: plt.Figure,
    phantom: "PhantomData",
    title: str,
    shared_extent: List[float],
    energy_idx: int = 6,
) -> None:
    """Ground-truth bubble scatter panel (dark background)."""
    ax = subfig.add_subplot(1, 1, 1)
    ax.set_facecolor("#0d0d0d")
    ax.grid(True, color="#2a2a2a", linewidth=0.5, zorder=0)

    materials  = phantom.materials
    n_mat      = len(materials)
    mu_x_vals  = np.array([m._mu_x_table[energy_idx] for m in materials])
    mu_n_vals  = np.array([m.mu_n for m in materials])
    vox_counts = np.array([(phantom.label_vol == i).sum()
                            for i in range(n_mat)], dtype=float)

    sqrt_c  = np.sqrt(vox_counts)
    s_range = sqrt_c.max() - sqrt_c.min() + 1e-9
    sizes   = 55 + 500 * (sqrt_c - sqrt_c.min()) / s_range
    colours = plt.cm.Set1(np.linspace(0, 0.9, n_mat))

    x_max = shared_extent[1]
    n_max = shared_extent[3]

    for i, (m, mx, mn, sz, col) in enumerate(
            zip(materials, mu_x_vals, mu_n_vals, sizes, colours)):
        # Cross-hairs
        ax.axvline(mx, color=col, linewidth=0.5, alpha=0.3, zorder=1)
        ax.axhline(mn, color=col, linewidth=0.5, alpha=0.3, zorder=1)
        # Bubble
        ax.scatter(mx, mn, s=sz, color=col, edgecolors="white",
                   linewidths=0.7, zorder=3, alpha=0.92)
        # Label
        dx = 0.04 * x_max * (1 if mx <= x_max * 0.55 else -1)
        dy = 0.04 * n_max * (1 if mn <= n_max * 0.55 else -1)
        ax.annotate(
            f"{m.symbol}\n({mx:.3f}, {mn:.3f})",
            xy=(mx, mn), xytext=(mx + dx, mn + dy),
            fontsize=7, color="white", fontweight="bold",
            ha="left" if dx > 0 else "right",
            va="bottom" if dy > 0 else "top",
            arrowprops=dict(arrowstyle="-", color=col, lw=0.7),
            zorder=4,
        )

    ax.set_xlim(shared_extent[0], shared_extent[1])
    ax.set_ylim(shared_extent[2], shared_extent[3])
    ax.set_xlabel(r"$\mu_x$ [cm$^{-1}$]", fontsize=9)
    ax.set_ylabel(r"$\mu_n$ [cm$^{-1}$]", fontsize=9)
    ax.set_title(title, fontsize=9, color="white", pad=4)
    ax.tick_params(colors="white", labelsize=7)
    subfig.set_facecolor("#0d0d0d")
    for spine in ax.spines.values():
        spine.set_edgecolor("#444444")


def _draw_survey_histogram_panel(
    subfig: plt.Figure,
    hist: "HistogramResult",
    title: str,
    log_scale: bool,
    cmap: str,
    shared_extent: Optional[List[float]],
    gt_mu_x: Optional[np.ndarray] = None,
    gt_mu_n: Optional[np.ndarray] = None,
    materials: Optional[list] = None,
) -> None:
    """Single histogram panel for the survey grid (no marginals, compact)."""
    gs      = subfig.add_gridspec(1, 2, width_ratios=[1, 0.06], wspace=0.03)
    ax_main = subfig.add_subplot(gs[0, 0])
    ax_cbar = subfig.add_subplot(gs[0, 1])

    extent = shared_extent if shared_extent is not None else hist.extent
    H      = hist.H.T
    H_plot = np.log1p(H) if log_scale else H
    H_plot = np.ma.masked_where(H == 0, H_plot)

    im = ax_main.imshow(
        H_plot, origin="lower", extent=extent,
        aspect="auto", cmap=cmap, interpolation="bilinear",
    )
    ax_main.set_xlim(extent[0], extent[1])
    ax_main.set_ylim(extent[2], extent[3])
    ax_main.set_xlabel(r"$\mu_x$ [cm$^{-1}$]", fontsize=8)
    ax_main.set_ylabel(r"$\mu_n$ [cm$^{-1}$]", fontsize=8)
    ax_main.set_title(title, fontsize=8, pad=3)
    ax_main.tick_params(labelsize=7)

    cbar = plt.colorbar(im, cax=ax_cbar)
    cbar.ax.tick_params(labelsize=6)

    # GT position markers (white diamonds + material symbol)
    if gt_mu_x is not None and gt_mu_n is not None:
        colours = plt.cm.Set1(np.linspace(0, 0.9, len(gt_mu_x)))
        for i, (mx, mn, col) in enumerate(zip(gt_mu_x, gt_mu_n, colours)):
            sym = materials[i].symbol if materials is not None else str(i)
            ax_main.plot(mx, mn, marker="D", color="white",
                         markersize=4, markeredgecolor=col,
                         markeredgewidth=1.2, zorder=5)
            ax_main.annotate(
                sym, xy=(mx, mn),
                xytext=(3, 3), textcoords="offset points",
                fontsize=6, color="white", fontweight="bold", zorder=6,
            )
