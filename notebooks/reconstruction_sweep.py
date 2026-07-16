"""
Reconstruction sweep: FBP, SART, SIRT, CGLS across multiple iteration counts.

For each (algorithm, n_iter) combination this:
  1. Reconstructs the X-ray / neutron volume pair from the (shared) sinogram.
  2. Saves the ground-truth-vs-reconstruction histogram figure.
  3. Saves the X-ray slice visualizations (phantom / XCT / difference).
  4. Saves the neutron slice visualizations (phantom / neutron / difference).

Everything is written to results/figure4/<algorithm>[_iter<N>]/.

Usage in a notebook:
    import reconstruction_sweep as rs
    results = rs.run_sweep()

Or from the command line:
    python reconstruction_sweep.py
"""

import sys
import pathlib
from pathlib import Path

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# Make the neutron_xray_sim package importable regardless of the current
# working directory. This script lives in Bimodal_simulation/notebooks/,
# and the package lives in Bimodal_simulation/neutron_xray_sim/, so we walk
# up one directory from this file's own location (not the cwd) to find it.
_THIS_DIR   = pathlib.Path(__file__).resolve().parent      # .../Bimodal_simulation/notebooks
_REPO_ROOT  = _THIS_DIR.parent                              # .../Bimodal_simulation
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import neutron_xray_sim as nxs
from neutron_xray_sim.artifacts import inject_sinogram_artifacts, inject_volume_artifacts

# ──────────────────────────────────────────────────────────────────────────
# Global settings — identical to the original notebook
# ──────────────────────────────────────────────────────────────────────────
PRESET    = 'battery'
N_ANGLES  = 1200          # projection angles (use 360-1200 for publication)
N         = 1024

KVP       = 100.0
AL_FILTER = 2.0

SLICE_NO    = 256
FILTER_NAME = 'shepp-logan'   # FBP / GRIDREC only
N_SUBSETS   = 10              # OSSART only
LAMBDA_TV   = 0.02            # TV_MIN only

BASE_SAVE_DIR = Path("results/figure4")
DPI      = 300
SAVE_PNG = True
SAVE_SVG = True

# ── Reconstruction sweep definition ───────────────────────────────────────
# FBP is a direct method -> run once, no iteration count.
# SART / SIRT / CGLS are iterative -> sweep over these iteration counts.
ALGORITHMS_NO_ITER   = ['FBP']
ALGORITHMS_WITH_ITER = ['SART', 'SIRT', 'CGLS']
ITERATIONS           = [100, 200, 50]

# Fixed color-scale settings for the slice visualizations (from your code)
XRAY_NORM    = dict(vmin=0, vcenter=2.5, vmax=5)
NEUTRON_NORM = dict(vmin=0, vcenter=1.25, vmax=2.5)


# ──────────────────────────────────────────────────────────────────────────
# Phantom / sinogram / artifact setup (run once, shared across all recons)
# ──────────────────────────────────────────────────────────────────────────
def build_phantom():
    return nxs.phantom.make_li_ion_battery_phantom(
        diameter_cm=0.5,
        length_cm=1.0,
        N=N,
    )


def build_sinograms(phantom):
    xray_sino, neut_sino = nxs.make_sinogram_pair(
        phantom,
        n_angles=N_ANGLES,
        angle_range_deg=360.0,
        kVp=KVP,
        filter_mm_Al=AL_FILTER,
        n_spectrum_bins=12,
        I0_xray=1e5,
        I0_neutron=1e5,
        use_astra=True,
    )
    print('X-ray sinogram :', xray_sino['sino_lam'].shape, '  (n_angles, N_slice, N_det)')
    print('Neutron sinogram:', neut_sino['sino_lam'].shape)
    return xray_sino, neut_sino


def apply_sinogram_artifacts(xray_sino, neut_sino, cfg):
    x_sino_art, n_sino_art = inject_sinogram_artifacts(
        {k: v.copy() if isinstance(v, np.ndarray) else v for k, v in xray_sino.items()},
        {k: v.copy() if isinstance(v, np.ndarray) else v for k, v in neut_sino.items()},
        cfg,
    )
    return x_sino_art, n_sino_art


# ──────────────────────────────────────────────────────────────────────────
# Reconstruction
# ──────────────────────────────────────────────────────────────────────────
def reconstruct(x_sino_art, n_sino_art, algorithm, n_iter, cfg):
    vol_x, vol_n = nxs.reconstruct_pair(
        x_sino_art, n_sino_art,
        algorithm=algorithm,
        filter_name=FILTER_NAME,
        n_iter=n_iter,
        n_subsets=N_SUBSETS,
        lambda_tv=LAMBDA_TV,
        remove_rings=True,
        use_astra=True,
        clip_negative=True,
    )
    vol_x, vol_n = inject_volume_artifacts(vol_x, vol_n, cfg)
    return vol_x, vol_n


# ──────────────────────────────────────────────────────────────────────────
# Saving helpers
# ──────────────────────────────────────────────────────────────────────────
def _save_fig(fig, out_dir, name):
    if SAVE_PNG:
        fig.savefig(out_dir / f"{name}.png", dpi=DPI, bbox_inches="tight")
    if SAVE_SVG:
        fig.savefig(out_dir / f"{name}.svg", bbox_inches="tight")


def save_histogram_figure(phantom, vol_x, vol_n, algorithm, out_dir):
    """Ground truth vs. reconstruction bimodal histogram (your existing code)."""
    hist = nxs.compute_bimodal_histogram(vol_x, vol_n, bins=128)
    print(f'Histogram shape: {hist.H.shape}  total voxels: {hist.total_voxels:,}')

    fig = nxs.plot_ground_truth_comparison(
        phantom, hist,
        title_gt='Ground Truth (exact positions)',
        title_recon=f'Reconstructed — {algorithm}',
        show_marginals=True,
    )
    _save_fig(fig, out_dir, "gt_vs_recon")
    plt.close(fig)
    return hist


def plot_normalized(data, out_dir, name, cmap='bwr', title='normalized'):
    """Diverging-colormap difference plot (your existing plot_normalized code)."""
    fig = plt.figure()
    plt.title(title)
    norm = mcolors.TwoSlopeNorm(vmin=data.min(), vcenter=0, vmax=data.max())
    plt.imshow(data, cmap=cmap, norm=norm)
    plt.colorbar()
    _save_fig(fig, out_dir, name)
    plt.close(fig)


def save_xray_visualizations(phantom, vol_x, out_dir):
    slice_x_ray = vol_x[SLICE_NO, :, :]
    slice_phantom = phantom.mu_x_vols[7, SLICE_NO, :, :]
    norm = mcolors.TwoSlopeNorm(**XRAY_NORM)

    fig = plt.figure()
    plt.title('Phantom')
    plt.imshow(slice_phantom, cmap='turbo', norm=norm)
    plt.colorbar()
    _save_fig(fig, out_dir, 'phantom')
    plt.close(fig)

    fig = plt.figure()
    plt.title('XCT')
    plt.imshow(slice_x_ray, cmap='turbo', norm=norm)
    plt.colorbar()
    _save_fig(fig, out_dir, 'XCT')
    plt.close(fig)

    plot_normalized(slice_x_ray - slice_phantom, out_dir,
                     'XCT_minus_Phantom', title='XCT - Phantom')


def save_neutron_visualizations(phantom, vol_n, out_dir):
    slice_n = vol_n[SLICE_NO, :, :]
    slice_phantom_n = phantom.mu_n_vol[SLICE_NO, :, :]
    norm = mcolors.TwoSlopeNorm(**NEUTRON_NORM)

    fig = plt.figure()
    plt.title('Phantom')
    plt.imshow(slice_phantom_n, cmap='turbo', norm=norm)
    plt.colorbar()
    _save_fig(fig, out_dir, 'phantom_N')
    plt.close(fig)

    fig = plt.figure()
    plt.title('Neutron')
    plt.imshow(slice_n, cmap='turbo', norm=norm)
    plt.colorbar()
    _save_fig(fig, out_dir, 'Neutron')
    plt.close(fig)

    plot_normalized(slice_n - slice_phantom_n, out_dir,
                     'Neutron_minus_Phantom', title='Neutron - Phantom')


# ──────────────────────────────────────────────────────────────────────────
# Sweep driver
# ──────────────────────────────────────────────────────────────────────────
def build_job_list():
    """[(algorithm, n_iter_or_None), ...] — None means 'no iteration count'."""
    jobs = [(alg, None) for alg in ALGORITHMS_NO_ITER]
    for alg in ALGORITHMS_WITH_ITER:
        for n_iter in ITERATIONS:
            jobs.append((alg, n_iter))
    return jobs


def run_sweep():
    matplotlib.rcParams.update({'figure.dpi': 110, 'axes.titlesize': 10})
    print(f'neutron_xray_sim v{nxs.__version__} loaded.')
    print('Available algorithms:', nxs.AVAILABLE_ALGORITHMS)
    print('Available phantoms:  ', list(nxs.PHANTOM_PRESETS.keys()))

    phantom = build_phantom()
    xray_sino, neut_sino = build_sinograms(phantom)

    cfg = nxs.ArtifactConfig()  # no artifacts, matches your "no artifact" run
    x_sino_art, n_sino_art = apply_sinogram_artifacts(xray_sino, neut_sino, cfg)

    results = {}
    for algorithm, n_iter in build_job_list():
        label = algorithm if n_iter is None else f"{algorithm}_iter{n_iter}"
        out_dir = BASE_SAVE_DIR / label
        out_dir.mkdir(parents=True, exist_ok=True)

        print(f'\n--- Reconstructing: {label} ---')
        vol_x, vol_n = reconstruct(
            x_sino_art, n_sino_art, algorithm,
            n_iter if n_iter is not None else 1,  # FBP ignores n_iter
            cfg,
        )
        print(f'vol_x: {vol_x.shape}  [{vol_x.min():.3f}, {vol_x.max():.3f}] cm⁻¹')
        print(f'vol_n: {vol_n.shape}  [{vol_n.min():.3f}, {vol_n.max():.3f}] cm⁻¹')

        hist = save_histogram_figure(phantom, vol_x, vol_n, algorithm, out_dir)
        save_xray_visualizations(phantom, vol_x, out_dir)
        save_neutron_visualizations(phantom, vol_n, out_dir)

        results[label] = dict(vol_x=vol_x, vol_n=vol_n, hist=hist)
        print(f'    saved to {out_dir.resolve()}')

    return results


if __name__ == "__main__":
    run_sweep()