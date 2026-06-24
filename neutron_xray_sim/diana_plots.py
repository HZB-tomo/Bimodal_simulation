"""
diana_plots.py
==============
Publication-quality plotting functions for the DIANA simulation pipeline.

Style standard
--------------
- plt.style.use('classic') + serif font throughout
- make_axes_locatable for all colorbars
- White figure background
- Every plot saved individually as a PDF with a clear descriptive filename
- No grouped subplots unless the data is inherently 2-D (e.g. cross-sweep matrices)

Usage
-----
Import and call each function directly, or run the module as a script with
synthetic dummy data to verify all functions execute without error:

    python diana_plots.py --demo

All functions write their output to OUTPUT_DIR (default: outputs_disc_sweep/).
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.ticker as ticker
from matplotlib import font_manager
from matplotlib.patches import Ellipse, Patch, Rectangle
from mpl_toolkits.axes_grid1 import make_axes_locatable
import pathlib

# ── Global style ─────────────────────────────────────────────────────────────
plt.style.use('classic')
plt.rcParams["font.family"] = "serif"
FONT = font_manager.FontProperties(family='serif', size=12)

# Colour palette shared across functions
MAT_COLORS = {
    'HAp': '#E8B87A',
    'Hem': '#C0392B',
    'Org': '#2ECC71',
    'Qtz': '#3498DB',
    'Air': '#1a1a2e',
}

OUTPUT_DIR = pathlib.Path('outputs_disc_sweep')
OUTPUT_DIR.mkdir(exist_ok=True)


def _line2d_handles(labels, colors, markers, linestyles=None,
                    ms=7, lw=2.0, mew=0.7):
    """
    Return one Line2D legend handle per series.

    matplotlib classic style renders markeredgecolor as a second separate
    legend entry when handles are auto-collected from ax.plot(). Building
    handles explicitly with Line2D prevents that duplication entirely.
    """
    from matplotlib.lines import Line2D
    _ls = linestyles if linestyles is not None else ['-'] * len(labels)
    return [
        Line2D([0], [0], marker=mk, ms=ms, color=col,
               markeredgecolor='black', markeredgewidth=mew,
               linestyle=ls, linewidth=lw, label=lbl)
        for lbl, col, mk, ls in zip(labels, colors, markers, _ls)
    ]


# ─────────────────────────────────────────────────────────────────────────────
# 1. Phantom label map
# ─────────────────────────────────────────────────────────────────────────────

def plot_phantom_label_map(label_2d, materials, voxel_size_mm,
                           geo_label='G1', output_dir=None):
    """
    Render a phantom label map as a colour-coded image.

    Parameters
    ----------
    label_2d      : (NY, NX) uint8 — label array
    materials     : list of Material, index == label value
    voxel_size_mm : float — physical size of one voxel side [mm]
    geo_label     : str — short geometry identifier used in title and filename
    output_dir    : Path or None — output directory (defaults to OUTPUT_DIR)
    """
    out = pathlib.Path(output_dir or OUTPUT_DIR)
    out.mkdir(parents=True, exist_ok=True)
    ny, nx = label_2d.shape
    n_labels = len(materials)

    colors = [MAT_COLORS.get(m.symbol, getattr(m, 'color', '#888888'))
              for m in materials]
    cmap = mcolors.ListedColormap(colors)
    bounds = np.arange(-0.5, n_labels + 0.5)
    norm = mcolors.BoundaryNorm(bounds, cmap.N)

    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    fig.patch.set_facecolor('white')
    fig.subplots_adjust(left=0.12, bottom=0.10, right=0.88, top=0.92)

    extent = [0, nx * voxel_size_mm, 0, ny * voxel_size_mm]
    im = ax.imshow(label_2d, cmap=cmap, norm=norm, origin='lower',
                   extent=extent, interpolation='nearest')

    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.06)
    cbar = fig.colorbar(im, cax=cax, ticks=range(n_labels))
    cbar.set_ticklabels([f'[{i}] {materials[i].symbol}' for i in range(n_labels)])
    cbar.ax.tick_params(labelsize=10)
    for label in cbar.ax.get_yticklabels():
        label.set_fontproperties(FONT)

    ax.set_xlabel('x (mm)', fontname='serif', fontsize=12)
    ax.set_ylabel('y (mm)', fontname='serif', fontsize=12)
    ax.set_title(f'{geo_label} — Phantom Label Map', fontname='serif', fontsize=13)
    ax.tick_params(labelsize=10)

    file_name = f'{geo_label}_phantom_label_map'
    fig.savefig(out / f'{file_name}.pdf', bbox_inches='tight')
    plt.close(fig)
    print(f'Saved  {out / file_name}.pdf')


# ─────────────────────────────────────────────────────────────────────────────
# 2. Bimodal histogram (single N_proj)
# ─────────────────────────────────────────────────────────────────────────────

def plot_bimodal_histogram(hist, gt_positions=None, table=None,
                           n_proj=None, condition='clean',
                           beam_label='ILL-NeXT', kvp=60,
                           output_dir=None):
    """
    Plot a 2-D bimodal (μ_x, μ_n) histogram with attached marginals.

    Layout
    ------
    - Top panel   : μ_x marginal (bar chart, log counts)
    - Main panel  : 2-D histogram heatmap (log counts, jet colourmap)
    - Right panel : μ_n marginal (barh chart, log counts)
    - Colorbar    : appended to the right of the right marginal panel

    Parameters
    ----------
    hist          : HistogramResult — from compute_bimodal_histogram()
    gt_positions  : dict {symbol: (mu_x_gt, mu_n_gt)} or None — kept for API
                    compatibility but NOT plotted (no diamonds, no labels)
    table         : HistogramMetricsTable or None — CE/DB in title line
    n_proj        : int or None
    condition     : str — 'clean' or 'dirty'
    beam_label    : str
    kvp           : float
    output_dir    : Path or None
    """
    out = pathlib.Path(output_dir or OUTPUT_DIR)
    out.mkdir(parents=True, exist_ok=True)

    x_lo, x_hi = float(hist.x_edges[0]), float(hist.x_edges[-1])
    n_lo, n_hi = float(hist.n_edges[0]), float(hist.n_edges[-1])
    bw_x = float(hist.x_edges[1] - hist.x_edges[0])   # bin width x
    bw_n = float(hist.n_edges[1] - hist.n_edges[0])   # bin width n

    fig, ax_main = plt.subplots(1, 1, figsize=(8, 7))
    fig.patch.set_facecolor('white')
    fig.subplots_adjust(left=0.11, bottom=0.09, right=0.87, top=0.87)

    # --- Attach marginal axes and colorbar axis via make_axes_locatable -----
    divider = make_axes_locatable(ax_main)
    ax_top   = divider.append_axes('top',   size='18%', pad=0.06)
    ax_right = divider.append_axes('right', size='18%', pad=0.06)
    cax      = divider.append_axes('right', size='4%',  pad=0.08)

    # --- 2-D histogram — pin limits to exact edge arrays -------------------
    H = hist.H.T
    H_plot = np.log1p(H.astype(float))
    H_plot = np.ma.masked_where(H == 0, H_plot)

    im = ax_main.pcolormesh(
        hist.x_edges, hist.n_edges, H_plot,
        cmap='jet', shading='flat', rasterized=True,
    )
    # Pin axes exactly to data extent so no whitespace appears at borders
    ax_main.set_xlim(x_lo, x_hi)
    ax_main.set_ylim(n_lo, n_hi)

    cbar = fig.colorbar(im, cax=cax, label='log(1 + counts)')
    cbar.ax.tick_params(labelsize=9)
    for lbl in cbar.ax.get_yticklabels():
        lbl.set_fontproperties(FONT)

    # --- Top marginal: μ_x — use bar() to avoid step/fill artifacts --------
    x_raw = hist.H.sum(axis=1).astype(float)
    pos_mask = x_raw > 0
    ax_top.bar(hist.x_centres[pos_mask], x_raw[pos_mask],
               width=bw_x * 0.92, color='#5DADE2', alpha=0.70,
               edgecolor='#2471A3', linewidth=0.5, align='center')
    ax_top.set_yscale('log')
    ax_top.set_ylabel('Counts', fontname='serif', fontsize=9)
    ax_top.tick_params(labelbottom=False, labelsize=8)
    ax_top.set_xlim(x_lo, x_hi)
    for sp in ax_top.spines.values():    # hide all spines on top marginal
        sp.set_visible(False)

    # --- Right marginal: μ_n — use barh() to avoid step/fill artifacts -----
    n_raw = hist.H.sum(axis=0).astype(float)
    pos_mask_n = n_raw > 0
    ax_right.barh(hist.n_centres[pos_mask_n], n_raw[pos_mask_n],
                  height=bw_n * 0.92, color='#58D68D', alpha=0.70,
                  edgecolor='#1E8449', linewidth=0.5, align='center')
    ax_right.set_xscale('log')
    ax_right.set_xlabel('Counts', fontname='serif', fontsize=9)
    ax_right.tick_params(labelleft=False, labelsize=7)
    ax_right.set_ylim(n_lo, n_hi)
    for sp in ax_right.spines.values():  # hide all spines on right marginal
        sp.set_visible(False)
    # Hide the y-axis of ax_right entirely (ticks + labels) — the tick marks
    # bleed through onto ax_main's right border and look like a second axis.
    ax_right.yaxis.set_visible(False)

    # --- Metrics: placed in the title line, not overlaid on data -----------
    ce_str, db_str = '', ''
    if table is not None:
        ce = table.scalars.get('CE')
        db = table.scalars.get('DB')
        if ce is not None:
            ce_str = f'  CE={ce:.4f} cm⁻¹'
        if db is not None:
            db_str = f'  DB={db:.4f}'

    ax_main.set_xlabel(r'$\mu_x$ (cm$^{-1}$)', fontname='serif', fontsize=12)
    ax_main.set_ylabel(r'$\mu_n$ (cm$^{-1}$)', fontname='serif', fontsize=12)
    ax_main.tick_params(labelsize=10)

    n_str = f'N={n_proj}' if n_proj is not None else ''
    ax_top.set_title(
        f'Bimodal Histogram  {n_str}  [{condition}]{ce_str}{db_str}\n'
        f'X-ray {kvp:.0f} kVp  |  Neutron {beam_label}',
        fontname='serif', fontsize=11,
    )

    n_tag = f'N{n_proj:04d}' if n_proj is not None else 'Nall'
    file_name = f'hist_{n_tag}_{condition}'
    fig.savefig(out / f'{file_name}.pdf', bbox_inches='tight')
    plt.close(fig)
    print(f'Saved  {out / file_name}.pdf')


# ─────────────────────────────────────────────────────────────────────────────
# 3 & 4. Standalone marginal functions — thin wrappers kept for compatibility.
# The marginals are now embedded inside plot_bimodal_histogram. These functions
# remain callable but simply delegate to the combined plot so that any existing
# notebook cells that call them explicitly still work without modification.
# ─────────────────────────────────────────────────────────────────────────────

def plot_xray_marginal(hist, materials=None, n_proj=None, condition='clean',
                       output_dir=None):
    """
    Compatibility wrapper — marginals are now part of plot_bimodal_histogram.
    Calling this function is a no-op; the combined histogram already contains
    the μ_x marginal panel. Remove calls to this function from new code.
    """
    pass  # marginal is embedded in plot_bimodal_histogram


def plot_neutron_marginal(hist, n_proj=None, condition='clean',
                          output_dir=None):
    """
    Compatibility wrapper — marginals are now part of plot_bimodal_histogram.
    Calling this function is a no-op; the combined histogram already contains
    the μ_n marginal panel. Remove calls to this function from new code.
    """
    pass  # marginal is embedded in plot_bimodal_histogram


# ─────────────────────────────────────────────────────────────────────────────
# 5. Reconstructed volume image (X-ray or neutron, single slice)
# ─────────────────────────────────────────────────────────────────────────────

def plot_reconstruction_slice(vol_2d, modality='xray', n_proj=None,
                              condition='clean', vmin=0, vmax=None,
                              voxel_size_mm=0.05, geo_label='G1',
                              output_dir=None):
    """
    Display one 2-D reconstruction slice.

    Parameters
    ----------
    vol_2d        : (NY, NX) float32, units cm⁻¹
    modality      : 'xray' or 'neutron'
    n_proj        : int or None
    condition     : str — 'clean' or 'dirty'
    vmin, vmax    : colour scale limits (None → data range for vmax)
    voxel_size_mm : float
    geo_label     : str
    output_dir    : Path or None
    """
    out = pathlib.Path(output_dir or OUTPUT_DIR)
    out.mkdir(parents=True, exist_ok=True)

    cmap = 'bone' if modality == 'xray' else 'viridis'
    label_str = r'X-ray $\mu$ (cm$^{-1}$)' if modality == 'xray' \
        else r'Neutron $\mu$ (cm$^{-1}$)'
    _vmax = float(vol_2d.max()) if vmax is None else vmax
    ny, nx = vol_2d.shape
    extent = [0, nx * voxel_size_mm, 0, ny * voxel_size_mm]

    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    fig.patch.set_facecolor('white')
    fig.subplots_adjust(left=0.12, bottom=0.10, right=0.88, top=0.90)

    im = ax.imshow(vol_2d, cmap=cmap, vmin=vmin, vmax=_vmax,
                   origin='lower', extent=extent, interpolation='nearest')

    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.06)
    cbar = fig.colorbar(im, cax=cax, label=label_str)
    cbar.ax.tick_params(labelsize=9)
    for lbl in cbar.ax.get_yticklabels():
        lbl.set_fontproperties(FONT)

    ax.set_xlabel('x (mm)', fontname='serif', fontsize=12)
    ax.set_ylabel('y (mm)', fontname='serif', fontsize=12)
    ax.tick_params(labelsize=10)

    n_str = f'N={n_proj}' if n_proj is not None else ''
    ax.set_title(
        f'{geo_label} — {modality.capitalize()} Reconstruction  '
        f'{n_str}  [{condition}]',
        fontname='serif', fontsize=12,
    )

    n_tag = f'N{n_proj:04d}' if n_proj is not None else 'Nall'
    file_name = f'{geo_label}_recon_{modality}_{n_tag}_{condition}'
    fig.savefig(out / f'{file_name}.pdf', bbox_inches='tight')
    plt.close(fig)
    print(f'Saved  {out / file_name}.pdf')


# ─────────────────────────────────────────────────────────────────────────────
# 6. CE vs N_proj
# ─────────────────────────────────────────────────────────────────────────────

def plot_CE_vs_nprojections(n_proj_vals, ce_vals, geo_label='G1',
                            condition='clean', output_dir=None):
    """
    Plot mean centroid error CE as a function of N_proj.

    Parameters
    ----------
    n_proj_vals : list[int]
    ce_vals     : list[float] — CE values in cm⁻¹
    geo_label   : str
    condition   : str
    output_dir  : Path or None
    """
    out = pathlib.Path(output_dir or OUTPUT_DIR)
    out.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(1, 1, figsize=(7, 5))
    fig.patch.set_facecolor('white')
    fig.subplots_adjust(left=0.13, bottom=0.12, right=0.95, top=0.90)

    ax.plot(n_proj_vals, ce_vals, 'o-', color='#2471A3', lw=2.0, ms=8,
            markeredgecolor='black', markeredgewidth=0.8)

    ax.set_xscale('log', base=2)
    ax.set_xticks(n_proj_vals)
    ax.set_xticklabels([str(n) for n in n_proj_vals], fontname='serif',
                       fontsize=11)
    ax.set_xlabel('Number of projection angles', fontname='serif', fontsize=12)
    ax.set_ylabel(r'Mean Centroid Error CE (cm$^{-1}$)',
                  fontname='serif', fontsize=12)
    ax.set_title(f'{geo_label} — CE vs N_proj  [{condition}]',
                 fontname='serif', fontsize=12)
    ax.tick_params(labelsize=10)
    ax.grid(True, linestyle='--', alpha=0.4)

    file_name = f'{geo_label}_CE_vs_Nproj_{condition}'
    fig.savefig(out / f'{file_name}.pdf', bbox_inches='tight')
    plt.close(fig)
    print(f'Saved  {out / file_name}.pdf')


# ─────────────────────────────────────────────────────────────────────────────
# 7. DB vs N_proj
# ─────────────────────────────────────────────────────────────────────────────

def plot_DB_vs_nprojections(n_proj_vals, db_vals, geo_label='G1',
                            condition='clean', output_dir=None):
    """
    Plot Davies-Bouldin index DB as a function of N_proj.

    Parameters
    ----------
    n_proj_vals : list[int]
    db_vals     : list[float]
    geo_label   : str
    condition   : str
    output_dir  : Path or None
    """
    out = pathlib.Path(output_dir or OUTPUT_DIR)
    out.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(1, 1, figsize=(7, 5))
    fig.patch.set_facecolor('white')
    fig.subplots_adjust(left=0.13, bottom=0.12, right=0.95, top=0.90)

    ax.plot(n_proj_vals, db_vals, 's-', color='#1E8449', lw=2.0, ms=8,
            markeredgecolor='black', markeredgewidth=0.8)

    ax.set_xscale('log', base=2)
    ax.set_xticks(n_proj_vals)
    ax.set_xticklabels([str(n) for n in n_proj_vals], fontname='serif',
                       fontsize=11)
    ax.set_xlabel('Number of projection angles', fontname='serif', fontsize=12)
    ax.set_ylabel('Davies\u2013Bouldin Index DB', fontname='serif', fontsize=12)
    ax.set_title(f'{geo_label} — DB vs N_proj  [{condition}]',
                 fontname='serif', fontsize=12)
    ax.tick_params(labelsize=10)
    ax.grid(True, linestyle='--', alpha=0.4)

    file_name = f'{geo_label}_DB_vs_Nproj_{condition}'
    fig.savefig(out / f'{file_name}.pdf', bbox_inches='tight')
    plt.close(fig)
    print(f'Saved  {out / file_name}.pdf')


# ─────────────────────────────────────────────────────────────────────────────
# 8. Per-cluster σ_x vs N_proj
# ─────────────────────────────────────────────────────────────────────────────

def plot_sigma_x_vs_nprojections(n_proj_vals, sigma_x_data,
                                  geo_label='G1', condition='clean',
                                  output_dir=None):
    """
    Plot X-ray cluster width σ_x for each material vs N_proj.

    Parameters
    ----------
    n_proj_vals   : list[int]
    sigma_x_data  : dict {symbol: list[float]} — σ_x values per material
    geo_label     : str
    condition     : str
    output_dir    : Path or None
    """
    out = pathlib.Path(output_dir or OUTPUT_DIR)
    out.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(1, 1, figsize=(7, 5))
    fig.patch.set_facecolor('white')
    fig.subplots_adjust(left=0.13, bottom=0.12, right=0.95, top=0.90)

    markers = ['o', 's', '^', 'D']
    syms = list(sigma_x_data.keys())
    cols = [MAT_COLORS.get(s, '#555555') for s in syms]
    for sym, mk, col in zip(syms, markers, cols):
        ax.plot(n_proj_vals, sigma_x_data[sym], f'{mk}-', color=col,
                lw=2.0, ms=7, markeredgecolor='black', markeredgewidth=0.7)

    ax.set_xscale('log', base=2)
    ax.set_xticks(n_proj_vals)
    ax.set_xticklabels([str(n) for n in n_proj_vals], fontname='serif',
                       fontsize=11)
    ax.set_xlabel('Number of projection angles', fontname='serif', fontsize=12)
    ax.set_ylabel(r'$\sigma_x$ (cm$^{-1}$) — X-ray cluster width',
                  fontname='serif', fontsize=12)
    ax.set_title(geo_label + r' — Per-material $\sigma_x$ vs N_proj  [' + condition + ']',
                 fontname='serif', fontsize=12)
    ax.legend(handles=_line2d_handles(syms, cols, markers),
              fontsize=11, prop=FONT, framealpha=0.7)
    ax.tick_params(labelsize=10)
    ax.grid(True, linestyle='--', alpha=0.4)

    file_name = f'{geo_label}_sigma_x_vs_Nproj_{condition}'
    fig.savefig(out / f'{file_name}.pdf', bbox_inches='tight')
    plt.close(fig)
    print(f'Saved  {out / file_name}.pdf')


# ─────────────────────────────────────────────────────────────────────────────
# 9. Per-cluster σ_n vs N_proj
# ─────────────────────────────────────────────────────────────────────────────

def plot_sigma_n_vs_nprojections(n_proj_vals, sigma_n_data,
                                  geo_label='G1', condition='clean',
                                  output_dir=None):
    """
    Plot neutron cluster width σ_n for each material vs N_proj.

    Parameters
    ----------
    n_proj_vals   : list[int]
    sigma_n_data  : dict {symbol: list[float]} — σ_n values per material
    geo_label     : str
    condition     : str
    output_dir    : Path or None
    """
    out = pathlib.Path(output_dir or OUTPUT_DIR)
    out.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(1, 1, figsize=(7, 5))
    fig.patch.set_facecolor('white')
    fig.subplots_adjust(left=0.13, bottom=0.12, right=0.95, top=0.90)

    markers = ['o', 's', '^', 'D']
    syms = list(sigma_n_data.keys())
    cols = [MAT_COLORS.get(s, '#555555') for s in syms]
    for sym, mk, col in zip(syms, markers, cols):
        ax.plot(n_proj_vals, sigma_n_data[sym], f'{mk}-', color=col,
                lw=2.0, ms=7, markeredgecolor='black', markeredgewidth=0.7)

    ax.set_xscale('log', base=2)
    ax.set_xticks(n_proj_vals)
    ax.set_xticklabels([str(n) for n in n_proj_vals], fontname='serif',
                       fontsize=11)
    ax.set_xlabel('Number of projection angles', fontname='serif', fontsize=12)
    ax.set_ylabel(r'$\sigma_n$ (cm$^{-1}$) — Neutron cluster width',
                  fontname='serif', fontsize=12)
    ax.set_title(geo_label + r' — Per-material $\sigma_n$ vs N_proj  [' + condition + ']',
                 fontname='serif', fontsize=12)
    ax.legend(handles=_line2d_handles(syms, cols, markers),
              fontsize=11, prop=FONT, framealpha=0.7)
    ax.tick_params(labelsize=10)
    ax.grid(True, linestyle='--', alpha=0.4)

    file_name = f'{geo_label}_sigma_n_vs_Nproj_{condition}'
    fig.savefig(out / f'{file_name}.pdf', bbox_inches='tight')
    plt.close(fig)
    print(f'Saved  {out / file_name}.pdf')


# ─────────────────────────────────────────────────────────────────────────────
# 10. CE vs N_proj — all geometries, clean vs dirty overlay
# ─────────────────────────────────────────────────────────────────────────────

def plot_CE_all_geometries(sweep_results, projection_counts,
                           geo_keys=None, geo_labels=None,
                           output_dir=None):
    """
    Plot CE vs N_proj for every geometry, with clean and dirty on the same axes.
    One PDF per geometry.

    Parameters
    ----------
    sweep_results    : dict {geo_key: {condition: {n_proj: result_dict}}}
    projection_counts: list[int]
    geo_keys         : list[str] or None — defaults to sorted(sweep_results.keys())
    geo_labels       : list[str] or None — human-readable labels for titles
    output_dir       : Path or None
    """
    out = pathlib.Path(output_dir or OUTPUT_DIR)
    out.mkdir(parents=True, exist_ok=True)

    _geo_keys = geo_keys if geo_keys is not None else sorted(sweep_results.keys())
    _geo_labels = geo_labels if geo_labels is not None else _geo_keys

    cond_style = {
        'clean': dict(ls='-',  lw=2.0, marker='o', ms=8,
                      markeredgecolor='black', markeredgewidth=0.7,
                      color='#2471A3'),
        'dirty': dict(ls='--', lw=2.0, marker='s', ms=8,
                      markeredgecolor='black', markeredgewidth=0.7,
                      color='#C0392B'),
    }

    for gkey, glabel in zip(_geo_keys, _geo_labels):
        fig, ax = plt.subplots(1, 1, figsize=(7, 5))
        fig.patch.set_facecolor('white')
        fig.subplots_adjust(left=0.13, bottom=0.12, right=0.95, top=0.90)

        plotted_conds, plotted_cols, plotted_mks, plotted_ls = [], [], [], []
        for cond, style in cond_style.items():
            if cond not in sweep_results.get(gkey, {}):
                continue
            sweep_c = sweep_results[gkey][cond]
            n_vals = sorted(sweep_c.keys())
            ce_vals = []
            for n in n_vals:
                t = sweep_c[n]['table_la']
                v = t.scalars.get('CE') if t else None
                ce_vals.append(v if v is not None else float('nan'))
            ax.plot(n_vals, ce_vals, style['ls'], color=style['color'],
                    lw=style['lw'], marker=style['marker'], ms=style['ms'],
                    markeredgecolor='black', markeredgewidth=style['markeredgewidth'])
            plotted_conds.append(cond.capitalize())
            plotted_cols.append(style['color'])
            plotted_mks.append(style['marker'])
            plotted_ls.append(style['ls'])

        ax.set_xscale('log', base=2)
        ax.set_xticks(projection_counts)
        ax.set_xticklabels([str(n) for n in projection_counts],
                           fontname='serif', fontsize=11)
        ax.set_xlabel('Number of projection angles', fontname='serif', fontsize=12)
        ax.set_ylabel(r'Mean Centroid Error CE (cm$^{-1}$)',
                      fontname='serif', fontsize=12)
        ax.set_title(f'{glabel} — CE vs N_proj  [clean vs dirty]',
                     fontname='serif', fontsize=12)
        ax.legend(handles=_line2d_handles(plotted_conds, plotted_cols,
                                          plotted_mks, plotted_ls, ms=8),
                  fontsize=11, prop=FONT, framealpha=0.7)
        ax.tick_params(labelsize=10)
        ax.grid(True, linestyle='--', alpha=0.4)

        file_name = f'{gkey}_CE_vs_Nproj_clean_vs_dirty'
        fig.savefig(out / f'{file_name}.pdf', bbox_inches='tight')
        plt.close(fig)
        print(f'Saved  {out / file_name}.pdf')


# ─────────────────────────────────────────────────────────────────────────────
# 11. DB vs N_proj — all geometries, clean vs dirty overlay
# ─────────────────────────────────────────────────────────────────────────────

def plot_DB_all_geometries(sweep_results, projection_counts,
                           geo_keys=None, geo_labels=None,
                           output_dir=None):
    """
    Plot DB vs N_proj for every geometry, with clean and dirty on the same axes.
    One PDF per geometry.

    Parameters
    ----------
    sweep_results    : dict {geo_key: {condition: {n_proj: result_dict}}}
    projection_counts: list[int]
    geo_keys         : list[str] or None
    geo_labels       : list[str] or None
    output_dir       : Path or None
    """
    out = pathlib.Path(output_dir or OUTPUT_DIR)
    out.mkdir(parents=True, exist_ok=True)

    _geo_keys = geo_keys if geo_keys is not None else sorted(sweep_results.keys())
    _geo_labels = geo_labels if geo_labels is not None else _geo_keys

    cond_style = {
        'clean': dict(ls='-',  lw=2.0, marker='o', ms=8,
                      markeredgecolor='black', markeredgewidth=0.7,
                      color='#1E8449'),
        'dirty': dict(ls='--', lw=2.0, marker='s', ms=8,
                      markeredgecolor='black', markeredgewidth=0.7,
                      color='#C0392B'),
    }

    for gkey, glabel in zip(_geo_keys, _geo_labels):
        fig, ax = plt.subplots(1, 1, figsize=(7, 5))
        fig.patch.set_facecolor('white')
        fig.subplots_adjust(left=0.13, bottom=0.12, right=0.95, top=0.90)

        plotted_conds, plotted_cols, plotted_mks, plotted_ls = [], [], [], []
        for cond, style in cond_style.items():
            if cond not in sweep_results.get(gkey, {}):
                continue
            sweep_c = sweep_results[gkey][cond]
            n_vals = sorted(sweep_c.keys())
            db_vals = []
            for n in n_vals:
                t = sweep_c[n]['table_la']
                v = t.scalars.get('DB') if t else None
                db_vals.append(v if v is not None else float('nan'))
            ax.plot(n_vals, db_vals, style['ls'], color=style['color'],
                    lw=style['lw'], marker=style['marker'], ms=style['ms'],
                    markeredgecolor='black', markeredgewidth=style['markeredgewidth'])
            plotted_conds.append(cond.capitalize())
            plotted_cols.append(style['color'])
            plotted_mks.append(style['marker'])
            plotted_ls.append(style['ls'])

        ax.set_xscale('log', base=2)
        ax.set_xticks(projection_counts)
        ax.set_xticklabels([str(n) for n in projection_counts],
                           fontname='serif', fontsize=11)
        ax.set_xlabel('Number of projection angles', fontname='serif', fontsize=12)
        ax.set_ylabel('Davies\u2013Bouldin Index DB', fontname='serif', fontsize=12)
        ax.set_title(f'{glabel} — DB vs N_proj  [clean vs dirty]',
                     fontname='serif', fontsize=12)
        ax.legend(handles=_line2d_handles(plotted_conds, plotted_cols,
                                          plotted_mks, plotted_ls, ms=8),
                  fontsize=11, prop=FONT, framealpha=0.7)
        ax.tick_params(labelsize=10)
        ax.grid(True, linestyle='--', alpha=0.4)

        file_name = f'{gkey}_DB_vs_Nproj_clean_vs_dirty'
        fig.savefig(out / f'{file_name}.pdf', bbox_inches='tight')
        plt.close(fig)
        print(f'Saved  {out / file_name}.pdf')


# ─────────────────────────────────────────────────────────────────────────────
# 12. Cross-sweep heatmap table (CE or DB)
# ─────────────────────────────────────────────────────────────────────────────

def plot_cross_sweep_heatmap(matrix, projection_counts, metric='CE',
                              good_is_low=True, output_dir=None):
    """
    Colour-coded table of CE or DB over the N_proj_X × N_proj_N cross-sweep.
    Rows = X-ray N_proj, columns = neutron N_proj.
    Diagonal cells (matched sweep) are outlined in gold.

    Parameters
    ----------
    matrix           : (N, N) float array
    projection_counts: list[int] — axis tick labels
    metric           : 'CE' or 'DB'
    good_is_low      : bool — if True, low values map to the green (good) end
    output_dir       : Path or None
    """
    out = pathlib.Path(output_dir or OUTPUT_DIR)
    out.mkdir(parents=True, exist_ok=True)

    N = len(projection_counts)
    labels_str = [str(n) for n in projection_counts]

    # Green → ivory → red diverging colourmap
    cmap_gr = mcolors.LinearSegmentedColormap.from_list(
        'good_bad',
        ['#1a6b3c', '#a8d5b5', '#f7f5ef', '#e8917a', '#8b1a1a'],
        N=256,
    )
    _cmap = cmap_gr if good_is_low else cmap_gr.reversed()

    _vmin = float(np.nanmin(matrix))
    _vmax = float(np.nanmax(matrix))
    norm = mcolors.Normalize(vmin=_vmin, vmax=_vmax)

    fig, ax = plt.subplots(1, 1, figsize=(8, 7))
    fig.patch.set_facecolor('white')
    fig.subplots_adjust(left=0.13, bottom=0.13, right=0.88, top=0.90)

    im = ax.imshow(matrix, cmap=_cmap, norm=norm,
                   aspect='equal', origin='upper',
                   interpolation='nearest')

    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.08)
    unit = r'cm$^{-1}$' if metric == 'CE' else ''
    cbar = fig.colorbar(im, cax=cax,
                        label=f'{metric} {unit}'.strip())
    cbar.ax.tick_params(labelsize=9)
    for lbl in cbar.ax.get_yticklabels():
        lbl.set_fontproperties(FONT)

    # Cell value annotations
    fmt = '.3f'
    for ix in range(N):
        for in_ in range(N):
            val = matrix[ix, in_]
            txt = '\u2014' if np.isnan(val) else format(val, fmt)
            brightness = norm(val) if not np.isnan(val) else 0.5
            txt_col = '#1a1a2e' if 0.25 < brightness < 0.75 else 'white'
            weight = 'bold' if ix == in_ else 'normal'
            ax.text(in_, ix, txt, ha='center', va='center',
                    fontsize=9, color=txt_col, fontweight=weight,
                    fontfamily='serif')

    # Gold border around diagonal (matched sweep)
    for i in range(N):
        rect = Rectangle((i - 0.5, i - 0.5), 1, 1,
                          linewidth=2.0, edgecolor='goldenrod',
                          facecolor='none', zorder=4)
        ax.add_patch(rect)

    # Subtle grid lines between cells
    for i in range(N + 1):
        ax.axhline(i - 0.5, color='#cccccc', lw=0.6, zorder=3)
        ax.axvline(i - 0.5, color='#cccccc', lw=0.6, zorder=3)

    ax.set_xticks(range(N))
    ax.set_yticks(range(N))
    ax.set_xticklabels(labels_str, fontname='serif', fontsize=10)
    ax.set_yticklabels(labels_str, fontname='serif', fontsize=10)
    ax.set_xlabel('Neutron $N_{\\mathrm{proj}}$', fontname='serif', fontsize=12)
    ax.set_ylabel('X-ray $N_{\\mathrm{proj}}$', fontname='serif', fontsize=12)
    ax.set_title(f'Cross-sweep {metric} — X-ray × Neutron $N_{{\\mathrm{{proj}}}}$',
                 fontname='serif', fontsize=12)
    ax.tick_params(length=0, labelsize=10)
    for sp in ax.spines.values():
        sp.set_visible(False)

    file_name = f'cross_sweep_{metric}'
    fig.savefig(out / f'{file_name}.pdf', bbox_inches='tight')
    plt.close(fig)
    print(f'Saved  {out / file_name}.pdf')


# ─────────────────────────────────────────────────────────────────────────────
# 13. X-ray attenuation spectrum per material
# ─────────────────────────────────────────────────────────────────────────────

def plot_xray_attenuation_spectra(materials, xray_e_kev, output_dir=None):
    """
    Plot μ_x vs photon energy [keV] for each material.

    Parameters
    ----------
    materials   : list of Material — each must have mu_x_at(energy_keV)
    xray_e_kev  : array-like — energy grid in keV
    output_dir  : Path or None
    """
    out = pathlib.Path(output_dir or OUTPUT_DIR)
    out.mkdir(parents=True, exist_ok=True)

    E = np.asarray(xray_e_kev)
    markers = ['o', 's', '^', 'D', 'v', 'P']

    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    fig.patch.set_facecolor('white')
    fig.subplots_adjust(left=0.12, bottom=0.12, right=0.95, top=0.90)

    plot_mats = [m for m in materials if m.symbol.lower() != 'air']
    mat_cols  = [MAT_COLORS.get(m.symbol, getattr(m, 'color', '#888888'))
                 for m in plot_mats]
    mat_labels = [f'{m.symbol} (\u03c1={m.density_gcc:.2f} g/cc)' for m in plot_mats]

    for mat, mk, col in zip(plot_mats, markers, mat_cols):
        mu_vals = [mat.mu_x_at(e) for e in E]
        ax.plot(E, mu_vals, f'{mk}-', color=col, lw=1.8, ms=7,
                markeredgecolor='black', markeredgewidth=0.7)

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Photon energy (keV)', fontname='serif', fontsize=12)
    ax.set_ylabel(r'$\mu_x$ (cm$^{-1}$)', fontname='serif', fontsize=12)
    ax.set_title('X-ray Linear Attenuation Coefficients vs Energy',
                 fontname='serif', fontsize=12)
    ax.legend(handles=_line2d_handles(mat_labels, mat_cols, markers[:len(plot_mats)]),
              fontsize=10, prop=FONT, framealpha=0.7)
    ax.tick_params(labelsize=10)
    ax.grid(True, which='both', linestyle='--', alpha=0.4)

    file_name = 'xray_attenuation_spectra'
    fig.savefig(out / f'{file_name}.pdf', bbox_inches='tight')
    plt.close(fig)
    print(f'Saved  {out / file_name}.pdf')


# ─────────────────────────────────────────────────────────────────────────────
# 14. Per-cluster centroid error ε_k vs N_proj
# ─────────────────────────────────────────────────────────────────────────────

def plot_epsilon_vs_nprojections(n_proj_vals, eps_data,
                                  geo_label='G1', condition='clean',
                                  output_dir=None):
    """
    Plot per-material centroid error ε_k vs N_proj.

    Parameters
    ----------
    n_proj_vals : list[int]
    eps_data    : dict {symbol: list[float]} — ε_k values per material
    geo_label   : str
    condition   : str
    output_dir  : Path or None
    """
    out = pathlib.Path(output_dir or OUTPUT_DIR)
    out.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(1, 1, figsize=(7, 5))
    fig.patch.set_facecolor('white')
    fig.subplots_adjust(left=0.13, bottom=0.12, right=0.95, top=0.90)

    markers = ['o', 's', '^', 'D']
    syms = list(eps_data.keys())
    cols = [MAT_COLORS.get(s, '#555555') for s in syms]
    for sym, mk, col in zip(syms, markers, cols):
        ax.plot(n_proj_vals, eps_data[sym], f'{mk}-', color=col,
                lw=2.0, ms=7, markeredgecolor='black', markeredgewidth=0.7)

    ax.set_xscale('log', base=2)
    ax.set_xticks(n_proj_vals)
    ax.set_xticklabels([str(n) for n in n_proj_vals], fontname='serif',
                       fontsize=11)
    ax.set_xlabel('Number of projection angles', fontname='serif', fontsize=12)
    ax.set_ylabel(r'Centroid error $\varepsilon_k$ (cm$^{-1}$)',
                  fontname='serif', fontsize=12)
    ax.set_title(
        geo_label + r' — Per-material centroid error $\varepsilon_k$ vs N_proj  [' + condition + ']',
        fontname='serif', fontsize=12,
    )
    ax.legend(handles=_line2d_handles(syms, cols, markers),
              fontsize=11, prop=FONT, framealpha=0.7)
    ax.tick_params(labelsize=10)
    ax.grid(True, linestyle='--', alpha=0.4)

    file_name = f'{geo_label}_epsilon_vs_Nproj_{condition}'
    fig.savefig(out / f'{file_name}.pdf', bbox_inches='tight')
    plt.close(fig)
    print(f'Saved  {out / file_name}.pdf')


# ─────────────────────────────────────────────────────────────────────────────
# 15. Pairwise material overlap vs N_proj
# ─────────────────────────────────────────────────────────────────────────────

def plot_pairwise_overlap_vs_nprojections(n_proj_vals, overlap_data,
                                           geo_label='G1', condition='clean',
                                           output_dir=None):
    """
    Plot pairwise material overlap fractions vs N_proj.

    Parameters
    ----------
    n_proj_vals  : list[int]
    overlap_data : dict {(sym_a, sym_b): list[float]} — overlap fraction per pair
    geo_label    : str
    condition    : str
    output_dir   : Path or None
    """
    out = pathlib.Path(output_dir or OUTPUT_DIR)
    out.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    fig.patch.set_facecolor('white')
    fig.subplots_adjust(left=0.13, bottom=0.12, right=0.95, top=0.90)

    linestyles = ['-', '--', ':', '-.', '-', '--']
    colors = ['#2471A3', '#C0392B', '#1E8449', '#8E44AD', '#D35400', '#7F8C8D']
    pair_labels = [f'{sa}\u2013{sb}' for sa, sb in overlap_data.keys()]

    for ((sa, sb), vals), ls, col in zip(overlap_data.items(), linestyles, colors):
        ax.plot(n_proj_vals, vals, ls, color=col, lw=2.0, ms=7,
                marker='o', markeredgecolor='black', markeredgewidth=0.7)

    ax.set_xscale('log', base=2)
    ax.set_xticks(n_proj_vals)
    ax.set_xticklabels([str(n) for n in n_proj_vals], fontname='serif',
                       fontsize=11)
    ax.set_xlabel('Number of projection angles', fontname='serif', fontsize=12)
    ax.set_ylabel('Overlap fraction', fontname='serif', fontsize=12)
    ax.set_title(f'{geo_label} — Pairwise material overlap vs N_proj  [{condition}]',
                 fontname='serif', fontsize=12)
    ax.set_ylim(bottom=0)
    ax.legend(handles=_line2d_handles(pair_labels, colors[:len(pair_labels)],
                                      ['o'] * len(pair_labels), linestyles[:len(pair_labels)]),
              fontsize=10, prop=FONT, framealpha=0.7)
    ax.tick_params(labelsize=10)
    ax.grid(True, linestyle='--', alpha=0.4)

    file_name = f'{geo_label}_pairwise_overlap_vs_Nproj_{condition}'
    fig.savefig(out / f'{file_name}.pdf', bbox_inches='tight')
    plt.close(fig)
    print(f'Saved  {out / file_name}.pdf')


# ─────────────────────────────────────────────────────────────────────────────
# 16. Artifact-signature fingerprint atlas  (Study 2)
# ─────────────────────────────────────────────────────────────────────────────

def plot_artifact_fingerprint(z_matrix, artifact_labels, signature_labels,
                              title='Artifact-signature fingerprint atlas',
                              file_name='artifact_fingerprint', output_dir=None):
    """
    Heatmap of standardised (z-scored) histogram-shape signatures per artifact.

    Rows = artifacts, columns = signatures.  A symmetric blue-ivory-red diverging
    map is centred at z = 0 so that the diagnostic feature of each artifact (the
    most extreme cell in its row) stands out in either direction.

    Parameters
    ----------
    z_matrix         : (n_artifacts, n_signatures) float array of z-scores
    artifact_labels  : list[str] — row labels
    signature_labels : list[str] — column labels
    title            : str
    file_name        : str — output file stem
    output_dir       : Path or None
    """
    out = pathlib.Path(output_dir or OUTPUT_DIR)
    out.mkdir(parents=True, exist_ok=True)

    Z = np.asarray(z_matrix, dtype=float)
    n_row, n_col = Z.shape

    cmap_div = mcolors.LinearSegmentedColormap.from_list(
        'z_div',
        ['#1a3c6b', '#5DADE2', '#f7f5ef', '#e8917a', '#8b1a1a'],
        N=256,
    )
    vmax = float(np.nanmax(np.abs(Z))) if np.isfinite(Z).any() else 1.0
    norm = mcolors.Normalize(vmin=-vmax, vmax=vmax)

    fig, ax = plt.subplots(1, 1, figsize=(1.6 + 1.1 * n_col, 1.2 + 0.46 * n_row))
    fig.patch.set_facecolor('white')
    fig.subplots_adjust(left=0.26, bottom=0.20, right=0.88, top=0.90)

    im = ax.imshow(Z, cmap=cmap_div, norm=norm, aspect='auto',
                   origin='upper', interpolation='nearest')

    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='4%', pad=0.08)
    cbar = fig.colorbar(im, cax=cax, label='z-score')
    cbar.ax.tick_params(labelsize=9)
    for lbl in cbar.ax.get_yticklabels():
        lbl.set_fontproperties(FONT)

    for ir in range(n_row):
        for ic in range(n_col):
            val = Z[ir, ic]
            if not np.isfinite(val):
                continue
            brightness = norm(val)
            txt_col = '#1a1a2e' if 0.30 < brightness < 0.70 else 'white'
            ax.text(ic, ir, format(val, '.2f'), ha='center', va='center',
                    fontsize=8, color=txt_col, fontfamily='serif')

    ax.set_xticks(range(n_col))
    ax.set_yticks(range(n_row))
    ax.set_xticklabels([s.replace('_score', '').replace('_', '\n')
                        for s in signature_labels],
                       fontname='serif', fontsize=9)
    ax.set_yticklabels(artifact_labels, fontname='serif', fontsize=10)
    ax.set_title(title, fontname='serif', fontsize=12)
    ax.tick_params(length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    for i in range(n_row + 1):
        ax.axhline(i - 0.5, color='#cccccc', lw=0.6, zorder=3)
    for i in range(n_col + 1):
        ax.axvline(i - 0.5, color='#cccccc', lw=0.6, zorder=3)

    fig.savefig(out / f'{file_name}.pdf', bbox_inches='tight')
    plt.close(fig)
    print(f'Saved  {out / file_name}.pdf')


# ─────────────────────────────────────────────────────────────────────────────
# 17. Scalar metric (CE / DB / …) per artifact  (Study 2)
# ─────────────────────────────────────────────────────────────────────────────

def plot_metric_by_artifact(values, artifact_labels, metric='CE',
                            baseline=None, baseline_label='clean',
                            file_name=None, output_dir=None):
    """
    Horizontal bar chart of one scalar metric across artifact conditions.

    Parameters
    ----------
    values          : list[float] — metric value per artifact (same order as labels)
    artifact_labels : list[str]
    metric          : str — used for the axis label and filename (e.g. 'CE', 'DB')
    baseline        : float or None — draws a reference line (e.g. the clean value)
    baseline_label  : str — legend label for the baseline line
    file_name       : str or None
    output_dir      : Path or None
    """
    out = pathlib.Path(output_dir or OUTPUT_DIR)
    out.mkdir(parents=True, exist_ok=True)

    vals = np.asarray(values, dtype=float)
    y = np.arange(len(artifact_labels))

    fig, ax = plt.subplots(1, 1, figsize=(7.5, 0.5 * len(artifact_labels) + 1.5))
    fig.patch.set_facecolor('white')
    fig.subplots_adjust(left=0.26, bottom=0.13, right=0.95, top=0.90)

    ax.barh(y, vals, color='#5DADE2', alpha=0.80,
            edgecolor='#2471A3', linewidth=0.8, align='center', zorder=2)
    if baseline is not None:
        ax.axvline(baseline, color='#C0392B', lw=1.8, linestyle='--', zorder=3)
        from matplotlib.lines import Line2D
        ax.legend(handles=[Line2D([0], [0], color='#C0392B', lw=1.8,
                                  linestyle='--', label=baseline_label)],
                  fontsize=10, prop=FONT, framealpha=0.7)

    ax.set_yticks(y)
    ax.set_yticklabels(artifact_labels, fontname='serif', fontsize=10)
    ax.invert_yaxis()
    unit = r' (cm$^{-1}$)' if metric.upper() == 'CE' else ''
    ax.set_xlabel(f'{metric}{unit}', fontname='serif', fontsize=12)
    ax.set_title(f'{metric} by artifact condition', fontname='serif', fontsize=12)
    ax.tick_params(labelsize=10)
    ax.grid(True, axis='x', linestyle='--', alpha=0.4)

    fname = file_name or f'metric_{metric}_by_artifact'
    fig.savefig(out / f'{fname}.pdf', bbox_inches='tight')
    plt.close(fig)
    print(f'Saved  {out / fname}.pdf')


# ─────────────────────────────────────────────────────────────────────────────
# 18. Cross-algorithm mismatch heatmap  (Study 3)
# ─────────────────────────────────────────────────────────────────────────────

def plot_algorithm_cross_heatmap(matrix, algorithms, metric='CE',
                                 good_is_low=True, file_name=None,
                                 output_dir=None):
    """
    Colour-coded table of a metric over the X-ray × neutron reconstruction grid.

    Rows = X-ray algorithm, columns = neutron algorithm.  The matched diagonal
    (alg_x == alg_n) is outlined in gold.  Mirrors plot_cross_sweep_heatmap but
    with algorithm labels instead of projection counts.

    Parameters
    ----------
    matrix      : (n_alg, n_alg) float array, indexed [x_alg, n_alg]
    algorithms  : list[str] — axis tick labels (same order on both axes)
    metric      : 'CE', 'DB', etc. — colourbar label / filename
    good_is_low : bool — if True, low values map to the green (good) end
    file_name   : str or None
    output_dir  : Path or None
    """
    out = pathlib.Path(output_dir or OUTPUT_DIR)
    out.mkdir(parents=True, exist_ok=True)

    M = np.asarray(matrix, dtype=float)
    N = len(algorithms)

    cmap_gr = mcolors.LinearSegmentedColormap.from_list(
        'good_bad',
        ['#1a6b3c', '#a8d5b5', '#f7f5ef', '#e8917a', '#8b1a1a'],
        N=256,
    )
    _cmap = cmap_gr if good_is_low else cmap_gr.reversed()
    norm = mcolors.Normalize(vmin=float(np.nanmin(M)), vmax=float(np.nanmax(M)))

    fig, ax = plt.subplots(1, 1, figsize=(7.5, 6.5))
    fig.patch.set_facecolor('white')
    fig.subplots_adjust(left=0.15, bottom=0.14, right=0.88, top=0.90)

    im = ax.imshow(M, cmap=_cmap, norm=norm, aspect='equal',
                   origin='upper', interpolation='nearest')

    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.08)
    unit = r'cm$^{-1}$' if metric.upper() == 'CE' else ''
    cbar = fig.colorbar(im, cax=cax, label=f'{metric} {unit}'.strip())
    cbar.ax.tick_params(labelsize=9)
    for lbl in cbar.ax.get_yticklabels():
        lbl.set_fontproperties(FONT)

    for ix in range(N):
        for in_ in range(N):
            val = M[ix, in_]
            txt = '\u2014' if np.isnan(val) else format(val, '.3f')
            brightness = norm(val) if not np.isnan(val) else 0.5
            txt_col = '#1a1a2e' if 0.25 < brightness < 0.75 else 'white'
            weight = 'bold' if ix == in_ else 'normal'
            ax.text(in_, ix, txt, ha='center', va='center',
                    fontsize=10, color=txt_col, fontweight=weight,
                    fontfamily='serif')

    for i in range(N):
        rect = Rectangle((i - 0.5, i - 0.5), 1, 1, linewidth=2.0,
                         edgecolor='goldenrod', facecolor='none', zorder=4)
        ax.add_patch(rect)
    for i in range(N + 1):
        ax.axhline(i - 0.5, color='#cccccc', lw=0.6, zorder=3)
        ax.axvline(i - 0.5, color='#cccccc', lw=0.6, zorder=3)

    ax.set_xticks(range(N))
    ax.set_yticks(range(N))
    ax.set_xticklabels(algorithms, fontname='serif', fontsize=11)
    ax.set_yticklabels(algorithms, fontname='serif', fontsize=11)
    ax.set_xlabel('Neutron reconstruction algorithm', fontname='serif', fontsize=12)
    ax.set_ylabel('X-ray reconstruction algorithm', fontname='serif', fontsize=12)
    ax.set_title(f'Cross-algorithm {metric} (gold = matched)',
                 fontname='serif', fontsize=12)
    ax.tick_params(length=0, labelsize=11)
    for sp in ax.spines.values():
        sp.set_visible(False)

    fname = file_name or f'cross_algorithm_{metric}'
    fig.savefig(out / f'{fname}.pdf', bbox_inches='tight')
    plt.close(fig)
    print(f'Saved  {out / fname}.pdf')


# ─────────────────────────────────────────────────────────────────────────────
# 19. Rare-phase metric failure: GMM (v1) vs label-anchored (v3)  (Study 5)
# ─────────────────────────────────────────────────────────────────────────────

def plot_rare_phase_failure(frac, eps_v1, eps_v3, ratio=None,
                            rare_symbol='Rare', file_name='rare_phase_failure',
                            output_dir=None):
    """
    Rare-phase centroid error under the GMM metric (v1) and the label-anchored
    metric (v3) as a function of the rare-phase voxel fraction.

    The x-axis is logarithmic and inverted, so the phase becomes rarer toward the
    right.  An optional v1/v3 ratio is overlaid on a secondary axis (dashed grey).

    Parameters
    ----------
    frac        : list[float] — rare-phase voxel fraction (0-1)
    eps_v1      : list[float] — GMM centroid error of the rare phase [cm^-1]
    eps_v3      : list[float] — label-anchored centroid error [cm^-1]
    ratio       : list[float] or None — eps_v1 / eps_v3
    rare_symbol : str — used in the title
    file_name   : str
    output_dir  : Path or None
    """
    out = pathlib.Path(output_dir or OUTPUT_DIR)
    out.mkdir(parents=True, exist_ok=True)

    frac = np.asarray(frac, dtype=float)
    order = np.argsort(frac)[::-1]            # large -> small (left -> right)
    frac = frac[order]
    eps_v1 = np.asarray(eps_v1, dtype=float)[order]
    eps_v3 = np.asarray(eps_v3, dtype=float)[order]

    cols = ['#C0392B', '#1E8449']
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    fig.patch.set_facecolor('white')
    fig.subplots_adjust(left=0.13, bottom=0.13, right=0.87, top=0.90)

    ax.plot(frac, eps_v1, 'o-', color=cols[0], lw=2.0, ms=8,
            markeredgecolor='black', markeredgewidth=0.7)
    ax.plot(frac, eps_v3, 's-', color=cols[1], lw=2.0, ms=8,
            markeredgecolor='black', markeredgewidth=0.7)

    ax.set_xscale('log')
    ax.invert_xaxis()
    ax.set_xlabel('Rare-phase voxel fraction', fontname='serif', fontsize=12)
    ax.set_ylabel(r'Rare-phase $\epsilon$ (cm$^{-1}$)', fontname='serif', fontsize=12)
    ax.set_title(rf'Rare-phase centroid error: GMM (v1) vs label-anchored (v3) — {rare_symbol}',
                 fontname='serif', fontsize=12)
    ax.tick_params(labelsize=10)
    ax.grid(True, which='both', linestyle='--', alpha=0.4)

    handles = _line2d_handles(['v1 GMM', 'v3 label-anchored'], cols, ['o', 's'])

    if ratio is not None:
        ratio = np.asarray(ratio, dtype=float)[order]
        ax2 = ax.twinx()
        ax2.plot(frac, ratio, 'd:', color='#7F8C8D', lw=1.6, ms=6,
                 markeredgecolor='black', markeredgewidth=0.6)
        ax2.set_ylabel('v1 / v3 ratio', fontname='serif', fontsize=12)
        ax2.set_ylim(bottom=0)
        ax2.tick_params(labelsize=10)
        from matplotlib.lines import Line2D
        handles.append(Line2D([0], [0], marker='d', ms=6, color='#7F8C8D',
                              markeredgecolor='black', markeredgewidth=0.6,
                              linestyle=':', linewidth=1.6, label='v1 / v3 ratio'))

    ax.legend(handles=handles, fontsize=10, prop=FONT, framealpha=0.7, loc='upper left')

    fig.savefig(out / f'{file_name}.pdf', bbox_inches='tight')
    plt.close(fig)
    print(f'Saved  {out / file_name}.pdf')


# ─────────────────────────────────────────────────────────────────────────────
# 20. Generic multi-series metric vs swept parameter
# ─────────────────────────────────────────────────────────────────────────────

def plot_metric_vs_param_multi(x_vals, series, xlabel, ylabel,
                               title='', file_name='metric_vs_param',
                               log_x=True, log_y=False, invert_x=False,
                               markers=None, colors=None, output_dir=None):
    """
    Plot several labelled series of a scalar metric against one swept parameter.

    Parameters
    ----------
    x_vals     : list[float] — shared x values
    series     : dict {label: list[float]} — one y-series per label
    xlabel     : str
    ylabel     : str
    title      : str
    file_name  : str — output stem
    log_x      : bool — log-scale x axis
    log_y      : bool — log-scale y axis
    invert_x   : bool — invert x (e.g. rarer phase toward the right)
    markers    : list[str] or None
    colors     : list[str] or None
    output_dir : Path or None
    """
    out = pathlib.Path(output_dir or OUTPUT_DIR)
    out.mkdir(parents=True, exist_ok=True)

    labels = list(series.keys())
    n = len(labels)
    _markers = markers or (['o', 's', '^', 'D', 'v', 'P', '*', 'X', '<', '>'] * 3)[:n]
    _palette = ['#2471A3', '#C0392B', '#1E8449', '#8E44AD', '#D35400',
                '#7F8C8D', '#16A085', '#B7950B', '#CB4335', '#5D6D7E']
    _colors = colors or (_palette * 3)[:n]

    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    fig.patch.set_facecolor('white')
    fig.subplots_adjust(left=0.13, bottom=0.13, right=0.95, top=0.90)

    for lbl, mk, col in zip(labels, _markers, _colors):
        ax.plot(x_vals, series[lbl], f'{mk}-', color=col, lw=2.0, ms=7,
                markeredgecolor='black', markeredgewidth=0.7)

    if log_x:
        ax.set_xscale('log')
    if log_y:
        ax.set_yscale('log')
    if invert_x:
        ax.invert_xaxis()
    ax.set_xlabel(xlabel, fontname='serif', fontsize=12)
    ax.set_ylabel(ylabel, fontname='serif', fontsize=12)
    ax.set_title(title, fontname='serif', fontsize=12)
    ax.legend(handles=_line2d_handles(labels, _colors, _markers),
              fontsize=10, prop=FONT, framealpha=0.7)
    ax.tick_params(labelsize=10)
    ax.grid(True, which='both', linestyle='--', alpha=0.4)

    fig.savefig(out / f'{file_name}.pdf', bbox_inches='tight')
    plt.close(fig)
    print(f'Saved  {out / file_name}.pdf')


# ─────────────────────────────────────────────────────────────────────────────
# 21. N*(tolerance) vs separability margin  (Study 1 generalisation)
# ─────────────────────────────────────────────────────────────────────────────

def plot_nstar_vs_margin(margins, nstar, fit_coeffs=None, tol=None,
                         file_name='nstar_vs_margin', output_dir=None):
    """
    Scatter of the required projection count N*(tau) against a dimensionless
    separability margin, with an optional power-law fit overlaid (log-log).

    Parameters
    ----------
    margins     : list[float] — separability margin (e.g. inter-pair distance / spread)
    nstar       : list[float] — required projection count for the tolerance tau
    fit_coeffs  : (a, b) or None — power law N* = a * margin**b, drawn if provided
    tol         : float or None — tolerance value, shown in the title
    file_name   : str
    output_dir  : Path or None
    """
    out = pathlib.Path(output_dir or OUTPUT_DIR)
    out.mkdir(parents=True, exist_ok=True)

    m = np.asarray(margins, dtype=float)
    ns = np.asarray(nstar, dtype=float)

    fig, ax = plt.subplots(1, 1, figsize=(7.5, 5))
    fig.patch.set_facecolor('white')
    fig.subplots_adjust(left=0.13, bottom=0.13, right=0.95, top=0.90)

    ax.plot(m, ns, 'o', color='#2471A3', ms=9,
            markeredgecolor='black', markeredgewidth=0.8, zorder=3)

    handles_lbls = ['N*(\u03c4) measured']
    handles_cols = ['#2471A3']
    handles_mk = ['o']
    handles_ls = ['none']
    if fit_coeffs is not None:
        a, b = fit_coeffs
        xx = np.linspace(m.min(), m.max(), 100)
        ax.plot(xx, a * xx ** b, '-', color='#C0392B', lw=2.0, zorder=2)
        handles_lbls.append(rf'fit: $N^* = {a:.0f}\,\Delta^{{{b:.2f}}}$')
        handles_cols.append('#C0392B'); handles_mk.append('None'); handles_ls.append('-')

    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlabel(r'Separability margin $\Delta_{ab}/\bar{\sigma}$',
                  fontname='serif', fontsize=12)
    ax.set_ylabel(r'Required projections $N^*(\tau)$', fontname='serif', fontsize=12)
    tol_str = f'  (\u03c4 = {tol:g})' if tol is not None else ''
    ax.set_title(f'Acquisition budget vs separability margin{tol_str}',
                 fontname='serif', fontsize=12)
    ax.legend(handles=_line2d_handles(handles_lbls, handles_cols, handles_mk, handles_ls),
              fontsize=10, prop=FONT, framealpha=0.7)
    ax.tick_params(labelsize=10)
    ax.grid(True, which='both', linestyle='--', alpha=0.4)

    fig.savefig(out / f'{file_name}.pdf', bbox_inches='tight')
    plt.close(fig)
    print(f'Saved  {out / file_name}.pdf')


# ─────────────────────────────────────────────────────────────────────────────
# 22. Recovery / recall heatmap (algorithms × condition)  (Study 5)
# ─────────────────────────────────────────────────────────────────────────────

def plot_recovery_heatmap(matrix, row_labels, col_labels,
                          value_label='Rare-phase recall', title='',
                          file_name='recovery_heatmap', fmt='.2f',
                          good_is_high=True, output_dir=None):
    """
    Rectangular annotated heatmap, e.g. rare-phase recall per clustering algorithm
    (rows) across a swept condition (columns).

    Parameters
    ----------
    matrix       : (n_rows, n_cols) float array in [0, 1] (or any range)
    row_labels   : list[str]
    col_labels   : list[str]
    value_label  : str — colourbar label
    title        : str
    file_name    : str
    fmt          : str — cell value format
    good_is_high : bool — if True, high values map to green (good)
    output_dir   : Path or None
    """
    out = pathlib.Path(output_dir or OUTPUT_DIR)
    out.mkdir(parents=True, exist_ok=True)

    M = np.asarray(matrix, dtype=float)
    nr, nc = M.shape

    cmap_gr = mcolors.LinearSegmentedColormap.from_list(
        'recall', ['#8b1a1a', '#e8917a', '#f7f5ef', '#a8d5b5', '#1a6b3c'], N=256)
    _cmap = cmap_gr if good_is_high else cmap_gr.reversed()
    norm = mcolors.Normalize(vmin=float(np.nanmin(M)), vmax=float(np.nanmax(M)))

    fig, ax = plt.subplots(1, 1, figsize=(1.6 + 0.9 * nc, 1.4 + 0.45 * nr))
    fig.patch.set_facecolor('white')
    fig.subplots_adjust(left=0.24, bottom=0.18, right=0.88, top=0.90)

    im = ax.imshow(M, cmap=_cmap, norm=norm, aspect='auto',
                   origin='upper', interpolation='nearest')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='4%', pad=0.08)
    cbar = fig.colorbar(im, cax=cax, label=value_label)
    cbar.ax.tick_params(labelsize=9)
    for lbl in cbar.ax.get_yticklabels():
        lbl.set_fontproperties(FONT)

    for ir in range(nr):
        for ic in range(nc):
            val = M[ir, ic]
            txt = '\u2014' if np.isnan(val) else format(val, fmt)
            brightness = norm(val) if not np.isnan(val) else 0.5
            txt_col = '#1a1a2e' if 0.30 < brightness < 0.70 else 'white'
            ax.text(ic, ir, txt, ha='center', va='center',
                    fontsize=8, color=txt_col, fontfamily='serif')

    ax.set_xticks(range(nc)); ax.set_yticks(range(nr))
    ax.set_xticklabels(col_labels, fontname='serif', fontsize=9, rotation=0)
    ax.set_yticklabels(row_labels, fontname='serif', fontsize=10)
    ax.set_title(title, fontname='serif', fontsize=12)
    ax.tick_params(length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    for i in range(nr + 1):
        ax.axhline(i - 0.5, color='#cccccc', lw=0.6, zorder=3)
    for i in range(nc + 1):
        ax.axvline(i - 0.5, color='#cccccc', lw=0.6, zorder=3)

    fig.savefig(out / f'{file_name}.pdf', bbox_inches='tight')
    plt.close(fig)
    print(f'Saved  {out / file_name}.pdf')


# ─────────────────────────────────────────────────────────────────────────────
# 23. Grouped bars with confidence intervals  (Study 3 Monte-Carlo)
# ─────────────────────────────────────────────────────────────────────────────

def plot_grouped_bars_with_ci(labels, means, ci_low, ci_high,
                              ylabel='', title='', baseline=None,
                              baseline_label='matched baseline',
                              file_name='grouped_bars_ci', output_dir=None):
    """
    Bar chart with asymmetric (e.g. bootstrap) confidence intervals.

    Parameters
    ----------
    labels        : list[str] — one bar per label
    means         : list[float]
    ci_low        : list[float] — lower CI bound (absolute value, not delta)
    ci_high       : list[float] — upper CI bound
    ylabel        : str
    title         : str
    baseline      : float or None — horizontal reference line
    baseline_label: str
    file_name     : str
    output_dir    : Path or None
    """
    out = pathlib.Path(output_dir or OUTPUT_DIR)
    out.mkdir(parents=True, exist_ok=True)

    means = np.asarray(means, dtype=float)
    lo = means - np.asarray(ci_low, dtype=float)
    hi = np.asarray(ci_high, dtype=float) - means
    x = np.arange(len(labels))

    fig, ax = plt.subplots(1, 1, figsize=(1.5 + 1.0 * len(labels), 5))
    fig.patch.set_facecolor('white')
    fig.subplots_adjust(left=0.14, bottom=0.16, right=0.95, top=0.90)

    ax.bar(x, means, yerr=np.vstack([lo, hi]), color='#5DADE2', alpha=0.85,
           edgecolor='#2471A3', linewidth=0.9, capsize=5,
           error_kw=dict(ecolor='#1a1a2e', lw=1.3), zorder=2)
    if baseline is not None:
        ax.axhline(baseline, color='#C0392B', lw=1.8, linestyle='--', zorder=3)
        from matplotlib.lines import Line2D
        ax.legend(handles=[Line2D([0], [0], color='#C0392B', lw=1.8,
                                  linestyle='--', label=baseline_label)],
                  fontsize=10, prop=FONT, framealpha=0.7)

    ax.set_xticks(x); ax.set_xticklabels(labels, fontname='serif', fontsize=10)
    ax.set_ylabel(ylabel, fontname='serif', fontsize=12)
    ax.set_title(title, fontname='serif', fontsize=12)
    ax.tick_params(labelsize=10)
    ax.grid(True, axis='y', linestyle='--', alpha=0.4)

    fig.savefig(out / f'{file_name}.pdf', bbox_inches='tight')
    plt.close(fig)
    print(f'Saved  {out / file_name}.pdf')


# ─────────────────────────────────────────────────────────────────────────────
# Demo / self-test
# ─────────────────────────────────────────────────────────────────────────────

def _make_dummy_hist(bins=64):
    """Produce a synthetic HistogramResult-like object for demo purposes."""

    class _H:
        pass

    rng = np.random.default_rng(42)
    H = np.zeros((bins, bins), dtype=np.float32)
    # Three blobs
    for cx, cy, amp in [(0.3, 0.4, 500), (1.2, 1.0, 300), (2.5, 2.0, 200)]:
        xs = np.clip((rng.normal(cx, 0.08, amp) / 4.0 * bins).astype(int), 0, bins - 1)
        ys = np.clip((rng.normal(cy, 0.08, amp) / 4.0 * bins).astype(int), 0, bins - 1)
        np.add.at(H, (xs, ys), 1)
    h = _H()
    h.H = H
    e_x = np.linspace(0, 4.0, bins + 1)
    e_n = np.linspace(0, 4.0, bins + 1)
    h.x_edges = e_x
    h.n_edges = e_n
    h.x_centres = 0.5 * (e_x[:-1] + e_x[1:])
    h.n_centres = 0.5 * (e_n[:-1] + e_n[1:])
    h.total_voxels = int(H.sum())
    return h


def _demo():
    """Run all plotting functions with synthetic data."""
    print('\n=== DIANA plots demo ===\n')

    PROJECTION_COUNTS = [32, 64, 128, 256, 512, 1024, 2048]
    rng = np.random.default_rng(0)

    # --- phantom ---
    NY, NX = 128, 128
    label_2d = np.zeros((NY, NX), dtype=np.uint8)
    label_2d[20:80, 20:80] = 1
    label_2d[40:60, 40:60] = 2

    class _Mat:
        def __init__(self, sym, dens, col):
            self.symbol = sym; self.name = sym
            self.density_gcc = dens; self.color = col
        def mu_x_at(self, e): return self.density_gcc * 0.1 * (80.0 / e) ** 0.3

    materials = [
        _Mat('Air', 0.001, '#1a1a2e'),
        _Mat('HAp', 1.95,  '#E8B87A'),
        _Mat('Hem', 4.50,  '#C0392B'),
    ]

    plot_phantom_label_map(label_2d, materials, voxel_size_mm=0.05, geo_label='Demo')

    # --- single histogram ---
    hist = _make_dummy_hist()
    gt_positions = {'HAp': (1.2, 1.0), 'Hem': (2.5, 2.0)}
    plot_bimodal_histogram(hist, gt_positions=gt_positions,
                           n_proj=512, condition='clean')

    # --- marginals ---
    plot_xray_marginal(hist, n_proj=512, condition='clean')
    plot_neutron_marginal(hist, n_proj=512, condition='clean')

    # --- reconstruction slice ---
    vol_2d = rng.uniform(0, 3.0, (NY, NX)).astype(np.float32)
    for mod in ['xray', 'neutron']:
        plot_reconstruction_slice(vol_2d, modality=mod, n_proj=512,
                                  condition='clean', geo_label='Demo')

    # --- metric vs N_proj curves ---
    n_vals = PROJECTION_COUNTS
    ce_vals = [1.0 / (n ** 0.4) for n in n_vals]
    db_vals = [2.0 / (n ** 0.35) for n in n_vals]
    plot_CE_vs_nprojections(n_vals, ce_vals, geo_label='Demo', condition='clean')
    plot_DB_vs_nprojections(n_vals, db_vals, geo_label='Demo', condition='clean')

    syms = ['HAp', 'Hem', 'Org', 'Qtz']
    sigma_x = {s: [0.1 + 0.5 / (n ** 0.3) + rng.uniform(-0.01, 0.01)
                   for n in n_vals] for s in syms}
    sigma_n = {s: [0.08 + 0.4 / (n ** 0.3) + rng.uniform(-0.01, 0.01)
                   for n in n_vals] for s in syms}
    eps_k = {s: [0.15 + 0.3 / (n ** 0.35) + rng.uniform(-0.005, 0.005)
                 for n in n_vals] for s in syms}
    overlap = {('HAp', 'Hem'): [max(0, 0.5 / (n ** 0.4)) for n in n_vals],
               ('Org', 'Qtz'): [max(0, 0.3 / (n ** 0.35)) for n in n_vals]}

    plot_sigma_x_vs_nprojections(n_vals, sigma_x, geo_label='Demo', condition='clean')
    plot_sigma_n_vs_nprojections(n_vals, sigma_n, geo_label='Demo', condition='clean')
    plot_epsilon_vs_nprojections(n_vals, eps_k,   geo_label='Demo', condition='clean')
    plot_pairwise_overlap_vs_nprojections(n_vals, overlap, geo_label='Demo',
                                          condition='clean')

    # --- cross-sweep heatmap ---
    N = len(n_vals)
    cross_ce = np.array([[1.0 / ((nx * nn) ** 0.2)
                          for nn in n_vals] for nx in n_vals])
    cross_db = np.array([[2.0 / ((nx * nn) ** 0.18)
                          for nn in n_vals] for nx in n_vals])
    plot_cross_sweep_heatmap(cross_ce, n_vals, metric='CE')
    plot_cross_sweep_heatmap(cross_db, n_vals, metric='DB')

    # --- multi-geometry CE/DB clean vs dirty ---

    class _FakeTable:
        def __init__(self, ce, db):
            self.scalars = {'CE': ce, 'DB': db}

    fake_sweep = {}
    for gkey in ['G1_disc', 'G2_multi', 'G3_strat']:
        fake_sweep[gkey] = {}
        for cond, offset in [('clean', 0), ('dirty', 0.3)]:
            fake_sweep[gkey][cond] = {
                n: {'table_la': _FakeTable(
                    ce=1.0 / (n ** 0.4) + offset,
                    db=2.0 / (n ** 0.35) + offset * 0.5,
                )}
                for n in n_vals
            }

    plot_CE_all_geometries(fake_sweep, n_vals,
                           geo_labels=['G1: Disc', 'G2: Multi-Inclusion',
                                       'G3: Stratigraphy'])
    plot_DB_all_geometries(fake_sweep, n_vals,
                           geo_labels=['G1: Disc', 'G2: Multi-Inclusion',
                                       'G3: Stratigraphy'])

    # --- X-ray attenuation spectra ---
    import numpy as _np
    XRAY_E_KEV = [20., 30., 40., 50., 60., 70., 80., 90., 100., 120., 150., 200., 300.]
    plot_xray_attenuation_spectra(materials[1:], XRAY_E_KEV)

    print('\n=== Demo complete — all PDFs written to', OUTPUT_DIR, '===')


if __name__ == '__main__':
    import sys
    if '--demo' in sys.argv:
        _demo()
    else:
        print('Run with --demo to test all plotting functions.')
        print('Import this module into your notebook and call functions individually.')
