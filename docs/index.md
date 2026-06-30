# Documentation — Bimodal_simulation (`neutron_xray_sim`)

Welcome to the documentation for the dual-modality neutron / X-ray tomography
simulator. These pages can be read directly on GitHub, served as GitHub Pages, or
copied into the project wiki.

## Start here

If you are new, read these in order:

1. [Installation](installation.md) — get the package and its optional GPU backend running.
2. [Concepts](concepts.md) — the physics of dual-modality imaging and why the bimodal histogram is the central object.
3. [The simulation pipeline](pipeline.md) — how a phantom becomes a histogram, stage by stage, with the data contracts between stages.
4. [Example scripts](examples.md) — four runnable scripts that reproduce the main figures.

## Reference pages

| Page | Covers |
|---|---|
| [Phantoms](phantoms.md) | `PhantomData`, `PhantomBuilder`, the six preset phantoms, building custom geometries |
| [Materials](materials.md) | The material database, `material_from_formula`, `make_composite_material`, the physics of the coefficients |
| [Artifacts](artifacts.md) | Every field of `ArtifactConfig`, the factory presets, and what each artifact does to the histogram |
| [Reconstruction](reconstruction.md) | The nine algorithms, their parameters, and the ASTRA / CPU backends |
| [Histogram analysis](histogram-analysis.md) | The 2-D histogram, GMM fitting, segmentation, quality metrics, and artifact signatures |
| [Neutron spectra](neutron-spectra.md) | Thermal, cold, and ILL-NeXT beam models, and energy-dependent attenuation |
| [Importing real data](importing-data.md) | Turning your own segmented volumes into phantoms |
| [API reference](api-reference.md) | A compact listing of the public functions and classes per module |

## Conventions used throughout

- **Linear attenuation coefficients** are always in **cm⁻¹** (written μₓ for X-ray, μₙ for neutron).
- **Volumes** are stored in **(Nz, Nx, Ny)** order; coordinates are (z, x, y).
- **Sinograms** are passed around as dictionaries (`sino_dict`) whose key field
  `sino_lam` holds the line-integral (optical-depth) projections; see
  [the pipeline page](pipeline.md#data-contracts) for the full contract.
- **Thermal-neutron cross-sections** are referenced at 25.3 meV (λ ≈ 1.80 Å).
- The **X-ray energy grid** `XRAY_E_KEV` is 13 points from 20 keV to 300 keV.

## A note on `DIANA`

You will see a module named `diana_plots` and reference notes prefixed `DIANA_`.
*DIANA* is the internal name of the analysis project this simulator grew out of;
`diana_plots` provides publication-style figures (metric-versus-projection-count
curves, attenuation spectra, geometry sweeps). It is optional — the analysis API in
[`histogram.py`](histogram-analysis.md) is self-contained.
