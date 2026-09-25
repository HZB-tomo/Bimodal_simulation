# DIANA tutorials

Seven self-contained notebooks that teach the package from scratch. Each one runs on a
laptop **CPU in a few minutes** (no GPU or data download needed) and ends with an exercise.

```bash
pip install -e ".[notebooks]"      # from the repository root
jupyter lab tutorials/
```

| # | Notebook | You will learn |
|---|---|---|
| 1 | [Getting started](01_getting_started.ipynb) | the dual-modality idea, a full simulation in a few lines, the bimodal histogram, CE / DB metrics |
| 2 | [Materials and phantoms](02_materials_and_phantoms.ipynb) | the material database, new materials from formulas, `PhantomBuilder`, presets, importing segmented volumes |
| 3 | [Projection and artifacts](03_projection_and_artifacts.ipynb) | sinograms, X-ray spectra and beam hardening, every artifact, the Poisson noise model |
| 4 | [Reconstruction](04_reconstruction.ipynb) | FBP vs iterative algorithms, filters, angular sampling, error against ground truth |
| 5 | [Histogram analysis](05_histogram_analysis.ipynb) | histograms, GMM clustering, segmentation, artifact signatures, the three metric families |
| 6 | [Artifact studies](06_artifact_studies.ipynb) | parameter sweeps, the artifact survey, caching results with `SimCache` |
| 7 | [Neutron spectra](07_neutron_spectra.ipynb) | cold and polychromatic beams, beam-dependent contrast, Bragg edges |

Follow them in order the first time; later ones assume the vocabulary of earlier ones.

After the tutorials, the research notebooks in [`../notebooks`](../notebooks) show full-scale
studies (these typically need a GPU), and [`../docs`](../docs/index.md) is the reference.

**Contributing a tutorial:** keep it CPU-friendly (N ≤ 64), explain before you compute,
commit it without outputs (`pre-commit` handles this), and check it runs with
`pytest --nbmake tutorials/`.
