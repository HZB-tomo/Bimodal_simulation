# Concepts: physics and the bimodal histogram

This page explains the ideas the package is built on. You do not need it to run a
simulation, but it makes the parameters and the output metrics meaningful.

## Why two modalities

A tomographic scan measures the **linear attenuation coefficient** μ of every voxel —
how strongly that material removes radiation from the beam. The crucial point is that
X-rays and thermal neutrons are attenuated by *different physics*:

- **X-rays** interact with the electron cloud. Attenuation rises steeply with atomic
  number Z and with density, so metals and bone mineral are very bright while light
  organic matter is nearly transparent.
- **Thermal neutrons** interact with the nucleus. Attenuation is governed by nuclear
  cross-sections that vary erratically across the periodic table. **Hydrogen** in
  particular has an enormous incoherent scattering cross-section, so water, polymers,
  and organic tissue are bright to neutrons even though they are dim to X-rays.

The two contrasts are *complementary*: where one modality saturates or starves, the
other often sees clearly. The package's preset phantoms are chosen to dramatise this —
for example a titanium implant beside bone (X-rays starve at the metal, neutrons do
not), or a battery whose electrolyte and separator are invisible to X-rays but obvious
to neutrons.

## The bimodal histogram H(μₓ, μₙ)

If you measure the same object with both modalities and register the two volumes, each
voxel has a pair of values *(μₓ, μₙ)*. Plotting the 2-D histogram of these pairs gives
the **bimodal histogram**: a joint density over X-ray attenuation (one axis) and
neutron attenuation (the other).

In an ideal, noise-free, perfectly registered scan, every material occupies a single
tight cluster, because all voxels of that material share the same *(μₓ, μₙ)*. The
collection of clusters is a fingerprint of the sample's composition. Materials that
overlap on one axis are usually separated on the other — that is the whole point of
acquiring both.

Real acquisitions blur, shift, split, and smear these clusters. The central question
this package is designed to study is: **how does each acquisition imperfection deform
the bimodal histogram, and by how much?** Every cluster that blurs into its neighbour
is a material you can no longer tell apart.

## What degrades the clusters

The [artifacts](artifacts.md) module injects the realistic culprits:

- **Counting noise** widens every cluster isotropically (more dose → tighter clusters).
- **Beam hardening** (a consequence of the polychromatic X-ray spectrum) pulls the
  X-ray attenuation of dense materials, bending and shifting clusters along the μₓ axis.
- **Scatter** adds a low-frequency background that biases attenuation values.
- **Detector blur** (point-spread function) mixes neighbouring voxels, smearing
  clusters along the line joining them.
- **Ring artifacts** add structured offsets.
- **Misalignment** between the two reconstructed volumes is the most damaging for the
  bimodal method specifically: a voxel's X-ray value is paired with a *different*
  voxel's neutron value, so clusters smear into **horizontal streaks**. This is the
  signature [example 02](examples.md) sweeps out.

## Measuring the damage

Because the phantom is synthetic, the *ground-truth* position of every material in
*(μₓ, μₙ)* space is known exactly. The package fits a Gaussian mixture model to the
reconstructed histogram, matches each component to a known material, and reports:

- **Centroid error** — how far each cluster moved from its true position (cm⁻¹).
- **Cluster spread** σₓ, σₙ — how much each cluster blurred along each axis.
- **Davies–Bouldin index** — overall cluster separability (lower is better).
- **Pairwise overlap** — for neighbouring materials, the fraction of voxels that get
  misclassified.

These are defined precisely in [Histogram analysis](histogram-analysis.md). Together
they turn the visual question "can I still tell these materials apart?" into numbers
you can plot against dose, projection count, algorithm, or beam energy.

## Units and reference conditions

- All attenuation values are **cm⁻¹**.
- X-ray coefficients are tabulated on the 13-point grid `XRAY_E_KEV` (20–300 keV) and
  log-interpolated to any energy in between.
- Neutron cross-sections are bound-atom values at the **thermal reference energy of
  25.3 meV** (λ ≈ 1.80 Å), split into absorption, coherent-scatter, and
  incoherent-scatter components. Cold-neutron beams rescale the absorption term by the
  1/v law; see [Neutron spectra](neutron-spectra.md).
