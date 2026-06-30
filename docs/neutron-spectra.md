# Neutron spectra and beam modes

By default the neutron projector uses thermal-neutron attenuation (the values baked
into each `Material`). The `neutron_spectra` module lets you go further: model cold and
polychromatic neutron beams, and recompute attenuation with the correct
energy dependence. This matters because cold neutrons change absorption contrast — and
the package is designed partly to study how the choice of beam affects material
separability.

## Pre-built beam modes

`NEUTRON_MODES` provides ready-made `NeutronBeam` objects:

| Key | Type | Energy | Wavelength | Use |
|---|---|---|---|---|
| `thermal` | monochromatic | 25.3 meV | 1.80 Å | matches the package baseline (the values in `Material.mu_n`) |
| `cold_5meV` | monochromatic | 5 meV | 4.04 Å | cold-source peak |
| `cold_2meV` | monochromatic | 2 meV | 6.40 Å | very cold |
| `ill_next` | polychromatic | ⟨~6 meV⟩ | ⟨~4.5 Å⟩ | ILL-NeXT cold guide spectrum |

```python
from neutron_xray_sim import NEUTRON_MODES
beam = NEUTRON_MODES["cold_5meV"]
```

## Building beams

| Function | Returns |
|---|---|
| `cold_mono_beam(E_meV=5.0)` | a monochromatic beam at the given energy |
| `cold_poly_beam(...)` | a polychromatic cold spectrum (Maxwellian + epithermal tail) |
| `ill_next_beam(...)` | the ILL-NeXT cold-guide spectrum |

A `NeutronBeam` carries its energy grid, the normalised flux weights `phi`, whether it
is monochromatic, and the flux-weighted mean energy and wavelength.

## Energy-dependent attenuation

To recompute attenuation for a specific beam rather than the thermal baseline:

| Function | Purpose |
|---|---|
| `mu_n_lut_for_beam(materials, beam)` | flux-weighted total neutron attenuation per material (cm⁻¹), as a lookup table |
| `mu_n_spectrum_lut(materials, beam, n_bins=None)` | per-energy-bin attenuation table for a polychromatic beam |
| `plot_spectra({name: beam, ...})` | overlay several beam spectra for inspection |

The physics applied:

- **Absorption** follows the 1/v law: σ_abs(E) = σ_abs(E₀)·√(E₀/E) with E₀ = 25.3 meV.
  Cold neutrons (lower E) are absorbed more strongly. The effect is large for
  absorbers like iron and small for light elements.
- **Coherent and incoherent scatter** are treated as energy-independent (bound-atom
  values).

**Consequence.** For hydrogen-dominated materials (water, HDPE, organics), cold versus
thermal makes little difference, because hydrogen's huge incoherent cross-section
dominates and is energy-independent. For absorbing elements the cold beam noticeably
raises neutron contrast. This is exactly the kind of trade-off you can quantify by
running the pipeline at several beam modes and comparing the
[cluster-quality metrics](histogram-analysis.md).

## A limitation to be aware of

Coherent scatter is modelled as energy-independent. Real cold-neutron transmission
shows **Bragg edges** — sharp steps in attenuation at wavelengths set by a crystal's
lattice spacings — which are not represented in the base model. The repository includes
notes on a planned wavelength-dependent extension (`DIANA_bragg_edge_extension.md`); the
current package treats the bound-atom coherent value as constant, which is the main
physical simplification in the neutron model.
