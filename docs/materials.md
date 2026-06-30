# Materials

The material database supplies the physical contrast that drives the simulation: for
every material, an X-ray attenuation curve over energy and the three thermal-neutron
attenuation components.

## The `Material` dataclass

| Field | Meaning |
|---|---|
| `name`, `symbol` | human name and short plot label |
| `density_gcc` | density, g/cm³ (alias `rho`) |
| `mu_n_abs` | neutron absorption attenuation, cm⁻¹ |
| `mu_n_coh` | neutron coherent-scatter attenuation, cm⁻¹ |
| `mu_n_inc` | neutron incoherent-scatter attenuation, cm⁻¹ |
| `_mu_x_table` | X-ray attenuation at the 13 `XRAY_E_KEV` energies, cm⁻¹ |
| `color` | matplotlib colour for plots |

Derived properties and methods:

- `mu_n` — total neutron attenuation = `mu_n_abs + mu_n_coh + mu_n_inc`.
- `mu_n_scatter` — `mu_n_coh + mu_n_inc`.
- `mu_x_at(energy_keV)` — log-interpolated X-ray attenuation at one energy.
- `mu_x_array(energies_keV)` — the same for an array of energies.

> The neutron total is computed from the three components; there is no single backing
> field for it. If you need to override a material's neutron attenuation (e.g. to
> inject a beam-specific value), set the three component fields, not `mu_n`.

## Built-in materials

Available as both `MATERIALS["..."]` and module-level constants:

| Constant | Material | Notable contrast |
|---|---|---|
| `AIR` | air | ~0 attenuation in both modalities |
| `WATER` | water (H₂O) | strong neutron, weak X-ray (hydrogen) |
| `ALUMINUM` | Al | light metal, modest both |
| `HDPE` | polyethylene | very strong neutron, weak X-ray (hydrogen-rich) |
| `IRON` | Fe | strong X-ray |
| `TITANIUM` | Ti | strong X-ray, low neutron |
| `COPPER` | Cu | strong X-ray |
| `LEAD` | Pb | very strong X-ray |
| `BONE` | hydroxyapatite | bone-mineral X-ray contrast |
| `TUNGSTEN` | W | extreme X-ray (photon starvation) |
| `ZINC` | Zn | strong X-ray |

The database also includes battery-relevant materials added via formula: lithium
metal, LiPF₆ electrolyte, steel, graphite, LFP, NMC811/532/622, LCO, and PE/PP
separators (with and without electrolyte). These power the battery phantoms.

## Building a material from a formula

`material_from_formula` computes both contrasts from a chemical formula and a density:

```python
from neutron_xray_sim import material_from_formula

quartz = material_from_formula(
    name="Quartz", symbol="SiO2",
    formula="SiO2", density_gcc=2.65,
    color="#cccc99",
)
```

How it works: the formula is parsed into element counts; X-ray attenuation is the
mass-fraction-weighted sum of the NIST element curves times density; neutron
attenuation sums the absorption, coherent, and incoherent cross-sections (barns) over
the atoms and scales by number density. An optional `incoherent_scale` factor lets you
down-weight the incoherent term — useful for some liquids and polymers where the bound
cross-section over-predicts.

**Formula syntax.** Standard chemical notation with optional decimal subscripts:
`SiO2`, `KAlSi3O8`, `Fe0.98Ni0.02`, `LiNi0.8Mn0.1Co0.1O2`. Supported elements are
those present in both `ATOMIC_MASS` / `NEUTRON_XS` and `lib/xray_data/`: H, Li, C, N,
O, F, Mg, Al, Si, P, S, Cl, K, Ca, Fe, Co, Ni, Mn, In. For minerals with hydroxyl or
bracketed groups, expand them fully (e.g. hydroxyapatite as `Ca10P6O26H2`, fluorapatite
as `Ca5P3O12F`, dolomite as `CaMgC2O6`).

## Building a composite (mixture) material

When a material is a mixture of phases — fossilised bone, a sedimentary matrix, a
multi-mineral aggregate — use `make_composite_material`. You give it the bulk density
and a list of `(formula, weight_fraction, end_member_density)` tuples:

```python
from neutron_xray_sim.materials import make_composite_material

bone = make_composite_material(
    name="Fossilised bone", symbol="Bone",
    bulk_density_gcc=2.00,
    components=[
        ("Ca5P3O12F", 0.70, 3.20),   # fluorapatite
        ("C10H13NO4", 0.15, 1.35),   # dry collagen proxy
        ("CaCO3",     0.12, 2.71),   # calcite
        ("SiO2",      0.03, 2.65),   # silicate proxy
    ],
)
```

The weight fractions must sum to 1 (within ±0.02; this is asserted). X-ray attenuation
is the mass-fraction-weighted μ/ρ times bulk density; neutron attenuation treats each
component as occupying its mass fraction of the bulk density and sums the macroscopic
cross-sections. The `end_member_density` entry is informational and does not affect the
calculation.

## The physics behind the numbers

- **X-ray.** Coefficients come from NIST XCOM mass-attenuation tables on a 13-point
  energy grid `XRAY_E_KEV = [20, 30, 40, 50, 60, 70, 80, 90, 100, 120, 150, 200, 300]`
  keV, log-interpolated between points. Units throughout are cm⁻¹ (μ/ρ × density).
- **Neutron.** Bound-atom cross-sections (absorption, coherent, incoherent) at the
  thermal reference energy 25.3 meV, from Sears (1992) and ENDF/B-VIII.0. Hydrogen's
  incoherent cross-section (~80 barns) dominates the neutron contrast of any
  hydrogen-bearing material. Cold-neutron beams rescale the **absorption** term by the
  1/v law; the scatter terms are treated as energy-independent. See
  [Neutron spectra](neutron-spectra.md).

## Generating an X-ray spectrum

`xray_spectrum(kVp, filter_mm_Al, filter_mm_Cu, n_bins)` returns
`(energies_keV, weights)` for a simplified bremsstrahlung tube spectrum (Kramers' law)
after aluminium/copper filtration, normalised to unit total. This is what the
polychromatic projector samples; harder filtration or higher kVp shifts the spectrum up
in energy and changes the amount of beam hardening you will see.
