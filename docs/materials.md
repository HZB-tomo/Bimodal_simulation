# Materials

The material database supplies the physical contrast that drives the simulation: for
every material, an X-ray attenuation curve over energy and the three thermal-neutron
attenuation components.

Since v1.2 `materials` is a package with one job per module:

| Module | Contents |
|---|---|
| `materials/elements.py` | atomic masses, neutron cross sections, NIST table reader |
| `materials/core.py` | `Material`, `MaterialSpec`, `build_material`, the validator |
| `materials/registry.py` | `MaterialRegistry` and the package-wide `MATERIALS` |
| `materials/database.py` | the built-in catalogue, as data — **edit this to add a material** |

Everything is re-exported, so `from neutron_xray_sim.materials import MATERIALS,
material_from_formula` works exactly as before.

## The `Material` dataclass

| Field | Meaning |
|---|---|
| `name`, `symbol` | human name and short plot label |
| `density_gcc` | density, g/cm³ (alias `rho`) |
| `mu_n_abs` | neutron absorption attenuation, cm⁻¹ |
| `mu_n_coh` | neutron coherent-scatter attenuation, cm⁻¹ |
| `mu_n_inc` | neutron incoherent-scatter attenuation, cm⁻¹ |
| `mu_x_table` | X-ray attenuation at the 13 `XRAY_E_KEV` energies, cm⁻¹ |
| `color` | matplotlib colour for plots |
| `key`, `reference`, `tags` | registry key, citation, grouping labels |

Derived properties and methods:

- `mu_n` — total neutron attenuation = `mu_n_abs + mu_n_coh + mu_n_inc`.
- `mu_n_scatter` — `mu_n_coh + mu_n_inc`.
- `mu_x_at(energy_keV)` — log-log-interpolated X-ray attenuation at one energy.
- `mu_x_array(energies_keV)` — the same for an array of energies.
- `as_dict()` — a plain-Python view for JSON export and metadata sidecars.

> The neutron total is computed from the three components; there is no single backing
> field for it. If you need to override a material's neutron attenuation (e.g. to
> inject a beam-specific value), set the three component fields, not `mu_n`.

> `_mu_x_table` still works as an alias for `mu_x_table`, so older scripts keep
> running. New code should use the public name.

Materials compare and hash **by identity**, not by value. Two materials with the same
numbers are two distinct phases, and `material in some_list` is safe — previously that
comparison reached a NumPy field and raised *"truth value of an array is ambiguous"*.

## Using the registry

`MATERIALS` behaves like the dict it replaced, and adds the operations a shared
codebase needs:

```python
from neutron_xray_sim import MATERIALS

MATERIALS["hdpe"]                   # a Material
"steel" in MATERIALS                # membership
list(MATERIALS.keys())              # every key

MATERIALS.names(tag="cathode")      # filter by tag
MATERIALS.tags()                    # every tag in use
MATERIALS.search("separator")       # substring match on key/name/symbol
print(MATERIALS.table())            # readable overview for a method section
MATERIALS.spec("nmc811")            # the recipe, for provenance
MATERIALS.audit()                   # data-quality report
```

Materials are built on **first access** and cached, so importing the package is free
and a missing element file only raises when that material is actually used.

## Built-in materials

Available as `MATERIALS["..."]` and as module-level constants:

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

Plus battery materials derived from formula: `LITHIUM`, `ELECTROLYTE_LIPF6_1M`,
`STEEL`, `GRAPHITE`, `LFP`, `NMC811`, `NMC622`, `NMC532`, `LCO`, and the PE/PP
separators with and without electrolyte.

## Adding a material

### For everyone: `materials/database.py`

Append a `MaterialSpec`. A spec declares **where each channel's numbers come from**,
and the two channels are independent:

| Channel | Sources (pick one) |
|---|---|
| X-ray | `formula` / `components` → derived from NIST element tables |
| | `xray_mu_cm` → explicit 13-point table, cm⁻¹ |
| Neutron | `formula` / `components` → derived from Sears cross sections |
| | `neutron_mu_cm` → explicit `(abs, coh, inc)`, cm⁻¹ |

```python
MaterialSpec(
    key="quartz", name="Quartz", symbol="SiO2",
    density_gcc=2.65, formula="SiO2",
    color="#cccc99", tags=("mineral",),
    reference="Deer, Howie & Zussman (2013), 3rd ed.",
)
```

Everything goes through `build_material`, which validates before computing: exactly
one source per channel, a positive density, a 13-entry table, weight fractions summing
to 1 ± 0.02. Physically implausible X-ray tables produce a warning naming the energies
involved.

### For one project: a file

Keep project-specific materials next to the analysis, under version control, out of
the package:

```json
[
  {"key": "petg", "name": "PETG", "symbol": "PETG", "density_gcc": 1.27,
   "formula": "C10H8O4", "tags": ["polymer"],
   "reference": "manufacturer datasheet, 2026-02"}
]
```

```python
MATERIALS.load_file("my_beamtime/materials.json")     # JSON always; YAML with PyYAML
MATERIALS.save_file("archive/materials_used.json")    # archive exactly what a run used
```

Registering a key that already exists raises rather than silently shadowing a built-in
— pass `overwrite=True` if that is really what you want.

### At runtime

```python
from neutron_xray_sim.materials import MaterialSpec

MATERIALS.register(MaterialSpec(
    key="steel_316l", name="316L stainless", symbol="316L",
    density_gcc=7.99, formula="Fe0.68Ni0.12Si0.01",
    reference="nominal composition",
))
```

## Building a material directly

`material_from_formula` and `make_composite_material` are thin wrappers over
`build_material`, kept because they read well in notebooks:

```python
from neutron_xray_sim import material_from_formula, make_composite_material

quartz = material_from_formula("Quartz", "SiO2", "SiO2", 2.65, color="#cccc99")

bone = make_composite_material(
    name="Fossilised bone", symbol="Bone", bulk_density_gcc=2.00,
    components=[
        ("Ca5P3O12F", 0.70, 3.20),   # fluorapatite
        ("C10H13NO4", 0.15, 1.35),   # dry collagen proxy
        ("CaCO3",     0.12, 2.71),   # calcite
        ("SiO2",      0.03, 2.65),   # silicate proxy
    ],
)
```

X-ray attenuation is the mass-fraction-weighted μ/ρ times bulk density; neutron
attenuation treats each component as occupying its mass fraction of the bulk density
and sums the macroscopic cross sections. `end_member_density` is recorded for
provenance and does not enter the calculation.

**Formula syntax.** Flat chemical notation with optional decimal subscripts: `SiO2`,
`KAlSi3O8`, `Fe0.98Ni0.02`, `LiNi0.8Mn0.1Co0.1O2`. Brackets and hydrate dots are not
supported — expand them (`Ca10P6O26H2` for hydroxyapatite, `Ca5P3O12F` for
fluorapatite, `CaMgC2O6` for dolomite). An unparseable formula says so explicitly
rather than silently dropping atoms.

`incoherent_scale` down-weights the incoherent neutron term. The bound-atom cross
section over-predicts scattering in liquids and soft polymers where molecular motion
is not frozen; 0.3–0.6 is typical there, 1.0 for dense inorganic solids.

## Element data

Formula-derived materials need a NIST table per element. Check what is installed:

```python
from neutron_xray_sim.materials import element_data_status, available_elements
[el for el, ok in element_data_status().items() if not ok]
```

The package currently ships **H, Li, C, O, F, P, Mn, Fe, Co, Ni**. Materials
containing Al, Ti, Cu, Zn, W, Pb, Si, Ca, … therefore use the tabulated X-ray path.
Asking for a formula with a missing element raises `MissingElementDataError` naming
the file, the download URL, and the elements that *are* available.

To add one:

1. Add entries to `ATOMIC_MASS` and `NEUTRON_XS` in `materials/elements.py`.
2. Export the element from [NIST XCOM](https://physics.nist.gov/PhysRefData/Xcom/html/xcom1.html),
   requesting *total attenuation with coherent scattering*, and save the plain text as
   `neutron_xray_sim/lib/xray_data/<Symbol>.txt`. A two-column
   `E[MeV]  mu/rho[cm2/g]` file is also accepted.
3. Confirm with `element_data_status()`.

## Data quality

`MATERIALS.audit()` re-checks every spec and reports anything implausible. As of
v1.2.0 it flags three of the shipped X-ray tables, whose values are **not**
self-consistent with NIST:

| Material | Problem | Impact |
|---|---|---|
| `hdpe` | rises across 60→70→80 keV where NIST falls monotonically | HDPE is the matrix of the `composite`, `industrial` and `HDPE_composite` phantoms; biases μ_x by up to ~9 % near 80 keV |
| `bone` | rises across 90→100 keV | affects the `bone_implant` phantom |
| `lead` | K-edge jump placed at 70–80 keV instead of 88.0 keV; several entries disagree with XCOM | no built-in phantom uses lead |

The numbers are preserved unchanged so that already-published results stay
reproducible. Fixing them is a data task: drop a verified NIST export into
`lib/xray_data/` and switch the spec to the derived path.

## The physics behind the numbers

- **X-ray.** NIST XCOM mass-attenuation tables, resampled onto the 13-point grid
  `XRAY_E_KEV = [20, 30, 40, 50, 60, 70, 80, 90, 100, 120, 150, 200, 300]` keV.
  Resampling and later interpolation are **log-log**, which is what NIST recommends:
  photoelectric attenuation follows an approximate power law, so a straight line in
  log-log space is a far better local model than one in linear space. (Before v1.2 the
  code interpolated linearly while the docs claimed log — off-node grid points differed
  by up to ~9 % for medium-Z elements.) Absorption edges appear as duplicated energies
  in NIST tables and are handled explicitly. Units throughout are cm⁻¹ (μ/ρ × density).
- **Neutron.** Bound-atom cross sections (absorption, coherent, incoherent) at the
  thermal reference energy 25.3 meV, from Sears (1992). Hydrogen's incoherent cross
  section (~80 barn) dominates the neutron contrast of any hydrogen-bearing material.
  Cold-neutron beams rescale the **absorption** term by the 1/v law; the scatter terms
  are treated as energy-independent. See [Neutron spectra](neutron-spectra.md).

## Generating an X-ray spectrum

`xray_spectrum(kVp, filter_mm_Al, filter_mm_Cu, n_bins)` returns
`(energies_keV, weights)` for a simplified bremsstrahlung tube spectrum (Kramers' law)
after aluminium/copper filtration, normalised to unit total. This is what the
polychromatic projector samples; harder filtration or higher kVp shifts the spectrum
up in energy and changes the amount of beam hardening you will see. Pass
`materials=` to use a different filter database — useful in tests.
