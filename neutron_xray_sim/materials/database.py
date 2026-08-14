"""
neutron_xray_sim.materials.database
───────────────────────────────────
The built-in material catalogue, as data.

**This is the file to edit when adding a material to the package.**  Append a
:class:`~neutron_xray_sim.materials.core.MaterialSpec` to :data:`BUILTIN_SPECS`
with a ``reference`` for its numbers and one or more ``tags``, and it becomes
available everywhere as ``MATERIALS["<key>"]``.  Nothing else needs to change.

For materials specific to one study or beamline, prefer a JSON/YAML file loaded
with ``MATERIALS.load_file(...)`` — see ``docs/materials.md``.  Reserve this
file for materials that several projects will share.

Two ways to specify a material
──────────────────────────────
*Derived* — give ``formula`` (or ``components``) and ``density_gcc``; both
channels are computed from the NIST and Sears element data.  Preferred, because
the result is reproducible from cited inputs.

*Tabulated* — give ``xray_mu_cm`` and/or ``neutron_mu_cm`` directly.  Needed
where the package does not ship a NIST file for one of the elements: Al, Ti,
Cu, Pb, Zn, W and Ca currently have no ``lib/xray_data/<El>.txt``, so materials
containing them must supply their X-ray table explicitly.  Run
``MATERIALS.audit()`` or ``element_data_status()`` to see the current state.

Known data issues
─────────────────
``MATERIALS.audit()`` currently flags three of the tabulated X-ray curves as
non-physical: attenuation must fall monotonically with energy except at an
absorption edge, and these do not.  The numbers are preserved **unchanged** so
that already-published results stay reproducible — replacing them requires a
verified NIST export, which is a data task, not a code change.

``lead``
    Places the K-edge jump at 70→80 keV, but the Pb K-edge is at 88.0 keV.
    Several entries also disagree with NIST XCOM (e.g. 7.58 cm⁻¹ at 300 keV
    against 4.58 from XCOM).  Affects the ``industrial`` phantom only if lead
    is added to it; no built-in phantom uses lead today.
``hdpe``
    Rises across 60→70→80 keV (0.153 → 0.157 → 0.167 cm⁻¹) where NIST gives a
    monotone fall (≈0.163 → 0.158 → 0.153).  HDPE is the matrix of the
    ``composite``, ``industrial`` and ``HDPE_composite`` phantoms, so this
    biases the X-ray channel by up to ~9 % near 80 keV.
``bone``
    Rises across 90→100 keV (0.421 → 0.438 cm⁻¹).

Fixing them means dropping a verified NIST export into ``lib/xray_data/`` and
converting the spec to the derived path — see ``docs/materials.md``.
"""

from __future__ import annotations

from typing import List

from .core import MaterialSpec

__all__ = ["BUILTIN_SPECS"]

_SEARS = "Sears, Neutron News 3(3):26-37 (1992), 25.3 meV"
_XCOM = "NIST XCOM total attenuation with coherent scattering"


BUILTIN_SPECS: List[MaterialSpec] = [

    # ── Reference and structural materials ───────────────────────────────────
    MaterialSpec(
        key="air", name="Air", symbol="Air", density_gcc=1.205e-3,
        xray_mu_cm=(1e-4,) * 13,
        neutron_mu_cm=(0.0, 0.0, 0.0),
        color="#FFFFFF", tags=("reference",),
        reference="Dry air at 20 °C, 1 atm; attenuation treated as negligible.",
    ),
    MaterialSpec(
        key="water", name="Water", symbol="H₂O", density_gcc=1.00,
        xray_mu_cm=(0.811, 0.380, 0.268, 0.228, 0.206, 0.196,
                    0.184, 0.178, 0.171, 0.163, 0.151, 0.137, 0.119),
        neutron_mu_cm=(0.022, 0.259, 1.099),
        color="#4488CC", tags=("reference", "hydrogenous"),
        reference=f"{_XCOM}; neutron components from {_SEARS}.",
    ),
    MaterialSpec(
        key="aluminum", name="Aluminum", symbol="Al", density_gcc=2.70,
        xray_mu_cm=(9.29, 2.47, 1.06, 0.616, 0.413, 0.293,
                    0.278, 0.258, 0.239, 0.219, 0.202, 0.193, 0.187),
        neutron_mu_cm=(0.014, 0.083, 0.001),
        color="#C8C8C8", tags=("metal",),
        reference=f"{_XCOM}; neutron components from {_SEARS}. Tabulated "
                  "because lib/xray_data/Al.txt is not shipped.",
    ),
    MaterialSpec(
        key="hdpe", name="HDPE", symbol="HDPE", density_gcc=0.95,
        xray_mu_cm=(0.288, 0.181, 0.159, 0.155, 0.153, 0.157,
                    0.167, 0.165, 0.160, 0.152, 0.141, 0.128, 0.111),
        neutron_mu_cm=(0.027, 0.369, 1.784),
        color="#88CC88", tags=("polymer", "hydrogenous"),
        reference=f"{_XCOM} for (C₂H₄)ₙ; neutron components from {_SEARS}.",
    ),
    MaterialSpec(
        key="iron", name="Iron", symbol="Fe", density_gcc=7.87,
        xray_mu_cm=(294.0, 80.3, 34.2, 17.5, 10.1, 6.41,
                    4.12, 3.05, 2.38, 1.68, 1.26, 1.05, 0.882),
        neutron_mu_cm=(0.217, 0.910, 0.033),
        color="#666666", tags=("metal",),
        reference=f"{_XCOM}; neutron components from {_SEARS}.",
    ),
    MaterialSpec(
        key="titanium", name="Titanium", symbol="Ti", density_gcc=4.51,
        xray_mu_cm=(92.3, 24.8, 9.97, 5.28, 3.11, 2.07,
                    1.48, 1.14, 0.943, 0.770, 0.699, 0.636, 0.568),
        neutron_mu_cm=(0.345, 0.175, 0.120),
        color="#AA9999", tags=("metal", "implant"),
        reference=f"{_XCOM}; neutron components from {_SEARS}.",
    ),
    MaterialSpec(
        key="copper", name="Copper", symbol="Cu", density_gcc=8.96,
        xray_mu_cm=(375.0, 110.0, 45.4, 22.5, 12.5, 7.87,
                    4.92, 3.70, 2.97, 2.10, 1.55, 1.26, 1.03),
        neutron_mu_cm=(0.313, 0.760, 0.047),
        color="#DD8833", tags=("metal", "battery"),
        reference=f"{_XCOM}; neutron components from {_SEARS}.",
    ),
    MaterialSpec(
        key="lead", name="Lead", symbol="Pb", density_gcc=11.35,
        xray_mu_cm=(340.0, 124.0, 62.4, 34.4, 20.6, 13.2,
                    22.8, 63.0, 50.7, 27.5, 16.4, 11.1, 7.58),
        neutron_mu_cm=(0.006, 0.366, 0.001),
        k_edge_keV=88.0,
        color="#556677", tags=("metal", "shielding"),
        reference="SUSPECT: this table places the K-edge jump at 70–80 keV but "
                  "the Pb K-edge is 88.0 keV, and several entries disagree with "
                  "NIST XCOM. Preserved for reproducibility — replace with a "
                  "verified export before quantitative use. See MATERIALS.audit().",
    ),
    MaterialSpec(
        key="bone", name="Bone (HAp)", symbol="HAp", density_gcc=1.92,
        xray_mu_cm=(6.37, 1.94, 0.980, 0.666, 0.557, 0.481,
                    0.469, 0.421, 0.438, 0.421, 0.400, 0.373, 0.322),
        neutron_mu_cm=(0.038, 0.130, 0.392),
        color="#F0DDB0", tags=("biological",),
        reference=f"Cortical bone, ICRU-44 composition; {_XCOM}; "
                  f"neutron components from {_SEARS}.",
    ),
    MaterialSpec(
        key="tungsten", name="Tungsten", symbol="W", density_gcc=19.3,
        xray_mu_cm=(1035.0, 313.0, 140.0, 74.4, 45.0, 88.5,
                    88.0, 64.0, 50.0, 30.5, 18.0, 11.0, 6.50),
        neutron_mu_cm=(1.157, 0.301, 0.102),
        k_edge_keV=69.5,
        color="#222244", tags=("metal", "shielding"),
        reference=f"{_XCOM}; the 60→70 keV rise is the W K-edge at 69.5 keV. "
                  f"Neutron components from {_SEARS}.",
    ),
    MaterialSpec(
        key="zinc", name="Zinc", symbol="Zn", density_gcc=7.13,
        xray_mu_cm=(215.0, 66.2, 28.2, 14.1, 7.97, 5.09,
                    3.25, 2.43, 1.97, 1.41, 1.05, 0.862, 0.713),
        neutron_mu_cm=(0.073, 0.272, 0.005),
        color="#BBBB44", tags=("metal", "battery"),
        reference=f"{_XCOM}; neutron components from {_SEARS}.",
    ),

    # ── Battery materials (derived from formula) ─────────────────────────────
    MaterialSpec(
        key="lithium", name="Lithium Metal", symbol="Li", density_gcc=0.534,
        formula="Li", color="#B0B0B0", tags=("metal", "battery"),
        reference="Derived from formula; Li absorption dominates (σ_abs = 70.5 b).",
    ),
    MaterialSpec(
        key="electrolyte_lipf6_1m", name="1 M LiPF6 Organic Electrolyte",
        symbol="LiPF6-sol", density_gcc=1.20,
        formula="Li1P1F6C5H10O3", incoherent_scale=0.5,
        color="#66CCFF", tags=("liquid", "battery", "hydrogenous"),
        reference="1 M LiPF₆ in EC:DMC, treated as a single averaged formula. "
                  "incoherent_scale=0.5 accounts for molecular motion in the "
                  "liquid, which the bound-atom cross section over-predicts.",
    ),
    MaterialSpec(
        key="steel", name="Steel (Fe-Ni)", symbol="Steel", density_gcc=7.85,
        formula="Fe0.98Ni0.02", color="#777777", tags=("metal",),
        reference="Low-alloy steel approximated as 98 wt% Fe / 2 wt% Ni.",
    ),
    MaterialSpec(
        key="graphite", name="Graphite", symbol="Graphite", density_gcc=2.26,
        formula="C", color="#444444", tags=("battery", "anode"),
        reference="Derived from formula at crystalline graphite density.",
    ),
    MaterialSpec(
        key="lfp", name="Lithium Iron Phosphate", symbol="LFP", density_gcc=3.60,
        formula="LiFePO4", color="#6B8E23", tags=("battery", "cathode"),
        reference="Derived from formula; olivine LiFePO₄.",
    ),
    MaterialSpec(
        key="nmc811", name="NMC811", symbol="NMC811", density_gcc=4.80,
        formula="LiNi0.8Mn0.1Co0.1O2", color="#CC6677",
        tags=("battery", "cathode"),
        reference="Derived from formula; layered oxide, crystallographic density.",
    ),
    MaterialSpec(
        key="nmc622", name="NMC622", symbol="NMC622", density_gcc=4.75,
        formula="LiNi0.6Mn0.2Co0.2O2", color="#BB5577",
        tags=("battery", "cathode"), reference="Derived from formula.",
    ),
    MaterialSpec(
        key="nmc532", name="NMC532", symbol="NMC532", density_gcc=4.70,
        formula="LiNi0.5Mn0.3Co0.2O2", color="#DD8899",
        tags=("battery", "cathode"), reference="Derived from formula.",
    ),
    MaterialSpec(
        key="lco", name="Lithium Cobalt Oxide", symbol="LCO", density_gcc=5.05,
        formula="LiCoO2", color="#3366AA", tags=("battery", "cathode"),
        reference="Derived from formula; Co absorption is significant (37.2 b).",
    ),
    MaterialSpec(
        key="separator_pe", name="PE Separator", symbol="PE-sep", density_gcc=0.94,
        formula="C2H4", incoherent_scale=0.35,
        color="#EEEEAA", tags=("polymer", "battery", "hydrogenous"),
        reference="Porous polyethylene; incoherent_scale=0.35 for the porous, "
                  "partly mobile polymer network.",
    ),
    MaterialSpec(
        key="separator_pp", name="PP Separator", symbol="PP-sep", density_gcc=0.90,
        formula="C3H6", incoherent_scale=0.35,
        color="#FFDDAA", tags=("polymer", "battery", "hydrogenous"),
        reference="Porous polypropylene; see separator_pe for incoherent_scale.",
    ),
    MaterialSpec(
        key="separator_pe_electrolyte",
        name="PE Separator + LiPF6 Electrolyte", symbol="separator + electrolyte",
        density_gcc=1.05, formula="C7H14O3Li1P1F6", incoherent_scale=0.5,
        color="#BFEFFF", tags=("battery", "hydrogenous"),
        reference="Electrolyte-soaked PE separator as a single averaged formula.",
    ),
]
