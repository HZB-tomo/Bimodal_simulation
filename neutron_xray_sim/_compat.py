"""
Backwards-compatible module paths.

Version 2.0 reorganised the package into subpackages (``physics``,
``phantoms``, ``acquisition``, ``reconstruction``, ``analysis``,
``plotting``).  So that existing scripts and notebooks keep working, the old
flat module names are registered as aliases:

    import neutron_xray_sim.histogram            # still works
    from neutron_xray_sim.artifacts import ArtifactConfig

A one-to-one rename is aliased to the new module object itself; the two old
modules that were split (``phantom`` and ``histogram``) become lightweight
modules re-exporting everything from their successors.

New code should import from the package top level or the new subpackages.
"""

from __future__ import annotations

import importlib
import sys
import types

_PKG = __name__.rsplit(".", 1)[0]

#: old module name → new module path(s), relative to the package
LEGACY_MODULES = {
    "materials":                ["physics.materials"],
    "neutron_spectra":          ["physics.neutron_spectra"],
    "ncrystal_bragg":           ["physics.ncrystal_bragg"],
    "phantom":                  ["phantoms.base", "phantoms.presets"],
    "volume_importer":          ["phantoms.importer"],
    "nmc_phantom":              ["phantoms.nmc"],
    "density_sweep":            ["phantoms.density_sweep"],
    "projector":                ["acquisition.projector"],
    "artifacts":                ["acquisition.artifacts"],
    "noise":                    ["acquisition.noise"],
    "cone3d_geometry":          ["acquisition.cone3d_geometry"],
    "laminography_projector":   ["acquisition.laminography"],
    "reconstructor":            ["reconstruction.reconstructor"],
    "fusion":                   ["reconstruction.fusion"],
    "histogram":                ["analysis.histogram", "analysis.gmm",
                                 "analysis.signatures", "analysis.quality",
                                 "analysis.cross_algorithm", "plotting.histograms"],
    "metrics_table":            ["analysis.metrics_table"],
    "metrics_table_morphology": ["analysis.metrics_morphology"],
    "fusion_metrics":           ["analysis.fusion_metrics"],
    "diana_plots":              ["plotting.publication"],
}

# Modules whose import needs an optional dependency; aliased lazily.
_OPTIONAL = {"nmc_phantom", "density_sweep", "fusion_metrics", "laminography_projector"}


class _LazyAlias(types.ModuleType):
    """Module placeholder that imports its target on first attribute access."""

    def __init__(self, name: str, target: str):
        super().__init__(name)
        self.__dict__["_target"] = target

    def __getattr__(self, attr):
        module = importlib.import_module(self.__dict__["_target"])
        sys.modules[self.__name__] = module
        return getattr(module, attr)


def install_aliases(package: types.ModuleType) -> None:
    for old, targets in LEGACY_MODULES.items():
        full_old = f"{_PKG}.{old}"
        full_targets = [f"{_PKG}.{t}" for t in targets]
        if len(targets) == 1 and old in _OPTIONAL:
            alias = _LazyAlias(full_old, full_targets[0])
        elif len(targets) == 1:
            alias = importlib.import_module(full_targets[0])
        else:
            alias = types.ModuleType(full_old, f"Legacy alias for {', '.join(full_targets)}")
            for target in full_targets:
                module = importlib.import_module(target)
                for key, value in vars(module).items():
                    if not key.startswith("__"):
                        alias.__dict__.setdefault(key, value)
        sys.modules[full_old] = alias
        setattr(package, old, alias)
