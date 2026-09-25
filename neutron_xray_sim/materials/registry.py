"""
neutron_xray_sim.materials.registry
───────────────────────────────────
The material registry: a lazily-built, mutable, ``Mapping``-compatible
catalogue of every material the package knows about.

``MATERIALS`` behaves exactly like the plain dict it replaced —
``MATERIALS["water"]``, ``"steel" in MATERIALS``, ``list(MATERIALS.keys())``
and iteration all work unchanged — while adding the operations a collaboration
needs:

* :meth:`MaterialRegistry.register` — add a spec from code;
* :meth:`MaterialRegistry.load_file` — add specs from a JSON or YAML file, so
  a contributor can ship materials without touching the package source;
* :meth:`MaterialRegistry.search` / :meth:`~MaterialRegistry.names` — find
  materials by tag or substring;
* :meth:`MaterialRegistry.table` — a readable overview for notebooks and
  method sections.

Materials are built on first access and cached, so importing the package costs
nothing and a missing NIST element file only raises when that material is
actually used.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Sequence, Union

from .core import Material, MaterialSpec, build_material, validate_spec

__all__ = ["MaterialRegistry", "MATERIALS"]

_PathLike = Union[str, Path]


class DuplicateMaterialError(KeyError):
    """Raised when registering a key that already exists without ``overwrite``."""


class MaterialRegistry(Mapping):
    """A lazily-built, mutable catalogue of :class:`Material` objects.

    Parameters
    ----------
    specs
        Initial specs. Later registrations may override them only when
        ``overwrite=True`` is passed explicitly.
    """

    def __init__(self, specs: Sequence[MaterialSpec] = ()):
        self._specs: Dict[str, MaterialSpec] = {}
        self._cache: Dict[str, Material] = {}
        for spec in specs:
            # Built-in specs are audited on demand via ``audit()`` rather than
            # warning on every import; see the known issues noted in database.py.
            self.register(spec, warn_physics=False)

    # ── Mapping protocol ─────────────────────────────────────────────────────

    def __getitem__(self, key: str) -> Material:
        if key in self._cache:
            return self._cache[key]
        try:
            spec = self._specs[key]
        except KeyError:
            raise KeyError(
                f"Unknown material {key!r}. "
                f"Available: {', '.join(sorted(self._specs))}"
            ) from None
        # Specs are validated on registration, so building skips re-validation
        # and its physics warnings — use ``audit()`` for a data-quality report.
        material = build_material(spec, validate=False)
        self._cache[key] = material
        return material

    def __iter__(self) -> Iterator[str]:
        return iter(self._specs)

    def __len__(self) -> int:
        return len(self._specs)

    def __contains__(self, key: object) -> bool:
        return key in self._specs

    def __repr__(self) -> str:
        return f"<MaterialRegistry: {len(self._specs)} materials>"

    # ── Registration ─────────────────────────────────────────────────────────

    def register(
        self,
        spec: MaterialSpec,
        *,
        overwrite: bool = False,
        warn_physics: bool = True,
    ) -> MaterialSpec:
        """Validate and add one spec. Returns the spec so it can be chained.

        Raises
        ------
        DuplicateMaterialError
            If *spec.key* is already registered and *overwrite* is False. This
            is deliberate: silently shadowing a built-in material would make
            two contributors' phantoms disagree about what ``'steel'`` means.
        """
        if not spec.key:
            raise ValueError("MaterialSpec.key must be a non-empty string.")
        if spec.key in self._specs and not overwrite:
            raise DuplicateMaterialError(
                f"Material {spec.key!r} is already registered "
                f"({self._specs[spec.key].name}). Pass overwrite=True to replace "
                "it, or choose a distinct key."
            )
        validate_spec(spec, warn_physics=warn_physics)
        self._specs[spec.key] = spec
        self._cache.pop(spec.key, None)
        return spec

    def register_material(
        self, key: str, material: Material, *, overwrite: bool = False
    ) -> Material:
        """Add an already-built :class:`Material` (e.g. from NCrystal).

        Prefer :meth:`register` where a spec is available — a spec is
        reproducible from its inputs, a bare Material is not.
        """
        if key in self._specs and not overwrite:
            raise DuplicateMaterialError(
                f"Material {key!r} is already registered. Pass overwrite=True "
                "to replace it."
            )
        spec = MaterialSpec(
            key=key, name=material.name, symbol=material.symbol,
            density_gcc=material.density_gcc,
            xray_mu_cm=tuple(float(v) for v in material.mu_x_table),
            neutron_mu_cm=(material.mu_n_abs, material.mu_n_coh, material.mu_n_inc),
            color=material.color, reference=material.reference,
            tags=tuple(material.tags),
        )
        self._specs[key] = spec
        self._cache[key] = material
        return material

    def unregister(self, key: str) -> None:
        """Remove a material. Mainly useful to undo a test fixture."""
        self._specs.pop(key, None)
        self._cache.pop(key, None)

    # ── File-based contribution ──────────────────────────────────────────────

    def load_file(self, path: _PathLike, *, overwrite: bool = False) -> List[str]:
        """Register every material in a JSON or YAML file. Returns the new keys.

        The file holds a list of spec objects, or a mapping of ``key → spec``::

            [
              {"key": "quartz", "name": "Quartz", "symbol": "SiO2",
               "density_gcc": 2.65, "formula": "SiO2",
               "color": "#cccc99", "tags": ["mineral"],
               "reference": "Deer, Howie & Zussman (2013)"}
            ]

        YAML needs PyYAML installed; JSON always works. This is the intended
        route for beamline- or project-specific materials: keep them in a file
        under version control next to the analysis, not as edits to the package.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Material file not found: {path}")

        text = path.read_text(encoding="utf-8")
        if path.suffix.lower() in {".yaml", ".yml"}:
            try:
                import yaml
            except ImportError as exc:
                raise ImportError(
                    f"Reading {path.name} needs PyYAML (`pip install pyyaml`). "
                    "Alternatively save the same content as .json."
                ) from exc
            payload = yaml.safe_load(text)
        else:
            payload = json.loads(text)

        if isinstance(payload, Mapping):
            records = [{**value, "key": key} for key, value in payload.items()]
        elif isinstance(payload, list):
            records = list(payload)
        else:
            raise ValueError(
                f"{path} must contain a list of material specs or a mapping of "
                f"key → spec, got {type(payload).__name__}."
            )

        added: List[str] = []
        for index, record in enumerate(records):
            try:
                spec = self._spec_from_dict(record)
            except Exception as exc:
                raise ValueError(f"{path}: entry {index} is invalid — {exc}") from exc
            self.register(spec, overwrite=overwrite)
            added.append(spec.key)
        return added

    @staticmethod
    def _spec_from_dict(record: Mapping) -> MaterialSpec:
        """Build a spec from a plain dict, rejecting unknown keys loudly."""
        allowed = set(MaterialSpec.__dataclass_fields__)
        unknown = sorted(set(record) - allowed)
        if unknown:
            raise ValueError(
                f"unknown field(s) {unknown}; allowed fields are {sorted(allowed)}"
            )
        data = dict(record)
        if data.get("components") is not None:
            data["components"] = tuple(
                (str(c[0]), float(c[1]), float(c[2])) for c in data["components"]
            )
        for tuple_field in ("xray_mu_cm", "neutron_mu_cm", "tags"):
            if data.get(tuple_field) is not None:
                data[tuple_field] = tuple(data[tuple_field])
        return MaterialSpec(**data)

    def save_file(self, path: _PathLike, keys: Optional[Sequence[str]] = None) -> None:
        """Write specs to JSON — for archiving exactly what a study used."""
        keys = list(self._specs) if keys is None else list(keys)
        records = []
        for key in keys:
            spec = self._specs[key]
            record = {
                field: getattr(spec, field)
                for field in MaterialSpec.__dataclass_fields__
                if getattr(spec, field) != MaterialSpec.__dataclass_fields__[field].default
            }
            record["key"] = spec.key
            records.append(record)
        Path(path).write_text(json.dumps(records, indent=2), encoding="utf-8")

    # ── Introspection ────────────────────────────────────────────────────────

    def spec(self, key: str) -> MaterialSpec:
        """Return the recipe behind a material, for provenance and metadata."""
        try:
            return self._specs[key]
        except KeyError:
            raise KeyError(f"Unknown material {key!r}.") from None

    def names(self, tag: Optional[str] = None) -> List[str]:
        """Registry keys, optionally filtered to those carrying *tag*."""
        if tag is None:
            return sorted(self._specs)
        return sorted(k for k, s in self._specs.items() if tag in s.tags)

    def tags(self) -> List[str]:
        """Every tag in use, sorted."""
        return sorted({t for s in self._specs.values() for t in s.tags})

    def search(self, text: str) -> List[str]:
        """Keys whose key, name, or symbol contains *text* (case-insensitive)."""
        needle = text.lower()
        return sorted(
            key for key, spec in self._specs.items()
            if needle in key.lower()
            or needle in spec.name.lower()
            or needle in spec.symbol.lower()
        )

    def audit(self, keys: Optional[Sequence[str]] = None) -> Dict[str, List[str]]:
        """Run the physics plausibility checks and collect the findings.

        Returns ``{key: [problem, …]}`` for every material that looks wrong —
        an X-ray table rising with energy away from a declared absorption edge,
        or a material that cannot be built at all because an element table is
        missing.  Empty means everything checks out.

        The built-in catalogue is *not* audited at import time, so this is the
        place to look when a material's contrast seems off::

            >>> from neutron_xray_sim import MATERIALS
            >>> MATERIALS.audit()
        """
        import warnings as _warnings

        findings: Dict[str, List[str]] = {}
        for key in (sorted(self._specs) if keys is None else list(keys)):
            problems: List[str] = []
            with _warnings.catch_warnings(record=True) as caught:
                _warnings.simplefilter("always")
                try:
                    validate_spec(self._specs[key], warn_physics=True)
                    build_material(self._specs[key], validate=False)
                except Exception as exc:
                    problems.append(str(exc).splitlines()[0])
            problems.extend(str(w.message).splitlines()[0] for w in caught)
            if problems:
                findings[key] = problems
        return findings

    def table(self, keys: Optional[Sequence[str]] = None, energy_keV: float = 80.0) -> str:
        """A plain-text overview of the catalogue, for notebooks and papers.

        Materials whose X-ray data cannot be built (a missing NIST element
        file) are listed with the reason rather than aborting the whole table.
        """
        keys = sorted(self._specs) if keys is None else list(keys)
        header = (
            f"{'key':<26} {'symbol':<12} {'rho [g/cm3]':>12} "
            f"{'mu_x(' + format(energy_keV, '.0f') + 'keV)':>14} {'mu_n':>9}  tags"
        )
        lines = [header, "-" * len(header)]
        for key in keys:
            spec = self._specs[key]
            try:
                mat = self[key]
                lines.append(
                    f"{key:<26} {mat.symbol[:12]:<12} {mat.density_gcc:>12.3f} "
                    f"{mat.mu_x_at(energy_keV):>14.4f} {mat.mu_n:>9.4f}  "
                    f"{','.join(mat.tags)}"
                )
            except Exception as exc:
                reason = str(exc).splitlines()[0]
                lines.append(f"{key:<26} {spec.symbol[:12]:<12} {'-- unavailable: ' + reason}")
        return "\n".join(lines)


# ──────────────────────────────────────────────────────────────────────────────
# The package-wide registry
# ──────────────────────────────────────────────────────────────────────────────

from .database import BUILTIN_SPECS  # noqa: E402  (avoids a circular import)

#: The package-wide material catalogue. Mutable at runtime — see
#: :meth:`MaterialRegistry.register` and :meth:`MaterialRegistry.load_file`.
MATERIALS = MaterialRegistry(BUILTIN_SPECS)
