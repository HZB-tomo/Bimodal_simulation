"""
neutron_xray_sim/io.py
-----------------------
Persistence layer for every stage of the dual-modality CT pipeline.

All intermediate results are saved as ``.npy`` files under a structured
directory tree so that any stage of a long simulation can be re-loaded
without re-running the steps before it.

Directory layout
----------------
::

    <root>/
    ├── phantom/
    │   ├── label_vol.npy          – (Nz, Nx, Ny) uint8 material labels
    │   ├── meta.json              – geometry + full material specs
    │   └── (optional, save_derived_volumes=True)
    │       ├── mu_x_vol_<e>keV.npy   – X-ray mu at each energy bin
    │       ├── mu_n_abs_vol.npy      – neutron absorption mu
    │       ├── mu_n_coh_vol.npy      – neutron coherent scatter mu
    │       └── mu_n_inc_vol.npy      – neutron incoherent scatter mu
    │
    ├── sinograms/
    │   ├── xray_sino_lam.npy      – (n_angles, N, N) log-attenuation
    │   ├── xray_sino_trans.npy    – (n_angles, N, N) transmission
    │   ├── neutron_sino_lam.npy
    │   ├── neutron_sino_trans.npy
    │   ├── neutron_sino_abs_lam.npy
    │   ├── neutron_sino_scatter_lam.npy
    │   └── sino_meta.json         – angles, kVp, I0, voxel_cm, …
    │
    ├── runs/
    │   └── <tag_slug>/
    │       ├── xray_sino_lam.npy         – sinogram after artifact injection
    │       ├── neutron_sino_lam.npy
    │       ├── vol_xray.npy              – reconstructed X-ray volume [cm⁻¹]
    │       ├── vol_neutron.npy           – reconstructed neutron volume [cm⁻¹]
    │       ├── histogram_H.npy           – 2-D histogram counts
    │       ├── histogram_x_edges.npy
    │       ├── histogram_n_edges.npy
    │       └── run_meta.json             – tag, algorithm, artifact cfg, …
    │
    └── survey/
        └── <survey_slug>/
            ├── metrics_table.csv          – DB, CE per tag
            └── <tag_slug>/  (same layout as runs/)

Naming rules
------------
* **Tag slugs** are produced by :func:`tag_to_slug`: spaces and special
  characters replaced with underscores, truncated to 64 characters, and
  lower-cased.  The slug is stable across platforms.
* **Energy-bin filenames** use the rounded keV value, e.g.
  ``mu_x_vol_080keV.npy``.
* All ``meta.json`` files carry a ``"schema_version"`` key so that future
  format changes can be handled transparently.

Usage
-----
::

    from neutron_xray_sim.io import SimCache

    # Writer side (inside DualModalitySimulation.run)
    cache = SimCache("my_results/")
    cache.save_phantom(phantom)
    cache.save_raw_sinograms(xray_sino, neutron_sino)
    cache.save_run(result)

    # Reader side (standalone re-analysis)
    cache = SimCache("my_results/")
    phantom     = cache.load_phantom()           # PhantomData
    xray_sino   = cache.load_raw_xray_sino()     # dict
    vol_xray    = cache.load_run_volume("clean", modality="xray")
    hist        = cache.load_run_histogram("clean")

    # Re-run only the histogram step on previously saved volumes
    for tag in cache.list_run_tags():
        vol_x = cache.load_run_volume(tag, "xray")
        vol_n = cache.load_run_volume(tag, "neutron")
        hist  = compute_bimodal_histogram(vol_x, vol_n)
        cache.save_run_histogram(tag, hist)

Schema version
--------------
Current version is ``"1.1"``: ``phantom/meta.json`` gained a ``material_specs``
block and the derived attenuation volumes became optional, since they are
reproducible from the label volume plus the materials.  A cache written by
schema 1.0 still loads for sinograms, volumes and histograms; only
:meth:`SimCache.load_phantom` needs the newer metadata.
"""

from __future__ import annotations

import json
import re
import shutil
from pathlib import Path
from typing import List

import numpy as np

__all__ = ["SimCache", "tag_to_slug"]

_SCHEMA_VERSION = "1.1"


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def tag_to_slug(tag: str, max_len: int = 64) -> str:
    """
    Convert a human-readable run tag to a safe directory/file slug.

    Rules:
    * Lower-cased
    * All runs of non-alphanumeric characters replaced with ``_``
    * Leading/trailing underscores stripped
    * Truncated to *max_len* characters

    Examples
    --------
    >>> tag_to_slug("Noise moderate\\n(I₀=5×10⁴)")
    'noise_moderate_i_5_10'
    >>> tag_to_slug("Clean (reference)")
    'clean_reference'
    """
    s = tag.lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    s = s.strip("_")
    return s[:max_len] or "run"


def _write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _save_npy(path: Path, arr: np.ndarray) -> None:
    """Save a float32 array, converting dtype if necessary."""
    np.save(path, arr.astype(np.float32))


def _load_npy(path: Path) -> np.ndarray:
    return np.load(path)


# ─────────────────────────────────────────────────────────────────────────────
# SimCache
# ─────────────────────────────────────────────────────────────────────────────

class SimCache:
    """
    File-system cache for all pipeline stages.

    Parameters
    ----------
    root : str or Path
        Root directory.  Created automatically if it does not exist.
    overwrite : bool
        If True, existing files are silently overwritten.
        If False (default), saving to an existing file raises ``FileExistsError``.
        Pass ``overwrite=True`` when re-running a simulation.
    """

    def __init__(self, root: str | Path, overwrite: bool = False):
        self.root      = Path(root)
        self.overwrite = overwrite
        self.root.mkdir(parents=True, exist_ok=True)

    # ── Directory accessors ───────────────────────────────────────────────────

    @property
    def phantom_dir(self) -> Path:
        d = self.root / "phantom"
        d.mkdir(exist_ok=True)
        return d

    @property
    def sino_dir(self) -> Path:
        d = self.root / "sinograms"
        d.mkdir(exist_ok=True)
        return d

    def run_dir(self, tag: str) -> Path:
        d = self.root / "runs" / tag_to_slug(tag)
        d.mkdir(parents=True, exist_ok=True)
        return d

    def survey_dir(self, survey_slug: str) -> Path:
        d = self.root / "survey" / survey_slug
        d.mkdir(parents=True, exist_ok=True)
        return d

    # ── Guard ─────────────────────────────────────────────────────────────────

    def _guard(self, path: Path) -> None:
        if path.exists() and not self.overwrite:
            raise FileExistsError(
                f"{path} already exists.  "
                "Pass overwrite=True to SimCache to allow overwriting."
            )

    # ─────────────────────────────────────────────────────────────────────────
    # Phantom
    # ─────────────────────────────────────────────────────────────────────────

    def save_phantom(self, phantom, save_derived_volumes: bool = False) -> None:
        """
        Persist the phantom.

        Saves
        -----
        * ``phantom/label_vol.npy``  – uint8 label volume
        * ``phantom/meta.json``      – geometry plus the full material specs

        The attenuation volumes are **not** written by default.  They are a
        pure function of the label volume and the materials, both of which are
        saved, so :meth:`load_phantom` rebuilds them exactly.  Writing them out
        used to cost 16 extra copies of the volume on disk — 13 X-ray energies
        plus 3 neutron channels, about 7 GB for a 512³ phantom.

        Parameters
        ----------
        phantom : PhantomData
        save_derived_volumes :
            Also write the neutron and per-energy X-ray volumes, for tools that
            read the cache directly instead of going through ``load_phantom``.
        """
        d = self.phantom_dir

        p = d / "label_vol.npy"
        self._guard(p)
        np.save(p, phantom.label_vol.astype(np.uint8))

        if save_derived_volumes:
            from .materials import XRAY_E_KEV
            for e_idx, e_kev in enumerate(XRAY_E_KEV):
                fname = d / f"mu_x_vol_{int(round(e_kev)):03d}keV.npy"
                self._guard(fname)
                _save_npy(fname, phantom.mu_x_at_index(e_idx))

            for name, vol in [
                ("mu_n_abs_vol", phantom.mu_n_abs_vol),
                ("mu_n_coh_vol", phantom.mu_n_coh_vol),
                ("mu_n_inc_vol", phantom.mu_n_inc_vol),
            ]:
                p = d / f"{name}.npy"
                self._guard(p)
                _save_npy(p, vol)

        meta = {
            "schema_version": _SCHEMA_VERSION,
            "name":           phantom.name,
            "N":              int(phantom.N),
            "Nz":             int(phantom.Nz),
            "Nx":             int(phantom.Nx),
            "Ny":             int(phantom.Ny),
            "voxel_cm":       float(phantom.voxel_cm),
            "materials":      [m.name for m in phantom.materials],
            "n_materials":    len(phantom.materials),
            # Full numeric definition of every material, so a cached phantom is
            # self-describing even if the material database changes later.
            "material_specs": [m.as_dict() for m in phantom.materials],
        }
        _write_json(d / "meta.json", meta)

    def load_phantom_meta(self) -> dict:
        """Return the phantom metadata dict."""
        return _read_json(self.phantom_dir / "meta.json")

    def load_phantom(self):
        """Rebuild the cached :class:`~neutron_xray_sim.phantom.PhantomData`.

        Materials are restored from the ``material_specs`` block written by
        :meth:`save_phantom`, so the reloaded phantom carries the exact numbers
        used in the original run rather than whatever the current material
        database happens to say.

        Raises
        ------
        FileNotFoundError
            If no phantom has been saved to this cache.
        """
        import numpy as _np

        from .materials import Material
        from .phantom import PhantomData

        meta_path = self.phantom_dir / "meta.json"
        if not meta_path.exists():
            raise FileNotFoundError(
                f"No phantom in this cache ({meta_path} is missing). "
                "Call save_phantom() first, or point SimCache at a populated "
                "results directory."
            )
        meta = _read_json(meta_path)
        label_vol = _load_npy(self.phantom_dir / "label_vol.npy")

        specs = meta.get("material_specs")
        if not specs:
            raise ValueError(
                f"{meta_path} predates schema 1.1 and has no 'material_specs' "
                "block, so the materials cannot be restored. Re-run "
                "save_phantom() with this version to upgrade the cache."
            )
        materials = [
            Material(
                name=s["name"], symbol=s["symbol"],
                density_gcc=s["density_gcc"],
                mu_n_abs=s["mu_n_abs"], mu_n_coh=s["mu_n_coh"],
                mu_n_inc=s["mu_n_inc"],
                mu_x_table=_np.asarray(s["mu_x_cm"], dtype=float),
                color=s.get("color", "#888888"),
                key=s.get("key", ""), reference=s.get("reference", ""),
                tags=tuple(s.get("tags", ())),
            )
            for s in specs
        ]

        Nz, Nx, Ny = label_vol.shape
        return PhantomData(
            Nz=int(meta.get("Nz", Nz)),
            Nx=int(meta.get("Nx", Nx)),
            Ny=int(meta.get("Ny", Ny)),
            voxel_cm=float(meta["voxel_cm"]),
            label_vol=label_vol.astype(np.uint8),
            materials=materials,
            name=meta.get("name", "phantom"),
        )

    def phantom_exists(self) -> bool:
        """Return True if a phantom has been saved."""
        return (self.phantom_dir / "meta.json").exists()

    # ─────────────────────────────────────────────────────────────────────────
    # Raw sinograms (clean, pre-artifact-injection)
    # ─────────────────────────────────────────────────────────────────────────

    def save_raw_sinograms(self, xray_sino: dict, neutron_sino: dict) -> None:
        """
        Save the clean (pre-artifact) sinogram pair.

        Saves
        -----
        * ``sinograms/xray_sino_lam.npy``
        * ``sinograms/xray_sino_trans.npy``
        * ``sinograms/neutron_sino_lam.npy``
        * ``sinograms/neutron_sino_trans.npy``
        * ``sinograms/neutron_sino_abs_lam.npy``
        * ``sinograms/neutron_sino_scatter_lam.npy``
        * ``sinograms/sino_meta.json``

        Parameters
        ----------
        xray_sino    : dict returned by ``project_xray()``
        neutron_sino : dict returned by ``project_neutron()``
        """
        d = self.sino_dir

        for key, arr in [
            ("xray_sino_lam",   xray_sino["sino_lam"]),
            ("xray_sino_trans", xray_sino["sino_trans"]),
        ]:
            p = d / f"{key}.npy"
            self._guard(p)
            _save_npy(p, arr)

        for key, arr in [
            ("neutron_sino_lam",          neutron_sino["sino_lam"]),
            ("neutron_sino_trans",         neutron_sino["sino_trans"]),
            ("neutron_sino_abs_lam",       neutron_sino.get("sino_abs_lam",
                                               neutron_sino["sino_lam"])),
            ("neutron_sino_scatter_lam",   neutron_sino.get("sino_scatter_lam",
                                               np.zeros_like(neutron_sino["sino_lam"]))),
        ]:
            p = d / f"{key}.npy"
            self._guard(p)
            _save_npy(p, arr)

        meta = {
            "schema_version":   _SCHEMA_VERSION,
            "angles_deg":       xray_sino["angles_deg"].tolist(),
            "I0_xray":          float(xray_sino.get("I0", 1e5)),
            "I0_neutron":       float(neutron_sino.get("I0", 1e5)),
            "voxel_cm":         float(xray_sino.get("voxel_cm", 0.0)),
            "kVp":              float(xray_sino.get("spectrum", {}).get("kVp", 0.0)
                                       if isinstance(xray_sino.get("spectrum"), dict)
                                       else 0.0),
            "scatter_D_over_L": float(neutron_sino.get("scatter_D_over_L", 100.0)),
            "n_angles":         int(len(xray_sino["angles_deg"])),
            "sino_shape":       list(xray_sino["sino_lam"].shape),
        }
        _write_json(d / "sino_meta.json", meta)

    def load_raw_xray_sino(self) -> dict:
        """
        Re-load the clean X-ray sinogram dict.

        Returns
        -------
        dict  with keys ``sino_lam``, ``sino_trans``, ``angles_deg``,
              ``voxel_cm``, ``I0``
        """
        d    = self.sino_dir
        meta = _read_json(d / "sino_meta.json")
        return {
            "sino_lam":   _load_npy(d / "xray_sino_lam.npy"),
            "sino_trans": _load_npy(d / "xray_sino_trans.npy"),
            "angles_deg": np.array(meta["angles_deg"], dtype=np.float32),
            "voxel_cm":   float(meta["voxel_cm"]),
            "I0":         float(meta["I0_xray"]),
        }

    def load_raw_neutron_sino(self) -> dict:
        """
        Re-load the clean neutron sinogram dict.

        Returns
        -------
        dict  with keys ``sino_lam``, ``sino_trans``, ``sino_abs_lam``,
              ``sino_scatter_lam``, ``angles_deg``, ``voxel_cm``, ``I0``
        """
        d    = self.sino_dir
        meta = _read_json(d / "sino_meta.json")
        return {
            "sino_lam":          _load_npy(d / "neutron_sino_lam.npy"),
            "sino_trans":        _load_npy(d / "neutron_sino_trans.npy"),
            "sino_abs_lam":      _load_npy(d / "neutron_sino_abs_lam.npy"),
            "sino_scatter_lam":  _load_npy(d / "neutron_sino_scatter_lam.npy"),
            "angles_deg":        np.array(meta["angles_deg"], dtype=np.float32),
            "voxel_cm":          float(meta["voxel_cm"]),
            "I0":                float(meta["I0_neutron"]),
            "scatter_D_over_L":  float(meta["scatter_D_over_L"]),
        }

    def raw_sinograms_exist(self) -> bool:
        """Return True if raw sinograms have been saved."""
        return (self.sino_dir / "sino_meta.json").exists()

    # ─────────────────────────────────────────────────────────────────────────
    # Per-run results
    # ─────────────────────────────────────────────────────────────────────────

    def save_run(self, result) -> None:
        """
        Save all arrays from one simulation run.

        Saves under ``runs/<slug>/``:

        * ``vol_xray.npy``              – reconstructed X-ray volume [cm⁻¹]
        * ``vol_neutron.npy``           – reconstructed neutron volume [cm⁻¹]
        * ``xray_sino_lam.npy``         – artifact-injected X-ray sinogram
        * ``neutron_sino_lam.npy``      – artifact-injected neutron sinogram
        * ``histogram_H.npy``           – 2-D histogram counts
        * ``histogram_x_edges.npy``     – μ_x bin edges
        * ``histogram_n_edges.npy``     – μ_n bin edges
        * ``run_meta.json``             – tag, artifact config, algorithm, …

        Parameters
        ----------
        result : SimulationResult
        """
        tag = result.tag
        d   = self.run_dir(tag)

        # Reconstructed volumes
        _save_npy(d / "vol_xray.npy",     result.vol_xray)
        _save_npy(d / "vol_neutron.npy",  result.vol_neutron)

        # Artifact-injected sinograms (log-attenuation only — largest arrays)
        _save_npy(d / "xray_sino_lam.npy",
                  result.xray_sino["sino_lam"])
        _save_npy(d / "neutron_sino_lam.npy",
                  result.neutron_sino["sino_lam"])

        # Histogram
        h = result.histogram
        _save_npy(d / "histogram_H.npy",       h.H)
        _save_npy(d / "histogram_x_edges.npy", h.x_edges)
        _save_npy(d / "histogram_n_edges.npy", h.n_edges)

        # Metadata
        meta = {
            "schema_version": _SCHEMA_VERSION,
            "tag":            tag,
            "slug":           tag_to_slug(tag),
            "elapsed_s":      float(result.elapsed_s),
            "artifact_cfg":   _artifact_cfg_to_dict(result.cfg),
        }
        _write_json(d / "run_meta.json", meta)

    def save_run_volume(self, tag: str, modality: str,
                        vol: np.ndarray) -> None:
        """
        Save a single reconstructed volume for *tag*.

        Parameters
        ----------
        tag      : run tag (slug is computed automatically)
        modality : ``'xray'`` or ``'neutron'``
        vol      : (N, N, N) float32 [cm⁻¹]
        """
        d  = self.run_dir(tag)
        p  = d / f"vol_{modality}.npy"
        self._guard(p)
        _save_npy(p, vol)

    def save_run_sinogram(self, tag: str, modality: str,
                          sino: np.ndarray) -> None:
        """
        Save an artifact-injected sinogram for *tag*.

        Parameters
        ----------
        tag      : run tag
        modality : ``'xray'`` or ``'neutron'``
        sino     : (n_angles, N, N) float32
        """
        d = self.run_dir(tag)
        p = d / f"{modality}_sino_lam.npy"
        self._guard(p)
        _save_npy(p, sino)

    def save_run_histogram(self, tag: str, hist) -> None:
        """
        Save the 2-D histogram for *tag* (useful when re-running only
        the histogram step without touching the volumes).

        Parameters
        ----------
        tag  : run tag
        hist : HistogramResult
        """
        d = self.run_dir(tag)
        _save_npy(d / "histogram_H.npy",       hist.H)
        _save_npy(d / "histogram_x_edges.npy", hist.x_edges)
        _save_npy(d / "histogram_n_edges.npy", hist.n_edges)

    # ── Loaders ───────────────────────────────────────────────────────────────

    def load_run_volume(self, tag: str, modality: str) -> np.ndarray:
        """
        Load a reconstructed volume.

        Parameters
        ----------
        tag      : run tag (as used when saving)
        modality : ``'xray'`` or ``'neutron'``

        Returns
        -------
        (N, N, N) float32 [cm⁻¹]
        """
        p = self.run_dir(tag) / f"vol_{modality}.npy"
        if not p.exists():
            raise FileNotFoundError(f"No saved volume for tag='{tag}', "
                                    f"modality='{modality}' at {p}")
        return _load_npy(p)

    def load_run_sinogram(self, tag: str, modality: str) -> np.ndarray:
        """
        Load an artifact-injected sinogram.

        Parameters
        ----------
        tag      : run tag
        modality : ``'xray'`` or ``'neutron'``

        Returns
        -------
        (n_angles, N, N) float32
        """
        p = self.run_dir(tag) / f"{modality}_sino_lam.npy"
        if not p.exists():
            raise FileNotFoundError(f"No sinogram for tag='{tag}', "
                                    f"modality='{modality}' at {p}")
        return _load_npy(p)

    def load_run_histogram(self, tag: str):
        """
        Load a saved histogram and return a ``HistogramResult``.

        Parameters
        ----------
        tag : run tag

        Returns
        -------
        HistogramResult
        """
        from .histogram import HistogramResult
        d = self.run_dir(tag)
        H       = _load_npy(d / "histogram_H.npy")
        x_edges = _load_npy(d / "histogram_x_edges.npy")
        n_edges = _load_npy(d / "histogram_n_edges.npy")
        x_centres = 0.5 * (x_edges[:-1] + x_edges[1:])
        n_centres = 0.5 * (n_edges[:-1] + n_edges[1:])
        # vol_x_flat / vol_n_flat are needed only for GMM fitting;
        # they are not stored (too large).  Provide empty arrays as stubs.
        return HistogramResult(
            H=H,
            x_edges=x_edges,
            n_edges=n_edges,
            x_centres=x_centres,
            n_centres=n_centres,
            vol_x_flat=np.array([], dtype=np.float32),
            vol_n_flat=np.array([], dtype=np.float32),
            total_voxels=int(H.sum()),
        )

    def load_run_meta(self, tag: str) -> dict:
        """Return the metadata dict for a run."""
        return _read_json(self.run_dir(tag) / "run_meta.json")

    # ── Survey helpers ────────────────────────────────────────────────────────

    def save_survey_metrics(self, survey_slug: str,
                             metrics: dict) -> None:
        """
        Save a cluster-quality metrics table for a full survey run to CSV.

        Parameters
        ----------
        survey_slug : identifier string for this survey (e.g. ``"SIRT_N512"``)
        metrics     : dict {tag -> ClusterQualityMetrics}
        """
        import csv
        d    = self.survey_dir(survey_slug)
        path = d / "metrics_table.csv"
        self._guard(path)

        rows = []
        for tag, m in metrics.items():
            row = {
                "tag":               tag,
                "slug":              tag_to_slug(tag),
                "mean_centroid_error": m.mean_centroid_error,
                "davies_bouldin":    m.davies_bouldin,
                "n_matched":         m.n_matched,
            }
            for mat, err in m.centroid_errors.items():
                row[f"ce_{mat}"] = err
            for mat, sx in m.sigma_x.items():
                row[f"sx_{mat}"] = sx
            for mat, sn in m.sigma_n.items():
                row[f"sn_{mat}"] = sn
            rows.append(row)

        if not rows:
            return
        fieldnames = list(rows[0].keys())
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames,
                                    extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)

    def load_survey_metrics_csv(self, survey_slug: str) -> list:
        """
        Load a previously saved metrics CSV as a list of dicts.

        Parameters
        ----------
        survey_slug : survey identifier

        Returns
        -------
        list of dicts, one per row
        """
        import csv
        path = self.survey_dir(survey_slug) / "metrics_table.csv"
        if not path.exists():
            raise FileNotFoundError(f"No metrics CSV at {path}")
        with open(path, newline="", encoding="utf-8") as f:
            return list(csv.DictReader(f))

    # ── Inventory ─────────────────────────────────────────────────────────────

    def list_run_tags(self) -> List[str]:
        """
        Return the list of run tags that have been saved, in the order
        they were saved (directory mtime).

        Returns
        -------
        list of str  (original tags, read from ``run_meta.json``)
        """
        runs_root = self.root / "runs"
        if not runs_root.exists():
            return []
        tags = []
        for slug_dir in sorted(runs_root.iterdir(), key=lambda p: p.stat().st_mtime):
            meta_path = slug_dir / "run_meta.json"
            if meta_path.exists():
                tags.append(_read_json(meta_path)["tag"])
        return tags

    def has_run(self, tag: str) -> bool:
        """Return True if a run with *tag* has been saved."""
        return (self.run_dir(tag) / "run_meta.json").exists()

    def clear_run(self, tag: str) -> None:
        """Delete all files for a single run."""
        d = self.root / "runs" / tag_to_slug(tag)
        if d.exists():
            shutil.rmtree(d)

    def clear_all(self) -> None:
        """Delete the entire cache root directory."""
        shutil.rmtree(self.root)

    def __repr__(self) -> str:
        n_runs = len(self.list_run_tags())
        return (f"SimCache(root='{self.root}', "
                f"runs={n_runs}, overwrite={self.overwrite})")


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _artifact_cfg_to_dict(cfg) -> dict:
    """Convert an ArtifactConfig to a JSON-serialisable dict."""
    import dataclasses
    return {k: v for k, v in dataclasses.asdict(cfg).items()
            if not isinstance(v, (np.ndarray,))}
