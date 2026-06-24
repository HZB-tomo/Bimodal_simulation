"""
ncrystal_bragg.py
═════════════════
Energy-resolved (wavelength-resolved) neutron cross sections from **NCrystal**,
packaged so they drop straight into the ``neutron_xray_sim`` (DIANA) pipeline.

Why this module exists
──────────────────────
The DIANA core (``materials.py`` / ``neutron_spectra.py``) treats the coherent
neutron scatter cross section as energy-independent (the bound-atom value).  That
is fine for broadband cold/thermal contrast, but it deliberately does **not**
model Bragg edges — see ``DIANA_technical_reference.md`` §12, open task *M7*.

Bragg edges are exactly the feature that separates crystallographic phases:
a body-centred-cubic phase (α-Fe / ferrite, the simplest stand-in for
**martensite**) and a face-centred-cubic phase (γ-Fe, i.e. **austenite**) have
different lattice symmetries and therefore different edge positions:

    α-Fe (BCC)  (110) edge ≈ 4.05 Å      (200) ≈ 2.87 Å   (211) ≈ 2.34 Å
    γ-Fe (FCC)  (111) edge ≈ 4.15 Å      (200) ≈ 3.59 Å   (220) ≈ 2.54 Å

NCrystal computes the full λ-dependent total cross section (coherent-elastic
Bragg + incoherent + inelastic + 1/v absorption) for polycrystalline materials,
so a monochromatic wavelength scan resolves the two phases.

What this module provides
─────────────────────────
* ``wavelength_A_to_energy_meV`` / ``energy_meV_to_wavelength_A`` — conversions.
* ``BraggPhase``  — a lazily-loaded NCrystal material exposing σ(λ) and the
  macroscopic attenuation μ(λ) in cm⁻¹.
* ``load_phase`` / ``DEFAULT_PHASE_CFG`` — convenience loaders for the two steels.
* ``mu_n_bragg_lut`` — per-label μ(λ) lookup table (the neutron analogue of the
  X-ray ``mu_lut``), with ``None`` entries (e.g. air) mapped to μ = 0.
* ``set_phantom_neutron_mu`` — overwrite a ``PhantomData`` object's neutron
  attenuation volumes with a single per-label μ (follows the documented patch
  pattern: everything goes in the absorption channel, coh/inc set to 0).
* ``run_bragg_edge_scan`` — orchestrate a full monochromatic scan: for every
  wavelength, project + reconstruct the neutron volume using DIANA's own
  ``project_neutron`` / ``reconstruct``.

Units
─────
The clean identity used throughout::

    μ [cm⁻¹] = n [atoms·Å⁻³] × σ [barn]

holds because 1 Å³ = 1e-24 cm³ and 1 barn = 1e-24 cm², so the factors cancel.
NCrystal reports number density in atoms·Å⁻³, so no extra conversion is needed.

NCrystal
────────
Tested against the modern pythonic API (``NCrystal.load`` → ``.info`` /
``.scatter`` / ``.absorption``), NCrystal ≥ 3.6 and the 4.x series.
Install with::

    pip install ncrystal            # or:  conda install -c conda-forge ncrystal

The two iron phases use data files shipped with NCrystal's standard library:
``Fe_sg229_Iron-alpha.ncmat`` (BCC) and ``Fe_sg225_Iron-gamma.ncmat`` (FCC).
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Union

import numpy as np


# ──────────────────────────────────────────────────────────────────────────────
#  Neutron energy ↔ wavelength
# ──────────────────────────────────────────────────────────────────────────────
# E[meV] = h² / (2 m_n λ²)  →  E[meV] = 81.80420 / λ[Å]²
NEUTRON_E_LAMBDA_CONST_meV_A2: float = 81.80420


def wavelength_A_to_energy_meV(wl_A) -> np.ndarray:
    """Convert neutron de Broglie wavelength (Å) to kinetic energy (meV)."""
    wl = np.asarray(wl_A, dtype=float)
    return NEUTRON_E_LAMBDA_CONST_meV_A2 / (wl * wl)


def energy_meV_to_wavelength_A(E_meV) -> np.ndarray:
    """Convert neutron kinetic energy (meV) to de Broglie wavelength (Å)."""
    E = np.asarray(E_meV, dtype=float)
    return np.sqrt(NEUTRON_E_LAMBDA_CONST_meV_A2 / E)


# ──────────────────────────────────────────────────────────────────────────────
#  NCrystal availability
# ──────────────────────────────────────────────────────────────────────────────
try:                                # pragma: no cover - depends on environment
    import NCrystal as _NC          # type: ignore
    NCRYSTAL_AVAILABLE = True
    NCRYSTAL_VERSION = getattr(_NC, "__version__", "unknown")
except Exception:                   # pragma: no cover
    _NC = None
    NCRYSTAL_AVAILABLE = False
    NCRYSTAL_VERSION = None


def _require_ncrystal() -> None:
    if not NCRYSTAL_AVAILABLE:
        raise ImportError(
            "NCrystal is required for Bragg-edge cross sections but is not "
            "installed.\n"
            "    pip install ncrystal\n"
            "  or\n"
            "    conda install -c conda-forge ncrystal\n"
            "See https://github.com/mctools/ncrystal for details."
        )


# Default NCrystal cfg strings for the two steel phases.
#   martensite  → α-Fe (BCC, space group 229).  Real lath/plate martensite is
#                 body-centred *tetragonal* (carbon supersaturation); the small
#                 tetragonality is neglected here, so BCC α-Fe is the standard
#                 first-order proxy and reproduces the diagnostic (110) edge.
#   austenite   → γ-Fe (FCC, space group 225).
# To model true stainless compositions (e.g. 304: Fe-Cr-Ni) supply your own
# multi-phase .ncmat cfg string instead.
#
# The iron oxides are NOT part of NCrystal's standard data library (checked
# against release v4.2.10 — only Fe_sg229 / Fe_sg225 ship for iron). We
# therefore model them as **isostructural analogs** built from data files that
# DO ship, using the ``atomdb`` cfg parameter to substitute the cation sites
# with Fe:
#   magnetite (Fe3O4) ≈ spinel  MgAl2O4 (Fd-3m, 227) with Mg,Al → Fe.
#   hematite  (Fe2O3) ≈ corundum Al2O3  (R-3c, 167)   with Al    → Fe.
# These reproduce the correct space group, site topology and neutron scatterers
# (Fe + O), giving each phase a distinct multi-edge Bragg fingerprint. They use
# the HOST lattice parameters, however, so absolute edge wavelengths are a few %
# off the real oxides (corundum a≈4.76 vs hematite 5.04 Å; MgAl2O4 a≈8.08 vs
# magnetite 8.40 Å) and μ magnitudes scale with the host number density.
#
# For quantitative work, generate exact files from a CIF and override the cfg:
#     ncrystal_cif2ncmat <hematite.cif>  -o Fe2O3_Hematite.ncmat
#     load_phase("Fe2O3_Hematite.ncmat", name="hematite")
# or override DEFAULT_PHASE_CFG / pass your own cfg to load_phase.
#
# Each value is a list of candidate cfgs tried in order: a real dedicated file
# first (used automatically if a plugin/newer library provides it), then the
# analog. Two atomdb separators are listed in case a given NCrystal build wants
# '@' vs a newline between entries.
DEFAULT_PHASE_CFG: Dict[str, Union[str, List[str]]] = {
    "martensite": "Fe_sg229_Iron-alpha.ncmat",
    "austenite":  "Fe_sg225_Iron-gamma.ncmat",
    "magnetite": [
        "Fe3O4_sg227_Magnetite.ncmat",
        "MgAl2O4_sg227_MAS.ncmat;atomdb=Mg is Fe@Al is Fe",
        "MgAl2O4_sg227_MAS.ncmat;atomdb=Mg is Fe\nAl is Fe",
    ],
    "hematite": [
        "Fe2O3_sg167_Hematite.ncmat",
        "Al2O3_sg167_Corundum.ncmat;atomdb=Al is Fe",
    ],
}


def available_ncmat(pattern: Optional[str] = None) -> List[str]:
    """List .ncmat data files known to the current NCrystal installation.

    Useful for confirming the exact file name of a phase before loading, since
    the data library grows between NCrystal releases.

    Parameters
    ----------
    pattern
        Optional case-insensitive substring filter (e.g. ``"Fe"`` or
        ``"Magnetite"``). When ``None`` every available file is returned.

    Returns
    -------
    list of str
        Sorted data-file names. Empty if NCrystal is unavailable or exposes no
        file-listing API.

    Examples
    --------
    >>> available_ncmat("Fe2O3")            # doctest: +SKIP
    ['Fe2O3_sg167_Hematite.ncmat']
    >>> available_ncmat("Magnetite")        # doctest: +SKIP
    ['Fe3O4_sg227_Magnetite.ncmat']
    """
    if not NCRYSTAL_AVAILABLE:
        return []
    names: List[str] = []
    # The browsing API has moved across NCrystal versions; try the known ones.
    for getter in ("browseFiles", "listAvailableFiles", "browse_files"):
        fn = getattr(_NC, getter, None)
        if fn is None:
            continue
        try:
            entries = fn()
        except Exception:
            continue
        for e in entries:
            # Entries may be strings or small records with a ``name`` attribute.
            name = getattr(e, "name", None) or (e if isinstance(e, str) else None)
            if name:
                names.append(str(name))
        if names:
            break
    names = sorted(set(names))
    if pattern:
        p = pattern.lower()
        names = [n for n in names if p in n.lower()]
    return names


# ──────────────────────────────────────────────────────────────────────────────
#  Low-level NCrystal helpers
# ──────────────────────────────────────────────────────────────────────────────

def _as_float(value) -> float:
    """Coerce NCrystal scalar wrappers (Density / NumberDensity) or plain
    numbers to a Python float, regardless of NCrystal version."""
    try:
        return float(value)
    except TypeError:
        return float(getattr(value, "value"))


def _xsect_barn(process, wl_A: np.ndarray) -> np.ndarray:
    """Cross section (barn) of an NCrystal process at the given wavelengths.

    Works whether ``xsect`` accepts a vectorised ``wl=`` argument or only
    scalars, so the helper is robust across NCrystal releases.
    """
    wl = np.atleast_1d(np.asarray(wl_A, dtype=float))
    try:
        out = np.asarray(process.xsect(wl=wl), dtype=float)
        if out.shape == wl.shape:
            return out
    except Exception:
        pass
    return np.array([float(process.xsect(wl=float(w))) for w in wl], dtype=float)


# ──────────────────────────────────────────────────────────────────────────────
#  BraggPhase
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class BraggPhase:
    """A crystalline phase loaded through NCrystal.

    Parameters
    ----------
    name           : human-readable label (e.g. ``'martensite'``)
    cfg            : NCrystal cfg string / .ncmat file name, OR a list of
                     candidate cfgs tried in order until one loads
                     (e.g. ``'Fe_sg229_Iron-alpha.ncmat'``)
    temperature_K  : sample temperature; appended to the cfg as ``;temp=...``
                     unless the cfg already specifies one.

    The NCrystal material is loaded lazily on first use so that a ``BraggPhase``
    can be created cheaply (e.g. in a config cell) before NCrystal is touched.
    """

    name: str
    cfg: Union[str, Sequence[str]]
    temperature_K: float = 293.15
    _loaded: object = field(default=None, init=False, repr=False)
    _cfg_used: Optional[str] = field(default=None, init=False, repr=False)

    # ── loading ──────────────────────────────────────────────────────────────
    def _candidate_cfgs(self) -> List[str]:
        cands = [self.cfg] if isinstance(self.cfg, str) else list(self.cfg)
        out = []
        for c in cands:
            if "temp" in c:
                out.append(c)
            else:
                sep = "" if c.rstrip().endswith(";") else ";"
                out.append(f"{c}{sep}temp={self.temperature_K}")
        return out

    def _full_cfg(self) -> str:
        # Back-compat: first candidate's fully-qualified cfg.
        return self._candidate_cfgs()[0]

    @property
    def loaded(self):
        if self._loaded is None:
            _require_ncrystal()
            errors = []
            for cfg in self._candidate_cfgs():
                try:
                    self._loaded = _NC.load(cfg)
                    self._cfg_used = cfg
                    break
                except Exception as exc:                     # try next candidate
                    errors.append(f"    {cfg!r}\n      -> {type(exc).__name__}: {exc}")
            if self._loaded is None:
                hint = ""
                if NCRYSTAL_AVAILABLE:
                    hint = ("\nUse available_ncmat('Fe') / nctool --browse to see what your "
                            "NCrystal build provides, then pass a working cfg to load_phase().")
                raise RuntimeError(
                    f"Could not load NCrystal phase {self.name!r}. Tried:\n"
                    + "\n".join(errors) + hint)
        return self._loaded

    @property
    def cfg_used(self) -> Optional[str]:
        """The cfg string that actually loaded (set after first use)."""
        return self._cfg_used

    @property
    def info(self):
        return self.loaded.info

    # ── bulk properties ────────────────────────────────────────────────────────
    @property
    def density_gcc(self) -> float:
        """Mass density [g/cm³]."""
        return _as_float(self.info.density)

    @property
    def number_density_per_A3(self) -> float:
        """Atomic number density [atoms·Å⁻³]."""
        return _as_float(self.info.numberdensity)

    # ── cross sections (barn / atom) ────────────────────────────────────────────
    def sigma_scatter_barn(self, wl_A) -> np.ndarray:
        return _xsect_barn(self.loaded.scatter, wl_A)

    def sigma_absorption_barn(self, wl_A) -> np.ndarray:
        return _xsect_barn(self.loaded.absorption, wl_A)

    def sigma_total_barn(self, wl_A) -> np.ndarray:
        """Total microscopic cross section = scatter + absorption [barn]."""
        return self.sigma_scatter_barn(wl_A) + self.sigma_absorption_barn(wl_A)

    # ── macroscopic attenuation (cm⁻¹) ──────────────────────────────────────────
    def mu_total_cm(self, wl_A) -> np.ndarray:
        """Macroscopic total attenuation μ(λ) [cm⁻¹].

        μ = n[atoms·Å⁻³] × σ_total[barn]   (the unit factors cancel exactly).
        """
        return self.number_density_per_A3 * self.sigma_total_barn(wl_A)

    def mu_at_energy_cm(self, E_meV) -> np.ndarray:
        """Macroscopic total attenuation as a function of neutron energy [cm⁻¹]."""
        return self.mu_total_cm(energy_meV_to_wavelength_A(E_meV))

    # ── Bragg edges ──────────────────────────────────────────────────────────────
    def bragg_edges_A(self, n_strongest: int = 6) -> np.ndarray:
        """Bragg-edge wavelengths λ = 2·d (Å), longest first.

        Pulled directly from NCrystal's reflection list when available; the
        list is ranked by structure factor × multiplicity so the strongest
        (most visible) edges come first.  Returns an empty array if the phase
        exposes no hkl list (e.g. a purely amorphous material).
        """
        try:
            entries = list(self.info.hklList())
        except Exception:
            return np.array([], dtype=float)

        rows = []
        for hkl in entries:
            d = getattr(hkl, "dspacing", None)
            if d is None:
                d = getattr(hkl, "d", None)
            if d is None:
                continue
            fsq = getattr(hkl, "fsquared", getattr(hkl, "f2", 1.0))
            mult = getattr(hkl, "multiplicity", getattr(hkl, "mult", 1))
            rows.append((2.0 * float(d), float(fsq) * float(mult)))

        if not rows:
            return np.array([], dtype=float)

        rows.sort(key=lambda r: r[1], reverse=True)          # by visibility
        lam = np.array([r[0] for r in rows[: max(n_strongest, 1)]], dtype=float)
        return np.sort(lam)[::-1]                             # longest first


def load_phase(
    name_or_cfg: Union[str, BraggPhase],
    temperature_K: float = 293.15,
    name: Optional[str] = None,
) -> BraggPhase:
    """Build a :class:`BraggPhase`.

    ``name_or_cfg`` may be:
      * a key of :data:`DEFAULT_PHASE_CFG` (``'martensite'`` / ``'austenite'``),
      * an NCrystal cfg string / .ncmat file name, or
      * an existing :class:`BraggPhase` (returned unchanged).
    """
    if isinstance(name_or_cfg, BraggPhase):
        return name_or_cfg
    if name_or_cfg in DEFAULT_PHASE_CFG:
        return BraggPhase(
            name=name or name_or_cfg,
            cfg=DEFAULT_PHASE_CFG[name_or_cfg],
            temperature_K=temperature_K,
        )
    return BraggPhase(
        name=name or name_or_cfg,
        cfg=name_or_cfg,
        temperature_K=temperature_K,
    )


# ──────────────────────────────────────────────────────────────────────────────
#  Per-label μ(λ) lookup table
# ──────────────────────────────────────────────────────────────────────────────

def mu_n_bragg_lut(
    phases_by_label: Sequence[Optional[Union[str, BraggPhase]]],
    wavelengths_A: np.ndarray,
    temperature_K: float = 293.15,
) -> np.ndarray:
    """Build a neutron μ(λ) lookup table indexed by material label.

    Parameters
    ----------
    phases_by_label : list indexed by label value.  Each entry is either a
                      :class:`BraggPhase` / cfg-string / preset key, or ``None``
                      for a non-crystalline / vacuum label (e.g. air → μ = 0).
    wavelengths_A   : 1-D array of monochromatic wavelengths (Å).
    temperature_K   : temperature for any phase given by string/key.

    Returns
    -------
    mu_lut : np.ndarray, shape ``(n_labels, n_wavelengths)``, float32, cm⁻¹.
    """
    wl = np.asarray(wavelengths_A, dtype=float)
    n_lab = len(phases_by_label)
    mu_lut = np.zeros((n_lab, wl.size), dtype=np.float32)

    for label, entry in enumerate(phases_by_label):
        if entry is None:
            continue                          # air / vacuum → μ = 0
        phase = load_phase(entry, temperature_K=temperature_K)
        mu_lut[label, :] = phase.mu_total_cm(wl).astype(np.float32)

    return mu_lut


# ──────────────────────────────────────────────────────────────────────────────
#  DIANA pipeline integration
# ──────────────────────────────────────────────────────────────────────────────

def set_phantom_neutron_mu(phantom, mu_per_label: np.ndarray) -> None:
    """Overwrite a ``PhantomData`` object's neutron attenuation volumes.

    Implements the documented patch pattern (DIANA technical reference §2.4):
    the full per-label μ is placed in the absorption channel and the coherent
    and incoherent channels are zeroed, so that the projector's total
    ``μ_abs + μ_coh + μ_inc`` reproduces exactly the requested μ(λ).

    Parameters
    ----------
    phantom      : ``neutron_xray_sim.PhantomData``
    mu_per_label : array of length ≥ (max label + 1), μ in cm⁻¹.
    """
    mu = np.asarray(mu_per_label, dtype=np.float32)
    labels = phantom.label_vol
    if int(labels.max()) >= mu.size:
        raise ValueError(
            f"mu_per_label has {mu.size} entries but label volume contains "
            f"label {int(labels.max())}."
        )

    abs_vol = mu[labels]                                   # fancy index → (Nz,Nx,Ny)
    phantom.mu_n_abs_vol = abs_vol.astype(np.float32)
    phantom.mu_n_coh_vol = np.zeros_like(abs_vol, dtype=np.float32)
    phantom.mu_n_inc_vol = np.zeros_like(abs_vol, dtype=np.float32)
    phantom.mu_n_vol = abs_vol.astype(np.float32)          # keep total consistent


def run_bragg_edge_scan(
    phantom,
    phases_by_label: Sequence[Optional[Union[str, BraggPhase]]],
    wavelengths_A: np.ndarray,
    angles_deg: np.ndarray,
    algorithm: str = "FBP",
    I0: float = 1e6,
    use_astra: bool = True,
    temperature_K: float = 293.15,
    recon_kwargs: Optional[dict] = None,
    verbose: bool = True,
) -> dict:
    """Project + reconstruct a monochromatic neutron wavelength scan.

    For every wavelength the per-label μ(λ) is written onto the phantom with
    :func:`set_phantom_neutron_mu`, the neutron sinogram is computed with the
    package's ``project_neutron`` and reconstructed with ``reconstruct``.

    Parameters
    ----------
    phantom         : ``PhantomData`` (geometry + labels).
    phases_by_label : phases indexed by label (``None`` → μ = 0); see
                      :func:`mu_n_bragg_lut`.
    wavelengths_A   : 1-D array of monochromatic wavelengths (Å).
    angles_deg      : projection angles (deg).
    algorithm       : reconstruction algorithm passed to ``reconstruct``.
    I0              : incident neutron count (kept high → essentially noiseless).
    use_astra       : prefer ASTRA GPU projection/reconstruction if available.
    temperature_K   : NCrystal sample temperature.
    recon_kwargs    : extra keyword arguments forwarded to ``reconstruct``.
    verbose         : print per-wavelength progress.

    Returns
    -------
    dict with keys:
      ``wavelengths_A``   (n_wl,)
      ``energies_meV``    (n_wl,)
      ``mu_lut``          (n_labels, n_wl)   ground-truth μ(λ) per label [cm⁻¹]
      ``vol_n``           (n_wl, Nz, N, N)    reconstructed neutron volumes [cm⁻¹]
    """
    # Local imports so this module can be used for physics only, without a
    # working DIANA/ASTRA install.
    from neutron_xray_sim import project_neutron, reconstruct

    wl = np.asarray(wavelengths_A, dtype=float)
    recon_kwargs = dict(recon_kwargs or {})
    mu_lut = mu_n_bragg_lut(phases_by_label, wl, temperature_K=temperature_K)

    n_wl = wl.size
    Nz = phantom.label_vol.shape[0]
    N = phantom.label_vol.shape[2]            # detector pixels (Ny)
    vol_n = np.zeros((n_wl, Nz, N, N), dtype=np.float32)

    for i, lam in enumerate(wl):
        if verbose:
            E = float(wavelength_A_to_energy_meV(lam))
            print(f"  [{i + 1:2d}/{n_wl}] λ = {lam:5.3f} Å  (E = {E:6.3f} meV)")
        set_phantom_neutron_mu(phantom, mu_lut[:, i])
        sino = project_neutron(phantom, angles_deg, use_astra=use_astra, I0=I0)
        vol_n[i] = reconstruct(sino, algorithm=algorithm,
                               use_astra=use_astra, **recon_kwargs)

    return {
        "wavelengths_A": wl,
        "energies_meV": wavelength_A_to_energy_meV(wl),
        "mu_lut": mu_lut,
        "vol_n": vol_n,
    }


# ──────────────────────────────────────────────────────────────────────────────
#  Phase classification from a measured wavelength stack
# ──────────────────────────────────────────────────────────────────────────────

def classify_phases(
    spectra: np.ndarray,
    reference_mu: np.ndarray,
    mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Assign each pixel to the reference phase its spectrum matches best.

    Uses cosine similarity (scale-invariant, so robust to a constant
    reconstruction bias) between each pixel's μ(λ) spectrum and every reference
    phase spectrum.

    Parameters
    ----------
    spectra      : (n_wl, ...) measured μ(λ) stack (any trailing spatial shape).
    reference_mu : (n_phases, n_wl) ground-truth μ(λ) per phase
                   (e.g. ``mu_lut[1:, :]`` excluding air).
    mask         : optional boolean array over the spatial shape; pixels where
                   ``mask`` is False are labelled 0 (background).

    Returns
    -------
    phase_index : integer array over the spatial shape.  0 = background;
                  ``k + 1`` = best-matching row ``k`` of ``reference_mu``.
    """
    n_wl = spectra.shape[0]
    spatial = spectra.shape[1:]
    S = spectra.reshape(n_wl, -1).T                         # (n_pix, n_wl)
    R = np.asarray(reference_mu, dtype=float)               # (n_phases, n_wl)

    S_norm = np.linalg.norm(S, axis=1, keepdims=True)
    R_norm = np.linalg.norm(R, axis=1, keepdims=True)
    S_norm[S_norm == 0] = 1.0
    R_norm[R_norm == 0] = 1.0

    cos = (S @ R.T) / (S_norm * R_norm.T)                   # (n_pix, n_phases)
    best = np.argmax(cos, axis=1) + 1                       # phase index (1-based)
    out = best.reshape(spatial).astype(np.int16)

    if mask is not None:
        out = np.where(mask, out, 0).astype(np.int16)
    return out
