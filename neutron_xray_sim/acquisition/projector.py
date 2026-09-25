"""
neutron_xray_sim.acquisition.projector
──────────────────────────────────────
Forward-projection engine for dual-modality CT simulation.

Both modalities use the Beer-Lambert law, with physics-appropriate
polychromatic extensions:

  X-ray (polychromatic, Kramers bremsstrahlung spectrum)
  --------------------------------------------------------
    I = Σ_E  W(E) · exp(−Σ_v μ_x(E,v) · Δl)

  Neutron (thermal, monochromatic approximation)
  -----------------------------------------------
    I = exp(−Σ_v [μ_abs(v) + μ_coh(v) + μ_inc(v)] · Δl)
        (scattered contribution is added in ``artifacts``)

Every projection goes through :func:`line_integrals`, which returns
``Σ_v μ(v)`` along each ray (in voxel units) for one volume.  Two backends:

* **ASTRA** (GPU) — used when ``astra`` is importable and ``use_astra=True``.
* **NumPy** (CPU) — rotation + summation.  Its geometry exactly matches
  ``skimage.transform.radon`` (and therefore the CPU FBP in
  ``reconstruction``), so CPU projection → CPU reconstruction reproduces the
  phantom voxel-for-voxel.

Sinogram layout: ``(n_angles, n_slices, n_det)`` with ``n_slices = Nz`` and
``n_det = Ny``.
"""

from __future__ import annotations

import warnings
from typing import Optional

import numpy as np
from scipy.ndimage import affine_transform, zoom

from ..phantoms.base import PhantomData
from ..physics.materials import xray_spectrum

__all__ = [
    "ASTRA_OK",
    "line_integrals",
    "project_xray",
    "project_xray_monochromatic",
    "project_neutron",
    "make_sinogram_pair",
]


# ──────────────────────────────────────────────────────────────────────────────
# Backend availability
# ──────────────────────────────────────────────────────────────────────────────

def _astra_available() -> bool:
    try:
        import astra  # noqa: F401
        return True
    except ImportError:
        return False


ASTRA_OK = _astra_available()


_warned_no_astra = False


def _use_astra(use_astra: bool, n_x: int, n_det: int) -> bool:
    """Decide the backend; warn (once per session) about a missing ASTRA."""
    global _warned_no_astra
    if not use_astra:
        return False
    if not ASTRA_OK:
        if not _warned_no_astra:
            warnings.warn("ASTRA not available — using the NumPy (CPU) projector. "
                          "Pass use_astra=False to silence this.", stacklevel=3)
            _warned_no_astra = True
        return False
    if n_x != n_det:
        warnings.warn(
            "Non-square projection slices: using the NumPy fallback because the "
            "ASTRA path assumes square 2-D slices.",
            stacklevel=3,
        )
        return False
    return True


# ──────────────────────────────────────────────────────────────────────────────
# Core line-integral operators
# ──────────────────────────────────────────────────────────────────────────────

def _ray_sum_numpy(vol3d: np.ndarray, angle_deg: float) -> np.ndarray:
    """
    Parallel-beam projection of every z-slice at one angle (CPU).

    Rotates each (x, y) slice about the detector centre ``n // 2`` and sums
    along x, matching ``skimage.transform.radon`` exactly.

    Returns
    -------
    (n_slices, n_det) array of line integrals in voxel units.
    """
    _, nx, ny = vol3d.shape
    centre = np.array([0.0, nx // 2, ny // 2])
    t = np.deg2rad(angle_deg)
    c, s = np.cos(t), np.sin(t)
    # Output voxel o samples input R @ o + offset (rotation by −angle), which
    # is skimage's convention for a projection at +angle.
    R = np.array([[1.0, 0.0, 0.0],
                  [0.0, c, -s],
                  [0.0, s, c]])
    offset = centre - R @ centre
    rotated = affine_transform(vol3d, R, offset=offset, order=1,
                               mode="constant", cval=0.0)
    return rotated.sum(axis=1)


def _magnify_numpy(proj2d: np.ndarray, SDD: float, SOD: float) -> np.ndarray:
    """First-order cone-beam approximation: magnify along the detector axis."""
    M = SDD / SOD
    n_det = proj2d.shape[1]
    proj_mag = zoom(proj2d, (1.0, M), order=1)
    n_mag = proj_mag.shape[1]
    if n_mag >= n_det:
        start = (n_mag - n_det) // 2
        return proj_mag[:, start:start + n_det]
    pad = n_det - n_mag
    return np.pad(proj_mag, ((0, 0), (pad // 2, pad - pad // 2)))


def _fit_detector(sino: np.ndarray, n_det: int) -> np.ndarray:
    """Centre-crop or zero-pad the last axis of *sino* to *n_det* pixels."""
    det = sino.shape[-1]
    if det >= n_det:
        start = (det - n_det) // 2
        return sino[..., start:start + n_det]
    pad = n_det - det
    return np.pad(sino, [(0, 0)] * (sino.ndim - 1) + [(pad // 2, pad - pad // 2)])


def line_integrals(
    vol3d: np.ndarray,
    angles_deg: np.ndarray,
    use_astra: bool = True,
    geometry: str = "parallel",
    SDD: float = 1000.0,
    SOD: float = 500.0,
) -> np.ndarray:
    """
    Line integrals ``Σ μ`` of a volume for every angle, slice and detector pixel.

    Parameters
    ----------
    vol3d      : (n_slices, n_x, n_det) attenuation volume (any units)
    angles_deg : projection angles [deg]
    use_astra  : use ASTRA (GPU) if available
    geometry   : ``'parallel'`` or ``'cone'`` (fan beam per slice)
    SDD, SOD   : source–detector / source–object distance [pixels]; cone only

    Returns
    -------
    (n_angles, n_slices, n_det) float32, in ``vol3d`` units × voxels.
    Multiply by ``voxel_cm`` to get optical depth.
    """
    if geometry not in {"parallel", "cone"}:
        raise ValueError(f"Unknown geometry={geometry!r}. Use 'parallel' or 'cone'.")
    if vol3d.ndim != 3:
        raise ValueError(f"Expected a 3-D volume, got shape {vol3d.shape}.")

    angles_deg = np.asarray(angles_deg, dtype=float)
    n_slices, n_x, n_det = vol3d.shape
    out = np.zeros((len(angles_deg), n_slices, n_det), dtype=np.float32)

    if _use_astra(use_astra, n_x, n_det):
        import astra

        angles_rad = np.radians(angles_deg)
        vol_geom = astra.create_vol_geom(n_x, n_det)
        if geometry == "cone":
            det_count = int(round(n_det * SDD / SOD))
            proj_geom = astra.create_proj_geom(
                "fanflat", 1.0, det_count, angles_rad, SOD, SDD - SOD)
        else:
            proj_geom = astra.create_proj_geom("parallel", 1.0, n_det, angles_rad)
        for s_idx in range(n_slices):
            sino_id, sino = astra.create_sino(
                np.ascontiguousarray(vol3d[s_idx], dtype=np.float32),
                proj_geom, vol_geom)
            astra.data2d.delete(sino_id)
            out[:, s_idx, :] = _fit_detector(sino, n_det)
        return out

    vol = np.asarray(vol3d, dtype=np.float32)
    for a_idx, angle in enumerate(angles_deg):
        proj = _ray_sum_numpy(vol, angle)
        if geometry == "cone":
            proj = _magnify_numpy(proj, SDD, SOD)
        out[a_idx] = proj
    return out


def _to_lambda(sino_trans: np.ndarray, I0: float) -> np.ndarray:
    """Transmission → optical depth, floored at 1/(10·I0) to avoid log(0)."""
    eps = 1.0 / (10 * I0)
    return -np.log(np.clip(sino_trans, eps, 1.0))


# ──────────────────────────────────────────────────────────────────────────────
# X-ray projectors
# ──────────────────────────────────────────────────────────────────────────────

def _build_xray_mu_volume(phantom: PhantomData, energy_keV: float) -> np.ndarray:
    """3-D X-ray attenuation volume [cm⁻¹] at one energy (via a per-label LUT)."""
    lut = np.array([m.mu_x_at(energy_keV) for m in phantom.materials],
                   dtype=np.float32)
    return lut[phantom.label_vol]


def project_xray_monochromatic(
    phantom: PhantomData,
    angles_deg: np.ndarray,
    energy_keV: float,
    use_astra: bool = True,
    I0: float = 1e5,
    geometry: str = "parallel",
    SDD: float = 1000.0,
    SOD: float = 500.0,
) -> dict:
    """
    Monochromatic X-ray sinograms (Beer-Lambert at a single energy).

    Returns
    -------
    dict with ``sino_lam`` / ``sino_trans`` of shape (n_angles, n_slices, n_det),
    plus ``angles_deg``, ``energy_keV``, ``I0``, ``voxel_cm``, ``geometry``.
    """
    dx = phantom.voxel_cm
    mu_vol = _build_xray_mu_volume(phantom, energy_keV)
    lam = line_integrals(mu_vol, angles_deg, use_astra, geometry, SDD, SOD) * dx
    sino_trans = np.exp(-lam).astype(np.float32)
    return {
        "sino_lam": _to_lambda(sino_trans, I0),
        "sino_trans": sino_trans,
        "angles_deg": np.asarray(angles_deg),
        "energy_keV": energy_keV,
        "I0": I0,
        "voxel_cm": dx,
        "geometry": geometry,
        "SDD": SDD,
        "SOD": SOD,
    }


def project_xray(
    phantom: PhantomData,
    angles_deg: np.ndarray,
    kVp: float = 120.0,
    filter_mm_Al: float = 2.0,
    filter_mm_Cu: float = 0.0,
    n_spectrum_bins: int = 12,
    use_astra: bool = True,
    I0: float = 1e5,
    geometry: str = "parallel",
    SDD: float = 1000.0,
    SOD: float = 500.0,
) -> dict:
    """
    Polychromatic X-ray sinograms.

    The transmitted intensity is the spectrum-weighted sum of per-energy
    Beer-Lambert transmissions, so beam hardening emerges naturally.

    Returns
    -------
    dict with ``sino_lam`` / ``sino_trans`` of shape (n_angles, n_slices, n_det),
    plus ``angles_deg``, ``spectrum`` (energies, weights, kVp, filters),
    ``I0``, ``voxel_cm``, ``geometry``, ``SDD``, ``SOD``.
    """
    energies_keV, weights = xray_spectrum(kVp, filter_mm_Al, filter_mm_Cu,
                                          n_spectrum_bins)
    dx = phantom.voxel_cm
    n_slices, _, n_det = phantom.label_vol.shape
    sino_trans = np.zeros((len(angles_deg), n_slices, n_det), dtype=np.float32)

    with warnings.catch_warnings():
        # Backend-fallback warnings would repeat once per energy bin.
        warnings.simplefilter("ignore")
        for E, W in zip(energies_keV, weights):
            mu_vol = _build_xray_mu_volume(phantom, E)
            lam = line_integrals(mu_vol, angles_deg, use_astra, geometry, SDD, SOD)
            sino_trans += (W * np.exp(-lam * dx)).astype(np.float32)
    if use_astra:
        _use_astra(True, phantom.label_vol.shape[1], n_det)  # emit warning once

    return {
        "sino_lam": _to_lambda(sino_trans, I0),
        "sino_trans": sino_trans,
        "angles_deg": np.asarray(angles_deg),
        "spectrum": {
            "energies_keV": energies_keV,
            "weights": weights,
            "kVp": kVp,
            "filter_mm_Al": filter_mm_Al,
            "filter_mm_Cu": filter_mm_Cu,
        },
        "I0": I0,
        "voxel_cm": dx,
        "geometry": geometry,
        "SDD": SDD,
        "SOD": SOD,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Neutron projector
# ──────────────────────────────────────────────────────────────────────────────

def project_neutron(
    phantom: PhantomData,
    angles_deg: np.ndarray,
    use_astra: bool = True,
    I0: float = 1e5,
    scatter_D_over_L: float = 100.0,
) -> dict:
    """
    Thermal-neutron sinograms (parallel beam).

    The absorption, coherent and incoherent components are projected
    separately so the scatter-only optical depth is available to the
    scatter artifact model.

    Returns
    -------
    dict with ``sino_lam``, ``sino_trans``, ``sino_abs_lam``,
    ``sino_scatter_lam`` (all (n_angles, n_slices, n_det)), plus
    ``angles_deg``, ``I0``, ``scatter_D_over_L``, ``voxel_cm``.
    """
    label_shape = phantom.label_vol.shape
    dx = phantom.voxel_cm
    components = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for name in ("mu_n_abs_vol", "mu_n_coh_vol", "mu_n_inc_vol"):
            vol = getattr(phantom, name)
            if vol.shape != label_shape:
                raise ValueError(
                    f"phantom.{name} shape {vol.shape} does not match "
                    f"phantom.label_vol shape {label_shape}."
                )
            components[name] = line_integrals(vol, angles_deg, use_astra) * dx
    if use_astra:
        _use_astra(True, label_shape[1], label_shape[2])

    lam_abs = components["mu_n_abs_vol"]
    lam_sca = components["mu_n_coh_vol"] + components["mu_n_inc_vol"]
    sino_trans = np.exp(-(lam_abs + lam_sca)).astype(np.float32)
    eps = 1.0 / (10 * I0)

    return {
        "sino_lam": _to_lambda(sino_trans, I0),
        "sino_trans": sino_trans,
        "sino_abs_lam": np.minimum(lam_abs, -np.log(eps)).astype(np.float32),
        "sino_scatter_lam": np.minimum(lam_sca, -np.log(eps)).astype(np.float32),
        "angles_deg": np.asarray(angles_deg),
        "I0": I0,
        "scatter_D_over_L": scatter_D_over_L,
        "voxel_cm": dx,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Convenience wrapper
# ──────────────────────────────────────────────────────────────────────────────

def make_sinogram_pair(
    phantom: PhantomData,
    n_angles: int = 180,
    angle_range_deg: float = 180.0,
    xray_mode: str = "polychromatic",
    kVp: float = 120.0,
    filter_mm_Al: float = 2.0,
    filter_mm_Cu: float = 0.0,
    n_spectrum_bins: int = 12,
    xray_energy_keV: Optional[float] = None,
    I0_xray: float = 1e5,
    I0_neutron: float = 1e5,
    use_astra: bool = True,
    scatter_D_over_L: float = 100.0,
    geometry: str = "parallel",
    SDD: float = 1000.0,
    SOD: float = 500.0,
    verbose: bool = True,
) -> tuple[dict, dict]:
    """
    Generate X-ray and neutron sinograms for a phantom.

    Parameters
    ----------
    phantom          : PhantomData
    n_angles         : number of projection angles
    angle_range_deg  : total angular range (180 = half-scan, 360 = full)
    xray_mode        : ``'polychromatic'`` or ``'monochromatic'``
    kVp, filter_mm_Al, filter_mm_Cu, n_spectrum_bins
                     : tube spectrum (polychromatic mode)
    xray_energy_keV  : X-ray energy [keV] (monochromatic mode)
    I0_xray, I0_neutron : incident counts (sets the log(0) floor; the noise
                     itself is added by ``artifacts``)
    use_astra        : use ASTRA (GPU) if available
    scatter_D_over_L : neutron beam collimation ratio D/L
    geometry         : X-ray beam geometry, ``'parallel'`` or ``'cone'``.
                       Neutron projection is always parallel beam.
    SDD, SOD         : source–detector / source–object distance [pixels]
                       (cone only; magnification = SDD / SOD)
    verbose          : print progress

    Returns
    -------
    (xray_sino, neutron_sino) dicts — see :func:`project_xray`,
    :func:`project_xray_monochromatic` and :func:`project_neutron`.
    """
    angles = np.linspace(0.0, angle_range_deg, n_angles, endpoint=False)
    log = print if verbose else (lambda *a, **k: None)

    backend = "ASTRA GPU" if (use_astra and ASTRA_OK) else "NumPy CPU"
    log(f"[projector] Projecting {n_angles} angles ({backend}) …")

    if xray_mode == "polychromatic":
        log("  → X-ray (polychromatic) …")
        xray = project_xray(
            phantom, angles,
            kVp=kVp, filter_mm_Al=filter_mm_Al, filter_mm_Cu=filter_mm_Cu,
            n_spectrum_bins=n_spectrum_bins, use_astra=use_astra, I0=I0_xray,
            geometry=geometry, SDD=SDD, SOD=SOD,
        )
    elif xray_mode == "monochromatic":
        if xray_energy_keV is None:
            raise ValueError(
                "xray_energy_keV must be provided when xray_mode='monochromatic'."
            )
        log(f"  → X-ray (monochromatic, {xray_energy_keV:.1f} keV) …")
        xray = project_xray_monochromatic(
            phantom, angles, energy_keV=xray_energy_keV, use_astra=use_astra,
            I0=I0_xray, geometry=geometry, SDD=SDD, SOD=SOD,
        )
    else:
        raise ValueError(
            f"Unknown xray_mode={xray_mode!r}. Use 'polychromatic' or 'monochromatic'."
        )

    log("  → Neutron (thermal) …")
    neutron = project_neutron(
        phantom, angles, use_astra=use_astra, I0=I0_neutron,
        scatter_D_over_L=scatter_D_over_L,
    )
    log("[projector] Done.")
    return xray, neutron
