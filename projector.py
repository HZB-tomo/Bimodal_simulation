"""
neutron_xray_sim/projector.py
──────────────────────────────
Forward-projection engine for dual-modality CT simulation.

Both modalities use the Beer-Lambert law, but with physics-appropriate
polychromatic extensions:

  X-ray (polychromatic, Kramers bremsstrahlung spectrum)
  --------------------------------------------------------
    I = Σ_E  W(E) · exp(−Σ_v μ_x(E,v) · Δl)

  Neutron (thermal, monochromatic approximation)
  -----------------------------------------------
    I = exp(−Σ_v [μ_abs(v) + μ_coh(v) + μ_inc(v)] · Δl)
        + S_scatter(v)     ← scattered contribution (added in artifacts.py)

ASTRA Toolbox is used for GPU-accelerated parallel-beam projection when
available; a pure-NumPy fallback (line sums along axis 1 after rotation)
is provided so that the package works on CPU-only machines.
"""

from __future__ import annotations

import warnings
import numpy as np
from scipy.ndimage import rotate as ndimage_rotate

from .phantom import PhantomData
from .materials import xray_spectrum, XRAY_E_KEV

__all__ = ["project_xray", "project_neutron", "make_sinogram_pair"]


# ──────────────────────────────────────────────────────────────────────────────
# ASTRA availability check
# ──────────────────────────────────────────────────────────────────────────────

def _astra_available() -> bool:
    try:
        import astra
        return True
    except ImportError:
        return False


ASTRA_OK = _astra_available()


# ──────────────────────────────────────────────────────────────────────────────
# Core projection helpers
# ──────────────────────────────────────────────────────────────────────────────

def _ray_sum_numpy(vol3d: np.ndarray, angle_deg: float) -> np.ndarray:
    """
    Project a 3-D volume along axis-1 (x) after rotating in the x-z plane.

    Parameters
    ----------
    vol3d     : (N, N, N) attenuation volume  [cm⁻¹]
    angle_deg : rotation angle in degrees

    Returns
    -------
    proj2d    : (N, N) projection [sum of μ · Δx, in cm⁻¹·cm = dimensionless
                  when multiplied by voxel_cm]
    """
    # Rotate in the horizontal (axis 0 = y = vertical, axes (1,2) = x-z plane)
    rotated = ndimage_rotate(vol3d, angle_deg, axes=(1, 2),
                             reshape=False, order=1, mode="constant", cval=0.0)
    return rotated.sum(axis=1)   # sum along x → (N, N) image


def _astra_project_2d(vol2d: np.ndarray, angles_rad: np.ndarray) -> np.ndarray:
    """
    GPU-accelerated 2-D parallel-beam forward projection via ASTRA.

    Parameters
    ----------
    vol2d      : (N, N) single slice  [any units]
    angles_rad : projection angles  [radians]

    Returns
    -------
    sino       : (n_angles, N_det) sinogram
    """
    import astra
    N = vol2d.shape[0]
    vol_geom  = astra.create_vol_geom(N, N)
    proj_geom = astra.create_proj_geom("parallel", 1.0, N, angles_rad)
    proj_id   = astra.create_projector("cuda", proj_geom, vol_geom)
    _, sino   = astra.create_sino(vol2d, proj_id)
    astra.projector.delete(proj_id)
    return sino




# ──────────────────────────────────────────────────────────────────────────────
# Cone-beam projection helper
# ──────────────────────────────────────────────────────────────────────────────

def _astra_project_cone_2d(
    vol2d: np.ndarray,
    angles_rad: np.ndarray,
    SDD: float,
    SOD: float,
) -> np.ndarray:
    """
    2-D fan-beam (cone-beam central slice) projection via ASTRA.

    Uses ASTRA's ``fanflat`` geometry, which is the 2-D equivalent of
    3-D cone-beam and produces a fan-beam sinogram identical to the central
    slice of a 3-D cone-beam acquisition.

    Parameters
    ----------
    vol2d      : (N, N) single axial slice  [any units]
    angles_rad : (n_angles,) projection angles [rad]
    SDD        : source-to-detector distance [pixels]
    SOD        : source-to-object (iso-centre) distance [pixels]
                 Magnification M = SDD / SOD.  For parallel beam: SOD → ∞.

    Returns
    -------
    sino2d : (n_angles, N) fan-beam sinogram  [same units as vol2d]

    Notes
    -----
    ASTRA ``fanflat`` geometry convention:
      - ``source_origin`` = SOD  (source to rotation centre)
      - ``origin_det``    = SDD − SOD  (rotation centre to detector)
      - ``det_spacing``   = 1.0 pixel
    """
    import astra
    N   = vol2d.shape[0]
    det_count = int(round(N * SDD / SOD))   # physical detector width at the detector plane

    vol_geom  = astra.create_vol_geom(N, N)
    proj_geom = astra.create_proj_geom(
        "fanflat",
        1.0,            # detector pixel spacing [pixels]
        det_count,      # number of detector pixels
        angles_rad,
        SOD,            # source–origin distance
        SDD - SOD,      # origin–detector distance
    )
    sino_id, sino = astra.create_sino(vol2d.astype(np.float32), proj_geom,
                                      vol_geom)
    astra.data2d.delete(sino_id)
    return sino   # (n_angles, det_count)


def _project_cone_numpy(
    vol3d: np.ndarray,
    angle_deg: float,
    SDD: float,
    SOD: float,
) -> np.ndarray:
    """
    CPU fallback for fan/cone-beam projection using ray-rescaling.

    Implements a flat-detector magnification correction: each parallel-beam
    projection line integral at detector position u is remapped to the
    fan-beam position u' = u · SOD / SDD, which approximates the cone-beam
    geometry to first order.  This is less accurate than ASTRA's exact
    fan-beam back-projection but provides a consistent fallback when ASTRA is
    unavailable.

    Parameters
    ----------
    vol3d     : (N, N, N) attenuation volume
    angle_deg : projection angle [°]
    SDD       : source-to-detector distance [pixels]
    SOD       : source-to-object distance [pixels]

    Returns
    -------
    proj2d : (N, N) magnified projection  [same units as vol3d]
    """
    from scipy.ndimage import zoom

    M       = SDD / SOD          # geometric magnification
    proj2d  = _ray_sum_numpy(vol3d, angle_deg)   # (N, N) parallel projection
    # Rescale in the detector direction (axis 1) to simulate magnification
    proj_mag = zoom(proj2d, (1.0, M), order=1)
    # Crop or pad to original N along the detector axis
    N_det = proj2d.shape[1]
    N_mag = proj_mag.shape[1]
    if N_mag >= N_det:
        start = (N_mag - N_det) // 2
        return proj_mag[:, start:start + N_det]
    else:
        pad = N_det - N_mag
        return np.pad(proj_mag, ((0, 0), (pad // 2, pad - pad // 2)))

# ──────────────────────────────────────────────────────────────────────────────
# X-ray projector
# ──────────────────────────────────────────────────────────────────────────────


def _build_xray_mu_volume(phantom: PhantomData, energy_keV: float) -> np.ndarray:
    """
    Build 3-D X-ray attenuation volume at a single energy.
    Works for non-cubic phantoms.
    """
    shape = phantom.label_vol.shape
    mu_vol = np.zeros(shape, dtype=np.float32)

    for m_idx, mat in enumerate(phantom.materials):
        mask = phantom.label_vol == m_idx
        if mask.any():
            mu_vol[mask] = mat.mu_x_at(energy_keV)

    return mu_vol


def project_xray_monochromatic(
    phantom: PhantomData,
    angles_deg: np.ndarray,
    energy_keV: float,
    use_astra: bool = True,
    I0: float = 1e5,
) -> dict:
    """
    Compute monochromatic X-ray sinograms.

    Uses Beer-Lambert law at a single X-ray energy:
        I = exp(-mu(E) * L)

    Parameters
    ----------
    phantom      : PhantomData object
    angles_deg   : 1-D array of projection angles [deg]
    energy_keV   : monochromatic X-ray energy [keV]
    use_astra    : use ASTRA GPU projection if available
    I0           : incident photon count (for Poisson noise later)

    Returns
    -------
    dict with keys:
        'sino_lam'   : (n_angles, N, N) log-attenuation sinogram
        'sino_trans' : (n_angles, N, N) transmission sinogram [0..1]
        'angles_deg' : copy of angles_deg
        'energy_keV' : monochromatic energy
        'I0'         : incident photon count
    """
    N = phantom.N
    dx = phantom.voxel_cm
    n_angles = len(angles_deg)
    angles_rad = np.radians(angles_deg)

    sino_trans = np.zeros((n_angles, N, N), dtype=np.float32)
    use_gpu = use_astra and ASTRA_OK

    mu_vol = _build_xray_mu_volume(phantom, energy_keV)

    if use_gpu:
        import astra
        vol_geom = astra.create_vol_geom(N, N)
        proj_geom = astra.create_proj_geom("parallel", 1.0, N, angles_rad)
        proj_id = astra.create_projector("cuda", proj_geom, vol_geom)

        for s_idx in range(N):
            _, sino_slice = astra.create_sino(mu_vol[s_idx], proj_id)
            sino_trans[:, s_idx, :] = np.exp(-sino_slice * dx)

        astra.projector.delete(proj_id)

    else:
        if use_astra and not ASTRA_OK:
            warnings.warn("ASTRA not available — using NumPy fallback (slower).")

        for a_idx, angle in enumerate(angles_deg):
            proj2d = _ray_sum_numpy(mu_vol, angle)
            sino_trans[a_idx] = np.exp(-proj2d * dx)

    eps = 1.0 / (10 * I0)
    sino_lam = -np.log(np.clip(sino_trans, eps, 1.0))

    return {
        "sino_lam": sino_lam,
        "sino_trans": sino_trans,
        "angles_deg": angles_deg,
        "energy_keV": energy_keV,
        "I0": I0,
        "voxel_cm": phantom.voxel_cm,
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
    Compute polychromatic X-ray sinograms.

    Supports non-cubic phantoms with shape:
        phantom.label_vol.shape == (n_slices, n_x, n_det)

    Output shape:
        sino_trans, sino_lam == (n_angles, n_slices, n_det)
    """
    energies_keV, weights = xray_spectrum(
        kVp,
        filter_mm_Al,
        filter_mm_Cu,
        n_spectrum_bins,
    )

    if geometry not in {"parallel", "cone"}:
        raise ValueError(
            f"Unknown geometry={geometry!r}. Use 'parallel' or 'cone'."
        )

    label_shape = phantom.label_vol.shape
    if len(label_shape) != 3:
        raise ValueError(
            f"phantom.label_vol must be 3-D, got shape {label_shape}."
        )

    n_slices, n_x, n_det = label_shape
    dx = phantom.voxel_cm
    n_angles = len(angles_deg)
    angles_rad = np.radians(angles_deg)

    sino_trans = np.zeros((n_angles, n_slices, n_det), dtype=np.float32)

    # ASTRA 2-D projector assumes square slices in this implementation.
    # For rectangular / non-cubic phantoms, use the NumPy fallback.
    astra_safe = (
        ASTRA_OK
        and use_astra
        and n_x == n_det
    )

    if use_astra and not ASTRA_OK:
        warnings.warn("ASTRA not available — using NumPy fallback.")

    if use_astra and ASTRA_OK and not astra_safe:
        warnings.warn(
            "Non-square projection slices detected. "
            "Using NumPy fallback because the current ASTRA path assumes "
            "square 2-D slices."
        )

    if astra_safe:
        import astra

        vol_geom = astra.create_vol_geom(n_x, n_det)

        if geometry == "cone":
            for E, W in zip(energies_keV, weights):
                mu_vol = _build_xray_mu_volume(phantom, E)

                for s_idx in range(n_slices):
                    fan_sino = _astra_project_cone_2d(
                        mu_vol[s_idx],
                        angles_rad,
                        SDD,
                        SOD,
                    )

                    det_count = fan_sino.shape[1]

                    if det_count >= n_det:
                        start = (det_count - n_det) // 2
                        fan_sino = fan_sino[:, start:start + n_det]
                    else:
                        pad = n_det - det_count
                        fan_sino = np.pad(
                            fan_sino,
                            ((0, 0), (pad // 2, pad - pad // 2)),
                        )

                    sino_trans[:, s_idx, :] += (
                        W * np.exp(-fan_sino * dx)
                    ).astype(np.float32)

        else:
            proj_geom = astra.create_proj_geom(
                "parallel",
                1.0,
                n_det,
                angles_rad,
            )
            proj_id = astra.create_projector("cuda", proj_geom, vol_geom)

            try:
                for E, W in zip(energies_keV, weights):
                    mu_vol = _build_xray_mu_volume(phantom, E)

                    for s_idx in range(n_slices):
                        _, sino_slice = astra.create_sino(
                            mu_vol[s_idx].astype(np.float32),
                            proj_id,
                        )
                        sino_trans[:, s_idx, :] += (
                            W * np.exp(-sino_slice * dx)
                        ).astype(np.float32)

            finally:
                astra.projector.delete(proj_id)

    else:
        for E, W in zip(energies_keV, weights):
            mu_vol = _build_xray_mu_volume(phantom, E)

            if geometry == "cone":
                for a_idx, angle in enumerate(angles_deg):
                    proj2d = _project_cone_numpy(mu_vol, angle, SDD, SOD)

                    if proj2d.shape != (n_slices, n_det):
                        raise ValueError(
                            "Cone NumPy projection returned unexpected shape "
                            f"{proj2d.shape}; expected {(n_slices, n_det)}."
                        )

                    sino_trans[a_idx] += (
                        W * np.exp(-proj2d * dx)
                    ).astype(np.float32)

            else:
                for a_idx, angle in enumerate(angles_deg):
                    proj2d = _ray_sum_numpy(mu_vol, angle)

                    if proj2d.shape != (n_slices, n_det):
                        raise ValueError(
                            "Parallel NumPy projection returned unexpected "
                            f"shape {proj2d.shape}; expected "
                            f"{(n_slices, n_det)}."
                        )

                    sino_trans[a_idx] += (
                        W * np.exp(-proj2d * dx)
                    ).astype(np.float32)

    eps = 1.0 / (10 * I0)
    sino_lam = -np.log(np.clip(sino_trans, eps, 1.0))

    return {
        "sino_lam": sino_lam,
        "sino_trans": sino_trans,
        "angles_deg": angles_deg,
        "spectrum": {
            "energies_keV": energies_keV,
            "weights": weights,
        },
        "I0": I0,
        "voxel_cm": phantom.voxel_cm,
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
    Compute thermal-neutron sinograms.

    Supports non-cubic phantoms with shape:
        phantom.label_vol.shape == (n_slices, n_x, n_det)

    Output shape:
        sino_trans, sino_lam == (n_angles, n_slices, n_det)
    """
    label_shape = phantom.label_vol.shape
    if len(label_shape) != 3:
        raise ValueError(
            f"phantom.label_vol must be 3-D, got shape {label_shape}."
        )

    n_slices, n_x, n_det = label_shape
    dx = phantom.voxel_cm
    n_angles = len(angles_deg)
    angles_rad = np.radians(angles_deg)

    for name, vol in [
        ("mu_n_abs_vol", phantom.mu_n_abs_vol),
        ("mu_n_coh_vol", phantom.mu_n_coh_vol),
        ("mu_n_inc_vol", phantom.mu_n_inc_vol),
    ]:
        if vol.shape != label_shape:
            raise ValueError(
                f"phantom.{name} shape {vol.shape} does not match "
                f"phantom.label_vol shape {label_shape}."
            )

    sino_abs_trans = np.zeros((n_angles, n_slices, n_det), dtype=np.float32)
    sino_coh_trans = np.zeros((n_angles, n_slices, n_det), dtype=np.float32)
    sino_inc_trans = np.zeros((n_angles, n_slices, n_det), dtype=np.float32)

    astra_safe = (
        ASTRA_OK
        and use_astra
        and n_x == n_det
    )

    if use_astra and not ASTRA_OK:
        warnings.warn("ASTRA not available — using NumPy fallback.")

    if use_astra and ASTRA_OK and not astra_safe:
        warnings.warn(
            "Non-square projection slices detected. "
            "Using NumPy fallback because the current ASTRA path assumes "
            "square 2-D slices."
        )

    if astra_safe:
        import astra

        vol_geom = astra.create_vol_geom(n_x, n_det)
        proj_geom = astra.create_proj_geom(
            "parallel",
            1.0,
            n_det,
            angles_rad,
        )
        proj_id = astra.create_projector("cuda", proj_geom, vol_geom)

        try:
            for vol, target in [
                (phantom.mu_n_abs_vol, sino_abs_trans),
                (phantom.mu_n_coh_vol, sino_coh_trans),
                (phantom.mu_n_inc_vol, sino_inc_trans),
            ]:
                for s_idx in range(n_slices):
                    _, sino_slice = astra.create_sino(
                        vol[s_idx].astype(np.float32),
                        proj_id,
                    )
                    target[:, s_idx, :] = np.exp(-sino_slice * dx)

        finally:
            astra.projector.delete(proj_id)

    else:
        for a_idx, angle in enumerate(angles_deg):
            abs_proj = _ray_sum_numpy(phantom.mu_n_abs_vol, angle)
            coh_proj = _ray_sum_numpy(phantom.mu_n_coh_vol, angle)
            inc_proj = _ray_sum_numpy(phantom.mu_n_inc_vol, angle)

            expected_shape = (n_slices, n_det)
            for name, proj in [
                ("absorption", abs_proj),
                ("coherent", coh_proj),
                ("incoherent", inc_proj),
            ]:
                if proj.shape != expected_shape:
                    raise ValueError(
                        f"{name} NumPy projection returned shape "
                        f"{proj.shape}; expected {expected_shape}."
                    )

            sino_abs_trans[a_idx] = np.exp(-abs_proj * dx)
            sino_coh_trans[a_idx] = np.exp(-coh_proj * dx)
            sino_inc_trans[a_idx] = np.exp(-inc_proj * dx)

    sino_trans = sino_abs_trans * sino_coh_trans * sino_inc_trans

    eps = 1.0 / (10 * I0)
    sino_lam = -np.log(np.clip(sino_trans, eps, 1.0))
    sino_abs_lam = -np.log(np.clip(sino_abs_trans, eps, 1.0))
    sino_scatter_lam = -np.log(
        np.clip(sino_coh_trans * sino_inc_trans, eps, 1.0)
    )

    return {
        "sino_lam": sino_lam,
        "sino_trans": sino_trans,
        "sino_abs_lam": sino_abs_lam,
        "sino_scatter_lam": sino_scatter_lam,
        "angles_deg": angles_deg,
        "I0": I0,
        "scatter_D_over_L": scatter_D_over_L,
        "voxel_cm": phantom.voxel_cm,
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
    xray_energy_keV: float | None = None,
    I0_xray: float = 1e5,
    I0_neutron: float = 1e5,
    use_astra: bool = True,
    scatter_D_over_L: float = 100.0,
    geometry: str = "parallel",
    SDD: float = 1000.0,
    SOD: float = 500.0,
) -> tuple[dict, dict]:
    """
    Generate X-ray and neutron sinogram pairs for a phantom.

    Parameters
    ----------
    phantom          : PhantomData
    n_angles         : number of projection angles
    angle_range_deg  : total angular range (180 = half-scan, 360 = full)
    xray_mode        : 'polychromatic' or 'monochromatic'
    kVp              : X-ray tube voltage [kV] for polychromatic mode
    filter_mm_Al     : aluminium pre-filter [mm] for polychromatic mode
    filter_mm_Cu     : copper pre-filter [mm] for polychromatic mode
    n_spectrum_bins  : polychromatic energy bins
    xray_energy_keV  : monochromatic X-ray energy [keV]
    I0_xray          : incident X-ray photon count
    I0_neutron       : incident neutron count
    use_astra        : use GPU projection if available
    scatter_D_over_L : neutron beam collimation ratio D/L
    geometry         : X-ray beam geometry: ``'parallel'`` (default) or
                       ``'cone'``.  Neutron projection always uses parallel
                       geometry (standard for neutron imaging beamlines).
    SDD              : source-to-detector distance [pixels].  Ignored unless
                       ``geometry='cone'``.
    SOD              : source-to-object distance [pixels].  Ignored unless
                       ``geometry='cone'``.  Magnification = SDD / SOD.

    Returns
    -------
    (xray_sino_dict, neutron_sino_dict) — see project_xray /
    project_xray_monochromatic / project_neutron
    """
    angles = np.linspace(0.0, angle_range_deg, n_angles, endpoint=False)

    _backend = "ASTRA GPU" if (use_astra and ASTRA_OK) else "NumPy CPU"
    print(f"[projector] Projecting {n_angles} angles ({_backend}) …")

    if xray_mode == "polychromatic":
        print("  → X-ray (polychromatic) …")
        xray = project_xray(
            phantom,
            angles,
            kVp=kVp,
            filter_mm_Al=filter_mm_Al,
            filter_mm_Cu=filter_mm_Cu,
            n_spectrum_bins=n_spectrum_bins,
            use_astra=use_astra,
            I0=I0_xray,
            geometry=geometry,
            SDD=SDD,
            SOD=SOD,
        )

    elif xray_mode == "monochromatic":
        if xray_energy_keV is None:
            raise ValueError(
                "xray_energy_keV must be provided when "
                "xray_mode='monochromatic'."
            )

        if (
            kVp != 120.0
            or filter_mm_Al != 2.0
            or filter_mm_Cu != 0.0
            or n_spectrum_bins != 12
        ):
            warnings.warn(
                "kVp, filter_mm_Al, filter_mm_Cu, and n_spectrum_bins "
                "are ignored when xray_mode='monochromatic'."
            )

        print(f"  → X-ray (monochromatic, {xray_energy_keV:.1f} keV) …")
        xray = project_xray_monochromatic(
            phantom,
            angles,
            energy_keV=xray_energy_keV,
            use_astra=use_astra,
            I0=I0_xray,
        )

    else:
        raise ValueError(
            f"Unknown xray_mode={xray_mode!r}. "
            "Use 'polychromatic' or 'monochromatic'."
        )

    print("  → Neutron (thermal) …")
    neutron = project_neutron(
        phantom,
        angles,
        use_astra=use_astra,
        I0=I0_neutron,
        scatter_D_over_L=scatter_D_over_L,
    )

    print("[projector] Done.")
    return xray, neutron