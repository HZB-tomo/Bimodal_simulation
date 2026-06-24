"""
neutron_xray_sim/artifacts.py
──────────────────────────────
Artifact injection for dual-modality CT sinograms and reconstructed volumes.

Every artifact is controlled by a field in ArtifactConfig.  Setting a field
to None or False disables that artifact completely, making it trivial to run
controlled experiments with one artifact at a time or in combinations.

Sinogram-domain artifacts (applied before reconstruction):
──────────────────────────────────────────────────────────
  1. Poisson photon / neutron noise       (photon_noise)
  2. X-ray beam hardening                 (beam_hardening)   emerges from polychromatic
                                          projection; BHC correction can be applied or
                                          deliberately skipped to study the effect
  3. Neutron scatter build-up             (neutron_scatter)  Gaussian halo convolution
  4. Detector point-spread function       (detector_psf)     Gaussian blur per projection
  5. Ring / bad-pixel artifacts           (ring_artifacts)   column offsets in sinogram

Volume-domain artifacts (applied after reconstruction):
────────────────────────────────────────────────────────
  6. Rigid-body misalignment              (misalignment)     affine transform of one volume
  7. Partial-volume effect                automatic from voxel resolution; no extra code
  8. Salt-and-pepper voxel noise          (salt_pepper)      random voxel corruption
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, List, Tuple
from scipy.ndimage import (
    gaussian_filter,
    affine_transform,
    map_coordinates,
)
from scipy.spatial.transform import Rotation

__all__ = ["ArtifactConfig", "inject_sinogram_artifacts", "inject_volume_artifacts"]


# ──────────────────────────────────────────────────────────────────────────────
# Configuration dataclass
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class ArtifactConfig:
    """
    Master configuration for all artifact injections.

    Each group of parameters is documented below.  Set any group's enable
    flag to False (or the relevant amplitude to 0) to disable it.

    Examples
    --------
    Clean (reference) run::

        cfg = ArtifactConfig.clean()

    Noise only::

        cfg = ArtifactConfig(photon_noise=True, I0_xray=5e4, I0_neutron=2e4)

    Full realistic run::

        cfg = ArtifactConfig.realistic()

    Custom combination::

        cfg = ArtifactConfig(
            photon_noise=True, I0_xray=1e4,
            neutron_scatter=True, scatter_fraction=0.08,
            misalignment=True, translation_voxels=(3.0, 0.0, 0.0),
        )
    """

    # ── 1. Photon / neutron noise ─────────────────────────────────────────────
    photon_noise: bool = False
    """Enable Poisson counting noise on sinograms."""

    I0_xray: float = 1e5
    """Incident X-ray photon count per detector pixel per projection."""

    I0_neutron: float = 1e5
    """Incident neutron count per detector pixel per projection."""

    # ── 2. X-ray beam hardening ───────────────────────────────────────────────
    # Beam hardening emerges automatically from polychromatic projection.
    # The flag below controls whether a polynomial BHC is applied
    # (True = correct it; False = leave the artifact in).
    apply_bh_correction: bool = False
    """Apply polynomial beam-hardening correction to X-ray sinogram."""

    bh_correction_order: int = 3
    """Polynomial order for BHC (2–4 is typical)."""

    # ── 3. Neutron scatter build-up ───────────────────────────────────────────
    neutron_scatter: bool = False
    """Add scattered neutron contribution to sinogram (Gaussian halo model)."""

    scatter_fraction: float = 0.05
    """Fraction of unscattered intensity that becomes scattered background."""

    scatter_sigma_pixels: float = 8.0
    """Gaussian blur σ of the scatter halo [detector pixels]."""

    scatter_D_over_L: float = 100.0
    """Beam collimation D/L ratio; larger → more geometric scatter contamination."""

    # ── 4. X-ray scatter ─────────────────────────────────────────────────────
    xray_scatter: bool = False
    """Add scattered X-ray photon contribution to sinogram."""

    xray_scatter_fraction: float = 0.03
    """Fraction of primary intensity that becomes X-ray scatter."""

    xray_scatter_sigma_pixels: float = 20.0
    """Gaussian blur σ of X-ray scatter halo [detector pixels]."""

    # ── 5. Detector PSF ────────────────────────────────────────────────────────
    detector_psf: bool = False
    """Convolve each projection with a Gaussian PSF (scintillator blur)."""

    psf_sigma_xray_pixels: float = 0.8
    """Gaussian PSF σ for X-ray detector [pixels]."""

    psf_sigma_neutron_pixels: float = 1.5
    """Gaussian PSF σ for neutron detector [pixels] (scintillators are coarser)."""

    # ── 6. Ring artifacts ─────────────────────────────────────────────────────
    ring_artifacts: bool = False
    """Introduce ring / band artifacts from bad detector columns."""

    n_bad_columns: int = 3
    """Number of bad detector columns."""

    ring_amplitude: float = 0.05
    """Offset amplitude added to bad columns (in log-attenuation units)."""

    ring_seed: int = 42
    """Random seed for bad-column selection."""

    # ── 7. Misalignment (volume domain) ───────────────────────────────────────
    misalignment: bool = False
    """Apply rigid-body misalignment to the neutron reconstructed volume."""

    translation_voxels: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    """Translation (Δy, Δx, Δz) in voxels applied to the neutron volume."""

    rotation_deg: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    """Euler angles (ry, rx, rz) in degrees applied to the neutron volume."""

    # ── 8. Salt-and-pepper voxel noise (volume domain) ────────────────────────
    salt_pepper: bool = False
    """Randomly corrupt a fraction of voxels in the reconstructed volumes."""

    salt_pepper_fraction: float = 0.001
    """Fraction of voxels to corrupt."""

    salt_pepper_seed: int = 99
    """Random seed for voxel corruption."""

    # ──────────────────────────────────────────────────────────────────────────
    # Factory methods
    # ──────────────────────────────────────────────────────────────────────────

    @classmethod
    def clean(cls) -> "ArtifactConfig":
        """All artifacts disabled — clean reference simulation."""
        return cls()

    @classmethod
    def noise_only(cls, I0: float = 5e4) -> "ArtifactConfig":
        """Poisson counting noise only."""
        return cls(photon_noise=True, I0_xray=I0, I0_neutron=I0)

    @classmethod
    def beam_hardening_only(cls) -> "ArtifactConfig":
        """Polychromatic BH artifact without any BHC."""
        return cls(apply_bh_correction=False)

    @classmethod
    def scatter_only(cls) -> "ArtifactConfig":
        """Scatter artifacts (neutron + X-ray) only."""
        return cls(
            neutron_scatter=True, scatter_fraction=0.06,
            xray_scatter=True,    xray_scatter_fraction=0.04,
        )

    @classmethod
    def misalignment_only(cls, translation: Tuple = (3.0, 0.0, 0.0),
                           rotation: Tuple = (0.0, 1.5, 0.0)) -> "ArtifactConfig":
        """Rigid-body misalignment between the two modalities only."""
        return cls(misalignment=True,
                   translation_voxels=translation, rotation_deg=rotation)

    @classmethod
    def realistic(cls) -> "ArtifactConfig":
        """All physical artifacts at moderate realistic levels."""
        return cls(
            photon_noise=True,         I0_xray=5e4, I0_neutron=3e4,
            apply_bh_correction=False,
            neutron_scatter=True,      scatter_fraction=0.05, scatter_sigma_pixels=9.0,
            xray_scatter=True,         xray_scatter_fraction=0.03,
            detector_psf=True,         psf_sigma_xray_pixels=0.7,
                                       psf_sigma_neutron_pixels=1.4,
            ring_artifacts=True,       n_bad_columns=2, ring_amplitude=0.04,
            misalignment=True,         translation_voxels=(2.0, 0.5, 0.0),
                                       rotation_deg=(0.0, 0.8, 0.0),
        )

    def summary(self) -> str:
        """One-line human-readable summary of active artifacts."""
        active = []
        if self.photon_noise:
            active.append(f"noise(I0_x={self.I0_xray:.0e}, I0_n={self.I0_neutron:.0e})")
        if self.apply_bh_correction:
            active.append(f"BHC(order={self.bh_correction_order})")
        else:
            active.append("BH_artifact(no_correction)")
        if self.neutron_scatter:
            active.append(f"n_scatter(f={self.scatter_fraction:.2f})")
        if self.xray_scatter:
            active.append(f"x_scatter(f={self.xray_scatter_fraction:.2f})")
        if self.detector_psf:
            active.append(f"PSF(σ_x={self.psf_sigma_xray_pixels:.1f},"
                          f"σ_n={self.psf_sigma_neutron_pixels:.1f})")
        if self.ring_artifacts:
            active.append(f"rings(N={self.n_bad_columns},"
                          f"amp={self.ring_amplitude:.2f})")
        if self.misalignment:
            t = self.translation_voxels
            r = self.rotation_deg
            active.append(f"misalign(T={t},R={r})")
        if self.salt_pepper:
            active.append(f"salt_pepper(f={self.salt_pepper_fraction:.4f})")
        return " | ".join(active) if active else "clean (no artifacts)"


# ──────────────────────────────────────────────────────────────────────────────
# Sinogram-domain artifact injection
# ──────────────────────────────────────────────────────────────────────────────

def inject_sinogram_artifacts(
    xray_sino: dict,
    neutron_sino: dict,
    cfg: ArtifactConfig,
    rng: Optional[np.random.Generator] = None,
) -> tuple[dict, dict]:
    """
    Apply all sinogram-domain artifacts to X-ray and neutron sinograms.

    Parameters
    ----------
    xray_sino    : output of projector.project_xray()
    neutron_sino : output of projector.project_neutron()
    cfg          : ArtifactConfig
    rng          : optional numpy random generator (for reproducibility)

    Returns
    -------
    (xray_sino_mod, neutron_sino_mod) — new dicts with 'sino_lam' modified
    """
    if rng is None:
        rng = np.random.default_rng(0)

    # Work on copies
    x_lam = xray_sino["sino_lam"].copy()
    n_lam = neutron_sino["sino_lam"].copy()
    x_trans = xray_sino["sino_trans"].copy()
    n_trans = neutron_sino["sino_trans"].copy()

    I0_x = cfg.I0_xray
    I0_n = cfg.I0_neutron

    # ── 1. Poisson photon noise ───────────────────────────────────────────────
    if cfg.photon_noise:
        x_lam, x_trans = _apply_poisson_noise(x_trans, I0_x, rng)
        n_lam, n_trans = _apply_poisson_noise(n_trans, I0_n, rng)

    # ── 2. Beam-hardening correction (optional) ───────────────────────────────
    if cfg.apply_bh_correction:
        x_lam = _apply_bh_correction(x_lam, order=cfg.bh_correction_order)

    # ── 3. Neutron scatter build-up ───────────────────────────────────────────
    if cfg.neutron_scatter:
        n_lam = _apply_neutron_scatter(
            n_lam, n_trans,
            fraction=cfg.scatter_fraction,
            sigma=cfg.scatter_sigma_pixels,
            D_over_L=cfg.scatter_D_over_L,
        )

    # ── 4. X-ray scatter ─────────────────────────────────────────────────────
    if cfg.xray_scatter:
        x_lam = _apply_xray_scatter(
            x_lam, x_trans,
            fraction=cfg.xray_scatter_fraction,
            sigma=cfg.xray_scatter_sigma_pixels,
        )

    # ── 5. Detector PSF ───────────────────────────────────────────────────────
    if cfg.detector_psf:
        x_lam = _apply_psf(x_lam, cfg.psf_sigma_xray_pixels)
        n_lam = _apply_psf(n_lam, cfg.psf_sigma_neutron_pixels)

    # ── 6. Ring artifacts ─────────────────────────────────────────────────────
    if cfg.ring_artifacts:
        rng_ring = np.random.default_rng(cfg.ring_seed)
        x_lam = _apply_ring_artifacts(x_lam, cfg.n_bad_columns,
                                       cfg.ring_amplitude, rng_ring)
        rng_ring2 = np.random.default_rng(cfg.ring_seed + 1)
        n_lam = _apply_ring_artifacts(n_lam, max(1, cfg.n_bad_columns - 1),
                                       cfg.ring_amplitude * 0.7, rng_ring2)

    x_out = dict(xray_sino)
    x_out["sino_lam"] = x_lam
    x_out["sino_trans"] = np.exp(-x_lam)

    n_out = dict(neutron_sino)
    n_out["sino_lam"] = n_lam
    n_out["sino_trans"] = np.exp(-n_lam)

    return x_out, n_out


# ──────────────────────────────────────────────────────────────────────────────
# Volume-domain artifact injection
# ──────────────────────────────────────────────────────────────────────────────

def inject_volume_artifacts(
    vol_xray: np.ndarray,
    vol_neutron: np.ndarray,
    cfg: ArtifactConfig,
    rng: Optional[np.random.Generator] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Apply volume-domain artifacts after CT reconstruction.

    Parameters
    ----------
    vol_xray    : (N, N, N) reconstructed X-ray volume  [cm⁻¹]
    vol_neutron : (N, N, N) reconstructed neutron volume  [cm⁻¹]
    cfg         : ArtifactConfig
    rng         : optional random generator

    Returns
    -------
    (vol_x_mod, vol_n_mod)
    """
    if rng is None:
        rng = np.random.default_rng(1)

    vx = vol_xray.copy()
    vn = vol_neutron.copy()

    # ── 7. Rigid-body misalignment of neutron volume ──────────────────────────
    if cfg.misalignment:
        vn = _apply_misalignment(vn, cfg.translation_voxels, cfg.rotation_deg)

    # ── 8. Salt-and-pepper voxel noise ────────────────────────────────────────
    if cfg.salt_pepper:
        rng_sp = np.random.default_rng(cfg.salt_pepper_seed)
        vx = _apply_salt_pepper(vx, cfg.salt_pepper_fraction, rng_sp)
        vn = _apply_salt_pepper(vn, cfg.salt_pepper_fraction, rng_sp)

    return vx, vn


# ──────────────────────────────────────────────────────────────────────────────
# Individual artifact implementations
# ──────────────────────────────────────────────────────────────────────────────

def _apply_poisson_noise(
    transmission: np.ndarray,
    I0: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Add Poisson counting noise to a transmission sinogram.

    Model:
        detected = Poisson(I0 · T)
        T_noisy  = detected / I0
        λ_noisy  = −log(T_noisy)
    """
    counts      = rng.poisson(np.clip(transmission, 0, 1) * I0).astype(np.float32)
    t_noisy     = counts / I0
    eps         = 0.5 / I0        # half-count floor avoids log(0)
    t_noisy     = np.clip(t_noisy, eps, 1.0)
    lam_noisy   = -np.log(t_noisy)
    return lam_noisy, t_noisy


def _apply_bh_correction(
    sino_lam: np.ndarray,
    order: int = 3,
    water_equivalent_thickness: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Polynomial beam-hardening correction (Joseph & Spital, 1981 model).

    Fits a polynomial p(λ) ≈ λ_mono that maps the log-attenuated (BH-biased)
    line-integral to the water-equivalent monochromatic one.
    Coefficients chosen empirically for 120 kVp with 2 mm Al filter.
    """
    # Polynomial correction: λ_corrected = a1·λ + a2·λ² + a3·λ³ + …
    # Coefficients derived by fitting to monochromatic reference.
    # For a 120 kVp spectrum + 2 mm Al these are reasonable defaults.
    coeffs = {
        2: [1.0, -0.02],
        3: [1.0, -0.025, 0.003],
        4: [1.0, -0.028, 0.005, -0.0003],
    }
    c = coeffs.get(order, coeffs[3])
    corrected = np.zeros_like(sino_lam)
    for power, coef in enumerate(c, start=1):
        corrected += coef * sino_lam ** power
    return np.clip(corrected, 0, None)


def _apply_neutron_scatter(
    sino_lam: np.ndarray,
    sino_trans: np.ndarray,
    fraction: float = 0.05,
    sigma: float = 8.0,
    D_over_L: float = 100.0,
) -> np.ndarray:
    """
    Add scattered neutron contribution modelled as a Gaussian halo.

    In neutron imaging the scattered fraction can be 5-15% for thick samples.
    The collimation ratio D/L determines the solid angle of scatter collection:
    larger D/L → more scatter reaches the detector.

    Model:
        I_detected = I_primary + f · I_primary_blurred
    """
    # Collimation factor: normalise so D/L=100 (tight) gives fraction as-is,
    # D/L=10 (loose) scales up by factor 3
    dl_scale = np.clip((D_over_L / 100.0) ** 0.5, 0.3, 3.0)
    f_eff    = fraction * dl_scale

    I_primary = sino_trans        # shape (n_angles, N, N)
    # Blur the scatter source (halo around sample edges)
    scatter_map = np.zeros_like(I_primary)
    for a_idx in range(I_primary.shape[0]):
        for s_idx in range(I_primary.shape[1]):
            scatter_map[a_idx, s_idx] = gaussian_filter(
                I_primary[a_idx, s_idx], sigma=sigma
            )

    I_detected = I_primary + f_eff * scatter_map
    eps = 1e-9
    return -np.log(np.clip(I_detected, eps, None))


def _apply_xray_scatter(
    sino_lam: np.ndarray,
    sino_trans: np.ndarray,
    fraction: float = 0.03,
    sigma: float = 20.0,
) -> np.ndarray:
    """
    Add Compton/Rayleigh scatter to X-ray sinogram (cupping model).

    X-ray scatter produces a broad low-frequency background that causes
    cupping in the reconstructed volume and diagonal smearing in the
    bimodal histogram.
    """
    I_primary = sino_trans
    scatter_map = np.zeros_like(I_primary)
    for a_idx in range(I_primary.shape[0]):
        for s_idx in range(I_primary.shape[1]):
            scatter_map[a_idx, s_idx] = gaussian_filter(
                I_primary[a_idx, s_idx], sigma=sigma
            )
    I_detected = I_primary + fraction * scatter_map
    eps = 1e-9
    return -np.log(np.clip(I_detected, eps, None))


def _apply_psf(sino_lam: np.ndarray, sigma: float) -> np.ndarray:
    """
    Convolve each projection image with a Gaussian PSF.

    Models scintillator light spread (X-ray: CsI, Gd₂O₂S; neutron: LiF/ZnS).
    Applied in the sinogram domain (before reconstruction) to simulate
    the reduced MTF of the detector system.
    """
    if sigma <= 0:
        return sino_lam
    blurred = np.zeros_like(sino_lam)
    for a_idx in range(sino_lam.shape[0]):
        for s_idx in range(sino_lam.shape[1]):
            blurred[a_idx, s_idx] = gaussian_filter(
                sino_lam[a_idx, s_idx], sigma=sigma
            )
    return blurred


def _apply_ring_artifacts(
    sino_lam: np.ndarray,
    n_bad: int,
    amplitude: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Introduce ring artifacts by adding constant offsets to detector columns.

    Bad detector pixels that have a consistent gain or offset error produce
    concentric rings in the reconstructed volume and vertical/horizontal
    streaks in the sinogram.
    """
    N_det   = sino_lam.shape[-1]
    bad_cols = rng.choice(N_det, size=n_bad, replace=False)
    offsets  = rng.uniform(-amplitude, amplitude, size=n_bad)

    result = sino_lam.copy()
    for col, off in zip(bad_cols, offsets):
        result[:, :, col] += off
    return result


def _apply_misalignment(
    vol: np.ndarray,
    translation_voxels: Tuple[float, float, float],
    rotation_deg: Tuple[float, float, float],
) -> np.ndarray:
    """
    Apply rigid-body misalignment to a 3-D volume via affine transform.

    This is the key artifact for bimodal histogram analysis: even sub-voxel
    misalignment smears compact Gaussian clusters into elongated streaks.

    The transform is applied around the volume centre to avoid offset bias.
    """
    N      = vol.shape[0]
    center = np.array(vol.shape) / 2.0

    # Build rotation matrix from Euler angles (extrinsic xyz convention)
    R = Rotation.from_euler("xyz", rotation_deg, degrees=True).as_matrix()

    # Affine transform: x_phantom = R @ (x_vol − centre) + centre + translation
    offset = center - R @ center + np.array(translation_voxels)

    misaligned = affine_transform(
        vol, R,
        offset=offset,
        order=1,                   # bilinear interpolation
        mode="constant", cval=0.0,
    )
    return misaligned.astype(vol.dtype)


def _apply_salt_pepper(
    vol: np.ndarray,
    fraction: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Randomly corrupt a fraction of voxels with outlier values.

    Simulates dead or stuck voxels, cosmic rays, or electronic read-out errors.
    """
    corrupted = vol.copy()
    n_total   = vol.size
    n_corrupt = int(fraction * n_total)
    flat_idx  = rng.choice(n_total, size=n_corrupt, replace=False)
    values    = rng.choice([vol.min(), vol.max()], size=n_corrupt)
    corrupted.ravel()[flat_idx] = values
    return corrupted


# ──────────────────────────────────────────────────────────────────────────────
# Preset configs exposed for convenience
# ──────────────────────────────────────────────────────────────────────────────

PRESET_CONFIGS = {
    "clean":           ArtifactConfig.clean,
    "noise_only":      ArtifactConfig.noise_only,
    "scatter_only":    ArtifactConfig.scatter_only,
    "misalignment":    ArtifactConfig.misalignment_only,
    "realistic":       ArtifactConfig.realistic,
}
