"""
neutron_xray_sim.acquisition.artifacts
──────────────────────────────────────
Artifact injection for dual-modality CT sinograms and reconstructed volumes.

Every artifact is controlled by a field in :class:`ArtifactConfig`.  Setting a
flag to False disables that artifact completely, which makes it easy to run
controlled experiments with one artifact at a time or in combination.

Sinogram-domain artifacts are applied in the physical order of the detection
chain (before reconstruction):

  1. Scatter build-up             (neutron_scatter, xray_scatter)
                                  Gaussian halo added to the primary intensity
  2. Detector point-spread        (detector_psf)
                                  2-D Gaussian blur of each projection image
  3. Poisson counting noise       (photon_noise)
  4. Ring / bad-pixel artifacts   (ring_artifacts)  column offsets
  5. Beam-hardening correction    (apply_bh_correction)
                                  Beam hardening itself emerges from the
                                  polychromatic projector; this optionally
                                  applies a polynomial correction.

Volume-domain artifacts (applied after reconstruction):

  6. Rigid-body misalignment      (misalignment)  of the neutron volume
  7. Salt-and-pepper voxels       (salt_pepper)

The partial-volume effect is automatic (finite voxel size) and needs no code.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from scipy.ndimage import affine_transform, gaussian_filter
from scipy.spatial.transform import Rotation

from .noise import add_poisson_noise

__all__ = [
    "ArtifactConfig",
    "inject_sinogram_artifacts",
    "inject_volume_artifacts",
    "PRESET_CONFIGS",
]


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
    """Translation in voxels along the storage axes (z, x, y) of the neutron
    volume.  Follows ``scipy.ndimage.affine_transform``: output voxel ``o``
    samples input ``o + t``, i.e. the content moves by ``−t``."""

    rotation_deg: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    """Extrinsic Euler angles in degrees about storage axes (z, x, y),
    applied about the volume centre."""

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
        """Polychromatic BH artifact without any BHC.

        Identical to :meth:`clean`: beam hardening is intrinsic to the
        polychromatic projector and is present unless corrected.
        """
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
    xray_sino    : output of ``project_xray()``
    neutron_sino : output of ``project_neutron()``
    cfg          : ArtifactConfig
    rng          : random generator for the noise (default: seeded with 0)

    Returns
    -------
    (xray_sino_mod, neutron_sino_mod) — new dicts; ``sino_lam`` and
    ``sino_trans`` are replaced, all other keys are shared with the input.
    """
    if rng is None:
        rng = np.random.default_rng(0)

    # ── X-ray: detection chain ────────────────────────────────────────────────
    x_lam = _detected_optical_depth(
        xray_sino,
        scatter=(cfg.xray_scatter_fraction, cfg.xray_scatter_sigma_pixels)
                if cfg.xray_scatter else None,
        psf_sigma=cfg.psf_sigma_xray_pixels if cfg.detector_psf else 0.0,
    )
    # ── Neutron: detection chain ──────────────────────────────────────────────
    n_frac = cfg.scatter_fraction * _collimation_scale(cfg.scatter_D_over_L)
    n_lam = _detected_optical_depth(
        neutron_sino,
        scatter=(n_frac, cfg.scatter_sigma_pixels) if cfg.neutron_scatter else None,
        psf_sigma=cfg.psf_sigma_neutron_pixels if cfg.detector_psf else 0.0,
    )

    # ── Poisson counting noise ────────────────────────────────────────────────
    if cfg.photon_noise:
        x_lam = add_poisson_noise(x_lam, cfg.I0_xray, rng=rng)
        n_lam = add_poisson_noise(n_lam, cfg.I0_neutron, rng=rng)

    # ── Ring artifacts (detector gain/offset errors) ──────────────────────────
    if cfg.ring_artifacts:
        x_lam = _apply_ring_artifacts(
            x_lam, cfg.n_bad_columns, cfg.ring_amplitude,
            np.random.default_rng(cfg.ring_seed))
        n_lam = _apply_ring_artifacts(
            n_lam, max(1, cfg.n_bad_columns - 1), cfg.ring_amplitude * 0.7,
            np.random.default_rng(cfg.ring_seed + 1))

    # ── Beam-hardening correction ─────────────────────────────────────────────
    if cfg.apply_bh_correction:
        x_lam = _apply_bh_correction(x_lam, order=cfg.bh_correction_order)

    x_out = dict(xray_sino)
    x_out["sino_lam"] = x_lam
    x_out["sino_trans"] = np.exp(-x_lam)

    n_out = dict(neutron_sino)
    n_out["sino_lam"] = n_lam
    n_out["sino_trans"] = np.exp(-n_lam)

    return x_out, n_out


def _detected_optical_depth(
    sino: dict,
    scatter: Optional[Tuple[float, float]],
    psf_sigma: float,
) -> np.ndarray:
    """
    Optical depth seen by the detector after scatter and detector blur.

    Returns a copy of ``sino['sino_lam']`` when neither effect is active, so a
    clean run reproduces the projector output exactly.
    """
    if scatter is None and psf_sigma <= 0:
        return np.array(sino["sino_lam"], dtype=np.float32, copy=True)

    intensity = np.asarray(sino["sino_trans"], dtype=np.float32)
    if scatter is not None:
        fraction, sigma = scatter
        intensity = intensity + fraction * _blur_projections(intensity, sigma)
    if psf_sigma > 0:
        intensity = _blur_projections(intensity, psf_sigma)

    eps = 1.0 / (10 * float(sino.get("I0", 1e5)))
    return (-np.log(np.clip(intensity, eps, None))).astype(np.float32)


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
    vol_xray, vol_neutron : reconstructed volumes [cm⁻¹]
    cfg                   : ArtifactConfig
    rng                   : unused; kept for API compatibility (the volume
                            artifacts use their own seeds from ``cfg``)

    Returns
    -------
    (vol_x_mod, vol_n_mod)
    """
    vx = vol_xray.copy()
    vn = vol_neutron.copy()

    if cfg.misalignment:
        vn = _apply_misalignment(vn, cfg.translation_voxels, cfg.rotation_deg)

    if cfg.salt_pepper:
        rng_sp = np.random.default_rng(cfg.salt_pepper_seed)
        vx = _apply_salt_pepper(vx, cfg.salt_pepper_fraction, rng_sp)
        vn = _apply_salt_pepper(vn, cfg.salt_pepper_fraction, rng_sp)

    return vx, vn


# ──────────────────────────────────────────────────────────────────────────────
# Individual artifact implementations
# ──────────────────────────────────────────────────────────────────────────────

def _blur_projections(sino: np.ndarray, sigma: float) -> np.ndarray:
    """
    Gaussian blur of every projection image.

    *sino* has shape (n_angles, n_slices, n_det); each projection image is the
    (n_slices, n_det) plane, so the blur acts on axes 1 and 2 only.
    """
    if sigma <= 0:
        return sino
    return gaussian_filter(sino, sigma=(0.0, sigma, sigma), mode="nearest")


def _collimation_scale(D_over_L: float) -> float:
    """
    Scale factor for the neutron scatter fraction from beam collimation.

    Normalised so that D/L = 100 gives the configured fraction unchanged;
    clipped to [0.3, 3].
    """
    return float(np.clip((D_over_L / 100.0) ** 0.5, 0.3, 3.0))


def _apply_bh_correction(
    sino_lam: np.ndarray,
    order: int = 3,
) -> np.ndarray:
    """
    Polynomial beam-hardening correction (Joseph & Spital, 1981 model).

    λ_corrected = Σ_k a_k · λ^k with coefficients chosen empirically for a
    120 kVp / 2 mm Al spectrum.  Orders 2–4 are supported (others use 3).
    """
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


def _apply_ring_artifacts(
    sino_lam: np.ndarray,
    n_bad: int,
    amplitude: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Add constant offsets to a few detector columns.

    Pixels with a consistent gain or offset error produce concentric rings in
    the reconstruction (vertical stripes in the sinogram).
    """
    n_det = sino_lam.shape[-1]
    n_bad = min(int(n_bad), n_det)
    bad_cols = rng.choice(n_det, size=n_bad, replace=False)
    offsets = rng.uniform(-amplitude, amplitude, size=n_bad)

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
    Rigid-body misalignment of a 3-D volume (about the volume centre).

    Even sub-voxel misalignment smears compact bimodal-histogram clusters into
    elongated streaks.  See ``ArtifactConfig.translation_voxels`` for the sign
    convention.
    """
    center = np.array(vol.shape) / 2.0
    R = Rotation.from_euler("xyz", rotation_deg, degrees=True).as_matrix()
    offset = center - R @ center + np.array(translation_voxels)
    misaligned = affine_transform(vol, R, offset=offset, order=1,
                                  mode="constant", cval=0.0)
    return misaligned.astype(vol.dtype)


def _apply_salt_pepper(
    vol: np.ndarray,
    fraction: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Set a random fraction of voxels to the volume min or max."""
    corrupted = vol.copy()
    n_corrupt = int(fraction * vol.size)
    flat_idx = rng.choice(vol.size, size=n_corrupt, replace=False)
    values = rng.choice([vol.min(), vol.max()], size=n_corrupt)
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
