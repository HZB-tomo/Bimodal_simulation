"""
neutron_xray_sim/reconstructor.py
-----------------------------------
CT reconstruction for dual-modality sinograms.

CPU fallback uses skimage.transform.iradon (correct geometry, no shadow bug).
All outputs are divided by voxel_cm to give linear attenuation in cm^-1.
"""

from __future__ import annotations

import warnings
import numpy as np
from scipy.ndimage import median_filter

__all__ = ["reconstruct", "reconstruct_pair"]


def _astra_ok() -> bool:
    try:
        import astra
        return True
    except ImportError:
        return False


def _remove_rings_vo(sinogram: np.ndarray, snr: float = 3.0, la: int = 11) -> np.ndarray:
    """Vo et al. (2018) ring removal: subtract systematic column offsets."""
    col_mean   = sinogram.mean(axis=0)
    col_smooth = median_filter(col_mean, size=la)
    residual   = col_mean - col_smooth
    corrected  = sinogram.copy()
    if residual.std() > 0:
        corrected -= residual[np.newaxis, :]
    return corrected


def _fbp_skimage(sinogram2d: np.ndarray, angles_deg: np.ndarray,
                 filter_name: str = "shepp-logan") -> np.ndarray:
    """
    2-D parallel-beam FBP using skimage.transform.iradon.

    Replaces the previous custom numpy FBP which had a coordinate-convention
    bug that placed FBP negative sidelobes INSIDE thin metal features,
    producing dark shadows (negative values) at iron/titanium rods.
    skimage iradon is validated and produces correct positive attenuation.

    Parameters
    ----------
    sinogram2d : (n_angles, n_det)
    angles_deg : (n_angles,)  projection angles [degrees]
    filter_name: 'ramp'|'shepp-logan'|'cosine'|'hann'
                 'shepp-logan' recommended for 1 cm samples to reduce ringing
                 at thin high-contrast features (Fe rod, Ti sphere).

    Returns
    -------
    recon : (n_det, n_det)  in same units as sinogram (OD/pixel)
    """
    from skimage.transform import iradon

    fmap = {
        "ram-lak":     "ramp",
        "ramp":        "ramp",
        "shepp-logan": "shepp-logan",
        "cosine":      "cosine",
        "hann":        "hann",
        "hamming":     "hamming",
    }
    sk_filter = fmap.get(filter_name.lower(), "shepp-logan")

    # skimage expects shape (n_det, n_angles)
    recon = iradon(sinogram2d.T, theta=angles_deg,
                   filter_name=sk_filter, interpolation="linear", circle=True)
    return recon


def reconstruct(
    sino_dict: dict,
    algorithm: str = "FBP",
    filter_name: str = "shepp-logan",
    n_iter: int = 50,
    remove_rings: bool = True,
    ring_snr: float = 3.0,
    center_offset: float = 0.0,
    use_astra: bool = True,
    clip_negative: bool = True,
) -> np.ndarray:
    """
    Reconstruct a 3-D volume from a sinogram dictionary.

    The reconstructor reads 'voxel_cm' from sino_dict (set by the projector)
    and divides the FBP result by it to convert from OD/pixel to cm^-1.

    Parameters
    ----------
    sino_dict   : projector output dict (keys: sino_lam, angles_deg, voxel_cm)
    algorithm   : 'FBP' | 'SIRT' | 'CGLS'
    filter_name : 'shepp-logan' (default) | 'ram-lak' | 'cosine' | 'hann'
    n_iter      : SIRT/CGLS iterations
    remove_rings: apply Vo ring removal
    ring_snr    : SNR threshold for ring removal
    center_offset: rotation centre offset [pixels]
    use_astra   : prefer ASTRA GPU
    clip_negative: clip result to >= 0

    Returns
    -------
    vol : (N, N, N) float32  [cm^-1]
    """
    sino_lam   = sino_dict["sino_lam"]       # (n_angles, N_slice, N_det)
    angles_deg = sino_dict["angles_deg"]
    angles_rad = np.radians(angles_deg)
    voxel_cm   = sino_dict.get("voxel_cm", None)

    n_angles, N_slice, N_det = sino_lam.shape
    vol     = np.zeros((N_slice, N_det, N_det), dtype=np.float32)
    use_gpu = use_astra and _astra_ok()
    alg     = algorithm.upper()

    if use_gpu:
        import astra

    for s_idx in range(N_slice):
        sino2d = sino_lam[:, s_idx, :].copy()

        if remove_rings:
            sino2d = _remove_rings_vo(sino2d, snr=ring_snr)

        if center_offset != 0.0:
            from scipy.ndimage import shift
            sino2d = shift(sino2d, (0, center_offset), mode="nearest")

        if use_gpu:
            vol_geom  = astra.create_vol_geom(N_det, N_det)
            proj_geom = astra.create_proj_geom("parallel", 1.0, N_det, angles_rad)
            sino_id   = astra.data2d.create("-sino", proj_geom, sino2d)
            rec_id    = astra.data2d.create("-vol", vol_geom)

            if alg == "FBP":
                cfg = astra.astra_dict("FBP_CUDA")
                cfg["ProjectionDataId"]     = sino_id
                cfg["ReconstructionDataId"] = rec_id
                cfg["FilterType"]           = _astra_filter(filter_name)
                alg_id = astra.algorithm.create(cfg)
                astra.algorithm.run(alg_id)
            elif alg in ("SIRT", "CGLS"):
                cfg = astra.astra_dict(f"{alg}_CUDA")
                cfg["ProjectionDataId"]     = sino_id
                cfg["ReconstructionDataId"] = rec_id
                alg_id = astra.algorithm.create(cfg)
                astra.algorithm.run(alg_id, n_iter)
            else:
                raise ValueError(
                    f"Unknown algorithm '{algorithm}'. Choose FBP, SIRT, or CGLS."
                )

            slice_recon = astra.data2d.get(rec_id)
            astra.algorithm.delete(alg_id)
            astra.data2d.delete([sino_id, rec_id])
        else:
            if alg != "FBP":
                warnings.warn(
                    f"ASTRA not available -- using skimage FBP fallback "
                    f"(requested: {alg})."
                )
            slice_recon = _fbp_skimage(sino2d, angles_deg, filter_name)

        # Scale from OD/pixel to cm^-1
        if voxel_cm is not None and voxel_cm > 0:
            slice_recon = slice_recon / voxel_cm

        vol[s_idx] = slice_recon

    if clip_negative:
        vol = np.clip(vol, 0, None)

    return vol


def reconstruct_pair(
    xray_sino: dict,
    neutron_sino: dict,
    algorithm: str = "FBP",
    filter_name: str = "shepp-logan",
    n_iter: int = 50,
    remove_rings: bool = True,
    use_astra: bool = True,
    clip_negative: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Reconstruct X-ray and neutron volumes from their sinogram dicts.

    Returns (vol_xray, vol_neutron) both (N,N,N) float32 [cm^-1].
    """
    print(f"[reconstructor] Reconstructing with {algorithm} ...")
    print("  -> X-ray ...")
    vol_x = reconstruct(
        xray_sino, algorithm=algorithm, filter_name=filter_name,
        n_iter=n_iter, remove_rings=remove_rings,
        use_astra=use_astra, clip_negative=clip_negative,
    )
    print("  -> Neutron ...")
    vol_n = reconstruct(
        neutron_sino, algorithm=algorithm, filter_name=filter_name,
        n_iter=n_iter, remove_rings=remove_rings,
        use_astra=use_astra, clip_negative=clip_negative,
    )
    print("[reconstructor] Done.")
    return vol_x, vol_n


def _astra_filter(name: str) -> str:
    mapping = {
        "ram-lak":     "Ram-Lak",
        "ramp":        "Ram-Lak",
        "shepp-logan": "SheppLogan",
        "cosine":      "Cosine",
        "hann":        "Hann",
    }
    return mapping.get(name.lower(), "SheppLogan")
