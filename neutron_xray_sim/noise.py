"""
noise.py
========
Poisson counting statistics for the DIANA dual-modality pipeline.

Place inside ``neutron_xray_sim/`` and add to ``__init__.py`` exports.

Scope
-----
DIANA's forward projectors return **optical depth** sinograms,
``lambda = integral(mu dl)`` (dimensionless, float32, shape ``(n_ang, nz, nx)``).
This module converts those noise-free line integrals into realistically noisy
ones by resampling the detected counts:

    I_obj  ~ Poisson(I0 * exp(-lambda))          object frame
    I_flat ~ Poisson(I0)                         open-beam (flat) frame, optional
    lambda_noisy = -log(I_obj / I_flat)

`I0` is the mean incident count **per detector pixel per projection**. It is the
only dose-like knob in the model, and it is modality-specific: a laboratory
microfocus X-ray tube reaches 1e5-1e6 counts/pixel in seconds, whereas an
imaging beamline at a research reactor or spallation source typically delivers
1e2-1e4 neutrons/pixel for the same wall-clock time. Running both modalities at
the same `I0` is the single most common way to make a dual-modality simulation
look better than the experiment ever will.

Why the flat field matters
--------------------------
With an ideal (noiseless) reference, Var[lambda] = (exp(lambda) - 1) / I0.
With a Poisson flat field of ``I0_flat`` counts the variances add:

    Var[lambda] = (exp(lambda) - 1) / I0 + 1 / I0 + 1 / I0_flat

For a well-counted flat field this is a small correction; for the
one-flat-per-scan practice common on neutron beamlines it is not. Default is
``flat_counts=None`` (ideal reference) to stay backwards-compatible with the
existing `artifacts._apply_poisson_noise`; set it explicitly for realism.

Memory
------
Fossil sinograms are multi-GB memmaps. Every function here streams over the
slice axis (`axis=1`) in chunks and never materialises the full array, so it
can read from one memmap and write to another.

Reproducibility
---------------
All randomness goes through an explicit `numpy.random.Generator`. Use
`rng_for(seed, tag)` to derive independent, reproducible streams per modality
and per dose point, so the X-ray and neutron noise realisations of a given run
are uncorrelated but the whole study is re-runnable from one master seed.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Iterable, Optional, Sequence

import numpy as np

__all__ = [
    "DoseConfig",
    "DOSE_PRESETS",
    "rng_for",
    "add_poisson_noise",
    "add_poisson_noise_streaming",
    "predicted_sigma_lambda",
    "measure_sigma_lambda",
    "cnr",
    "d_prime",
    "joint_d_prime",
    "material_stats",
    "rose_dose_threshold",
]


# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class DoseConfig:
    """
    Incident counts per detector pixel per projection, per modality.

    Parameters
    ----------
    I0_xray, I0_neutron : float
        Mean incident counts/pixel/projection. ``np.inf`` disables noise for
        that modality (useful for isolating one modality's dose dependence).
    flat_counts_xray, flat_counts_neutron : float or None
        Counts in the open-beam reference frame. ``None`` = ideal noiseless
        reference. A realistic value is ``n_flat_frames * I0``.
    label : str
        Short tag used in filenames and figure legends.
    """

    I0_xray: float = 1e5
    I0_neutron: float = 1e4
    flat_counts_xray: Optional[float] = None
    flat_counts_neutron: Optional[float] = None
    label: str = ""

    def __post_init__(self) -> None:
        for name in ("I0_xray", "I0_neutron"):
            v = getattr(self, name)
            if not (v > 0):
                raise ValueError(f"{name} must be > 0 (got {v!r})")
        for name in ("flat_counts_xray", "flat_counts_neutron"):
            v = getattr(self, name)
            if v is not None and not (v > 0):
                raise ValueError(f"{name} must be > 0 or None (got {v!r})")
        if not self.label:
            object.__setattr__(
                self, "label", f"Ix{self.I0_xray:.0e}_In{self.I0_neutron:.0e}"
            )

    @property
    def noiseless(self) -> bool:
        return np.isinf(self.I0_xray) and np.isinf(self.I0_neutron)

    def describe(self) -> str:
        def _fmt(i0, flat):
            if np.isinf(i0):
                return "noiseless"
            f = "ideal flat" if flat is None else f"flat={flat:.0e}"
            return f"I0={i0:.0e}, {f}"

        return (f"X-ray: {_fmt(self.I0_xray, self.flat_counts_xray)} | "
                f"neutron: {_fmt(self.I0_neutron, self.flat_counts_neutron)}")


#: Dose points spanning "synchrotron-clean" to "one-shot neutron radiograph".
#: The neutron values are deliberately 10x lower than the X-ray ones at each
#: rung, which is roughly the flux ratio between a lab microfocus tube and a
#: cold-neutron imaging beamline at equal acquisition time.
DOSE_PRESETS: dict[str, DoseConfig] = {
    "noiseless": DoseConfig(np.inf, np.inf, label="noiseless"),
    "high":      DoseConfig(1e6, 1e5, label="high"),
    "standard":  DoseConfig(1e5, 1e4, label="standard"),
    "low":       DoseConfig(1e4, 1e3, label="low"),
    "very_low":  DoseConfig(1e3, 1e2, label="very_low"),
}


def rng_for(seed: int, *tags: object) -> np.random.Generator:
    """
    Derive a reproducible, independent `Generator` from a master seed and tags.

    ``rng_for(42, 'xray', 1e4)`` and ``rng_for(42, 'neutron', 1e4)`` give
    uncorrelated streams that are both fully determined by the master seed, so
    a dose sweep can be re-run point-by-point without regenerating the rest.
    """
    key = "|".join(repr(t) for t in tags).encode()
    digest = hashlib.blake2b(key, digest_size=8).digest()
    return np.random.default_rng([int(seed), int.from_bytes(digest, "little")])


# ─────────────────────────────────────────────────────────────────────────────
# Core noise model
# ─────────────────────────────────────────────────────────────────────────────

def add_poisson_noise(
    sino_lam: np.ndarray,
    I0: float,
    rng: Optional[np.random.Generator] = None,
    flat_counts: Optional[float] = None,
    dtype: np.dtype = np.float32,
) -> np.ndarray:
    """
    Add Poisson counting noise to an optical-depth sinogram (in memory).

    Parameters
    ----------
    sino_lam : ndarray
        Noise-free optical depth ``integral(mu dl)``, any shape.
    I0 : float
        Mean incident counts per detector pixel per projection. ``np.inf``
        returns a copy unchanged.
    rng : np.random.Generator, optional
        Defaults to ``np.random.default_rng()`` (not reproducible — pass
        `rng_for(...)` for study work).
    flat_counts : float, optional
        Counts in the open-beam reference. ``None`` = ideal reference.

    Returns
    -------
    ndarray of `dtype`, same shape — noisy optical depth.

    Notes
    -----
    Zero detected counts are floored at 0.5 (half a count) rather than 1.0.
    Flooring at 1 biases lambda low in the thick-object limit; the half-count
    floor is the standard convention and keeps the estimator closer to unbiased
    where it matters. This matters for fossils, where the matrix path length
    through a hand specimen can push exp(-lambda) below 1/I0.
    """
    if np.isinf(I0):
        return np.asarray(sino_lam, dtype=dtype).copy()
    if rng is None:
        rng = np.random.default_rng()

    lam = np.asarray(sino_lam, dtype=np.float64)
    transmission = np.exp(-lam)

    counts = rng.poisson(I0 * transmission).astype(np.float64)
    np.maximum(counts, 0.5, out=counts)

    if flat_counts is None:
        ref = float(I0)
    else:
        ref = rng.poisson(flat_counts, size=lam.shape).astype(np.float64)
        np.maximum(ref, 0.5, out=ref)
        ref = ref * (I0 / flat_counts)     # rescale to the I0 normalisation

    return (-np.log(counts / ref)).astype(dtype)


def add_poisson_noise_streaming(
    src: np.ndarray,
    dst: np.ndarray,
    I0: float,
    rng: Optional[np.random.Generator] = None,
    flat_counts: Optional[float] = None,
    slice_axis: int = 1,
    chunk: int = 32,
    progress: bool = False,
) -> np.ndarray:
    """
    Memmap-safe version of `add_poisson_noise` for full fossil volumes.

    Streams over `slice_axis` (default 1, the nz axis of DIANA's
    ``(n_ang, nz, nx)`` sinograms) in blocks of `chunk` slices. `src` and `dst`
    may both be `np.memmap`; they must have the same shape. Passing the same
    array as both `src` and `dst` works and overwrites in place.

    Returns `dst`.
    """
    if src.shape != dst.shape:
        raise ValueError(f"shape mismatch: src {src.shape} vs dst {dst.shape}")
    if np.isinf(I0):
        if src is not dst:
            dst[...] = src
        return dst
    if rng is None:
        rng = np.random.default_rng()

    n = src.shape[slice_axis]
    for start in range(0, n, chunk):
        stop = min(start + chunk, n)
        sl = [slice(None)] * src.ndim
        sl[slice_axis] = slice(start, stop)
        sl = tuple(sl)
        dst[sl] = add_poisson_noise(
            np.asarray(src[sl]), I0, rng=rng,
            flat_counts=flat_counts, dtype=dst.dtype,
        )
        if progress:
            print(f"    noise {stop:5d}/{n}", end="\r", flush=True)

    if hasattr(dst, "flush"):
        dst.flush()
    if progress:
        print(f"    noise {n:5d}/{n}  done")
    return dst


# ─────────────────────────────────────────────────────────────────────────────
# Analytic reference — used to validate the sampler and to annotate figures
# ─────────────────────────────────────────────────────────────────────────────

def predicted_sigma_lambda(
    lam: np.ndarray | float,
    I0: float,
    flat_counts: Optional[float] = None,
) -> np.ndarray:
    """
    First-order analytic standard deviation of the noisy optical depth.

    Error propagation through ``lambda = -log(I/I0)`` with ``I ~ Poisson``:

        Var[lambda] ~= (exp(lambda) - 1) / I0 + 1 / I0   [+ 1 / flat_counts]

    The ``exp(lambda)`` factor is the whole story of dose in tomography: noise
    grows exponentially with the attenuation along the ray, so the densest part
    of the specimen sets the required dose, not the mean.

    Plotting this against `measure_sigma_lambda` is a cheap correctness check
    on the sampler, and makes a good supplementary panel.
    """
    lam = np.asarray(lam, dtype=np.float64)
    if np.isinf(I0):
        return np.zeros_like(lam)
    var = np.exp(lam) / I0
    if flat_counts is not None:
        var = var + 1.0 / flat_counts
    return np.sqrt(var)


def measure_sigma_lambda(
    lam_clean: np.ndarray,
    lam_noisy: np.ndarray,
    n_bins: int = 24,
    lam_range: Optional[tuple[float, float]] = None,
) -> dict[str, np.ndarray]:
    """
    Empirical sigma(lambda) from a clean/noisy sinogram pair, binned in lambda.

    Returns dict with ``'lam_centres'``, ``'sigma'``, ``'count'`` — ready to
    overlay on `predicted_sigma_lambda`.
    """
    a = np.asarray(lam_clean, dtype=np.float64).ravel()
    b = np.asarray(lam_noisy, dtype=np.float64).ravel()
    if lam_range is None:
        lam_range = (float(a.min()), float(np.percentile(a, 99.5)))

    edges = np.linspace(lam_range[0], lam_range[1], n_bins + 1)
    idx = np.clip(np.digitize(a, edges) - 1, 0, n_bins - 1)
    resid = b - a

    sigma = np.full(n_bins, np.nan)
    count = np.zeros(n_bins, dtype=np.int64)
    for k in range(n_bins):
        m = idx == k
        count[k] = int(m.sum())
        if count[k] > 8:
            sigma[k] = float(resid[m].std())

    return {
        "lam_centres": 0.5 * (edges[:-1] + edges[1:]),
        "sigma": sigma,
        "count": count,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Separability metrics — how the dose actually affects the science
# ─────────────────────────────────────────────────────────────────────────────

def material_stats(
    vol: np.ndarray,
    label_vol: np.ndarray,
    labels: Sequence[int],
    erode: int = 0,
) -> dict[int, tuple[float, float, int]]:
    """
    Per-label mean, standard deviation and voxel count of a reconstructed volume.

    Parameters
    ----------
    erode : int
        Erode each label mask by this many iterations before sampling. Voxels
        at a phase boundary are partial-volume mixtures, not the pure phase;
        including them inflates sigma and conflates a *resolution* limit with a
        *counting* limit. Set ``erode=1`` or ``2`` when quoting per-phase sigma
        as a noise measurement. Requires scipy; ignored if unavailable.
    """
    out: dict[int, tuple[float, float, int]] = {}
    for lb in labels:
        m = label_vol == lb
        if erode > 0 and m.any():
            try:
                from scipy import ndimage
                m = ndimage.binary_erosion(m, iterations=erode)
            except ImportError:
                pass
        v = vol[m]
        if v.size == 0:
            out[int(lb)] = (np.nan, np.nan, 0)
        else:
            out[int(lb)] = (float(v.mean()), float(v.std()), int(v.size))
    return out


def cnr(vol: np.ndarray, label_vol: np.ndarray,
        label_a: int, label_b: int, erode: int = 0) -> float:
    """
    Contrast-to-noise ratio between two phases in one reconstructed volume.

        CNR = |mu_a - mu_b| / sqrt((sigma_a^2 + sigma_b^2) / 2)

    The Rose criterion puts reliable visual detection at CNR >~ 5 for
    reasonably sized features; `rose_dose_threshold` interpolates the dose at
    which a sweep crosses it.
    """
    s = material_stats(vol, label_vol, [label_a, label_b], erode=erode)
    ma, sa, na = s[label_a]
    mb, sb, nb = s[label_b]
    if na == 0 or nb == 0:
        return np.nan
    denom = np.sqrt(0.5 * (sa ** 2 + sb ** 2))
    return float(abs(ma - mb) / denom) if denom > 0 else np.inf


def d_prime(vol: np.ndarray, label_vol: np.ndarray,
            label_a: int, label_b: int, erode: int = 0) -> float:
    """
    Univariate separability (identical to `cnr` for equal-variance clusters).

    Kept as a separate name because it is the 1-D special case of
    `joint_d_prime`, and the paper's argument is the comparison between them.
    """
    return cnr(vol, label_vol, label_a, label_b, erode=erode)


def joint_d_prime(
    vol_x: np.ndarray,
    vol_n: np.ndarray,
    label_vol: np.ndarray,
    label_a: int,
    label_b: int,
    erode: int = 0,
    max_voxels: Optional[int] = None,
    seed: int = 0,
) -> dict[str, float]:
    """
    Mahalanobis separability of two phases in the joint (mu_x, mu_n) plane.

        d_joint = sqrt( (m_a - m_b)^T S^-1 (m_a - m_b) )

    with `S` the pooled within-phase covariance. This is the quantitative form
    of the dual-modality claim: **d_joint >= max(d_x, d_n) always**, and the
    size of the excess is exactly what a bimodal experiment buys over the
    better single modality. When the two phases' offsets are parallel in the
    two modalities (both driven by bulk density) the excess is ~0 and neutrons
    add nothing; when the offsets are orthogonal (X-ray tracks Fe, neutrons
    track H) it approaches the quadrature sum.

    Parameters
    ----------
    max_voxels : int, optional
        Randomly subsample each phase to at most this many voxels before
        computing means and covariances. A 64-slice slab of a 1128^2 fossil
        holds ~1.5e7 matrix voxels; stacking those into a float64 (N, 2) array
        costs a few hundred MB per phase for an estimate that is already
        converged at ~1e6 samples. Subsampling is deterministic given `seed`.

    Returns
    -------
    dict with ``d_x``, ``d_n``, ``d_joint``, ``gain`` (= d_joint / max(d_x, d_n)),
    and ``angle_deg`` — the angle between the phase-offset vector and the
    X-ray axis after whitening, i.e. how much of the separation is neutron-borne.
    """
    ma = label_vol == label_a
    mb = label_vol == label_b
    if erode > 0:
        try:
            from scipy import ndimage
            ma = ndimage.binary_erosion(ma, iterations=erode)
            mb = ndimage.binary_erosion(mb, iterations=erode)
        except ImportError:
            pass

    def _sample(mask, tag):
        ax, an = vol_x[mask], vol_n[mask]
        if max_voxels is not None and ax.size > max_voxels:
            r = np.random.default_rng([seed, tag])
            idx = r.choice(ax.size, size=max_voxels, replace=False)
            ax, an = ax[idx], an[idx]
        return np.column_stack([ax, an]).astype(np.float64)

    A = _sample(ma, int(label_a))
    B = _sample(mb, int(label_b))
    if A.shape[0] < 2 or B.shape[0] < 2:
        return {k: np.nan for k in
                ("d_x", "d_n", "d_joint", "gain", "angle_deg")}

    delta = A.mean(0) - B.mean(0)
    na, nb = A.shape[0], B.shape[0]
    S = ((na - 1) * np.cov(A, rowvar=False)
         + (nb - 1) * np.cov(B, rowvar=False)) / (na + nb - 2)
    S = np.atleast_2d(S)
    S += np.eye(2) * 1e-12 * max(np.trace(S), 1e-12)

    d_joint = float(np.sqrt(delta @ np.linalg.solve(S, delta)))
    d_x = float(abs(delta[0]) / np.sqrt(S[0, 0]))
    d_n = float(abs(delta[1]) / np.sqrt(S[1, 1]))

    # Whitened offset direction: how much of d_joint comes from the neutron axis
    L = np.linalg.cholesky(np.linalg.inv(S))
    w = L.T @ delta
    angle = float(np.degrees(np.arctan2(abs(w[1]), abs(w[0]))))

    best = max(d_x, d_n)
    return {
        "d_x": d_x,
        "d_n": d_n,
        "d_joint": d_joint,
        "gain": float(d_joint / best) if best > 0 else np.nan,
        "angle_deg": angle,
    }


def rose_dose_threshold(
    I0_values: Iterable[float],
    metric_values: Iterable[float],
    threshold: float = 5.0,
) -> float:
    """
    Log-linear interpolation of the dose at which a metric first crosses
    `threshold` (default: the Rose criterion, CNR = 5).

    Returns ``np.nan`` if the sweep never crosses. Reported per specimen, this
    is the paper's practical output: *"this fossil needs >= 3e3 neutrons/pixel
    before bone and matrix separate; that one never does."*
    """
    I0 = np.asarray(list(I0_values), dtype=np.float64)
    y = np.asarray(list(metric_values), dtype=np.float64)
    order = np.argsort(I0)
    I0, y = I0[order], y[order]

    ok = np.isfinite(y) & np.isfinite(I0) & (I0 > 0)
    I0, y = I0[ok], y[ok]
    if y.size < 2 or y.max() < threshold or y.min() > threshold:
        return float(I0.min()) if y.size and y.min() > threshold else np.nan

    for i in range(len(y) - 1):
        if (y[i] - threshold) * (y[i + 1] - threshold) <= 0 and y[i] != y[i + 1]:
            lo, hi = np.log10(I0[i]), np.log10(I0[i + 1])
            f = (threshold - y[i]) / (y[i + 1] - y[i])
            return float(10 ** (lo + f * (hi - lo)))
    return np.nan
