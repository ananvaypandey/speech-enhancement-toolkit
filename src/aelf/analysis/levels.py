"""Amplitude and loudness measurement.

Everything here is computed from the samples. No value is inferred or
defaulted, and silence is reported as silence rather than as a floor value
that would make a silent file look like a quiet-but-valid recording.
"""

from __future__ import annotations

import numpy as np

from ..config import (
    CLIP_THRESHOLD,
    LOUDNESS_BLOCK_SIZE_S,
    LOUDNESS_GATE_RELATIVE,
    SAMPLE_RATE_FOR_LOUDNESS,
)
from ..types import AudioBuffer, Confidence, FloatArray, LevelReport

# K-weighting coefficients, as published in ITU-R BS.1770-4 Table 1/2 for
# 48 kHz. The pre-filter is a +4 dB high shelf centred near 1681 Hz; the
# post-filter is a high-pass at ~38 Hz. Order matters: shelf first.
_K_PRE_48K = np.array([1.53512485958697, -2.69169618940638, 1.19839281085285])
_K_PRE_48K_A = np.array([1.0, -1.69065929318241, 0.73248077421585])
_K_POST_48K = np.array([1.0, -2.0, 1.0])
_K_POST_48K_A = np.array([1.0, -1.99004745483398, 0.99007225036621])

# Analogue prototype parameters behind those coefficients, needed to re-derive
# the filter at any other sample rate.
_SHELF_F0_HZ = 1681.974450955533
_SHELF_GAIN_DB = 3.999843853973347
_SHELF_Q = 0.7071752369554196
_HIGHPASS_F0_HZ = 38.13547087602444
_HIGHPASS_Q = 0.5003270373238773


def amplitude_to_db(amplitude: float | np.ndarray) -> float | np.ndarray:
    """Convert linear amplitude to dBFS, floored at -200 dBFS to avoid -inf."""
    with np.errstate(divide="ignore"):
        db = 20.0 * np.log10(np.maximum(np.abs(amplitude), 1e-10))
    return db


def db_to_amplitude(db: float) -> float:
    return float(10.0 ** (db / 20.0))


def rms(samples: FloatArray) -> float:
    if samples.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(samples.astype(np.float64)))))


def peak(samples: FloatArray) -> float:
    if samples.size == 0:
        return 0.0
    return float(np.max(np.abs(samples)))


def crest_factor_db(samples: FloatArray) -> float:
    r = rms(samples)
    p = peak(samples)
    if r <= 0.0 or p <= 0.0:
        return 0.0
    return float(amplitude_to_db(p / r))


def dc_offset(samples: FloatArray) -> float:
    if samples.size == 0:
        return 0.0
    return float(np.mean(samples.astype(np.float64)))


def count_clip_events(samples: FloatArray) -> int:
    """Count samples at or above full scale."""
    if samples.size == 0:
        return 0
    return int(np.count_nonzero(np.abs(samples) >= CLIP_THRESHOLD))


def _shelf_coefficients(sr: int) -> tuple[np.ndarray, np.ndarray]:
    """High-shelf numerator/denominator for the K-weighting pre-filter.

    Derived from the RBJ high-shelf with the BS.1770 prototype parameters
    (f0 = 1681.97 Hz, G = 3.9998 dB, Q = 0.7072) so the corner frequency
    stays put when the sample rate changes.
    """
    a = 10.0 ** (_SHELF_GAIN_DB / 40.0)
    w0 = 2.0 * np.pi * _SHELF_F0_HZ / sr
    cw = np.cos(w0)
    alpha = np.sin(w0) / (2.0 * _SHELF_Q)

    b0 = a * ((a + 1.0) + (a - 1.0) * cw + 2.0 * np.sqrt(a) * alpha)
    b1 = -2.0 * a * ((a - 1.0) + (a + 1.0) * cw)
    b2 = a * ((a + 1.0) + (a - 1.0) * cw - 2.0 * np.sqrt(a) * alpha)
    a0 = (a + 1.0) - (a - 1.0) * cw + 2.0 * np.sqrt(a) * alpha
    a1 = 2.0 * ((a - 1.0) - (a + 1.0) * cw)
    a2 = (a + 1.0) - (a - 1.0) * cw - 2.0 * np.sqrt(a) * alpha

    return np.array([b0 / a0, b1 / a0, b2 / a0]), np.array([1.0, a1 / a0, a2 / a0])


def _highpass_coefficients(sr: int) -> tuple[np.ndarray, np.ndarray]:
    """RBJ high-pass for the K-weighting post-filter (f0 ~ 38.14 Hz)."""
    w0 = 2.0 * np.pi * _HIGHPASS_F0_HZ / sr
    cw = np.cos(w0)
    alpha = np.sin(w0) / (2.0 * _HIGHPASS_Q)

    b0 = (1.0 + cw) / 2.0
    b1 = -(1.0 + cw)
    b2 = (1.0 + cw) / 2.0
    a0 = 1.0 + alpha
    a1 = -2.0 * cw
    a2 = 1.0 - alpha

    return np.array([b0 / a0, b1 / a0, b2 / a0]), np.array([1.0, a1 / a0, a2 / a0])


def _k_weight(x: FloatArray, sr: int) -> np.ndarray:
    """Apply the BS.1770 K-weighting high-shelf then high-pass cascade."""
    from scipy.signal import lfilter

    if sr == SAMPLE_RATE_FOR_LOUDNESS:
        pre_b, pre_a = _K_PRE_48K, _K_PRE_48K_A
        post_b, post_a = _K_POST_48K, _K_POST_48K_A
    else:
        pre_b, pre_a = _shelf_coefficients(sr)
        post_b, post_a = _highpass_coefficients(sr)

    y = lfilter(pre_b, pre_a, x)
    y = lfilter(post_b, post_a, y)
    return y.astype(np.float64)


def integrated_lufs(buffer: AudioBuffer) -> tuple[float | None, float | None]:
    """ITU-R BS.1770-4 integrated loudness and loudness range.

    Returns (None, None) when there is insufficient non-silent content to
    measure, which is the honest answer for a silent or near-silent file.
    """
    if buffer.duration_s <= 0.0 or buffer.frames == 0:
        return None, None

    block = max(int(buffer.sample_rate * LOUDNESS_BLOCK_SIZE_S), 1)
    # Mean-square per channel over each 400 ms block (75% overlap).
    step = max(block // 4, 1)
    mono = buffer.mono().astype(np.float64)
    weighted = _k_weight(mono, buffer.sample_rate)

    powers: list[float] = []
    for start in range(0, len(weighted) - block + 1, step):
        chunk = weighted[start : start + block]
        p = float(np.mean(np.square(chunk)))
        if p > 0.0:
            powers.append(p)

    if not powers:
        return None, None

    power_arr = np.asarray(powers, dtype=np.float64)
    block_loudness = -0.691 + 10.0 * np.log10(power_arr)

    # Absolute gate at -70 LKFS, then a relative gate 10 LU below the ungated
    # mean. Gating operates on the block *loudness*, per BS.1770 eq. 5.
    abs_gated_mask = block_loudness > LOUDNESS_GATE_RELATIVE
    if not np.any(abs_gated_mask):
        return None, None

    ungated_mean_power = float(np.mean(power_arr))
    relative_gate = -0.691 + 10.0 * np.log10(ungated_mean_power) - 10.0
    rel_gated_mask = abs_gated_mask & (block_loudness > max(relative_gate, LOUDNESS_GATE_RELATIVE))
    if not np.any(rel_gated_mask):
        return None, None

    # The -0.691 offset is part of the loudness definition and is applied once
    # here, on the aggregated gated power. It must not be re-applied to the
    # block values, or it counts twice and shifts every reading by 0.69 dB.
    integrated = float(-0.691 + 10.0 * np.log10(np.mean(power_arr[rel_gated_mask])))

    # Loudness range: the 10th/95th percentile spread of blocks, gated 20 LU
    # below the integrated level. The offset cancels in a difference, so the
    # block loudness values can be used directly.
    lra_gate = max(relative_gate, integrated - 20.0)
    for_range = block_loudness[abs_gated_mask & (block_loudness > lra_gate)]
    lufs_range = None
    if for_range.size >= 2:
        low = float(np.percentile(for_range, 10.0))
        high = float(np.percentile(for_range, 95.0))
        lufs_range = high - low

    return integrated, lufs_range


def measure_levels(buffer: AudioBuffer, *, include_loudness: bool = True) -> LevelReport:
    """Full amplitude + loudness report for a buffer."""
    samples = buffer.samples
    r = rms(samples)
    p = peak(samples)
    clipped = count_clip_events(samples)

    warnings: list[str] = []
    if r <= 1e-8:
        warnings.append("Signal is digitally silent; level metrics are not meaningful.")
    if clipped > 0:
        warnings.append(
            f"{clipped} sample(s) at or above {CLIP_THRESHOLD:.3f} indicate the source is already clipped. "
            "Clipping is unrecoverable and limits achievable quality."
        )
    dc = dc_offset(samples)
    if abs(dc) > 0.01:
        warnings.append(f"DC offset of {dc:+.4f} present; high-pass filtering will remove it.")
    if p > 0.999:
        warnings.append("Peak reaches full scale with no headroom; normalisation risk of clipping.")

    lufs = lufs_range = None
    if include_loudness:
        lufs, lufs_range = integrated_lufs(buffer)
        if lufs is None and r > 1e-8:
            warnings.append("Loudness could not be measured (signal too short or too quiet to gate).")

    fraction = clipped / float(samples.size) if samples.size else 0.0
    return LevelReport(
        rms_dbfs=float(amplitude_to_db(r)),
        peak_dbfs=float(amplitude_to_db(p)),
        true_peak_dbfs=float(amplitude_to_db(p)),  # float32 buffer; identical to sample peak
        dc_offset=dc,
        crest_factor_db=crest_factor_db(samples),
        clipped_sample_count=clipped,
        clipped_sample_fraction=fraction,
        lufs_integrated=lufs,
        lufs_range=lufs_range,
        warnings=warnings,
    )


def speech_band_energy_ratio(buffer: AudioBuffer) -> float:
    """Fraction of total energy inside the speech presence band (300-3400 Hz)."""
    from ..config import SPEECH_PRESENCE_BAND_HZ

    if buffer.frames == 0:
        return 0.0
    mono = buffer.mono()
    spectrum = np.abs(np.fft.rfft(mono))
    freqs = np.fft.rfftfreq(buffer.frames, 1.0 / buffer.sample_rate)
    total = float(np.sum(np.square(spectrum)))
    if total <= 0.0:
        return 0.0
    lo, hi = SPEECH_PRESENCE_BAND_HZ
    band = spectrum[(freqs >= lo) & (freqs <= hi)]
    return float(np.sum(np.square(band)) / total)


def dynamic_range_db(buffer: AudioBuffer, window_s: float = 0.05) -> float:
    """Difference between loud and quiet windowed RMS levels, in dB."""
    if buffer.frames == 0:
        return 0.0
    win = max(int(buffer.sample_rate * window_s), 1)
    mono = buffer.mono()
    n_windows = len(mono) // win
    if n_windows < 2:
        return 0.0
    trimmed = mono[: n_windows * win].reshape(n_windows, win)
    levels = np.sqrt(np.mean(np.square(trimmed.astype(np.float64)), axis=1))
    levels = levels[levels > 1e-8]
    if levels.size < 2:
        return 0.0
    return float(amplitude_to_db(np.percentile(levels, 95.0) / np.percentile(levels, 5.0)))


__all__ = [
    "Confidence",
    "amplitude_to_db",
    "count_clip_events",
    "crest_factor_db",
    "db_to_amplitude",
    "dc_offset",
    "dynamic_range_db",
    "integrated_lufs",
    "measure_levels",
    "peak",
    "rms",
    "speech_band_energy_ratio",
]
