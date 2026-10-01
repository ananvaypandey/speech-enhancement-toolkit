"""Loudness normalisation with a true-peak guarantee.

Two normalisation targets are supported:

  - `normalise_lufs` measures with the BS.1770 K-weighting filter and scales
    to an integrated loudness target. This is the perceptually correct choice
    for speech and is the default.
  - `normalise_peak` scales to a peak target. Simpler, but it does not
    account for programme material, so a sparse recording with a single
    transient ends up quieter than intended.

Both finish with the same true-peak limiter, because integrated loudness
says nothing about sample peaks: a target of -16 LUFS can still contain
samples at 0.0 dBFS. That is exactly how clipping happens in practice.
"""

from __future__ import annotations

import numpy as np

from ..analysis.levels import integrated_lufs
from ..config import CLIP_CEILING_DBFS, TARGET_LUFS_SPEECH
from ..io.encode import true_peak_limit
from ..types import AudioBuffer, FloatArray

# A second pass can only help if the first pass was limited; iterate a bounded
# number of times because K-weighting is only approximately invariant to gain.
MAX_PASSES = 3
GAIN_TOLERANCE_DB = 0.1


def normalise_lufs(
    buffer: AudioBuffer,
    target_lufs: float = TARGET_LUFS_SPEECH,
    ceiling_dbfs: float = CLIP_CEILING_DBFS,
    max_passes: int = MAX_PASSES,
) -> AudioBuffer:
    """Scale to an integrated-loudness target, then limit true peaks.

    Falls back to peak normalisation when loudness cannot be measured (a
    silent or extremely short buffer), because returning the input unchanged
    would be less useful than a level the operator can actually work with.
    """
    samples = buffer.samples.astype(np.float32)
    if samples.size == 0 or not np.any(samples):
        return buffer

    current, _ = integrated_lufs(buffer)
    if current is None:
        return normalise_peak(buffer, target_peak_dbfs=ceiling_dbfs - 3.0, ceiling_dbfs=ceiling_dbfs)

    data = samples
    for _ in range(max_passes):
        gain_db = target_lufs - current
        if abs(gain_db) < GAIN_TOLERANCE_DB:
            break
        # True-peak limit *after* each gain step, so the measurement is of
        # what would actually be written.
        data = true_peak_limit(data * (10.0 ** (gain_db / 20.0)), ceiling_dbfs)
        candidate = AudioBuffer(
            samples=data,
            sample_rate=buffer.sample_rate,
            source_path=buffer.source_path,
            original_format=buffer.original_format,
        )
        current, _ = integrated_lufs(candidate)
        if current is None:
            break

    return AudioBuffer(
        samples=true_peak_limit(data, ceiling_dbfs),
        sample_rate=buffer.sample_rate,
        source_path=buffer.source_path,
        original_format=buffer.original_format,
    )


def normalise_peak(buffer: AudioBuffer, target_peak_dbfs: float = -3.0, ceiling_dbfs: float = CLIP_CEILING_DBFS) -> AudioBuffer:
    """Scale so the loudest sample lands on `target_peak_dbfs`."""
    samples = buffer.samples.astype(np.float32)
    peak = float(np.max(np.abs(samples))) if samples.size else 0.0
    if peak <= 0.0:
        return buffer
    target = float(10.0 ** (target_peak_dbfs / 20.0))
    scaled = (samples * (target / peak)).astype(np.float32)
    # Target above the ceiling: the limiter wins.
    if target_peak_dbfs > ceiling_dbfs:
        scaled = true_peak_limit(scaled, ceiling_dbfs)
    return AudioBuffer(
        samples=scaled,
        sample_rate=buffer.sample_rate,
        source_path=buffer.source_path,
        original_format=buffer.original_format,
    )


def speech_presence_gain(
    buffer: AudioBuffer,
    target_presence_lufs: float = -20.0,
    max_boost_db: float = 12.0,
) -> FloatArray:
    """Compute the gain that centres speech-band energy on a target level.

    Returned as a gain array rather than applied, so callers can clamp it
    before committing to a level change. Boosting a recording whose speech is
    already buried in noise makes the noise louder too, so this is advisory.
    """
    from ..config import SPEECH_PRESENCE_BAND_HZ

    mono = buffer.mono().astype(np.float64)
    if mono.size == 0:
        return np.zeros(buffer.frames, dtype=np.float32)

    spectrum = np.abs(np.fft.rfft(mono))
    freqs = np.fft.rfftfreq(buffer.frames, 1.0 / buffer.sample_rate)
    lo, hi = SPEECH_PRESENCE_BAND_HZ
    band = spectrum[(freqs >= lo) & (freqs <= hi)]
    if band.size == 0 or not np.any(band > 0):
        return np.zeros(buffer.frames, dtype=np.float32)

    band_rms = float(np.sqrt(np.mean(np.square(band))))
    current_db = 20.0 * np.log10(max(band_rms, 1e-12))
    gain_db = float(np.clip(target_presence_lufs - current_db, -max_boost_db, max_boost_db))
    return np.full(buffer.frames, 10.0 ** (gain_db / 20.0), dtype=np.float32)
