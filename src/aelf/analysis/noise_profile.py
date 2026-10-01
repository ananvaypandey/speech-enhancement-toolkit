"""Noise characteristic estimation.

The approach is minimum-statistics style: frame the signal, rank the
short-term energy within each frame, and take a low percentile as the noise
estimate. Percentiles are used rather than an absolute threshold because
absolute thresholds fail across recording levels; the 10th percentile of a
speech frame reliably sits in the inter-word gaps.

Reported values are estimates and are labelled as such. The `confidence`
field degrades when the signal gives us little to work with - a file with no
detected quiet frames cannot yield a trustworthy noise floor, and the module
says so instead of returning a number derived from speech.
"""

from __future__ import annotations

import numpy as np

from ..config import SPEECH_BAND_HZ
from ..types import AudioBuffer, Confidence, NoiseProfile

FRAME_S = 0.030
HOP_S = 0.010
NOISE_PERCENTILE = 10.0
# Below this many usable noise frames the estimate is not worth reporting.
MIN_NOISE_FRAMES = 8
# Broadband hum shows up as a narrow spike standing well above the local
# spectral median; require this much prominence (dB) before calling it hum.
HUM_PROMINENCE_DB = 12.0
MAINS_HZ = (50.0, 60.0)
# Interquartile spread under which the background counts as stationary.
STATIONARY_SPREAD_MAX_DB = 10.0


def _frames(mono: np.ndarray, sr: int) -> tuple[np.ndarray, int, int]:
    frame_len = max(int(sr * FRAME_S), 16)
    hop = max(int(sr * HOP_S), 1)
    if len(mono) < frame_len:
        # Too short to frame: treat the whole signal as one frame.
        return mono[np.newaxis, :], frame_len, hop
    n = 1 + (len(mono) - frame_len) // hop
    idx = np.arange(frame_len)[None, :] + hop * np.arange(n)[:, None]
    return mono[idx], frame_len, hop


def _spectral_tilt_db_per_octave(power: np.ndarray, freqs: np.ndarray) -> float:
    """Least-squares slope of power vs log2(frequency), in dB per octave."""
    valid = (freqs > 40.0) & (power > 0) & np.isfinite(power)
    if valid.sum() < 8:
        return 0.0
    log_f = np.log2(freqs[valid])
    log_p = 10.0 * np.log10(power[valid])
    slope = np.polyfit(log_f, log_p, 1)[0]
    return float(slope)


def estimate_noise_profile(buffer: AudioBuffer) -> NoiseProfile:
    """Estimate noise floor, spectral tilt, hum and SNR from the signal."""
    notes: list[str] = []
    sr = buffer.sample_rate
    mono = buffer.mono().astype(np.float64)

    if mono.size == 0:
        return NoiseProfile(
            noise_floor_dbfs=None,
            noise_floor_hz=0.0,
            spectral_tilt_db_per_octave=0.0,
            is_stationary=False,
            estimated_hum_hz=None,
            estimated_snr_db=None,
            dominant_noise_band_hz=SPEECH_BAND_HZ,
            method="empty-signal",
            confidence=Confidence.NONE,
            frame_count=0,
            notes=["Signal is empty; no noise characteristics can be estimated."],
        )

    framed, frame_len, _hop = _frames(mono, sr)
    window = np.hanning(frame_len)
    if framed.shape[0] == 1 and framed.shape[1] < frame_len:
        pad = np.pad(framed, ((0, 0), (0, frame_len - framed.shape[1])))
        framed = pad
        window = np.hanning(frame_len)

    spectra = np.fft.rfft(framed * window, axis=1)
    power = np.square(np.abs(spectra))

    # Convert the mean power per bin back into a time-domain mean-square
    # amplitude. Without this the reported floor is wrong by tens of dB.
    # By Parseval, sum_k |X_k|^2 = N * sum_n (x*w)_n^2, so
    #   mean_square = mean_k|X_k|^2 * n_bins / (N * sum_n w_n^2)
    n_bins = power.shape[1]
    window_power = float(np.sum(np.square(window)))
    to_mean_square = n_bins / (frame_len * window_power)
    frame_power = np.mean(power, axis=1) * to_mean_square

    active = frame_power[frame_power > 0.0]
    if active.size == 0:
        return NoiseProfile(
            noise_floor_dbfs=None,
            noise_floor_hz=0.0,
            spectral_tilt_db_per_octave=0.0,
            is_stationary=False,
            estimated_hum_hz=None,
            estimated_snr_db=None,
            dominant_noise_band_hz=SPEECH_BAND_HZ,
            method="digital-silence",
            confidence=Confidence.NONE,
            frame_count=int(framed.shape[0]),
            notes=["Signal is digitally silent; noise floor is undefined."],
        )

    # Minimum statistics: the 10th percentile of frame energy approximates the
    # noise-only level, since speech is present in well under half of frames.
    noise_threshold = float(np.percentile(active, NOISE_PERCENTILE))
    noise_mask = frame_power <= noise_threshold
    noise_frames = int(noise_mask.sum())
    freqs = np.fft.rfftfreq(frame_len, 1.0 / sr)

    if noise_frames < MIN_NOISE_FRAMES:
        # Not enough quiet material. Fall back to the quietest frames we do
        # have, but downgrade confidence rather than implying robustness.
        order = np.argsort(frame_power)[: min(MIN_NOISE_FRAMES, framed.shape[0])]
        noise_mask = np.zeros_like(frame_power, dtype=bool)
        noise_mask[order] = True
        noise_frames = int(noise_mask.sum())
        notes.append(
            f"Only {noise_frames} frame(s) fall below the noise threshold; the recording may be "
            "continuous speech with no pauses. Noise estimates are low confidence."
        )

    noise_power_spectrum = np.mean(power[noise_mask], axis=0)

    # Broadband floor level. The per-frame power is already a mean-square
    # quantity, so frame_power and the per-bin mean both need scaling out of
    # the FFT's implicit 1/N. Deriving the floor from the percentile
    # *threshold* rather than from the mean of the selected frames matters:
    # averaging only the quietest decile biases the result upward.
    threshold_power = noise_threshold
    noise_rms = float(np.sqrt(max(threshold_power, 0.0)))
    noise_floor_dbfs = float(20.0 * np.log10(max(noise_rms, 1e-12)))

    # The frequency where noise energy is most concentrated. For white noise
    # every bin is equally loaded, so report the band centre rather than
    # whichever bin happened to win a coin toss.
    band_mask = freqs <= sr / 2.0
    band_power = noise_power_spectrum[band_mask]
    band_freqs = freqs[band_mask]
    peak_bin = int(np.argmax(band_power))
    dominant_hz = float(band_freqs[peak_bin])

    nonzero_bins = band_power[band_power > 0.0]
    if nonzero_bins.size > 4:
        # If the top bin is within a few dB of the median bin, the spectrum is
        # effectively flat and there is no meaningful "dominant" frequency.
        spread_db = 10.0 * np.log10(max(nonzero_bins.max(), 1e-20) / max(np.median(nonzero_bins), 1e-20))
        if spread_db < 6.0:
            dominant_hz = float(np.sqrt(SPEECH_BAND_HZ[0] * min(sr / 2.0, 8_000.0)))
            notes.append("Noise spectrum is broadband (no dominant frequency); reporting a nominal band centre.")

    tilt = _spectral_tilt_db_per_octave(noise_power_spectrum, freqs)

    # Mains hum: a narrow peak near 50 or 60 Hz that stands well above the
    # local spectral floor. Comparing against the median bin (not the sum of
    # the search band) makes this a real outlier test, so broadband noise does
    # not trip it. The search band is padded to the analysis resolution
    # because a 30 ms frame cannot resolve 50 Hz any more finely than that.
    hum_hz: float | None = None
    bin_hz = float(sr / frame_len)
    pad_hz = max(3.0, bin_hz)
    valid_bins = noise_power_spectrum[band_power > 0.0]
    local_median = float(np.median(valid_bins)) if valid_bins.size else 0.0
    for nominal in MAINS_HZ:
        sel = (freqs >= nominal - pad_hz) & (freqs <= nominal + pad_hz)
        if not np.any(sel):
            continue
        local = noise_power_spectrum[sel]
        bin_idx = int(np.where(sel)[0][int(np.argmax(local))])
        peak_power = float(local.max())
        if local_median <= 0.0:
            continue
        prominence_db = 10.0 * np.log10(max(peak_power, 1e-20) / local_median)
        if prominence_db > HUM_PROMINENCE_DB:
            hum_hz = float(freqs[bin_idx])
            notes.append(
                f"Narrowband component at {hum_hz:.1f} Hz stands {prominence_db:.0f} dB above the noise "
                "spectrum, consistent with mains hum."
            )
            break

    # SNR from the mean frame power vs the noise threshold.
    speech_power = float(np.mean(active))
    snr_db = float(10.0 * np.log10(max(speech_power, 1e-20) / max(noise_threshold, 1e-20)))

    # Stationarity. Comparing a low percentile against the median only detects
    # noise that gets *louder* than the rest. A recording that swings from
    # very quiet to very loud is equally non-stationary, so compare the
    # interquartile spread of the whole distribution.
    q25 = float(np.percentile(active, 25.0))
    q75 = float(np.percentile(active, 75.0))
    spread_db = 10.0 * np.log10(max(q75, 1e-20) / max(q25, 1e-20))
    is_stationary = spread_db < STATIONARY_SPREAD_MAX_DB
    if not is_stationary:
        notes.append(
            f"Signal level varies by {spread_db:.0f} dB (interquartile spread) across the recording; "
            "the background is non-stationary."
        )

    lo_hz = float(band_freqs[max(0, peak_bin - 3)])
    hi_hz = float(band_freqs[min(len(band_freqs) - 1, peak_bin + 3)])

    if noise_frames < MIN_NOISE_FRAMES:
        confidence = Confidence.LOW
    elif is_stationary and noise_frames >= framed.shape[0] * 0.15:
        confidence = Confidence.HIGH
    else:
        confidence = Confidence.MEDIUM

    return NoiseProfile(
        noise_floor_dbfs=noise_floor_dbfs,
        noise_floor_hz=dominant_hz,
        spectral_tilt_db_per_octave=tilt,
        is_stationary=is_stationary,
        estimated_hum_hz=hum_hz,
        estimated_snr_db=snr_db,
        dominant_noise_band_hz=(lo_hz, hi_hz),
        method=f"minimum-statistics p{NOISE_PERCENTILE:.0f} over {FRAME_S * 1000:.0f} ms frames",
        confidence=confidence,
        frame_count=noise_frames,
        notes=notes,
    )


def noise_psd(buffer: AudioBuffer, profile: NoiseProfile) -> np.ndarray:
    """Average noise power spectrum, used to initialise the spectral suppressor."""
    sr = buffer.sample_rate
    mono = buffer.mono().astype(np.float64)
    framed, frame_len, _hop = _frames(mono, sr)
    window = np.hanning(frame_len)
    spectra = np.square(np.abs(np.fft.rfft(framed * window, axis=1)))
    frame_power = np.mean(spectra, axis=1)
    active = frame_power[frame_power > 0.0]
    if active.size == 0:
        nfft = frame_len // 2 + 1
        return np.full(nfft, 1e-12, dtype=np.float64)
    threshold = float(np.percentile(active, NOISE_PERCENTILE))
    mask = frame_power <= threshold
    if not np.any(mask):
        order = np.argsort(frame_power)[: min(MIN_NOISE_FRAMES, framed.shape[0])]
        mask = np.zeros_like(frame_power, dtype=bool)
        mask[order] = True
    return np.mean(spectra[mask], axis=0)
