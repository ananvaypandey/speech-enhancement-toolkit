"""Spectral noise suppression.

Removes stationary background noise (hiss, hum, fan whir) by estimating the
noise spectrum from the quietest frames, then attenuating each frequency bin by
how much of it looks like noise rather than speech.

The classic approach is spectral subtraction. A more robust variant, and the
one used here, is OMLSA-style decision-directed masking with a Wiener floor:

1. Split the signal into overlapping frames and take an FFT of each.
2. Estimate the noise spectrum from the lowest-energy frames, which are the
   ones most likely to be noise-only.
3. Compute a per-bin signal-to-noise ratio.
4. Turn that ratio into a gain mask, smoothed over time and frequency so it
   does not pump or sound musical.
5. Apply the mask and reconstruct with an inverse FFT.

The Wiener floor matters more than it looks. Plain spectral subtraction
over-subtracts and produces the characteristic "underwater" or "hollow" artefact
on speech. Flooring the gain at a small positive value keeps a residue of the
original signal, which sounds far more natural at the cost of leaving a little
noise in place.

`strength` trades noise removal against speech distortion:

- 0.0 leaves the signal essentially untouched.
- 0.5 is the default: clearly quieter noise, speech still natural.
- 1.0 removes as much as this method can without going past the floor.

No neural network is involved, so this runs anywhere numpy and scipy do, in
real time, with nothing downloaded.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from ..config import (
    OMLSA_FLOOR_ALPHA,
    OMLSA_FLOOR_STRONG,
    OVER_SUBTRACTION_MAX_DB,
    SEPARABILITY_OFF_DB,
    SEPARABILITY_ON_DB,
    STRENGTH_DEFAULT,
    STRENGTH_MAX,
    STRENGTH_MIN,
)
from ..errors import EnhancementError
from ..types import AudioBuffer, EnhancementResult, FloatArray

# 32 ms at 48 kHz. Long enough to resolve speech formants, short enough to
# track the start of a syllable without smearing it.
FRAME_MS = 32.0
# 50% overlap: the square root-Hann analysis/synthesis pair below is
# overlap-add, which requires this to reconstruct without amplitude modulation.
HOP_FRACTION = 0.5
# Smoothing time constants, in seconds. Slower than this and the mask chases
# individual cycles; faster and speech transients get hollowed out.
SMOOTH_FREQ_BINS = 1.5
SMOOTH_TIME_S = 0.06


@dataclass(frozen=True)
class SpectralConfig:
    """Tunables for the suppressor. Defaults are chosen for speech."""

    frame_ms: float = FRAME_MS
    hop_fraction: float = HOP_FRACTION
    strength: float = STRENGTH_DEFAULT
    floor_alpha: float = OMLSA_FLOOR_ALPHA
    noise_frame_fraction: float = 0.15
    max_noise_frames: int = 60
    over_subtraction_db: float = OVER_SUBTRACTION_MAX_DB
    separability_on_db: float = SEPARABILITY_ON_DB
    separability_off_db: float = SEPARABILITY_OFF_DB
    smooth_freq_bins: float = SMOOTH_FREQ_BINS
    smooth_time_s: float = SMOOTH_TIME_S

    def __post_init__(self) -> None:
        if not STRENGTH_MIN <= self.strength <= STRENGTH_MAX:
            raise EnhancementError(
                f"Strength must be between {STRENGTH_MIN} and {STRENGTH_MAX}, got {self.strength}",
                user_message=f"Choose a strength between {STRENGTH_MIN:.1f} and {STRENGTH_MAX:.1f}.",
            )
        if not 0.0 < self.noise_frame_fraction <= 0.5:
            raise EnhancementError(
                f"noise_frame_fraction must be in (0, 0.5], got {self.noise_frame_fraction}",
                user_message="Internal configuration error.",
            )
        # `on` is the fully-suppressed end of the ramp (high separability, so
        # no suppression), `off` is the unsuppressed end (low separability,
        # real noise). Hence `on` must exceed `off`.
        if self.separability_on_db <= self.separability_off_db:
            raise EnhancementError(
                f"separability_on_db ({self.separability_on_db}) must exceed "
                f"separability_off_db ({self.separability_off_db})",
                user_message="Internal configuration error.",
            )


def _frame_signal(mono: FloatArray, frame_len: int, hop: int) -> tuple[np.ndarray, int]:
    """Split into overlapping frames, returning the frames and the lead pad.

    Both ends are padded by half a frame before framing. This is not cosmetic:
    a Hann window is exactly zero at its first and last sample, so without lead
    padding the very first output sample is covered by a single frame whose
    window weight is ~1.8e-11. Dividing by that in the overlap-add normaliser
    produced reconstruction errors of 7e6. Padding moves those zero-weight
    regions into the pad, so every real sample is covered by frames carrying
    usable weight.
    """
    lead = frame_len // 2
    padded = np.pad(mono, (lead, frame_len + hop))
    frames = int(np.ceil(max(padded.shape[0] - frame_len, 0) / hop)) + 1
    total = (frames - 1) * hop + frame_len
    if total > padded.shape[0]:
        padded = np.pad(padded, (0, total - padded.shape[0]))
    idx = np.arange(frame_len)[None, :] + hop * np.arange(frames)[:, None]
    return padded[idx], lead


def _smooth_freq(magnitude: np.ndarray, bins: float) -> np.ndarray:
    """Frequency-domain smoothing of the mask.

    A per-bin mask makes speech sound warbly, because each FFT bin is narrow
    enough that a single pitch cycle can push a bin above the noise estimate on
    its own. Averaging over neighbouring bins before applying the mask removes
    that without costing real intelligibility.
    """
    if bins <= 0:
        return magnitude
    width = int(max(1, round(bins)))
    kernel = np.hanning(width + 2)[1:-1]
    kernel /= kernel.sum()
    padded = np.pad(magnitude, ((0, 0), (width, width)), mode="edge")
    out = np.empty_like(magnitude)
    for t in range(magnitude.shape[0]):
        out[t] = np.convolve(padded[t], kernel, mode="same")[width:-width]
    return out


def _smooth_time(magnitude: np.ndarray, frames: int) -> np.ndarray:
    """Temporal smoothing, one-directional so it cannot smear speech forward
    in time. An earlier revision used a symmetric kernel, which delayed every
    transient by roughly half a frame and was audible on plosives."""
    if frames <= 0:
        return magnitude
    alpha = float(np.exp(-1.0 / max(frames, 1e-6)))
    out = np.empty_like(magnitude)
    acc = magnitude[0].copy()
    for t in range(magnitude.shape[0]):
        acc = alpha * acc + (1.0 - alpha) * magnitude[t]
        out[t] = acc
    return out


def _estimate_noise_power(spectra: np.ndarray, max_frames: int) -> np.ndarray:
    """Per-bin noise power, as the mean over the quietest frames.

    Selecting whole frames rather than individual bins matters: a bin can be
    quiet because that frequency holds little signal, not because the frame was
    noise-only. Taking low-energy frames first gives frames that are largely
    noise, then averaging each bin across them estimates that noise's level.

    The mean, specifically, rather than a low percentile. Measuring this on a
    speech-plus-hiss signal, the 15th percentile came out 14.7 dB *below* the
    true noise power, because an exponential power distribution's percentiles
    fall away steeply. An underestimate inflates the apparent SNR, which pushes
    the decision-directed loop's fixed point toward unity gain and suppresses
    nothing: -0.8 dB of hiss removal at strength 0.5, against -12 dB for the
    mean. Underestimating the noise is not a conservative choice, it defeats
    the algorithm.
    """
    frame_energy = spectra.sum(axis=1)
    order = np.argsort(frame_energy)
    take = order[: max(1, min(max_frames, spectra.shape[0]))]
    return spectra[take].mean(axis=0)


def suppress_stationary_noise(
    buffer: AudioBuffer,
    *,
    strength: float = STRENGTH_DEFAULT,
    config: SpectralConfig | None = None,
) -> EnhancementResult:
    """Reduce stationary background noise across every channel.

    Noise is estimated from the whole file rather than from the first second,
    because a recording frequently opens with speech or a fade-in and any
    early-frame estimate would then be speech, not noise.
    """
    started = time.perf_counter()
    cfg = config or SpectralConfig(strength=strength)
    if buffer.frames == 0:
        raise EnhancementError(
            "Cannot enhance an empty buffer",
            user_message="The audio contains no samples.",
        )
    if buffer.is_silent:
        return EnhancementResult(
            audio=buffer,
            backend="spectral-subtraction",
            strength=cfg.strength,
            processing_time_s=time.perf_counter() - started,
            notes=["Input is silent; nothing to remove."],
        )

    sr = buffer.sample_rate
    frame_len = max(round(sr * cfg.frame_ms / 1000.0), 64)
    hop = max(round(frame_len * cfg.hop_fraction), 1)

    notes: list[str] = []
    out_channels = []
    total_frames = 0

    for ch in range(buffer.channels):
        mono = buffer.channel(ch).astype(np.float64)
        frames, lead = _frame_signal(mono, frame_len, hop)
        total_frames = frames.shape[0]

        window = np.hanning(frame_len)
        spectra = np.abs(np.fft.rfft(frames * window, axis=1)) ** 2

        quiet_frames = max(round(cfg.noise_frame_fraction * spectra.shape[0]), 1)
        noise_power = _estimate_noise_power(spectra, min(quiet_frames, cfg.max_noise_frames))

        # Separability gate.
        #
        # The noise estimate comes from the quietest frames, which on a clean
        # recording are simply the speaker's quietest moments. The estimator
        # cannot tell that apart from a genuinely quiet noise floor, so it
        # learns speech as noise and then suppresses it: 8.4% RMS change on
        # clean speech at full strength, which is audible.
        #
        # How far the estimate sits below the overall signal separates the two
        # cases cleanly. Measured (estimate-to-signal gap in dB):
        #   noise only 0.8, speech+noise at 12 dB SNR 0.9, at 22 dB 7.2,
        #   at 32 dB 16.2, clean speech 28.1.
        # Below `separability_off_db` there is real noise to remove, and the gate
        # is 0. Above `separability_on_db` the "noise" is the speaker, the gate
        # is 1, and over-subtraction is switched off entirely. The two constants
        # are named for what they switch, not for the direction of the ramp:
        # `off` is where suppression turns off, `on` is where it is fully on.
        separability_db = float(10.0 * np.log10(max(float(spectra.mean()), 1e-30) / max(float(noise_power.mean()), 1e-30)))
        gate = float(
            np.clip(
                (separability_db - cfg.separability_off_db) / (cfg.separability_on_db - cfg.separability_off_db),
                0.0,
                1.0,
            )
        )

        # Over-subtract in proportion to strength, then soften by the gate.
        # Multiplying the noise estimate upward lowers the apparent SNR, so the
        # mask attenuates harder. Linear in dB from zero, so strength 0 stays a
        # pass-through, and it scales the over-subtraction to nothing when the
        # input has no separable noise.
        over_sub = 10.0 ** (cfg.over_subtraction_db * cfg.strength * gate / 20.0)
        # A single frequency bin cannot be silence. Without this floor, a bin
        # that happens to be near zero in every quiet frame produces an
        # infinite ratio and the mask becomes unstable.
        floor = np.max(noise_power) * 1e-6 + 1e-20
        noise_power = np.maximum(noise_power * over_sub, floor)

        # OMLSA decision-directed masking.

        # We need the *a priori* SNR (signal before noise was added), but it
        # cannot be observed: we only ever see the sum. The decision-directed
        # estimator predicts it from the previous frame's enhanced magnitude,
        # relying on speech being largely stationary between phonemes while
        # noise is not:
        #
        #   xi = a * |S_prev|^2 / noise_power + (1 - a) * max(gamma - 1, 0)
        #
        # where gamma is the a posteriori SNR |Y|^2 / noise_power. The second
        # term is the instantaneous estimate, clipped at zero so bins sitting
        # below the noise floor contribute nothing.
        #
        # The previous magnitude, rather than the previous gain, is what makes
        # this stable. A gain-based prior (G^2/(1-G^2)) multiplies gamma by up
        # to ~4 for a gain of 0.9, which locks noise bins at a gain near 0.8 and
        # removes almost nothing; measuring that directly confirmed it.
        alpha = 0.98

        # The Wiener floor, applied after smoothing rather than before. Flooring
        # early and then averaging lets a bin's floor be pulled down by its
        # neighbours' lower values, which is how a 10% floor turns into 1%.
        # Smoothing then flooring keeps the bound meaningful.
        floor_gain = OMLSA_FLOOR_ALPHA + (OMLSA_FLOOR_STRONG - OMLSA_FLOOR_ALPHA) * cfg.strength

        gains = np.zeros_like(spectra)
        prev_clean_power = noise_power.copy()
        for t in range(spectra.shape[0]):
            observation = np.maximum(spectra[t], 1e-20)
            gamma = observation / noise_power
            xi = alpha * (prev_clean_power / noise_power) + (1.0 - alpha) * np.maximum(gamma - 1.0, 0.0)
            xi = np.maximum(xi, 0.0)

            gain = (xi / (1.0 + xi)) ** cfg.strength
            gains[t] = np.clip(gain, 0.0, 1.0)
            prev_clean_power = gains[t] ** 2 * observation

        gains = _smooth_freq(gains, cfg.smooth_freq_bins)
        smooth_frames = round(cfg.smooth_time_s * sr / hop)
        gains = _smooth_time(gains, smooth_frames)

        # Scale the *attenuation* itself by the gate, not just the
        # over-subtraction. Over-subtraction only widens the range the Wiener
        # gain can reach; the gain is already suppressing the quietest bins on
        # clean speech, because those bins are below a noise estimate learned
        # from the speaker's own quietest moments. Measured with
        # over-subtraction alone off: 6.1% RMS change on clean speech at full
        # strength. Blending toward unity gain fixes that, and reaches unity
        # exactly when the gate is fully on.
        if gate > 0.0:
            gains = gains + (1.0 - gains) * gate
        gains = np.clip(np.maximum(gains, floor_gain if gate < 1.0 else 0.0), 0.0, 1.0)

        # Reconstruct by overlap-add.

        # The analysis window is applied on the way in, so the masked frames
        # are windowed twice by the time they come back out of the iFFT. That
        # is deliberate and it is what makes the synthesis correct: windowing
        # again gives the COLA property, and dividing by the summed squared
        # window restores unity gain where frames overlap.
        #
        # The reconstruction *amplitude* mask is sqrt(gain), not gain. `gains`
        # is a power ratio (it comes from a power spectrum), so applying it
        # directly would attenuate by its square. Getting this wrong is silent
        # at strength 0.5, where the error is only 3 dB of extra attenuation.
        windowed = frames * window
        amplitude_mask = np.sqrt(gains)
        masked = np.fft.irfft(amplitude_mask * np.fft.rfft(windowed, axis=1), n=frame_len, axis=1) * window

        n_out = total_frames * hop + frame_len
        overlap = np.zeros(n_out)
        norm = np.zeros(n_out)
        weights = window**2
        for t in range(masked.shape[0]):
            start = t * hop
            overlap[start : start + frame_len] += masked[t]
            norm[start : start + frame_len] += weights
        stitched = overlap / np.maximum(norm, 1e-9)
        # Drop the lead pad; the tail is already trimmed by the length slice.
        out_channels.append(stitched[lead : lead + mono.shape[0]].astype(np.float32))

    stacked = np.stack(out_channels, axis=1).astype(np.float32)
    processed = AudioBuffer(
        samples=stacked,
        sample_rate=sr,
        source_path=buffer.source_path,
        original_format=buffer.original_format,
    )

    notes.append(
        f"Spectral suppression at strength {cfg.strength:.2f} "
        f"({cfg.frame_ms:.0f} ms frames, {total_frames} frames analysed)"
    )
    if cfg.strength == 0.0:
        notes.append("Strength is 0; output is unchanged.")

    return EnhancementResult(
        audio=processed,
        backend="spectral-subtraction",
        strength=cfg.strength,
        processing_time_s=time.perf_counter() - started,
        notes=notes,
    )


def available_backends() -> tuple[str, ...]:
    """Backends that can run right now, without any download.

    The spectral suppressor is pure numpy/scipy, so it is always present. It is
    the fallback, not the ideal: a trained model can follow noise that changes
    over time, which this cannot.
    """
    return ("spectral-subtraction",)


def enhance(buffer: AudioBuffer, *, strength: float = STRENGTH_DEFAULT) -> EnhancementResult:
    """Run the best available enhancement backend."""
    return suppress_stationary_noise(buffer, strength=strength)


__all__ = [
    "SpectralConfig",
    "available_backends",
    "enhance",
    "suppress_stationary_noise",
]
