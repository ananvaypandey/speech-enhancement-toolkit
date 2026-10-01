"""Time-domain filtering: high-pass, speech-band EQ, de-essing, DC block.

All filters are linear-phase (FIR where quality matters, and a cascaded
biquad for the shelving EQ) and applied with `sosfilt` so the whole cascade
runs in one numerically stable pass. Filter order is deliberate: DC removal
first, then shelving, then the de-esser, because each stage assumes the
previous one has already removed out-of-band content.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import butter, sosfiltfilt, tf2sos

from ..analysis.levels import amplitude_to_db
from ..config import (
    CLIP_CEILING_DBFS,
    DEESSER_MAX_HZ,
    DEESSER_MIN_HZ,
    HIGHPASS_DEFAULT_HZ,
    HIGHPASS_MIN_HZ,
    SPEECH_BAND_HZ,
)
from ..errors import EnhancementError
from ..io.encode import true_peak_limit
from ..types import AudioBuffer, FloatArray

FILTER_ORDER = 4
# Order of magnitude below the passband that counts as "stopband reached".
_EPS = 1e-9


@dataclass(frozen=True)
class FilterResult:
    audio: AudioBuffer
    applied: list[str]
    notes: list[str]

    @property
    def changed(self) -> bool:
        return bool(self.applied)


def highpass(
    samples: FloatArray,
    sr: int,
    cutoff_hz: float = HIGHPASS_DEFAULT_HZ,
    order: int = FILTER_ORDER,
) -> tuple[FloatArray, bool]:
    """Remove rumble and DC below `cutoff_hz`.

    Returns (filtered, applied). Filters below ~20 Hz are skipped: at 48 kHz
    they sit below the noise floor of any real recording and only cost
    processing time.
    """
    if cutoff_hz <= 0 or cutoff_hz >= sr / 2:
        return samples, False
    if cutoff_hz < 20.0:
        return samples, False

    nyquist = sr / 2.0
    sos = butter(order, cutoff_hz / nyquist, btype="highpass", output="sos")
    # filtfilt doubles the effective order and gives zero phase distortion,
    # which matters because we are not allowed to shift speech formants.
    filtered = sosfiltfilt(sos, samples.astype(np.float64), axis=0).astype(np.float32)
    return filtered, True


def lowpass_shelf(samples: FloatArray, sr: int, cutoff_hz: float = SPEECH_BAND_HZ[1]) -> tuple[FloatArray, bool]:
    """Taper content above the speech band, where post-enhancement hiss lives."""
    nyquist = sr / 2.0
    if cutoff_hz <= 0 or cutoff_hz >= nyquist * 0.98:
        return samples, False
    sos = butter(FILTER_ORDER, cutoff_hz / nyquist, btype="lowpass", output="sos")
    return sosfiltfilt(sos, samples.astype(np.float64), axis=0).astype(np.float32), True


def peaking_eq(
    samples: FloatArray,
    sr: int,
    center_hz: float,
    gain_db: float,
    q: float = 1.0,
) -> FloatArray:
    """RBJ peaking (bell) EQ - the standard building block for tone shaping.

    `gain_db` is the gain the caller actually gets. The filter is applied with
    `sosfiltfilt` to keep it zero-phase, which runs the design twice and
    therefore squares its magnitude response: a bell designed for +6 dB would
    come out at +12 dB. Each pass is designed at half the requested gain so
    the pair lands on the requested value.
    """
    if gain_db == 0.0:
        return samples

    a_ = 10.0 ** (gain_db / 80.0)  # half the requested gain, because filtfilt doubles it
    w0 = 2.0 * np.pi * center_hz / sr
    alpha = np.sin(w0) / (2.0 * q)

    b0 = 1.0 + alpha * a_
    b1 = -2.0 * np.cos(w0)
    b2 = 1.0 - alpha * a_
    a0 = 1.0 + alpha / a_
    a1 = -2.0 * np.cos(w0)
    a2 = 1.0 - alpha / a_

    sos = tf2sos([b0, b1, b2], [a0, a1, a2])
    return sosfiltfilt(sos, samples.astype(np.float64), axis=0).astype(np.float32)


def speech_band_eq(
    samples: FloatArray,
    sr: int,
    *,
    presence_boost_db: float = 2.0,
    mud_cut_db: float = -2.0,
    enable: bool = True,
) -> tuple[FloatArray, list[str]]:
    """Gentle intelligibility shaping.

    A small cut through the 200-500 Hz muddiness region and a small lift
    across the 2-4 kHz presence band. Gains are intentionally modest: this
    stage runs after denoising and should not undo the work of the stage
    before it.
    """
    if not enable or (presence_boost_db == 0.0 and mud_cut_db == 0.0):
        return samples, []

    applied: list[str] = []
    out = samples.astype(np.float32)
    if mud_cut_db != 0.0:
        out = peaking_eq(out, sr, center_hz=320.0, gain_db=mud_cut_db, q=0.9)
        applied.append(f"-{abs(mud_cut_db):.1f} dB @ 320 Hz (mud reduction)")
    if presence_boost_db != 0.0:
        out = peaking_eq(out, sr, center_hz=2_800.0, gain_db=presence_boost_db, q=0.8)
        applied.append(f"+{presence_boost_db:.1f} dB @ 2.8 kHz (presence/intelligibility)")
        # A +6 dB shelf doubles amplitude, so a source already near full scale
        # would clip here. Scale back to the ceiling rather than waiting for a
        # downstream stage, so each stage is safe on its own.
        if out.size:
            limited = true_peak_limit(out, CLIP_CEILING_DBFS)
            if not np.array_equal(limited, out):
                out = limited
                applied.append(f"gain scaled to hold {CLIP_CEILING_DBFS:.1f} dBTP")
    return out, applied


def deess(
    samples: FloatArray,
    sr: int,
    sensitivity_db: float = 6.0,
    amount_db: float = -6.0,
    smooth_seconds: float = 0.03,
    presence_gate_db: float = -30.0,
    enable: bool = True,
) -> tuple[FloatArray, list[str]]:
    """Dynamic de-esser operating only on the sibilance band.

    Isolates 5.5-9 kHz energy with a bandpass, derives a control curve from
    that band's short-term envelope relative to its own long-term median, and
    subtracts the attenuated portion from the full-band signal. Sibilance reads
    as harsh after enhancement because the suppressor leaves broadband content
    the ear attributes to the voice rather than to the noise floor.

    The trigger is referenced to the band's own median level, never to
    absolute dBFS. Two earlier attempts used an absolute threshold and both
    were wrong: a band-passed 5.5-9 kHz signal sits 20-30 dB below full scale
    even on loud material, so an absolute trigger never fires. Keying to the
    band's median makes the stage behave identically regardless of input
    gain, which is what a de-esser has to do.
    """
    if not enable or amount_db == 0.0:
        return samples, []

    lo = DEESSER_MIN_HZ / (sr / 2.0)
    hi = DEESSER_MAX_HZ / (sr / 2.0)
    if hi >= 0.99:
        hi = 0.99
    if lo <= 0 or lo >= hi:
        return samples, []

    data = np.asarray(samples, dtype=np.float64)
    if data.size == 0 or not np.any(data):
        return samples, []

    sos = butter(2, [lo, hi], btype="bandpass", output="sos")
    band = sosfiltfilt(sos, data, axis=0)

    env_len = max(int(sr * 0.005), 1)
    kernel = np.ones(env_len) / env_len

    def _envelope(x: FloatArray) -> np.ndarray:
        mean_square = np.apply_along_axis(lambda row: np.convolve(row, kernel, mode="same"), 0, x)
        # The convolution is of the squared signal, so it must be square-rooted
        # to become an amplitude. Comparing the mean-square value against a
        # linear trigger leaves the control curve permanently at zero and
        # silently disables the stage.
        return np.sqrt(np.maximum(mean_square, 0.0))

    env = np.maximum(_envelope(band), 1e-12)
    env_db = 20.0 * np.log10(env)

    full_db = 20.0 * np.log10(max(float(np.max(_envelope(data))), 1e-12))

    # Presence gate: if the band holds essentially nothing relative to the
    # full-band level, what we are measuring is filter leakage, not sibilance.
    # Without this the stage engages on any low-frequency material.
    band_present_db = float(np.max(env_db)) - full_db
    if band_present_db < presence_gate_db:
        return samples, []

    median_db = float(np.median(env_db))
    over_db = np.maximum(env_db - median_db - sensitivity_db, 0.0)
    if not np.any(over_db > 0.0):
        return samples, []

    # Normalise by the largest excursion above the trigger so the full
    # requested depth is reached exactly at the loudest sibilant moment.
    span = max(float(np.max(over_db)), 1e-6)
    ramp = np.clip(over_db / span, 0.0, 1.0)
    ramp = ramp * ramp * (3.0 - 2.0 * ramp)  # C1-continuous

    # Smooth the control curve so the gain does not pump on individual frames.
    smooth_len = max(int(sr * smooth_seconds), 1)
    ramp = np.apply_along_axis(
        lambda row: np.convolve(row, np.ones(smooth_len) / smooth_len, mode="same"),
        0,
        ramp,
    )

    gain = np.clip(10.0 ** (amount_db * ramp / 20.0), 10.0 ** (amount_db / 20.0), 1.0)
    removed = band * (1.0 - gain)
    out = (data - removed).astype(np.float32)

    peak_reduction = amount_db * float(np.max(ramp))
    notes = [
        f"de-esser engaged (band present at {band_present_db:.1f} dB relative to programme)",
        f"up to {peak_reduction:.1f} dB reduction in "
        f"{DEESSER_MIN_HZ / 1000:.1f}-{DEESSER_MAX_HZ / 1000:.1f} kHz",
    ]
    return out, notes


def apply_filter_chain(
    buffer: AudioBuffer,
    *,
    highpass_hz: float = HIGHPASS_DEFAULT_HZ,
    lowpass_hz: float = SPEECH_BAND_HZ[1],
    presence_boost_db: float = 2.0,
    mud_cut_db: float = -2.0,
    deess_amount_db: float = -6.0,
    dc_block: bool = True,
    limit_peaks: bool = True,
) -> FilterResult:
    """Run the full filter chain over every channel.

    Filters can add gain (a +6 dB EQ shelf multiplies amplitude by 2), so the
    chain ends with the true-peak limiter. Without it a boosting configuration
    would clip on write, which is the one thing post-processing must not do.
    """
    if buffer.frames == 0:
        raise EnhancementError("Cannot filter an empty buffer", user_message="The audio contains no samples.")

    applied: list[str] = []
    notes: list[str] = []
    out = buffer.samples.astype(np.float32)

    if dc_block:
        before = float(np.mean(out))
        if abs(before) > 1e-4:
            out = out - before
            applied.append(f"DC offset removed ({before:+.5f})")
        else:
            notes.append("DC offset below 1e-4; no blocking needed.")

    effective_highpass = max(highpass_hz, HIGHPASS_MIN_HZ if highpass_hz > 0 else 0.0)
    filtered = np.empty_like(out)
    hp_applied = False
    for ch in range(out.shape[1]):
        filtered[:, ch], hp_applied = highpass(out[:, ch], buffer.sample_rate, effective_highpass)
    if hp_applied:
        applied.append(f"high-pass {effective_highpass:.0f} Hz (order {FILTER_ORDER * 2}, zero-phase)")
    else:
        notes.append(f"High-pass at {effective_highpass:.0f} Hz skipped as inaudible for this rate.")
    out = filtered

    filtered = np.empty_like(out)
    lp_applied = False
    for ch in range(out.shape[1]):
        filtered[:, ch], lp_applied = lowpass_shelf(out[:, ch], buffer.sample_rate, lowpass_hz)
    if lp_applied:
        applied.append(f"low-pass {lowpass_hz:.0f} Hz")
    out = filtered

    filtered = np.empty_like(out)
    eq_notes: list[str] = []
    for ch in range(out.shape[1]):
        filtered[:, ch], eq_notes = speech_band_eq(
            out[:, ch],
            buffer.sample_rate,
            presence_boost_db=presence_boost_db,
            mud_cut_db=mud_cut_db,
        )
    applied.extend(eq_notes)
    out = filtered

    filtered = np.empty_like(out)
    deess_notes: list[str] = []
    for ch in range(out.shape[1]):
        filtered[:, ch], deess_notes = deess(out[:, ch], buffer.sample_rate, amount_db=deess_amount_db)
    applied.extend(deess_notes)
    out = filtered

    if limit_peaks:
        before_peak = float(np.max(np.abs(out))) if out.size else 0.0
        out = true_peak_limit(out, CLIP_CEILING_DBFS)
        after_peak = float(np.max(np.abs(out))) if out.size else 0.0
        if before_peak > after_peak * 1.001:
            applied.append(f"true-peak limit to {CLIP_CEILING_DBFS:.1f} dBTP (from {amplitude_to_db(before_peak):.1f} dBFS)")

    audio = AudioBuffer(
        samples=out.astype(np.float32),
        sample_rate=buffer.sample_rate,
        source_path=buffer.source_path,
        original_format=buffer.original_format,
    )
    return FilterResult(audio=audio, applied=applied, notes=notes)
