"""Objective comparison against a clean reference.

Only usable when a clean recording of the same speech exists. Everything here
is reference-dependent, and the project refuses to report these numbers
otherwise: without a reference there is no way to know whether enhancement
improved anything or just made the file different.

Two metrics, because each fails differently:

- **SDR** compares the output to the reference directly. It is sensitive to
  any gain or timing error, which makes it a good honesty check but a harsh
  score.
- **SI-SDR** first removes the best possible scaling between the two, so it
  ignores a volume difference. It answers "did the underlying waveform survive?"
  without being confused by loudness.

Both require the two signals to be aligned. A misalignment shows up as a poor
score in both, which is why alignment is checked before either is computed.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..config import REF_MAX_DURATION_DRIFT_S, REF_MAX_SAMPLE_DRIFT
from ..errors import MetricUnavailableError
from ..types import AudioBuffer, Confidence, FloatArray


@dataclass(frozen=True)
class ComparisonReport:
    """Reference-based scores. Only valid when a reference was supplied."""

    sdr_db: float
    sisdr_db: float
    sample_drift: int
    duration_drift_s: float
    confidence: Confidence
    verdict: str
    notes: list[str]

    @property
    def is_valid(self) -> bool:
        return True


def _project(signal: FloatArray, reference: FloatArray) -> tuple[float, float]:
    """Least-squares scale factor and residual energy for one projection."""
    denom = float(np.dot(reference, reference))
    if denom <= 1e-20:
        return 0.0, float(np.dot(signal, signal))
    scale = float(np.dot(signal, reference)) / denom
    residual = signal - scale * reference
    return scale, float(np.dot(residual, residual))


def sdr_db(enhanced: FloatArray, reference: FloatArray) -> float:
    """Signal-to-distortion ratio, in dB. Scale-sensitive.

    Distortion is measured as `enhanced - reference` directly, with no scaling.
    A gain difference therefore counts as distortion, which is what makes this
    the honest check: if the output is the right waveform at the wrong volume,
    SDR will say so.
    """
    e = np.asarray(enhanced, dtype=np.float64)
    r = np.asarray(reference, dtype=np.float64)
    signal_power = float(np.dot(r, r))
    noise_power = float(np.dot(e - r, e - r))
    if noise_power <= 1e-20:
        return float("inf")
    return 10.0 * np.log10(max(signal_power / noise_power, 1e-20))


def sisdr_db(enhanced: FloatArray, reference: FloatArray) -> float:
    """Scale-invariant SDR, in dB. Ignores an overall gain difference.

    Projects onto the reference first and measures the residual, so the same
    waveform at a different level scores the same as the original. That makes
    it the right metric for "did the speech survive", and the wrong one for
    "is this file correctly levelled", which is what SDR covers.
    """
    ref = np.asarray(reference, dtype=np.float64)
    scale, residual_power = _project(np.asarray(enhanced, dtype=np.float64), ref)
    signal_power = float(np.dot(scale * ref, scale * ref))
    if residual_power <= 1e-20:
        return float("inf")
    return 10.0 * np.log10(max(signal_power / residual_power, 1e-20))


def best_lag_samples(enhanced: FloatArray, reference: FloatArray, max_lag: int = 4 * REF_MAX_SAMPLE_DRIFT) -> int:
    """Lag that best aligns the reference to the output, by cross-correlation.

    Searched over a range wider than the tolerance the caller warns about, so
    that an offset past the tolerance is still *measured* rather than clamped
    into looking acceptable.

    Reported but never silently applied. Correcting an offset hides the very
    problem the user needs to know about, so the caller is expected to show it.
    """
    a = np.asarray(enhanced, dtype=np.float64) - np.mean(enhanced)
    b = np.asarray(reference, dtype=np.float64) - np.mean(reference)
    corr = np.correlate(a, b, mode="full")
    lag = int(np.argmax(corr)) - (len(b) - 1)
    return int(np.clip(lag, -max_lag, max_lag))


def compare_to_reference(enhanced: AudioBuffer, reference: AudioBuffer) -> ComparisonReport:
    """Score an enhanced file against a clean reference.

    Raises rather than returning a number when the two cannot be compared. A
    score computed on mismatched audio is worse than no score, because it looks
    like an answer.
    """
    if enhanced.sample_rate != reference.sample_rate:
        raise MetricUnavailableError(
            f"Sample rates differ: {enhanced.sample_rate} Hz vs {reference.sample_rate} Hz",
            user_message="These recordings have different sample rates, so they cannot be compared directly.",
        )

    drift_s = abs(enhanced.duration_s - reference.duration_s)
    if drift_s > REF_MAX_DURATION_DRIFT_S:
        raise MetricUnavailableError(
            f"Duration differs by {drift_s * 1000:.0f} ms (limit {REF_MAX_DURATION_DRIFT_S * 1000:.0f} ms)",
            user_message=(
                "These recordings differ in length by more than 50 ms, so no quality score is reported. "
                "Beyond that point a score would measure the length mismatch rather than the audio."
            ),
        )

    n = min(enhanced.frames, reference.frames)
    e = enhanced.mono()[:n].astype(np.float64)
    r = reference.mono()[:n].astype(np.float64)

    if not np.any(r):
        raise MetricUnavailableError(
            "Reference is silent",
            user_message="The reference recording contains no audio, so there is nothing to compare against.",
        )
    if not np.any(e):
        raise MetricUnavailableError(
            "Enhanced audio is silent",
            user_message="The processed file contains no audio at all.",
        )

    drift_samples = enhanced.frames - reference.frames
    lag = best_lag_samples(e, r)
    notes: list[str] = []

    if abs(drift_samples) > REF_MAX_SAMPLE_DRIFT:
        notes.append(f"Output is {drift_samples} samples longer or shorter than the reference.")
    if abs(lag) > REF_MAX_SAMPLE_DRIFT:
        notes.append(
            f"Best alignment is {lag} samples ({lag / enhanced.sample_rate * 1000:.1f} ms) off. "
            "Scores below reflect that offset and are correspondingly pessimistic."
        )

    sdr = sdr_db(e, r)
    sisdr = sisdr_db(e, r)

    if not np.isfinite(sdr):
        verdict = "The output matches the reference exactly. That is worth double-checking: real enhancement never produces this."
        confidence = Confidence.LOW
    elif sdr >= 20.0:
        verdict = "Strong. Most of the original waveform survived."
        confidence = Confidence.HIGH
    elif sdr >= 10.0:
        verdict = "Good. Some change is audible, which is expected from denoising."
        confidence = Confidence.HIGH
    elif sdr >= 3.0:
        verdict = "Moderate. Substantial change to the waveform; listen before trusting it."
        confidence = Confidence.MEDIUM
    else:
        verdict = "Weak. The waveform bears little resemblance to the reference; check for over-processing."
        confidence = Confidence.LOW

    if abs(lag) > REF_MAX_SAMPLE_DRIFT:
        confidence = Confidence.LOW

    return ComparisonReport(
        sdr_db=sdr,
        sisdr_db=sisdr,
        sample_drift=drift_samples,
        duration_drift_s=drift_s,
        confidence=confidence,
        verdict=verdict,
        notes=notes,
    )


def describe_missing_reference() -> str:
    """What to show when no reference exists, instead of a score."""
    return (
        "No clean reference was supplied, so no quality score is reported.\n\n"
        "A reference-based score needs a clean recording of the same speech, captured the same "
        "way. Without one, there is no way to distinguish an improvement from a change that only "
        "sounds different.\n\n"
        "What can be reported without a reference:\n"
        "  - measured noise floor, before and after\n"
        "  - loudness and peak levels\n"
        "  - count of samples that were already distorting\n\n"
        "Those are measurements of the audio. They are not a judgement of how good it sounds."
    )
