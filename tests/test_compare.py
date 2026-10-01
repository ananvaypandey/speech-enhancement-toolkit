"""Tests for reference-based comparison.

The behaviour under test is as much about refusing to answer as about
answering: the project's rule is that a metric which cannot be computed is
reported as unavailable, never approximated.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from aelf.errors import MetricUnavailableError
from aelf.evaluate.compare import (
    best_lag_samples,
    compare_to_reference,
    describe_missing_reference,
    sdr_db,
    sisdr_db,
)
from aelf.types import AudioBuffer
from conftest import SR, silence, speech_like, white_noise


def buffer_of(samples: np.ndarray, sr: int = SR) -> AudioBuffer:
    return AudioBuffer(samples=np.asarray(samples, dtype=np.float32)[:, None], sample_rate=sr)


class TestMetrics:
    def test_identical_signals_score_infinite(self) -> None:
        """A perfect match has no distortion. Reporting a large finite number
        here would hide a bug in the projection maths."""
        sig = speech_like(1.0, SR, 120.0, 4)
        assert np.isinf(sdr_db(sig, sig))
        assert np.isinf(sisdr_db(sig, sig))

    def test_pure_gain_is_invisible_to_sisdr_but_not_sdr(self) -> None:
        """The reason both metrics exist, and the observable difference between them.

        Doubling the level leaves the waveform shape intact. SI-SDR ignores that
        entirely and reports a perfect score, because it projects the scale out
        first. SDR counts the doubling as distortion: the error is exactly as
        large as the signal, so it lands on 0 dB. That 0 dB is the honest
        reading, and it is why a loudness mistake cannot hide behind the
        scale-invariant number.
        """
        sig = speech_like(1.0, SR, 120.0, 4)
        louder = sig * 2.0
        assert np.isinf(sisdr_db(louder, sig))
        assert sdr_db(louder, sig) == pytest.approx(0.0, abs=0.01)

    def test_added_noise_lowers_the_score(self) -> None:
        sig = speech_like(1.0, SR, 120.0, 4)
        noisy = sdr_db(sig + white_noise(1.0, SR, 0.02, seed=3), sig)
        assert noisy < 15.0

    def test_more_noise_scores_lower(self) -> None:
        sig = speech_like(1.0, SR, 120.0, 4)
        mild = sdr_db(sig + white_noise(1.0, SR, 0.005, seed=3), sig)
        harsh = sdr_db(sig + white_noise(1.0, SR, 0.05, seed=3), sig)
        assert harsh < mild

    def test_finds_known_lag(self) -> None:
        """Cross-correlation must recover a shift that was actually applied."""
        sig = speech_like(1.0, SR, 120.0, 4)
        shift = 200
        delayed = np.concatenate([np.zeros(shift), sig])[: len(sig)]
        assert abs(best_lag_samples(delayed, sig)) == shift


class TestRefusals:
    """Cases where producing a number would be misleading."""

    def test_rejects_mismatched_sample_rates(self) -> None:
        sig = speech_like(1.0, SR, 120.0, 4)
        with pytest.raises(MetricUnavailableError):
            compare_to_reference(buffer_of(sig), buffer_of(sig, sr=16_000))

    def test_rejects_large_duration_difference(self) -> None:
        """Trimming 200 ms off a reference does not make it a valid reference;
        it makes a comparison that would score the trim, not the audio."""
        ref = speech_like(2.0, SR, 120.0, 4)
        with pytest.raises(MetricUnavailableError) as excinfo:
            compare_to_reference(buffer_of(ref[: int(SR * 1.8)]), buffer_of(ref))
        assert "50 ms" in str(excinfo.value)

    def test_accepts_small_duration_difference(self) -> None:
        """Under the limit, a few samples of difference is resampling noise."""
        ref = speech_like(1.0, SR, 120.0, 4)
        report = compare_to_reference(buffer_of(ref[: len(ref) - 10]), buffer_of(ref))
        assert report.duration_drift_s < 0.05

    def test_rejects_silent_reference(self) -> None:
        sig = speech_like(1.0, SR, 120.0, 4)
        with pytest.raises(MetricUnavailableError):
            compare_to_reference(buffer_of(sig), buffer_of(silence(1.0)))

    def test_rejects_silent_output(self) -> None:
        """An output that lost everything must not be able to score as a match."""
        ref = speech_like(1.0, SR, 120.0, 4)
        with pytest.raises(MetricUnavailableError):
            compare_to_reference(buffer_of(silence(1.0)), buffer_of(ref))

    def test_reports_missing_reference_without_a_number(self) -> None:
        """The text shown in place of a score. It must not contain a figure
        that could be mistaken for one."""
        text = describe_missing_reference()
        assert "No clean reference" in text
        assert "noise floor" in text


class TestReport:
    def test_reports_confidence_and_verdict(self) -> None:
        ref = speech_like(2.0, SR, 120.0, 4)
        report = compare_to_reference(buffer_of(ref + white_noise(2.0, SR, 0.001, seed=3)), buffer_of(ref))
        assert report.confidence.label in ("Verified", "Plausible", "Uncertain")
        assert report.verdict

    def test_flags_a_perfect_score_as_suspicious(self) -> None:
        """Real enhancement never reproduces the reference exactly. A perfect
        score usually means the reference was compared against itself."""
        ref = speech_like(2.0, SR, 120.0, 4)
        report = compare_to_reference(buffer_of(ref), buffer_of(ref))
        assert report.confidence.value == "low"
        assert "exactly" in report.verdict

    def test_notes_alignment_offset(self) -> None:
        """A large offset is reported, not silently corrected. Hiding it would
        conceal exactly the defect the user needs to know about."""
        ref = speech_like(2.0, SR, 120.0, 4)
        shift = int(SR * 0.03)
        shifted = np.concatenate([np.zeros(shift), ref])[: len(ref)]
        report = compare_to_reference(buffer_of(shifted), buffer_of(ref))
        assert report.confidence.value == "low"
        assert any("alignment" in n.lower() for n in report.notes)

    def test_is_valid_only_with_a_reference(self) -> None:
        """`is_valid` exists to make the absent-reference case unrepresentable
        as a report. A ComparisonReport implies a reference was supplied."""
        ref = speech_like(1.0, SR, 120.0, 4)
        assert compare_to_reference(buffer_of(ref), buffer_of(ref)).is_valid is True
