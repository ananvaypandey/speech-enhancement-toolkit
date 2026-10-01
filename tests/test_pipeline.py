"""Tests for the shared processing chain.

The chain exists so the command line and the browser page cannot disagree about
what happens to a file. These tests therefore assert on `process()` itself
rather than on either caller: whatever holds here holds for both.
"""

from __future__ import annotations

import numpy as np
import pytest

from aelf.analysis.levels import measure_levels
from aelf.config import CLIP_CEILING_DBFS
from aelf.pipeline import process
from aelf.types import AudioBuffer
from conftest import SR, sibilant_burst, silence, speech_like, speech_plus_noise, white_noise


def buf(samples: np.ndarray) -> AudioBuffer:
    return AudioBuffer(samples=samples, sample_rate=SR, original_format="wav")


class TestStages:
    def test_noise_suppression_is_always_reported(self) -> None:
        result = process(buf(speech_plus_noise(noise_sigma=0.05, speech_gain=0.3)), strength=0.5)
        assert any("noise suppression" in line for line in result.applied)

    def test_tidy_off_leaves_only_the_noise_stage(self) -> None:
        """The switch has to be a real switch, not a display preference."""
        result = process(buf(speech_plus_noise(noise_sigma=0.05, speech_gain=0.3)), strength=0.5, tidy=False)
        assert len(result.applied) == 1
        assert "noise suppression" in result.applied[0]

    def test_tidy_on_reports_the_filter_stages(self) -> None:
        result = process(buf(speech_like(2.0, SR, 120.0, 4)), strength=0.5, tidy=True)
        assert len(result.applied) > 1
        assert any("high-pass" in line for line in result.applied)

    def test_normalisation_is_opt_in(self) -> None:
        """Loudness must not change unless it was asked for."""
        source = speech_like(2.0, SR, 120.0, 4) * 0.05
        untouched = process(buf(source), strength=0.5, tidy=False).audio
        evened = process(buf(source), strength=0.5, tidy=False, target_lufs=-16.0).audio
        assert measure_levels(evened).lufs_integrated == pytest.approx(-16.0, abs=0.5)
        assert measure_levels(untouched).lufs_integrated != measure_levels(evened).lufs_integrated

    def test_every_stage_is_reported_in_order(self) -> None:
        """The report is a record of what ran, so order has to survive."""
        result = process(
            buf(speech_like(2.0, SR, 120.0, 4)),
            strength=0.75,
            tidy=True,
            target_lufs=-18.0,
        )
        assert "noise suppression" in result.applied[0]
        assert result.applied[-1].startswith("loudness evened out")
        assert result.summary == "; ".join(result.applied)


class TestGuarantees:
    def test_the_chain_cannot_clip(self) -> None:
        """Every stage after the limiter can add gain, so the ceiling must hold
        for the whole chain rather than only for the writer at the end."""
        loud = sibilant_burst(1.0, SR, amplitude=0.9) + speech_like(1.0, SR, 120.0, 3) * 0.8
        result = process(buf(loud), strength=1.0, tidy=True, target_lufs=-9.0)
        report = measure_levels(result.audio)
        assert not report.is_clipped
        assert float(np.max(np.abs(result.audio.samples))) <= 10 ** (CLIP_CEILING_DBFS / 20.0) + 1e-6

    def test_channel_count_and_sample_rate_survive(self) -> None:
        stereo = np.stack(
            [speech_like(1.5, SR, 120.0, 3) * 0.3, speech_like(1.5, SR, 190.0, 4) * 0.3],
            axis=1,
        )
        result = process(buf(stereo + white_noise(1.5, SR, 0.03, seed=6)[:, None]), strength=0.75, tidy=True)
        assert result.audio.channels == 2
        assert result.audio.sample_rate == SR
        assert result.audio.samples.shape == stereo.shape

    def test_length_is_unchanged(self) -> None:
        source = speech_plus_noise(seconds=2.0, noise_sigma=0.05, speech_gain=0.3)
        result = process(buf(source), strength=0.75, tidy=True, target_lufs=-16.0)
        assert result.audio.samples.shape[0] == source.shape[0]

    def test_silence_is_handled_rather_than_crashing(self) -> None:
        """An all-zero file has no floor and no loudness to target. It has to
        come back as silence with the failure explained, not as a crash."""
        result = process(buf(silence(1.0)), strength=0.5, tidy=True, target_lufs=-16.0)
        assert float(np.max(np.abs(result.audio.samples))) == 0.0

    def test_an_empty_buffer_is_refused_clearly(self) -> None:
        from aelf.errors import EnhancementError

        with pytest.raises(EnhancementError):
            process(AudioBuffer(samples=np.zeros((0, 1), dtype=np.float32), sample_rate=SR), strength=0.5)


class TestEffect:
    def test_a_noisy_recording_ends_up_quieter_in_the_gaps(self) -> None:
        """The point of the chain. Measured on the quietest frames, which is
        where background noise lives and speech does not."""
        source = speech_plus_noise(seconds=3.0, noise_sigma=0.05, speech_gain=0.3)
        result = process(buf(source), strength=0.75, tidy=False)
        quiet_before = np.percentile(np.abs(source[: int(SR * 0.15)]), 99)
        quiet_after = np.percentile(np.abs(result.audio.samples[: int(SR * 0.15)]), 99)
        assert quiet_after < quiet_before
