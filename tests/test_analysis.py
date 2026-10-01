"""Tests for level measurement and noise characterisation.

Assertions here check against analytically derivable values rather than
golden numbers, so they verify the maths is right rather than merely
self-consistent. For example a -20 dBFS sine must measure -20 dBFS RMS
after its DC-free window, and white noise at a known sigma must have a
noise floor near 20*log10(sigma).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from aelf.analysis.levels import (
    amplitude_to_db,
    crest_factor_db,
    db_to_amplitude,
    dc_offset,
    dynamic_range_db,
    integrated_lufs,
    measure_levels,
    peak,
    rms,
    speech_band_energy_ratio,
)
from aelf.analysis.noise_profile import estimate_noise_profile, noise_psd
from aelf.analysis.probe import inspect, probe_file
from aelf.types import AudioBuffer, Confidence
from conftest import (
    SR,
    clipped,
    pink_noise,
    silence,
    speech_like,
    speech_plus_noise,
    tone,
    two_speakers,
    white_noise,
    write_fixture,
)


def buffer_of(samples: np.ndarray, sr: int = SR) -> AudioBuffer:
    return AudioBuffer(samples=samples, sample_rate=sr, original_format="wav")


class TestAmplitudeMath:
    def test_dbfs_roundtrip(self) -> None:
        assert amplitude_to_db(1.0) == pytest.approx(0.0)
        assert amplitude_to_db(0.1) == pytest.approx(-20.0)
        assert amplitude_to_db(db_to_amplitude(-40.0)) == pytest.approx(-40.0)

    def test_silence_floors_instead_of_negative_infinity(self) -> None:
        assert np.isfinite(amplitude_to_db(0.0))
        assert amplitude_to_db(0.0) <= -190.0

    def test_rms_of_sine_matches_amplitude_over_sqrt2(self) -> None:
        amp = 0.5
        assert rms(tone(1000.0, 0.5, SR, amplitude=amp)) == pytest.approx(amp / np.sqrt(2.0), rel=1e-3)

    def test_rms_dbfs_of_known_tone(self) -> None:
        # -20 dBFS amplitude sine -> RMS at -23.01 dBFS.
        report = measure_levels(buffer_of(tone(1000.0, 0.5, SR, amplitude=0.1)), include_loudness=False)
        assert report.rms_dbfs == pytest.approx(-23.01, abs=0.1)
        assert report.peak_dbfs == pytest.approx(-20.0, abs=0.1)

    def test_crest_factor_of_sine_is_3db(self) -> None:
        assert crest_factor_db(tone(1000.0, 0.5, SR, amplitude=0.5)) == pytest.approx(3.01, abs=0.05)

    def test_peak_and_rms_handle_empty(self) -> None:
        empty = np.zeros(0, dtype=np.float32)
        assert peak(empty) == 0.0
        assert rms(empty) == 0.0
        assert dc_offset(empty) == 0.0


class TestLevelReport:
    def test_detects_clipped_source(self) -> None:
        report = measure_levels(buffer_of(clipped()), include_loudness=False)
        assert report.is_clipped
        assert report.clipped_sample_count > 0
        assert report.clipped_sample_fraction > 0.0
        assert any("clipped" in w.lower() for w in report.warnings)

    def test_silence_is_flagged_not_measured(self) -> None:
        report = measure_levels(buffer_of(silence()))
        assert report.rms_dbfs <= -190.0
        assert report.lufs_integrated is None
        assert any("silent" in w.lower() for w in report.warnings)

    def test_dc_offset_detected(self) -> None:
        signal = tone(440.0, 0.5, SR, 0.3) + 0.05
        report = measure_levels(buffer_of(signal), include_loudness=False)
        assert report.dc_offset == pytest.approx(0.05, abs=1e-3)
        assert any("DC offset" in w for w in report.warnings)

    def test_loudness_measures_conversational_level(self) -> None:
        """A -26 dBFS RMS signal should land in the plausible speech LUFS range."""
        report = measure_levels(buffer_of(speech_like() * 0.3))
        assert report.lufs_integrated is not None
        assert -40.0 < report.lufs_integrated < -10.0

    def test_loudness_unmeasurable_on_very_short_buffer(self) -> None:
        # Shorter than one 400 ms gating block.
        report = measure_levels(buffer_of(tone(1000.0, 0.1, SR, 0.3)))
        assert report.lufs_integrated is None

    def test_lufs_matches_itu_calibration_point(self) -> None:
        """BS.1770 defines a full-scale 1 kHz sine as -3.01 LUFS. This is the
        single most important calibration check in the whole module: the
        -0.691 offset must be applied exactly once."""
        buf = buffer_of(tone(1000.0, 10.0, SR, amplitude=1.0))
        lufs, _ = integrated_lufs(buf)
        assert lufs == pytest.approx(-3.01, abs=0.05)

    def test_lufs_is_offset_by_exactly_the_signal_level(self) -> None:
        """Halving the amplitude must lower LUFS by exactly 6.02 dB. Catches
        any error in the absolute reference level."""
        loud, _ = integrated_lufs(buffer_of(tone(1000.0, 10.0, SR, 1.0)))
        quiet, _ = integrated_lufs(buffer_of(tone(1000.0, 10.0, SR, 0.5)))
        assert loud - quiet == pytest.approx(6.0206, abs=0.01)

    def test_lufs_falls_as_level_drops(self) -> None:
        loud, _ = integrated_lufs(buffer_of(speech_like() * 0.5))
        quiet, _ = integrated_lufs(buffer_of(speech_like() * 0.05))
        assert loud > quiet
        assert loud - quiet == pytest.approx(20.0, abs=1.5)

    def test_stereo_uses_mono_downmix(self) -> None:
        left = tone(440.0, 0.5, SR, 0.4)
        right = tone(440.0, 0.5, SR, 0.4)
        stereo = buffer_of(np.stack([left, right], axis=1))
        assert measure_levels(stereo).rms_dbfs == pytest.approx(measure_levels(buffer_of(left)).rms_dbfs, abs=0.1)

    def test_speech_band_ratio_prefers_speech_like(self) -> None:
        speech_ratio = speech_band_energy_ratio(buffer_of(speech_like()))
        rumbly = speech_band_energy_ratio(buffer_of(tone(60.0, 1.0, SR, 0.5)))
        assert speech_ratio > rumbly

    def test_dynamic_range_exceeds_0_for_modulated_signal(self) -> None:
        assert dynamic_range_db(buffer_of(speech_like())) > 0.0

    def test_dynamic_range_of_constant_tone_is_near_zero(self) -> None:
        assert dynamic_range_db(buffer_of(tone(1000.0, 1.0, SR, 0.5))) < 0.5


class TestNoiseProfile:
    def test_estimates_floor_close_to_known_sigma(self) -> None:
        """A minimum-statistics estimator reads the *low* tail of the frame
        distribution, so it sits below the true mean of stationary noise. For
        Gaussian noise the p10 of the mean-square distribution is about 0.69x
        the mean, i.e. roughly -1.6 dB. Assert the estimator lands in that
        neighbourhood rather than expecting an exact match to the sigma."""
        sigma = 0.01  # -40 dBFS
        profile = estimate_noise_profile(buffer_of(white_noise(4.0, SR, sigma)))
        assert -44.0 < profile.noise_floor_dbfs < -38.0

    def test_floor_tracks_known_sigma(self) -> None:
        """The estimate must move with the true level, by the same amount."""
        quiet = estimate_noise_profile(buffer_of(white_noise(4.0, SR, 0.005)))
        loud = estimate_noise_profile(buffer_of(white_noise(4.0, SR, 0.05)))
        assert (loud.noise_floor_dbfs - quiet.noise_floor_dbfs) == pytest.approx(20.0, abs=2.0)

    def test_detects_stationary_white_noise(self) -> None:
        profile = estimate_noise_profile(buffer_of(white_noise(3.0, SR, 0.02)))
        assert profile.is_stationary
        assert profile.confidence in (Confidence.HIGH, Confidence.MEDIUM)

    def test_flags_non_stationary_noise_as_low_confidence(self) -> None:
        """Noise whose level changes over time is not stationary."""
        segments = [white_noise(1.0, SR, 0.001, seed=1), white_noise(1.0, SR, 0.3, seed=2), white_noise(1.0, SR, 0.001, seed=3)]
        signal = np.concatenate(segments)
        profile = estimate_noise_profile(buffer_of(signal))
        assert not profile.is_stationary

    def test_silence_yields_no_confidence(self) -> None:
        profile = estimate_noise_profile(buffer_of(silence()))
        assert profile.confidence == Confidence.NONE
        assert profile.noise_floor_dbfs == -np.inf

    def test_speech_plus_noise_gives_positive_snr(self) -> None:
        profile = estimate_noise_profile(buffer_of(speech_plus_noise(noise_sigma=0.005, speech_gain=0.3)))
        assert profile.estimated_snr_db > 0.0
        assert profile.frame_count > 0

    def test_snr_drops_as_noise_rises(self) -> None:
        clean = estimate_noise_profile(buffer_of(speech_plus_noise(noise_sigma=0.002, speech_gain=0.3))).estimated_snr_db
        noisy = estimate_noise_profile(buffer_of(speech_plus_noise(noise_sigma=0.08, speech_gain=0.3))).estimated_snr_db
        assert clean > noisy

    def test_detects_mains_hum(self) -> None:
        """A 50 Hz tone must be flagged as hum, not absorbed into a broadband
        floor. The exact reported frequency is limited by the 30 ms analysis
        window (bin spacing is sr/frame_len), so assert the estimate lands
        within one bin of the true 50 Hz rather than to sub-Hz precision."""
        t = np.arange(int(SR * 2.0)) / SR
        hum = 0.05 * np.sin(2.0 * np.pi * 50.0 * t)
        frame_len = int(SR * 0.030)
        bin_hz = SR / frame_len
        profile = estimate_noise_profile(buffer_of(hum.astype(np.float32) + white_noise(2.0, SR, 0.005, seed=3)))
        assert profile.estimated_hum_hz is not None
        assert abs(profile.estimated_hum_hz - 50.0) <= bin_hz
        assert any("hum" in n.lower() for n in profile.notes)

    def test_broadband_noise_is_not_called_hum(self) -> None:
        """Flat noise must not trip the narrowband detector."""
        profile = estimate_noise_profile(buffer_of(white_noise(3.0, SR, 0.02, seed=9)))
        assert profile.estimated_hum_hz is None

    def test_pink_noise_is_not_mislabelled_stationary_at_wrong_floor(self) -> None:
        profile = estimate_noise_profile(buffer_of(pink_noise(2.0, SR, 0.01)))
        assert np.isfinite(profile.noise_floor_dbfs)
        assert profile.confidence != Confidence.NONE

    def test_noise_psd_shape_matches_fft(self) -> None:
        buf = buffer_of(white_noise(1.0, SR, 0.01))
        psd = noise_psd(buf, estimate_noise_profile(buf))
        assert psd.ndim == 1
        assert psd.size > 0
        assert np.all(psd >= 0)

    def test_reports_method_for_provenance(self) -> None:
        profile = estimate_noise_profile(buffer_of(white_noise(1.0, SR, 0.01)))
        assert "minimum-statistics" in profile.method


class TestProbe:
    def test_probe_reads_header(self, tmp_path: Path) -> None:
        path = Path(write_fixture(tmp_path / "a.wav", tone(440.0, 0.5, SR, 0.3)))
        probe = probe_file(path)
        assert probe.exists
        assert probe.sample_rate == SR
        assert probe.channels == 1
        assert probe.duration_s == pytest.approx(0.5, abs=1e-3)
        assert probe.bit_depth == 16

    def test_probe_reports_missing_file(self, tmp_path: Path) -> None:
        probe = probe_file(tmp_path / "nope.wav")
        assert not probe.exists
        assert probe.warnings

    def test_probe_reports_zero_byte_file(self, tmp_path: Path) -> None:
        path = tmp_path / "empty.wav"
        path.write_bytes(b"")
        assert probe_file(path).warnings

    def test_stereo_layout_named(self, tmp_path: Path) -> None:
        path = Path(write_fixture(tmp_path / "s.wav", np.stack([tone(440.0, 0.2, SR, 0.3), tone(660.0, 0.2, SR, 0.2)], axis=1)))
        probe = probe_file(path)
        assert probe.channels == 2
        assert probe.channel_layout == "stereo"

    def test_cross_check_flags_rate_mismatch(self, tmp_path: Path) -> None:
        path = Path(write_fixture(tmp_path / "a.wav", tone(440.0, 0.2, 44_100, 0.3), sr=44_100))
        probe, _buffer = inspect(path)
        # Header said 44.1k, decode canonicalises to 48k. The mismatch must be
        # reported rather than silently accepted.
        assert probe.declared_sample_rate == 44_100
        assert probe.decoded_sample_rate == 48_000
        assert any("44100" in w or "44,100" in w for w in probe.warnings)

    def test_channel_layout_for_unusual_count(self) -> None:
        from aelf.analysis.probe import FileProbe

        probe = FileProbe(path="x", filename="x", exists=True, size_bytes=1, decoded_channels=6)
        assert probe.channel_layout == "6 channels"


class TestTwoSpeakerFixture:
    def test_fixture_is_a_sum_of_two_voices(self) -> None:
        data = two_speakers()
        assert data.shape[0] == int(SR * 2.0)
        # Two voices mean the low-frequency harmonic content is denser than a
        # single voice, so energy below 200 Hz is present and substantial.
        spectrum = np.abs(np.fft.rfft(data))
        freqs = np.fft.rfftfreq(len(data), 1.0 / SR)
        low = spectrum[freqs < 200.0]
        assert low.sum() > 0.0
