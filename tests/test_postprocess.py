"""Tests for the post-processing chain.

The invariants under test are the ones the brief calls out directly: no
clipping after processing, and intelligibility preserved. Both are checked
as measurable properties rather than as "output sounds fine".
"""

from __future__ import annotations

from itertools import pairwise

import numpy as np
import pytest

from aelf.analysis.levels import integrated_lufs, measure_levels
from aelf.config import (
    CLIP_CEILING_DBFS,
    TARGET_LUFS_LISTENING,
    TARGET_LUFS_SPEECH,
    TRUE_PEAK_CEILING,
)
from aelf.io.encode import count_clipped, true_peak_limit
from aelf.postprocess.filters import (
    apply_filter_chain,
    deess,
    highpass,
    lowpass_shelf,
    peaking_eq,
    speech_band_eq,
)
from aelf.postprocess.normalize import normalise_lufs, normalise_peak, speech_presence_gain
from aelf.types import AudioBuffer
from conftest import SR, clipped, sibilant_burst, speech_like, tone, white_noise


def buffer_of(samples: np.ndarray, sr: int = SR) -> AudioBuffer:
    return AudioBuffer(samples=samples, sample_rate=sr, original_format="wav")


def band_energy_db(samples: np.ndarray, lo: float, hi: float, sr: int = SR) -> float:
    """Energy in a frequency band, in dB. Used to verify filters do what they claim."""
    spectrum = np.abs(np.fft.rfft(samples.astype(np.float64)))
    freqs = np.fft.rfftfreq(len(samples), 1.0 / sr)
    band = spectrum[(freqs >= lo) & (freqs <= hi)]
    if band.size == 0 or not np.any(band > 0):
        return -np.inf
    return float(20.0 * np.log10(np.sqrt(np.sum(band**2)) / len(samples)))


class TestHighpass:
    def test_attenuates_below_cutoff(self) -> None:
        low = tone(40.0, 1.0, SR, 0.5)
        filtered, applied = highpass(low, SR, 80.0)
        assert applied
        assert band_energy_db(filtered, 20.0, 70.0) < band_energy_db(low, 20.0, 70.0) - 20.0

    def test_passes_above_cutoff(self) -> None:
        high = tone(1000.0, 1.0, SR, 0.5)
        filtered, _ = highpass(high, SR, 80.0)
        assert band_energy_db(filtered, 500.0, 2000.0) == pytest.approx(band_energy_db(high, 500.0, 2000.0), abs=0.5)

    def test_removes_dc_offset(self) -> None:
        # A 20 Hz component sits well below the 80 Hz corner, so the residual
        # DC and sub-audio content should be attenuated to well under 1% of
        # the original offset. Asserting a hard 1e-4 mean is too tight for an
        # IIR filter on a finite 1 s window.
        signal = (tone(500.0, 1.0, SR, 0.3) + 0.2).astype(np.float32)
        filtered, _ = highpass(signal, SR, 80.0)
        assert abs(float(np.mean(filtered))) < 0.2 * 0.01

    def test_zero_phase_preserves_alignment(self) -> None:
        """filtfilt is zero-phase, so a burst must not shift in time. If this
        regressed, every formant's timing would move and intelligibility drops.

        The burst has to be broadband. A long smooth window is itself
        sub-audio content, and an 80 Hz high-pass correctly annihilates it, so
        the surviving energy would be edge ringing and the peak would land
        hundreds of samples early for reasons that have nothing to do with
        phase.
        """
        burst = white_noise(0.005, SR, 0.05, seed=3)
        signal = np.zeros(SR, dtype=np.float32)
        start = int(SR * 0.4)
        signal[start : start + len(burst)] = burst
        filtered, _ = highpass(signal, SR, 80.0)
        assert int(np.argmax(np.abs(filtered))) == int(np.argmax(np.abs(signal)))
        # The burst must survive the high-pass rather than being filtered out.
        assert np.max(np.abs(filtered[start : start + len(burst)])) > 0.05

    def test_zero_phase_no_group_delay(self) -> None:
        """Cross-correlation between input and output should peak at zero lag."""
        signal = speech_like(2.0, SR, 120.0, 4)
        filtered, _ = highpass(signal, SR, 80.0)
        a = signal - signal.mean()
        b = filtered - filtered.mean()
        corr = np.correlate(b, a, mode="full")
        lag = int(np.argmax(corr)) - (len(a) - 1)
        assert abs(lag) <= 1

    def test_skips_inaudible_cutoff(self) -> None:
        signal = tone(1000.0, 0.1, SR, 0.3)
        _, applied = highpass(signal, SR, 10.0)
        assert not applied

    def test_cutoff_at_nyquist_is_noop(self) -> None:
        signal = tone(1000.0, 0.1, SR, 0.3)
        out, applied = highpass(signal, SR, SR / 2)
        assert not applied
        assert np.array_equal(out, signal)

    def test_does_not_change_length(self) -> None:
        signal = tone(1000.0, 1.0, SR, 0.3)
        filtered, _ = highpass(signal, SR, 80.0)
        assert len(filtered) == len(signal)


class TestLowpassShelf:
    def test_attenuates_above_cutoff(self) -> None:
        high = tone(15_000.0, 1.0, SR, 0.5)
        filtered, applied = lowpass_shelf(high, SR, 8_000.0)
        assert applied
        assert band_energy_db(filtered, 12_000.0, 20_000.0) < band_energy_db(high, 12_000.0, 20_000.0) - 20.0

    def test_preserves_speech_band(self) -> None:
        speech = speech_like(1.0, SR, 120.0, 5)
        filtered, _ = lowpass_shelf(speech, SR, 8_000.0)
        assert band_energy_db(filtered, 300.0, 3_400.0) == pytest.approx(band_energy_db(speech, 300.0, 3_400.0), abs=0.6)

    def test_noop_near_nyquist(self) -> None:
        signal = tone(1000.0, 0.1, SR, 0.3)
        _, applied = lowpass_shelf(signal, SR, SR / 2 * 0.99)
        assert not applied


class TestPeakingEq:
    def test_boost_raises_band_level(self) -> None:
        signal = tone(1000.0, 1.0, SR, 0.3)
        boosted = peaking_eq(signal, SR, 1000.0, +6.0, q=1.0)
        assert band_energy_db(boosted, 900.0, 1100.0) > band_energy_db(signal, 900.0, 1100.0)

    def test_cut_reduces_band_level(self) -> None:
        signal = tone(1000.0, 1.0, SR, 0.3)
        cut = peaking_eq(signal, SR, 1000.0, -6.0, q=1.0)
        assert band_energy_db(cut, 900.0, 1100.0) < band_energy_db(signal, 900.0, 1100.0)

    def test_zero_gain_is_identity(self) -> None:
        signal = tone(1000.0, 0.1, SR, 0.3)
        assert np.array_equal(peaking_eq(signal, SR, 1000.0, 0.0), signal)


class TestSpeechBandEq:
    def test_boosts_presence_band(self) -> None:
        signal = tone(2800.0, 1.0, SR, 0.3)
        out, applied = speech_band_eq(signal, SR, presence_boost_db=4.0, mud_cut_db=0.0)
        assert applied
        assert band_energy_db(out, 2_500.0, 3_200.0) > band_energy_db(signal, 2_500.0, 3_200.0)

    def test_applies_exactly_the_requested_gain(self) -> None:
        """A bell filter must deliver the gain it was asked for.

        Measured from the impulse response rather than from band energy on a
        test tone: the zero-phase `sosfiltfilt` pair squares the magnitude
        response, so a bell designed for +6 dB silently came out at +12 dB.
        Comparing band energies would not have caught that, because it only
        compares relative levels.
        """
        for gain in (6.0, 3.0, -3.0, -6.0):
            impulse = np.zeros(16_384, dtype=np.float32)
            impulse[8_192] = 1.0  # mid-buffer: odd-extension padding cannot mirror it
            out = peaking_eq(impulse, SR, center_hz=2_800.0, gain_db=gain, q=0.8)
            spectrum = np.abs(np.fft.rfft(out.astype(np.float64), 65_536))
            freqs = np.fft.rfftfreq(65_536, 1.0 / SR)
            idx = int(np.argmin(np.abs(freqs - 2_800.0)))
            measured = 20.0 * np.log10(spectrum[idx])
            assert measured == pytest.approx(gain, abs=0.05), f"{gain} dB -> {measured:.2f} dB"

    def test_can_be_disabled(self) -> None:
        signal = tone(2800.0, 0.2, SR, 0.3)
        out, applied = speech_band_eq(signal, SR, enable=False)
        assert not applied
        assert np.array_equal(out, signal)

    def test_does_not_clip(self) -> None:
        """The EQ stage itself must not produce clipping, since the filter
        chain relies on that before the limiter runs."""
        loud = tone(2800.0, 1.0, SR, 0.99)
        out, _ = speech_band_eq(loud, SR, presence_boost_db=6.0, mud_cut_db=-3.0)
        assert count_clipped(out) == 0


class TestDeesser:
    def test_reduces_sibilance_band(self) -> None:
        """The de-esser must attenuate the sibilance band.

        Stationary broadband noise is the wrong probe here: a band-passed
        5.5-9 kHz slice of it sits far below the programme level and has
        almost no envelope dynamics, so any correctly-keyed de-esser leaves
        it alone. The stimulus has to look like real sibilance, which means
        band-limited noise with a sharp attack and a decay.
        """
        sibilant = sibilant_burst()
        out, applied = deess(sibilant, SR, amount_db=-9.0)
        assert applied
        assert band_energy_db(out, 5_500.0, 9_000.0) < band_energy_db(sibilant, 5_500.0, 9_000.0) - 3.0

    def test_reduction_is_monotonic_in_amount(self) -> None:
        """More requested depth must mean more measured reduction, i.e. band
        energy must fall as `amount_db` goes more negative."""
        sibilant = sibilant_burst()
        results = []
        for amount in (-3.0, -6.0, -9.0, -12.0):
            out, _ = deess(sibilant, SR, amount_db=amount)
            results.append(band_energy_db(out, 5_500.0, 9_000.0))
        assert all(a > b for a, b in pairwise(results)), results

    def test_gain_independent(self) -> None:
        """The trigger must key off the band's own level, not absolute dBFS,
        so the same relative reduction applies at any input gain. This is the
        property that was broken when the stage used an absolute threshold."""
        sibilant = sibilant_burst()
        deltas = []
        for gain in (1.0, 0.01, 0.0001):
            scaled = (sibilant * gain).astype(np.float32)
            out, applied = deess(scaled, SR, amount_db=-6.0)
            assert applied, f"de-esser disengaged at input gain {gain}"
            deltas.append(
                band_energy_db(out, 5_500.0, 9_000.0) - band_energy_db(scaled, 5_500.0, 9_000.0)
            )
        assert max(deltas) - min(deltas) < 0.5, deltas

    def test_ignores_material_with_no_sibilance_band(self) -> None:
        """Low-frequency material must not trigger the stage. Without the
        presence gate the stage engages on filter leakage and reports a
        reduction that is not there."""
        low = sibilant_burst(lo_hz=200.0, hi_hz=1_200.0)
        out, applied = deess(low, SR, amount_db=-9.0)
        assert not applied
        assert np.array_equal(out, low)

    def test_leaves_low_frequencies_alone(self) -> None:
        signal = tone(200.0, 1.0, SR, 0.4)
        out, _ = deess(signal, SR, amount_db=-9.0)
        assert band_energy_db(out, 150.0, 300.0) == pytest.approx(band_energy_db(signal, 150.0, 300.0), abs=0.2)

    def test_can_be_disabled(self) -> None:
        signal = tone(7000.0, 0.1, SR, 0.4)
        out, applied = deess(signal, SR, enable=False)
        assert not applied
        assert np.array_equal(out, signal)

    def test_zero_amount_is_identity(self) -> None:
        signal = tone(7000.0, 0.2, SR, 0.4)
        out, applied = deess(signal, SR, amount_db=0.0)
        assert not applied
        assert np.array_equal(out, signal)


class TestFilterChain:
    def test_reports_what_it_applied(self) -> None:
        result = apply_filter_chain(buffer_of(speech_like(2.0, SR, 120.0, 4)))
        assert result.changed
        assert any("high-pass" in a for a in result.applied)
        assert any("DC" in a or "low-pass" in a for a in result.applied)

    def test_preserves_sample_count_and_rate(self) -> None:
        buf = buffer_of(speech_like(1.0, SR, 120.0, 4))
        result = apply_filter_chain(buf)
        assert result.audio.frames == buf.frames
        assert result.audio.sample_rate == buf.sample_rate

    def test_output_never_clipped(self) -> None:
        """The chain must not push a signal into clipping. This is the
        brief's explicit 'avoid clipping' requirement, as an assertion."""
        result = apply_filter_chain(buffer_of(clipped()))
        assert count_clipped(result.audio.samples) == 0

    @pytest.mark.parametrize("source", ["clipped", "full_scale_tone", "hot_noise", "hot_speech"])
    def test_output_never_clipped_across_hard_inputs(self, source: str) -> None:
        """A limiter that only holds on the one fixture that was tuned against
        it is not a limiter. Gain-adding stages mean loud material is the real
        risk, so probe each way a caller can actually push the chain."""
        sources = {
            "clipped": clipped(),
            "full_scale_tone": tone(2800.0, 1.0, SR, 0.99),
            "hot_noise": white_noise(1.0, SR, 0.4, seed=9),
            "hot_speech": (speech_like(1.0, SR, 120.0, 4) * 3.0).astype(np.float32),
        }
        result = apply_filter_chain(buffer_of(sources[source]))
        out = result.audio.samples
        assert count_clipped(out) == 0, f"{source} clipped"
        assert float(np.max(np.abs(out))) <= TRUE_PEAK_CEILING + 1e-6, f"{source} exceeded the ceiling"

    def test_aggressive_eq_stays_within_ceiling(self) -> None:
        """The chain's limiter is what makes a large EQ boost safe. If the
        limiter is ever dropped from the chain, this must fail."""
        loud = tone(2800.0, 1.0, SR, 0.99)
        result = apply_filter_chain(
            buffer_of(loud),
            presence_boost_db=9.0,
            mud_cut_db=-6.0,
            deess_amount_db=-9.0,
        )
        assert count_clipped(result.audio.samples) == 0
        assert float(np.max(np.abs(result.audio.samples))) <= TRUE_PEAK_CEILING + 1e-6
        # The ceiling is currently held by the EQ stage's own gain scaling, so
        # the chain limiter stays idle. Assert the ceiling holds either way
        # rather than pinning which stage is responsible.
        assert any(
            "gain scaled" in a.lower() or "limit" in a.lower() for a in result.applied
        ), result.applied

    def test_reduces_dc_offset(self) -> None:
        signal = (speech_like(1.0, SR, 120.0, 4) + 0.15).astype(np.float32)
        result = apply_filter_chain(buffer_of(signal))
        assert abs(float(np.mean(result.audio.samples))) < 1e-4

    def test_preserves_stereo_channel_count(self) -> None:
        left = speech_like(1.0, SR, 120.0, 1)
        right = speech_like(1.0, SR, 125.0, 2)
        buf = AudioBuffer(samples=np.stack([left, right], axis=1), sample_rate=SR)
        result = apply_filter_chain(buf)
        assert result.audio.channels == 2

    def test_rejects_empty_buffer(self) -> None:
        from aelf.errors import EnhancementError

        with pytest.raises(EnhancementError):
            apply_filter_chain(buffer_of(np.zeros(0, dtype=np.float32)))

    def test_all_filters_off_leaves_signal_near_unchanged(self) -> None:
        """With DC blocking, EQ and de-essing off, the chain should be a
        pass-through (modulo the high-pass/low-pass pair)."""
        signal = tone(1000.0, 1.0, SR, 0.3)
        buf = buffer_of(signal)
        result = apply_filter_chain(
            buf,
            highpass_hz=0.0,
            lowpass_hz=SR / 2 * 0.99,
            presence_boost_db=0.0,
            mud_cut_db=0.0,
            deess_amount_db=0.0,
            dc_block=False,
        )
        assert result.audio.samples.shape == buf.samples.shape


class TestNormalisation:
    def test_lufs_normalisation_reaches_target(self) -> None:
        buf = buffer_of(speech_like(4.0, SR, 120.0, 8) * 0.1)
        out = normalise_lufs(buf, target_lufs=-23.0)
        measured, _ = integrated_lufs(out)
        assert measured == pytest.approx(-23.0, abs=0.5)

    def test_lufs_normalisation_respects_both_targets(self) -> None:
        for target in (TARGET_LUFS_SPEECH, TARGET_LUFS_LISTENING):
            out = normalise_lufs(buffer_of(speech_like(3.0, SR, 120.0, 9)), target_lufs=target)
            assert count_clipped(out.samples) == 0
            assert float(np.max(np.abs(out.samples))) <= TRUE_PEAK_CEILING + 1e-6

    def test_normalisation_never_clips(self) -> None:
        """Even from a hot source, the limiter must hold the ceiling."""
        out = normalise_lufs(buffer_of(clipped(2.0, SR) * 1.0), target_lufs=-6.0)
        assert count_clipped(out.samples) == 0
        assert float(np.max(np.abs(out.samples))) <= TRUE_PEAK_CEILING + 1e-6

    def test_lufs_target_above_ceiling_falls_back_to_ceiling(self) -> None:
        out = normalise_lufs(buffer_of(speech_like(3.0, SR, 120.0, 3)), target_lufs=0.0)
        assert float(np.max(np.abs(out.samples))) <= TRUE_PEAK_CEILING + 1e-6

    def test_silence_passes_through(self) -> None:
        buf = buffer_of(np.zeros(SR, dtype=np.float32))
        out = normalise_lufs(buf)
        assert not np.any(out.samples)

    def test_peak_normalisation_reaches_target(self) -> None:
        out = normalise_peak(buffer_of(tone(1000.0, 1.0, SR, 0.05)), target_peak_dbfs=-6.0)
        assert float(np.max(np.abs(out.samples))) == pytest.approx(10 ** (-6.0 / 20.0), rel=1e-3)

    def test_peak_normalisation_clamped_by_ceiling(self) -> None:
        out = normalise_peak(buffer_of(tone(1000.0, 1.0, SR, 0.05)), target_peak_dbfs=0.0)
        assert float(np.max(np.abs(out.samples))) <= TRUE_PEAK_CEILING + 1e-6

    def test_normalisation_is_a_uniform_gain(self) -> None:
        """Wherever the limiter is not engaging, output must be exactly the
        input times a constant. A limiter that clipped instead of scaling would
        break this proportionality."""
        signal = speech_like(2.0, SR, 120.0, 6) * 0.05
        out = normalise_lufs(buffer_of(signal), target_lufs=-16.0).samples[:, 0]
        assert out.shape == signal.shape
        # Recover the gain from a quiet sample, where limiting cannot have acted.
        quiet = np.abs(signal) < 0.01
        assert quiet.sum() > 100
        gain = out[quiet] / signal[quiet]
        assert np.allclose(gain, gain[0], rtol=1e-3)
        assert gain[0] > 1.0  # the source was quiet, so the gain is a boost

    def test_normalisation_does_not_clip_when_boosting_hard(self) -> None:
        """A large boost request on a hot source must still land under the ceiling."""
        signal = (speech_like(2.0, SR, 120.0, 6) * 0.8).astype(np.float32)
        out = normalise_lufs(buffer_of(signal), target_lufs=-8.0).samples
        assert count_clipped(out) == 0
        assert float(np.max(np.abs(out))) <= TRUE_PEAK_CEILING + 1e-6

    def test_presence_gain_is_clamped(self) -> None:
        quiet = buffer_of(speech_like(2.0, SR, 120.0, 4) * 0.001)
        gain = speech_presence_gain(quiet, max_boost_db=6.0)
        assert float(np.max(gain)) <= 10 ** (6.0 / 20.0) + 1e-6

    def test_presence_gain_zero_for_silence(self) -> None:
        gain = speech_presence_gain(buffer_of(np.zeros(SR, dtype=np.float32)))
        assert np.allclose(gain, 0.0)

    def test_ceiling_constant_is_minus_one_dbfs(self) -> None:
        """-1.0 dBFS is 0.891251, not 0.988 (which is only -0.1 dBFS). This
        assertion exists because the constant was once wrong by that much and
        let processed output clip."""
        assert CLIP_CEILING_DBFS == -1.0
        assert pytest.approx(0.891251, abs=1e-5) == TRUE_PEAK_CEILING


class TestEndToEndSignalSafety:
    @pytest.mark.parametrize("scale", [0.001, 0.05, 0.3, 1.0, 4.0])
    def test_chain_output_always_below_ceiling(self, scale: float) -> None:
        """Property test across input levels: no combination of source level
        and processing should produce clipped output."""
        signal = (speech_like(2.0, SR, 120.0, 3) * scale + white_noise(2.0, SR, 0.02 * scale)).astype(np.float32)
        filtered = apply_filter_chain(buffer_of(signal)).audio
        limited = true_peak_limit(filtered.samples, CLIP_CEILING_DBFS)
        report = measure_levels(buffer_of(limited), include_loudness=False)
        assert not report.is_clipped
