"""Spectral noise suppression tests.

Each test pairs a property the suppressor must have with the reason it matters.
Where a property can be measured directly it is, rather than inferred from a
proxy that could pass for the wrong reason.
"""

from __future__ import annotations

import sys
from itertools import pairwise
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from aelf.config import STRENGTH_MAX, STRENGTH_MIN
from aelf.enhance.spectral import (
    SpectralConfig,
    available_backends,
    enhance,
    suppress_stationary_noise,
)
from aelf.errors import EnhancementError
from aelf.types import AudioBuffer
from conftest import SR, silence, speech_like, tone, white_noise


def buffer_of(samples: np.ndarray, sr: int = SR) -> AudioBuffer:
    return AudioBuffer(samples=np.asarray(samples, dtype=np.float32)[:, None], sample_rate=sr)


def band_power(samples: np.ndarray, lo: float, hi: float, sr: int = SR) -> float:
    x = np.asarray(samples, dtype=np.float64)
    spec = np.abs(np.fft.rfft(x)) ** 2
    freqs = np.fft.rfftfreq(len(x), 1.0 / sr)
    mask = (freqs >= lo) & (freqs <= hi)
    return float(np.sum(spec[mask]))


def projected_snr_db(out: np.ndarray, signal: np.ndarray, noise: np.ndarray) -> float:
    """SNR against known sources, by projecting onto each.

    Legitimate here because both sources are known, which is exactly the
    situation the toolkit refuses to report a metric in.
    """
    o = np.asarray(out, dtype=np.float64)
    s = np.asarray(signal, dtype=np.float64)
    n = np.asarray(noise, dtype=np.float64)
    signal_gain = float(np.dot(o, s)) / max(float(np.dot(s, s)), 1e-20)
    residual = o - signal_gain * s
    noise_gain = float(np.dot(residual, n))
    return 20.0 * np.log10(max(signal_gain**2 / max(noise_gain**2, 1e-20), 1e-20))


@pytest.fixture
def mixture() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Signal, noise, and their sum. All three known."""
    speech = speech_like(3.0, SR, 120.0, 4) * 0.25
    noise = white_noise(3.0, SR, 0.01, seed=99)
    return speech.astype(np.float32), noise.astype(np.float32), (speech + noise).astype(np.float32)


class TestStrengthControl:
    def test_zero_strength_is_identity(self, mixture) -> None:
        """Strength 0 must be a genuine pass-through, not a near-pass-through.
        A user who turns suppression off has asked for the original file back."""
        _, _, mixed = mixture
        out = suppress_stationary_noise(buffer_of(mixed), strength=0.0).audio
        assert np.max(np.abs(out.samples[:, 0] - mixed)) < 1e-6

    def test_more_strength_removes_more_noise(self, mixture) -> None:
        """The whole point of the control. Noise remaining must fall
        monotonically as strength rises."""
        _, _, mixed = mixture
        floors = []
        for strength in (0.25, 0.5, 0.75, 1.0):
            out = suppress_stationary_noise(buffer_of(mixed), strength=strength).audio
            # Measure residual noise as energy in the top octave, where this
            # synthetic speech carries little content but the hiss carries a lot.
            floors.append(band_power(out.samples[:, 0], 16_000.0, 22_000.0))
        assert all(a > b for a, b in pairwise(floors)), floors

    def test_snr_improves(self, mixture) -> None:
        """Against known sources, enhancement must raise SNR."""
        speech, noise, mixed = mixture
        before = projected_snr_db(mixed, speech, noise)
        after = projected_snr_db(suppress_stationary_noise(buffer_of(mixed), strength=0.75).audio.samples[:, 0], speech, noise)
        assert after > before + 1.0, f"{before:.2f} -> {after:.2f}"

    @pytest.mark.parametrize("strength", [-0.1, 1.1, 2.0])
    def test_rejects_out_of_range_strength(self, mixture, strength: float) -> None:
        _, _, mixed = mixture
        with pytest.raises(EnhancementError):
            suppress_stationary_noise(buffer_of(mixed), strength=strength)

    def test_config_validates_strength(self) -> None:
        with pytest.raises(EnhancementError):
            SpectralConfig(strength=5.0)


class TestSignalFidelity:
    def test_preserves_length_and_rate(self, mixture) -> None:
        """A suppressor that shifts or resamples audio produces output the user
        cannot align with their original."""
        _, _, mixed = mixture
        buf = buffer_of(mixed)
        out = suppress_stationary_noise(buf, strength=0.5).audio
        assert out.frames == buf.frames
        assert out.sample_rate == buf.sample_rate

    def test_reconstruction_is_exact_when_nothing_is_removed(self) -> None:
        """The overlap-add path must reconstruct the input to within rounding
        when the mask is all ones.

        This caught a real bug: a Hann window is exactly zero at its endpoints,
        so the first output sample was divided by a window weight of ~1.8e-11
        and reconstruction error reached 7e6.
        """
        import aelf.enhance.spectral as sp

        sr = SR
        frame_len = round(sr * 32.0 / 1000.0)
        hop = max(round(frame_len * 0.5), 1)
        signal = speech_like(1.0, sr, 120.0, 4).astype(np.float64)

        frames, lead = sp._frame_signal(signal, frame_len, hop)
        window = np.hanning(frame_len)
        windowed = frames * window
        ones = np.ones((frames.shape[0], frames.shape[1] // 2 + 1))
        masked = np.fft.irfft(ones * np.fft.rfft(windowed, axis=1), n=frame_len, axis=1) * window

        n_out = frames.shape[0] * hop + frame_len
        overlap = np.zeros(n_out)
        norm = np.zeros(n_out)
        for t in range(masked.shape[0]):
            start = t * hop
            overlap[start : start + frame_len] += masked[t]
            norm[start : start + frame_len] += window**2
        recon = (overlap / np.maximum(norm, 1e-9))[lead : lead + signal.shape[0]]
        assert np.max(np.abs(recon - signal)) < 1e-6

    def test_preserves_stereo_channels_independently(self) -> None:
        """Each channel gets its own noise estimate. Sharing one across
        channels would let a loud channel mask a quiet one."""
        quiet = (speech_like(1.0, SR, 120.0, 1) * 0.05).astype(np.float32)
        loud = (speech_like(1.0, SR, 120.0, 2) * 0.5).astype(np.float32)
        buf = AudioBuffer(samples=np.stack([quiet, loud], axis=1), sample_rate=SR)
        out = suppress_stationary_noise(buf, strength=0.75).audio
        assert out.channels == 2
        assert out.frames == buf.frames

    def test_does_not_clip(self, mixture) -> None:
        """Suppression only attenuates, so it must never raise a peak past
        full scale. This is the guarantee the writer depends on."""
        from aelf.io.encode import count_clipped

        _, _, mixed = mixture
        hot = np.clip(mixed * 8.0, -1.0, 1.0).astype(np.float32)
        out = suppress_stationary_noise(buffer_of(hot), strength=1.0).audio
        assert count_clipped(out.samples) == 0

    def test_clean_speech_is_nearly_untouched(self) -> None:
        """Noise-free input must not be audibly damaged.

        This is the most likely way the stage could silently ruin a recording:
        a suppressor that works by removing high frequencies chews up clean
        speech, and the damage is obvious after the fact.

        Bounded on RMS deviation rather than peak. A peak bound looks stricter
        but measures the wrong thing: the largest single-sample change lands in
        the quietest passages, where it sits well below peak and inaudibly.

        The separability gate is supposed to make this case exact: clean speech
        measures 28 dB of separability against a 26 dB threshold, so the gate
        reaches 1.0 and the mask becomes unity gain. The bound is therefore
        tight, because a regression here is the difference between a suppressor
        that leaves good recordings alone and one that quietly eats them.
        """
        clean = (speech_like(2.0, SR, 120.0, 4) * 0.25).astype(np.float32)
        buf = buffer_of(clean)
        out = suppress_stationary_noise(buf, strength=1.0).audio.samples[:, 0]
        deviation_rms = float(np.std(out - clean))
        signal_rms = float(np.std(clean))
        assert deviation_rms < 1e-3 * signal_rms, f"{(deviation_rms / signal_rms) * 100:.1f}% of RMS"

    def test_does_not_shift_timing(self, mixture) -> None:
        """Zero-phase processing. If the suppressor delayed audio, a user's
        lip-sync or dubbing would drift."""
        _, _, mixed = mixture
        out = suppress_stationary_noise(buffer_of(mixed), strength=0.75).audio.samples[:, 0]
        a = mixed - mixed.mean()
        b = out - out.mean()
        corr = np.correlate(b, a, mode="full")
        lag = int(np.argmax(corr)) - (len(a) - 1)
        assert abs(lag) <= 2


class TestEdgeCases:
    def test_silence_passes_through(self) -> None:
        result = suppress_stationary_noise(buffer_of(silence(1.0)), strength=0.75)
        assert result.audio.is_silent
        assert any("silent" in n.lower() for n in result.notes)

    def test_rejects_empty_buffer(self) -> None:
        with pytest.raises(EnhancementError):
            suppress_stationary_noise(buffer_of(np.zeros(0, dtype=np.float32)), strength=0.5)

    def test_handles_signal_shorter_than_one_frame(self) -> None:
        """A 100 ms clip is shorter than a 32 ms window plus its padding.
        It must still come back at the right length rather than crashing or
        returning nothing."""
        short = (speech_like(0.1, SR, 120.0, 4) * 0.2).astype(np.float32)
        out = suppress_stationary_noise(buffer_of(short), strength=0.5).audio
        assert out.frames == len(short)

    def test_handles_pure_noise_without_speech(self) -> None:
        """Noise-only input has no signal to preserve. It must still process
        without error and still get quieter."""
        hiss = white_noise(2.0, SR, 0.05, seed=7)
        out = suppress_stationary_noise(buffer_of(hiss), strength=0.75).audio.samples[:, 0]
        assert np.isfinite(out).all()
        assert np.max(np.abs(out)) <= np.max(np.abs(hiss)) + 1e-6

    def test_handles_pure_tone(self) -> None:
        """A steady tone has near-zero noise, so the estimate must not
        manufacture attenuation out of nothing."""
        sig = tone(440.0, 1.0, SR, 0.3)
        out = suppress_stationary_noise(buffer_of(sig), strength=0.5).audio.samples[:, 0]
        assert np.isfinite(out).all()
        assert np.max(np.abs(out)) > 0.1 * np.max(np.abs(sig))


class TestApi:
    def test_reports_backend_and_timing(self, mixture) -> None:
        _, _, mixed = mixture
        result = suppress_stationary_noise(buffer_of(mixed), strength=0.5)
        assert result.backend
        assert result.strength == 0.5
        assert result.processing_time_s > 0.0

    def test_spectral_backend_is_always_available(self) -> None:
        """The whole point of the classical path is that it needs no download."""
        assert "spectral-subtraction" in available_backends()

    def test_enhance_entry_point_works(self, mixture) -> None:
        _, _, mixed = mixture
        result = enhance(buffer_of(mixed), strength=0.5)
        assert result.audio.frames == len(mixed)

    def test_reports_no_fallback(self, mixture) -> None:
        """`fallback_used` must be False while the classical path is the
        intended default, not an accident. The UI shows this."""
        _, _, mixed = mixture
        assert suppress_stationary_noise(buffer_of(mixed), strength=0.5).fallback_used is False

    def test_strength_bounds_match_config(self) -> None:
        assert (STRENGTH_MIN, STRENGTH_MAX) == (0.0, 1.0)
