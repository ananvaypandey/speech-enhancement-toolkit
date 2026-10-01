"""Shared fixtures.

Every fixture is synthetic and generated with a fixed seed, so the suite
never touches a user recording and never needs a network download. Signals
are built from primitives with analytically known properties (a 1 kHz tone
has exactly 0 dBFS RMS at full scale; white noise at a given sigma has a
predictable floor) so assertions can be exact rather than approximate.
"""

from __future__ import annotations

import numpy as np
import pytest
import soundfile as sf

from aelf.types import AudioBuffer

SR = 48_000


def rng(seed: int = 1234) -> np.random.Generator:
    return np.random.default_rng(seed)


def tone(freq_hz: float = 1000.0, seconds: float = 1.0, sr: int = SR, amplitude: float = 0.5) -> np.ndarray:
    t = np.arange(int(sr * seconds)) / sr
    return (amplitude * np.sin(2.0 * np.pi * freq_hz * t)).astype(np.float32)


def white_noise(seconds: float = 1.0, sr: int = SR, sigma: float = 0.01, seed: int = 7) -> np.ndarray:
    return (rng(seed).normal(0.0, sigma, int(sr * seconds))).astype(np.float32)


def pink_noise(seconds: float = 1.0, sr: int = SR, sigma: float = 0.01, seed: int = 7) -> np.ndarray:
    """1/f noise via spectral shaping - a realistic stand-in for room noise."""
    n = int(sr * seconds)
    white = rng(seed).normal(0.0, 1.0, n)
    spectrum = np.fft.rfft(white)
    freqs = np.fft.rfftfreq(n, 1.0 / sr)
    scale = np.ones_like(freqs)
    nonzero = freqs > 0
    scale[nonzero] = 1.0 / np.sqrt(freqs[nonzero])
    shaped = np.fft.irfft(spectrum * scale, n=n)
    rms_value = float(np.sqrt(np.mean(shaped**2))) or 1.0
    return (shaped / rms_value * sigma).astype(np.float32)


def speech_like(
    seconds: float = 2.0,
    sr: int = SR,
    f0_hz: float = 120.0,
    seed: int = 42,
) -> np.ndarray:
    """A crude voiced-speech surrogate: harmonic stack with formant shaping.

    Not real speech, but it has speech-like spectral structure, a realistic
    harmonic-to-noise ratio, and syllabic amplitude modulation, which is what
    the noise and level estimators actually key on.
    """
    generator = rng(seed)
    n = int(sr * seconds)
    t = np.arange(n) / sr
    signal = np.zeros(n, dtype=np.float64)

    for harmonic in range(1, 26):
        f = f0_hz * harmonic
        if f > sr / 2.2:
            break
        # Formant-like emphasis around 700 Hz and 2200 Hz.
        gain = 1.0 / harmonic
        gain *= 1.0 + 1.6 * np.exp(-((f - 700.0) ** 2) / (2 * 300.0**2))
        gain *= 1.0 + 1.1 * np.exp(-((f - 2200.0) ** 2) / (2 * 450.0**2))
        signal += gain * np.sin(2.0 * np.pi * f * t + generator.uniform(0, 2 * np.pi))

    signal /= np.max(np.abs(signal)) or 1.0

    # Syllabic envelope at ~4 Hz, then 12 ms raised-cosine transitions.
    env = 0.5 + 0.5 * np.sin(2.0 * np.pi * 4.0 * t)
    env = np.clip(env, 0.0, 1.0) ** 1.5
    fade = max(int(sr * 0.012), 1)
    ramp = np.hanning(2 * fade)[:fade]
    env[:fade] *= ramp
    env[-fade:] *= ramp[::-1]

    signal = signal * env
    aspiration = generator.normal(0.0, 0.01, n)
    return (signal + aspiration).astype(np.float32) * 0.3


def speech_plus_noise(
    seconds: float = 2.0,
    sr: int = SR,
    noise_sigma: float = 0.02,
    speech_gain: float = 0.3,
    seed: int = 42,
) -> np.ndarray:
    return (speech_like(seconds, sr, seed=seed) * speech_gain + white_noise(seconds, sr, noise_sigma, seed + 1)).astype(
        np.float32
    )


def two_speakers(sr: int = SR, seconds: float = 2.0) -> np.ndarray:
    """Two simultaneous, non-identical voices at the same frequency.

    Deliberately constructed so that the sources are genuinely confusable:
    separation cannot succeed cleanly on this, which is exactly the case the
    toolkit must report as uncertain rather than present as recovered.
    """
    a = speech_like(seconds, sr, f0_hz=120.0, seed=11) * 0.25
    b = speech_like(seconds, sr, f0_hz=125.0, seed=22) * 0.25
    return (a + b).astype(np.float32)


def sibilant_burst(
    seconds: float = 1.0,
    sr: int = SR,
    lo_hz: float = 5_800.0,
    hi_hz: float = 8_200.0,
    seed: int = 5,
    amplitude: float = 0.035,
) -> np.ndarray:
    """Band-limited noise with a sharp attack and a decay, in the sibilance band.

    Probes for the de-esser must look like real sibilance. Stationary
    broadband noise does not work: a band-passed 5.5-9 kHz slice of it sits
    20-30 dB below the programme level and has almost no envelope dynamics,
    so a correctly-keyed de-esser correctly leaves it alone. Pass `lo_hz`/`hi_hz`
    to build material with no sibilance band for the negative cases.
    """
    from scipy.signal import butter, sosfiltfilt

    noise = white_noise(seconds, sr, 1.0, seed).astype(np.float64)
    sos = butter(2, [lo_hz / (sr / 2.0), hi_hz / (sr / 2.0)], btype="bandpass", output="sos")
    band = sosfiltfilt(sos, noise, axis=0)
    peak = float(np.max(np.abs(band)))
    if peak > 0:
        band /= peak
    t = np.arange(int(seconds * sr)) / sr
    env = np.exp(-t / 0.22)
    env = np.where(t < 0.02, t / 0.02, env)
    return (band * env * amplitude).astype(np.float32)


def silence(seconds: float = 1.0, sr: int = SR) -> np.ndarray:
    return np.zeros(int(sr * seconds), dtype=np.float32)


def clipped(seconds: float = 0.5, sr: int = SR) -> np.ndarray:
    return np.clip(tone(440.0, seconds, sr, amplitude=2.0), -1.0, 1.0).astype(np.float32)


def write_fixture(path, samples: np.ndarray, sr: int = SR, subtype: str = "PCM_16") -> str:
    data = np.asarray(samples)
    if data.ndim == 1:
        data = data[:, None]
    sf.write(str(path), data.astype(np.float32), sr, subtype=subtype)
    return str(path)


@pytest.fixture
def sr() -> int:
    return SR


@pytest.fixture
def tone_buffer() -> AudioBuffer:
    return AudioBuffer(samples=tone(), sample_rate=SR, original_format="wav")


@pytest.fixture
def noise_buffer() -> AudioBuffer:
    return AudioBuffer(samples=white_noise(sigma=0.01), sample_rate=SR, original_format="wav")


@pytest.fixture
def speech_noise_buffer() -> AudioBuffer:
    return AudioBuffer(samples=speech_plus_noise(), sample_rate=SR, original_format="wav")


@pytest.fixture
def silence_buffer() -> AudioBuffer:
    return AudioBuffer(samples=silence(), sample_rate=SR, original_format="wav")


@pytest.fixture
def stereo_buffer() -> AudioBuffer:
    left = tone(440.0, 1.0, SR, 0.4)
    right = tone(660.0, 1.0, SR, 0.3)
    return AudioBuffer(samples=np.stack([left, right], axis=1), sample_rate=SR, original_format="wav")
