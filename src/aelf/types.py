"""Core data types shared across the pipeline.

Audio is carried as float32 numpy arrays in [-1.0, 1.0]. Float is the
internal representation because every processing stage (STFT, filters,
limiter) operates in linear amplitude; integer PCM is only produced at the
encode boundary.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float32]
IntArray = NDArray[np.int16]


class Confidence(StrEnum):
    """How much a result can be trusted. Surfaced directly in the UI.

    A `StrEnum` so the value renders as "high" rather than "Confidence.HIGH"
    when formatted for display.
    """

    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    NONE = "none"

    @property
    def label(self) -> str:
        return {
            Confidence.HIGH: "Verified",
            Confidence.MEDIUM: "Plausible",
            Confidence.LOW: "Uncertain",
            Confidence.NONE: "Not assessed",
        }[self]


class Availability(StrEnum):
    AVAILABLE = "available"
    UNAVAILABLE = "unavailable"
    ERROR = "error"


@dataclass(frozen=True)
class AudioBuffer:
    """A decoded, normalised-to-float audio buffer plus its provenance.

    `source_path` is retained so downstream stages can verify that no output
    was ever written over an input.
    """

    samples: FloatArray
    sample_rate: int
    source_path: Path | None = None
    original_format: str | None = None

    def __post_init__(self) -> None:
        if self.samples.ndim == 1:
            # Frozen dataclass, so bypass the assignment guard.
            object.__setattr__(self, "samples", self.samples[:, None])
        if self.samples.ndim != 2:
            raise ValueError(f"AudioBuffer expects 1-D or 2-D samples, got shape {self.samples.shape}")
        if self.samples.dtype != np.float32:
            object.__setattr__(self, "samples", self.samples.astype(np.float32))
        if self.sample_rate <= 0:
            raise ValueError(f"AudioBuffer requires a positive sample rate, got {self.sample_rate}")

    @property
    def channels(self) -> int:
        return int(self.samples.shape[1])

    @property
    def frames(self) -> int:
        return int(self.samples.shape[0])

    @property
    def duration_s(self) -> float:
        if self.sample_rate <= 0:
            return 0.0
        return self.frames / self.sample_rate

    @property
    def is_silent(self) -> bool:
        return not np.any(self.samples)

    def mono(self) -> FloatArray:
        """Downmix to mono by averaging. Returns a copy."""
        if self.channels == 1:
            return self.samples[:, 0].copy()
        return self.samples.mean(axis=1).astype(np.float32)

    def channel(self, index: int) -> FloatArray:
        return self.samples[:, index].copy()

    def replace_mono(self, mono: FloatArray) -> AudioBuffer:
        """Return a copy with `mono` duplicated to all channels."""
        m = np.asarray(mono, dtype=np.float32)
        return AudioBuffer(
            samples=np.repeat(m[:, None], self.channels, axis=1),
            sample_rate=self.sample_rate,
            source_path=self.source_path,
            original_format=self.original_format,
        )


@dataclass(frozen=True)
class SourceIntegrity:
    """Provenance record proving the input file was never mutated."""

    path: Path
    size_bytes: int
    sha256: str
    mtime_ns: int
    verified_at_end: bool = False

    def matches(self, path: Path, size_bytes: int, sha256: str) -> bool:
        return self.path == path and self.size_bytes == size_bytes and self.sha256 == sha256


@dataclass
class LevelReport:
    """Amplitude measurements. All dBFS values are derived, never assumed."""

    rms_dbfs: float
    peak_dbfs: float
    true_peak_dbfs: float
    dc_offset: float
    crest_factor_db: float
    clipped_sample_count: int
    clipped_sample_fraction: float
    lufs_integrated: float | None = None
    lufs_range: float | None = None
    warnings: list[str] = field(default_factory=list)

    @property
    def is_clipped(self) -> bool:
        return self.clipped_sample_count > 0


@dataclass
class NoiseProfile:
    """Estimated noise characteristics, derived from low-energy frames."""

    # Both are None when there is nothing to measure (empty or digitally
    # silent signal). A -inf floor and a 0 dB SNR would both read as real
    # numbers, which is exactly the sort of confident nonsense this toolkit
    # refuses to emit.
    noise_floor_dbfs: float | None
    noise_floor_hz: float
    spectral_tilt_db_per_octave: float
    is_stationary: bool
    estimated_hum_hz: float | None
    estimated_snr_db: float | None
    dominant_noise_band_hz: tuple[float, float]
    method: str
    confidence: Confidence
    frame_count: int
    notes: list[str] = field(default_factory=list)


@dataclass
class SpeechSegment:
    start_s: float
    end_s: float

    @property
    def duration_s(self) -> float:
        return self.end_s - self.start_s

    def as_dict(self) -> dict[str, float]:
        return {"start_s": round(self.start_s, 3), "end_s": round(self.end_s, 3)}


@dataclass
class VadResult:
    segments: list[SpeechSegment]
    speech_ratio: float
    speech_duration_s: float
    backend: str
    confidence: Confidence
    notes: list[str] = field(default_factory=list)

    @property
    def is_speech_detected(self) -> bool:
        return bool(self.segments)


@dataclass
class EnhancementResult:
    audio: AudioBuffer
    backend: str
    strength: float
    processing_time_s: float
    notes: list[str] = field(default_factory=list)
    fallback_used: bool = False


@dataclass
class SeparationStem:
    name: str
    audio: AudioBuffer
    rms_dbfs: float
    energy_ratio: float


@dataclass
class SeparationResult:
    stems: list[SeparationStem]
    backend: str
    processing_time_s: float
    confidence: Confidence
    stem_credibility_db: float | None
    verified: bool
    warning: str | None = None
    notes: list[str] = field(default_factory=list)

    @property
    def source_count(self) -> int:
        return len(self.stems)


@dataclass
class TranscriptWord:
    word: str
    start_s: float
    end_s: float
    probability: float


@dataclass
class TranscriptSegment:
    start_s: float
    end_s: float
    text: str
    avg_logprob: float
    no_speech_prob: float
    words: list[TranscriptWord] = field(default_factory=list)

    @property
    def confidence(self) -> Confidence:
        if self.avg_logprob > -0.3 and self.no_speech_prob < 0.3:
            return Confidence.HIGH
        if self.avg_logprob > -0.8 and self.no_speech_prob < 0.6:
            return Confidence.MEDIUM
        return Confidence.LOW


@dataclass
class Transcript:
    text: str
    language: str | None
    segments: list[TranscriptSegment]
    model: str
    compute_type: str
    processing_time_s: float
    source_label: str
    audio_duration_s: float
    confidence: Confidence
    notes: list[str] = field(default_factory=list)

    @property
    def is_empty(self) -> bool:
        return not self.text.strip()
