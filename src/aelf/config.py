"""Central configuration and pinned constants for AELF.

All numeric targets are module-level so tests can assert against a single
source of truth rather than magic numbers scattered across the codebase.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

PACKAGE_NAME = "aelf"
CANONICAL_SR = 48_000
SUPPORTED_INPUT_SUFFIXES = frozenset({".wav", ".mp3", ".m4a", ".aac", ".flac", ".ogg", ".opus"})

# Amplitude guards. Clipping at |x| >= 1.0 is the hard digital ceiling.
# -1.0 dBFS = 0.891251 leaves headroom for downstream resampling and lossy
# encoding, which can overshoot the sample peak.
CLIP_CEILING_DBFS = -1.0
TRUE_PEAK_CEILING = 10.0 ** (CLIP_CEILING_DBFS / 20.0)  # 0.891251
CLIP_THRESHOLD = 0.999

# Level targets. -23 LUFS is the EBU R128 broadcast recommendation for
# spoken-word distribution; -16 suits headphone listening.
TARGET_LUFS_SPEECH = -23.0
TARGET_LUFS_LISTENING = -16.0
# Loudness measurement.
LOUDNESS_BLOCK_SIZE_S = 0.400
LOUDNESS_GATE_RELATIVE = -70.0  # blocks below target-70 LUFS are ignored
# BS.1770 K-weighting coefficients are specified at 48 kHz; other rates are
# derived from the analogue prototype.
SAMPLE_RATE_FOR_LOUDNESS = 48_000

# Speech band weighting. Broadband speech intelligibility lives roughly in
# 100-8000 Hz; energy below 80 Hz is rumble, above 12 kHz is near-inaudible
# sibilance that only adds hiss after enhancement.
SPEECH_BAND_HZ = (80.0, 8_000.0)
SPEECH_PRESENCE_BAND_HZ = (300.0, 3_400.0)
HIGHPASS_DEFAULT_HZ = 80.0
HIGHPASS_MIN_HZ = 40.0
DEESSER_MIN_HZ = 5_500.0
DEESSER_MAX_HZ = 9_000.0

# Enhancement strength -> spectral over-subtraction factor and Wiener floor.
# strength 0.0 leaves the signal essentially untouched, 1.0 is aggressive.
STRENGTH_MIN = 0.0
STRENGTH_MAX = 1.0
STRENGTH_DEFAULT = 0.5
OMLSA_FLOOR_ALPHA = 0.05
OMLSA_FLOOR_STRONG = 0.01

# Minimum separation of stems before we are willing to call a split credible.
# Two sources within 1 dB of each other is a strong hint the model did not
# actually separate anything.
STEM_CREDIBILITY_MIN_DB = 1.0
STEM_COLLAPSE_RATIO = 0.02  # stem energy below 2% of input -> degenerate
SEPARATION_UNCERTAIN = "separation could not be verified"

# Time-aligned reference comparison tolerances. Beyond these we refuse to
# compute metrics rather than silently resampling a mismatch.
REF_MAX_DURATION_DRIFT_S = 0.05
REF_MAX_SAMPLE_DRIFT = 1024

# Whisper defaults. "base" is the largest size that comfortably fits the
# 6 GB VRAM budget of the target machine alongside other loaded models.
WHISPER_DEFAULT_MODEL = "base"
WHISPER_AVAILABLE_SIZES = ("tiny", "base", "small", "medium", "large-v3")
WHISPER_DEFAULT_COMPUTE = "int8"
WHISPER_DEFAULT_BEAM = 5


def _default_root() -> Path:
    override = os.environ.get("AELF_HOME")
    if override:
        return Path(override).expanduser().resolve()
    return Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Paths:
    """Filesystem layout. `uploads` is read-only from the pipeline's view."""

    root: Path = field(default_factory=_default_root)
    work: Path = field(init=False)
    uploads: Path = field(init=False)
    outputs: Path = field(init=False)
    models: Path = field(init=False)
    reports: Path = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "work", self.root / "work")
        object.__setattr__(self, "uploads", self.root / "work" / "uploads")
        object.__setattr__(self, "outputs", self.root / "work" / "outputs")
        object.__setattr__(self, "models", self.root / "work" / "models")
        object.__setattr__(self, "reports", self.root / "work" / "reports")

    def ensure(self) -> None:
        for p in (self.work, self.uploads, self.outputs, self.models, self.reports):
            p.mkdir(parents=True, exist_ok=True)


PATHS = Paths()

# Models are cached here rather than in the user profile so that the whole
# runtime footprint is relocatable and inspectable.
os.environ.setdefault("HF_HOME", str(PATHS.models / "hf"))
os.environ.setdefault("TORCH_HOME", str(PATHS.models / "torch"))


def model_cache_dir() -> Path:
    PATHS.models.mkdir(parents=True, exist_ok=True)
    return PATHS.models


def torch_home() -> Path:
    return Path(os.environ["TORCH_HOME"])


def hf_home() -> Path:
    return Path(os.environ["HF_HOME"])
