"""Encoding back to disk.

Two safeguards live here:
  - `assert_writable` blocks any target that collides with a protected source
  - `true_peak_limit` guarantees the written float32 WAV stays below
    -1 dBTP, so no downstream resampler or codec can push it into clipping

16-bit PCM is the default output subtype because it is lossless-enough for
speech, universally supported, and half the size of 32-bit float.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import soundfile as sf

from ..config import CLIP_CEILING_DBFS, CLIP_THRESHOLD
from ..errors import EmptyAudioError
from ..io.integrity import IntegrityGuard
from ..types import AudioBuffer


def assert_writable(target: Path, guard: IntegrityGuard | None = None) -> Path:
    """Resolve `target` and refuse to write over a protected source."""
    resolved = Path(target).expanduser().resolve()
    resolved.parent.mkdir(parents=True, exist_ok=True)
    if guard is not None:
        guard.assert_targets_safe([resolved])
    return resolved


def count_clipped(samples: np.ndarray) -> int:
    return int(np.count_nonzero(np.abs(samples) >= CLIP_THRESHOLD))


def true_peak_limit(samples: np.ndarray, ceiling_dbfs: float = CLIP_CEILING_DBFS) -> np.ndarray:
    """Scale the whole signal down so its peak sits just under `ceiling_dbfs`.

    Operates on a global gain rather than per-sample clamping. Clipping
    individual samples introduces broadband distortion that is far more
    damaging to intelligibility than a uniform level reduction, so the
    limiter's job is to prevent hitting the ceiling, not to repair samples
    that already did.
    """
    data = np.asarray(samples, dtype=np.float32)
    peak = float(np.max(np.abs(data))) if data.size else 0.0
    if peak <= 0.0:
        return data
    ceiling = float(10.0 ** (ceiling_dbfs / 20.0))
    if peak <= ceiling:
        return data
    return (data * (ceiling / peak)).astype(np.float32)


def normalise(samples: np.ndarray, target_peak_dbfs: float = -3.0) -> np.ndarray:
    """Peak-normalise to `target_peak_dbfs`, then apply the true-peak limiter."""
    data = np.asarray(samples, dtype=np.float32)
    peak = float(np.max(np.abs(data))) if data.size else 0.0
    if peak <= 0.0:
        return data
    target = float(10.0 ** (target_peak_dbfs / 20.0))
    data = (data * (target / peak)).astype(np.float32)
    return true_peak_limit(data)


def write_wav(
    buffer: AudioBuffer,
    target: Path,
    *,
    subtype: str = "PCM_16",
    guard: IntegrityGuard | None = None,
    limit: bool = True,
) -> Path:
    """Write a buffer to `target` as WAV, protecting any registered source."""
    if buffer.frames == 0:
        raise EmptyAudioError(f"Refusing to write an empty buffer to {target}")

    resolved = assert_writable(target, guard)
    data = buffer.samples.astype(np.float32)
    if limit:
        data = true_peak_limit(data)
    if not np.all(np.isfinite(data)):
        raise ValueError(f"Refusing to write non-finite samples to {resolved}")

    # The limiter above guarantees the sample peak, but guard anyway: a caller
    # that disabled limiting must not be able to write a clipped file.
    if count_clipped(data) > 0:
        data = true_peak_limit(data, CLIP_CEILING_DBFS)

    sf.write(str(resolved), data, buffer.sample_rate, subtype=subtype, format="WAV")
    return resolved


def write_stems(stems: Sequence[tuple[str, AudioBuffer]], out_dir: Path, *, guard: IntegrityGuard | None = None) -> list[Path]:
    """Write each named stem into `out_dir`, returning the written paths."""
    out_dir = Path(out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for name, buffer in stems:
        safe = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in name)
        written.append(write_wav(buffer, out_dir / f"{safe}.wav", guard=guard))
    return written


def to_int16(samples: np.ndarray) -> np.ndarray:
    """Convert float samples to int16 with TPDF-safe rounding and hard clipping."""
    data = np.clip(np.asarray(samples, dtype=np.float64), -1.0, 1.0)
    return (data * 32767.0).astype(np.int16)
