"""Decoding and encoding.

Decode strategy, in order of preference:
  1. soundfile (libsndfile) - handles WAV/FLAC/OGG natively, no extra deps
  2. ffmpeg subprocess - the reliable path for MP3/M4A on every platform
  3. error, with the reason reported

Nothing is ever written back to the input path; `decode_to_canonical` only
reads. Integer PCM is normalised to float32 here so that all downstream
arithmetic is in linear amplitude with a known ceiling of 1.0.
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from functools import lru_cache
from pathlib import Path

import numpy as np
import soundfile as sf

from ..config import CANONICAL_SR, SUPPORTED_INPUT_SUFFIXES
from ..errors import DecodeError, EmptyAudioError, UnsupportedFormatError
from ..types import AudioBuffer

# Below this duration the file is treated as a failed capture rather than
# audio. 20 ms is shorter than any VAD window we use.
MIN_DURATION_S = 0.020


@lru_cache(maxsize=1)
def ffmpeg_path() -> str | None:
    """Locate a usable ffmpeg binary, or None if unavailable."""
    return shutil.which("ffmpeg")


def check_supported(path: Path) -> None:
    suffix = Path(path).suffix.lower()
    if suffix not in SUPPORTED_INPUT_SUFFIXES:
        shown = suffix if suffix else "unknown"
        raise UnsupportedFormatError(
            f"Unsupported suffix {suffix!r}; supported={sorted(SUPPORTED_INPUT_SUFFIXES)}",
            user_message=f"Files of type '{shown}' are not supported.",
            remedy="Upload a WAV, MP3, M4A, FLAC, OGG or Opus file.",
        )


def _normalise(raw: np.ndarray, source_sr: int, path: Path) -> AudioBuffer:
    """Convert raw libsndfile output into a canonical AudioBuffer.

    libsndfile returns int16/int32/float64 depending on the file subtype.
    We scale integers by their full-scale range, which is what the subtype
    declares, so a 16-bit file at half amplitude reads as 0.5 rather than
    being silently normalised up to 1.0.
    """
    data = np.asarray(raw)
    if data.dtype == np.int16:
        data = data.astype(np.float32) / 32768.0
    elif data.dtype == np.int32:
        data = data.astype(np.float32) / 2147483648.0
    elif data.dtype == np.uint8:
        data = (data.astype(np.float32) - 128.0) / 128.0
    elif data.dtype in (np.float32, np.float64):
        data = data.astype(np.float32)
    else:
        raise DecodeError(
            f"Unsupported sample dtype {data.dtype} from {path}",
            user_message="The audio uses a sample format this tool cannot read.",
            remedy="Convert the file to 16-bit or 24-bit PCM WAV and retry.",
        )

    buffer = AudioBuffer(
        samples=data,
        sample_rate=source_sr,
        source_path=Path(path),
        original_format=source_sr and Path(path).suffix.lower().lstrip("."),
    )
    _validate(buffer)
    return buffer


def _validate(buffer: AudioBuffer) -> None:
    if buffer.frames == 0:
        raise EmptyAudioError(f"No samples decoded from {buffer.source_path}")
    if buffer.duration_s < MIN_DURATION_S:
        raise EmptyAudioError(
            f"Decoded duration {buffer.duration_s:.4f}s is below the {MIN_DURATION_S}s minimum",
            user_message="The audio file is too short to process.",
            remedy="Provide a recording of at least a few hundred milliseconds.",
        )
    if not np.all(np.isfinite(buffer.samples)):
        raise DecodeError(
            f"Non-finite samples in {buffer.source_path}",
            user_message="The audio file contains unreadable data.",
            remedy="The file may be corrupt. Try re-encoding it as 16-bit WAV.",
        )


def _decode_with_soundfile(path: Path) -> AudioBuffer:
    raw, sr = sf.read(str(path), dtype="auto", always_2d=True)
    return _normalise(raw, int(sr), path)


def _decode_with_ffmpeg(path: Path) -> AudioBuffer:
    binary = ffmpeg_path()
    if binary is None:
        raise DecodeError("ffmpeg not found on PATH", detail="soundfile failed and no ffmpeg fallback is available.")

    with tempfile.TemporaryDirectory(prefix="aelf-decode-") as tmp:
        target = Path(tmp) / "decoded.wav"
        cmd = [
            binary,
            "-hide_banner",
            "-loglevel",
            "error",
            "-nostdin",
            "-i",
            str(path),
            "-map",
            "0:a:0",
            "-acodec",
            "pcm_f32le",
            "-ar",
            str(CANONICAL_SR),
            "-f",
            "wav",
            str(target),
        ]
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=600, check=False)
        except subprocess.TimeoutExpired as exc:
            raise DecodeError(
                f"ffmpeg timed out decoding {path}",
                user_message="Decoding took too long and was stopped.",
                remedy="For very long files, trim the audio to the segment you need before uploading.",
            ) from exc

        if proc.returncode != 0 or not target.exists():
            raise DecodeError(
                f"ffmpeg failed (code {proc.returncode}): {proc.stderr.strip()[:500]}",
                user_message="The audio file could not be decoded.",
                remedy="The file may be corrupt or use an unusual codec. Re-encode it as 16-bit WAV and retry.",
            )
        raw, sr = sf.read(str(target), dtype="float32", always_2d=True)
        return _normalise(raw, int(sr), path)


def load(path: Path) -> AudioBuffer:
    """Read an audio file into an AudioBuffer without modifying it."""
    path = Path(path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"No such audio file: {path}")
    check_supported(path)

    errors: list[str] = []
    # Track the most specific failure. A file that decodes to zero samples
    # or is too short is not a codec problem, and telling the operator it
    # "could not be decoded" would send them chasing the wrong fix.
    specific: EmptyAudioError | None = None

    for decoder in (_decode_with_soundfile, _decode_with_ffmpeg):
        try:
            return decoder(path)
        except EmptyAudioError as exc:
            specific = exc
            errors.append(f"{decoder.__name__}: {exc}")
        except DecodeError as exc:
            errors.append(f"{decoder.__name__}: {exc}")
        except Exception as exc:
            errors.append(f"{decoder.__name__}: {type(exc).__name__}: {exc}")

    if specific is not None:
        raise specific

    raise DecodeError(
        " | ".join(errors),
        user_message="The audio file could not be decoded by any available backend.",
        remedy=(
            "Try re-encoding as 16-bit PCM WAV. If ffmpeg is installed but not on PATH, "
            "add it to PATH and retry."
        ),
    )


def resample(buffer: AudioBuffer, target_sr: int = CANONICAL_SR) -> AudioBuffer:
    """Resample to `target_sr` using a polyphase FIR (linear-phase)."""
    if buffer.sample_rate == target_sr:
        return buffer

    gcd = np.gcd(buffer.sample_rate, target_sr)
    up, down = target_sr // gcd, buffer.sample_rate // gcd

    if up == 1 and down == 1:
        return buffer

    from scipy.signal import resample_poly

    resampled = resample_poly(buffer.samples, up, down, axis=0).astype(np.float32)
    out = AudioBuffer(
        samples=resampled,
        sample_rate=target_sr,
        source_path=buffer.source_path,
        original_format=buffer.original_format,
    )
    _validate(out)
    return out


def decode_to_canonical(path: Path, target_sr: int = CANONICAL_SR) -> AudioBuffer:
    """Decode and resample in one step. The canonical pre-processing entry point."""
    return resample(load(path), target_sr)
