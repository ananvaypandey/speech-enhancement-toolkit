"""Container and stream inspection.

Reports what the file *claims* to be, straight from the header, separately
from what the decoder actually produced. When those two disagree (a file
labelled 48 kHz that decodes at 44.1 kHz, a stream count that does not match
the channel count) the discrepancy is surfaced rather than silently accepted.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path

import soundfile as sf

from ..errors import DecodeError
from ..types import AudioBuffer


@dataclass
class FileProbe:
    """Everything known about a file before any processing."""

    path: str
    filename: str
    exists: bool
    size_bytes: int
    container: str | None = None
    codec: str | None = None
    declared_sample_rate: int | None = None
    declared_channels: int | None = None
    declared_frames: int | None = None
    declared_duration_s: float | None = None
    subtype: str | None = None
    format_long: str | None = None
    decoded_sample_rate: int | None = None
    decoded_channels: int | None = None
    decoded_duration_s: float | None = None
    warnings: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def duration_s(self) -> float:
        return self.decoded_duration_s or self.declared_duration_s or 0.0

    @property
    def sample_rate(self) -> int:
        return self.decoded_sample_rate or self.declared_sample_rate or 0

    @property
    def channels(self) -> int:
        return self.decoded_channels or self.declared_channels or 0

    @property
    def channel_layout(self) -> str:
        n = self.channels
        return {1: "mono", 2: "stereo"}.get(n, f"{n} channels" if n else "unknown")

    @property
    def bit_depth(self) -> int | None:
        mapping = {
            "PCM_S8": 8,
            "PCM_U8": 8,
            "PCM_16": 16,
            "PCM_24": 24,
            "PCM_32": 32,
            "FLOAT": 32,
            "DOUBLE": 64,
        }
        return mapping.get((self.subtype or "").upper())

    def to_dict(self) -> dict[str, object]:
        data = asdict(self)
        data["duration_s"] = self.duration_s
        data["sample_rate"] = self.sample_rate
        data["channels"] = self.channels
        data["channel_layout"] = self.channel_layout
        data["bit_depth"] = self.bit_depth
        return data


def probe_file(path: Path) -> FileProbe:
    """Read header metadata without decoding the full signal."""
    path = Path(path).expanduser().resolve()
    if not path.exists():
        return FileProbe(
            path=str(path),
            filename=path.name,
            exists=False,
            size_bytes=0,
            warnings=["File does not exist."],
        )

    probe = FileProbe(path=str(path), filename=path.name, exists=True, size_bytes=path.stat().st_size)

    try:
        info = sf.info(str(path))
        probe.container = info.format
        probe.format_long = (info.subtype and f"{info.format}/{info.subtype}") or info.format
        probe.subtype = info.subtype
        probe.codec = info.format
        probe.declared_sample_rate = int(info.samplerate)
        probe.declared_channels = int(info.channels)
        probe.declared_frames = int(info.frames)
        probe.declared_duration_s = float(info.duration) if info.duration else None
    except Exception as exc:
        probe.notes.append(f"Header inspection unavailable ({type(exc).__name__}); container will be probed during decode.")
        probe.container = path.suffix.lower().lstrip(".") or None

    if probe.size_bytes == 0:
        probe.warnings.append("File is zero bytes.")
    if probe.declared_frames == 0:
        probe.warnings.append("Header reports zero frames.")
    return probe


def cross_check(probe: FileProbe, buffer: AudioBuffer) -> FileProbe:
    """Compare header claims against the decoded signal."""
    probe.decoded_sample_rate = buffer.sample_rate
    probe.decoded_channels = buffer.channels
    probe.decoded_duration_s = buffer.duration_s

    if probe.declared_sample_rate and probe.declared_sample_rate != buffer.sample_rate:
        probe.warnings.append(
            f"Header declares {probe.declared_sample_rate} Hz but the decoded signal is {buffer.sample_rate} Hz. "
            "The file may be mislabelled; decoded values are used."
        )
    if probe.declared_channels and probe.declared_channels != buffer.channels:
        probe.warnings.append(
            f"Header declares {probe.declared_channels} channel(s) but the decoded signal has {buffer.channels}."
        )
    return probe


def inspect(path: Path) -> tuple[FileProbe, AudioBuffer]:
    """Probe and decode in one call, returning a cross-checked probe."""
    from ..io.decode import decode_to_canonical

    probe = probe_file(path)
    if not probe.exists:
        raise DecodeError(
            f"No such file: {path}",
            user_message="The selected audio file could not be found.",
            remedy="Re-upload the file and try again.",
        )
    buffer = decode_to_canonical(path)
    return cross_check(probe, buffer), buffer
