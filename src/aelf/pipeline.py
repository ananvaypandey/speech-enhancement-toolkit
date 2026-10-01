"""The processing chain, in one place.

Both entry points — the command line and the browser page — used to assemble
this sequence themselves: enhance, optionally even out the loudness, write.
That is a duplication waiting to go wrong. The two callers already disagreed
about which stages ran, and once the filter chain was added the duplication
would have made it possible for the CLI to skip it while the page did not, so
the documented behaviour and the actual behaviour would quietly diverge.

So the chain lives here, once, and both callers ask for a result. What is left
in the callers is presentation: what to print, what to draw.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from .enhance import enhance
from .postprocess.filters import apply_filter_chain
from .postprocess.normalize import normalise_lufs
from .types import AudioBuffer


@dataclass
class PipelineResult:
    """The processed audio plus an honest account of what was done to it."""

    audio: AudioBuffer
    backend: str
    processing_time_s: float
    # One line per stage that actually changed something, for the user to read.
    applied: list[str] = field(default_factory=list)
    # Anything a stage declined to do, and why.
    notes: list[str] = field(default_factory=list)

    @property
    def summary(self) -> str:
        if not self.applied:
            return "nothing needed changing"
        return "; ".join(self.applied)


def process(
    buffer: AudioBuffer,
    *,
    strength: float = 0.5,
    tidy: bool = True,
    target_lufs: float | None = None,
) -> PipelineResult:
    """Clean up a recording: suppress noise, tidy it, optionally even it out.

    `strength` scales noise suppression only. The tidying stages are either on
    or off, because a user who wants the voice left alone asks for that with a
    switch, not by guessing at a strength number that means something else.

    `target_lufs` of None leaves the loudness alone. The value is a target,
    not a promise: a recording that is already quiet will be brought up to it,
    and one that is not speech will land somewhere else, which is why the
    caller reports the measured result rather than the requested one.
    """
    started = time.perf_counter()
    applied: list[str] = []
    notes: list[str] = []

    result = enhance(buffer, strength=strength)
    applied.append(f"background noise suppression at strength {strength:.2f} ({result.backend})")
    notes.extend(result.notes)
    audio = result.audio

    if tidy:
        filtered = apply_filter_chain(audio)
        audio = filtered.audio
        applied.extend(filtered.applied)
        notes.extend(filtered.notes)

    if target_lufs is not None:
        audio = normalise_lufs(audio, target_lufs=target_lufs)
        applied.append(f"loudness evened out to a {target_lufs:.0f} LUFS target")

    return PipelineResult(
        audio=audio,
        backend=result.backend,
        processing_time_s=time.perf_counter() - started,
        applied=applied,
        notes=notes,
    )


__all__ = ["PipelineResult", "process"]
