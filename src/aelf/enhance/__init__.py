"""AELF enhancement backends.

`spectral` is always available and needs nothing downloaded. A neural backend
can be registered here later; `best_available()` prefers the best one present
and reports which was used and whether it fell back.
"""

from __future__ import annotations

from ..types import AudioBuffer, EnhancementResult
from .spectral import SpectralConfig, available_backends, suppress_stationary_noise


def enhance(buffer: AudioBuffer, *, strength: float = 0.5) -> EnhancementResult:
    """Enhance with the best backend currently available."""
    return suppress_stationary_noise(buffer, strength=strength)


__all__ = [
    "SpectralConfig",
    "available_backends",
    "enhance",
    "suppress_stationary_noise",
]
