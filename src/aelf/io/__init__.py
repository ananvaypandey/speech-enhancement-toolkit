"""Audio input/output. Import `decode.load` for the canonical read path."""

from __future__ import annotations

from .decode import (
    CANONICAL_SR,
    check_supported,
    decode_to_canonical,
    ffmpeg_path,
    load,
    resample,
)
from .encode import (
    assert_writable,
    count_clipped,
    normalise,
    to_int16,
    true_peak_limit,
    write_stems,
    write_wav,
)
from .integrity import IntegrityGuard, fingerprint, sha256_file

__all__ = [
    "CANONICAL_SR",
    "IntegrityGuard",
    "assert_writable",
    "check_supported",
    "count_clipped",
    "decode_to_canonical",
    "ffmpeg_path",
    "fingerprint",
    "load",
    "normalise",
    "resample",
    "sha256_file",
    "to_int16",
    "true_peak_limit",
    "write_stems",
    "write_wav",
]
