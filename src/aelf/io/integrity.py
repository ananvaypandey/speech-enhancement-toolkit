"""Source-integrity guard.

The brief requires that the original file is preserved and never
overwritten. This module is the single enforcement point: it fingerprints
inputs on read and re-verifies them on completion, and it rejects any write
whose target collides with a known source path.

Deliberately paranoid: hash the bytes, not just the size and mtime, because a
file can be modified in place and preserve both.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable
from pathlib import Path

from ..errors import SourceIntegrityError, SourceMutatedError
from ..types import SourceIntegrity

_HASH_CHUNK = 1 << 20  # 1 MiB


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(_HASH_CHUNK), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fingerprint(path: Path) -> SourceIntegrity:
    """Hash and stat a file. Raises if the path is not a readable file."""
    resolved = Path(path).expanduser().resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"Input file does not exist: {resolved}")
    if not resolved.is_file():
        raise SourceIntegrityError(
            f"Input path is not a regular file: {resolved}",
            user_message="The selected path is not a regular audio file.",
            remedy="Select a single audio file rather than a directory.",
        )
    stat = resolved.stat()
    return SourceIntegrity(
        path=resolved,
        size_bytes=stat.st_size,
        sha256=sha256_file(resolved),
        mtime_ns=stat.st_mtime_ns,
    )


class IntegrityGuard:
    """Tracks protected inputs and refuses writes that would clobber them.

    Usage:
        with IntegrityGuard([upload_path]) as guard:
            process(upload_path)
            guard.assert_targets_safe([enhanced_path, transcript_path])
            guard.verify_unchanged()
    """

    def __init__(self, sources: Iterable[Path] = ()) -> None:
        self._protected: dict[Path, SourceIntegrity] = {}
        for source in sources:
            self.protect(source)

    def protect(self, path: Path) -> SourceIntegrity:
        """Register a path as read-only for the lifetime of this guard."""
        record = fingerprint(path)
        self._protected[record.path] = record
        return record

    @property
    def protected_paths(self) -> tuple[Path, ...]:
        return tuple(self._protected)

    def is_protected(self, path: Path) -> bool:
        return Path(path).expanduser().resolve() in self._protected

    def assert_targets_safe(self, targets: Iterable[Path]) -> None:
        """Raise if any target write would overwrite a protected source.

        Compares resolved paths, so a superficially different path that
        resolves to the same file (symlink, `..`, case change) is still caught.
        """
        for target in targets:
            resolved = Path(target).expanduser().resolve()
            if resolved in self._protected:
                raise SourceIntegrityError(
                    f"Blocked write to protected source: {resolved}",
                    user_message="The operation was stopped to protect your original audio file.",
                    remedy="Outputs are written to the work/outputs directory, which is separate from uploads.",
                )

    def verify_unchanged(self) -> None:
        """Re-hash every protected source. Raises if any input was mutated."""
        for record in self._protected.values():
            if not record.path.exists():
                raise SourceMutatedError(
                    f"Protected source disappeared: {record.path}",
                    user_message="The original audio file was deleted while processing was running.",
                    remedy="Restore the file and retry.",
                )
            current_size = record.path.stat().st_size
            current_hash = sha256_file(record.path)
            if current_size != record.size_bytes or current_hash != record.sha256:
                raise SourceMutatedError(
                    f"Protected source mutated: {record.path} "
                    f"(size {record.size_bytes}->{current_size}, sha {record.sha256[:12]}->{current_hash[:12]})",
                    user_message="The original audio file changed during processing, so results were discarded.",
                    remedy="Do not modify the input while it is processing. Reload the file and retry.",
                )
            object.__setattr__(record, "verified_at_end", True)

    def summary(self) -> list[dict[str, object]]:
        return [
            {
                "path": str(r.path),
                "size_bytes": r.size_bytes,
                "sha256": r.sha256,
                "verified_unchanged": r.verified_at_end,
            }
            for r in self._protected.values()
        ]

    def __enter__(self) -> IntegrityGuard:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        # Only verify on clean exit; a mid-pipeline failure has already
        # surfaced its own error and needs no extra noise.
        if exc_type is None:
            self.verify_unchanged()
        return None
