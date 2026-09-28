"""Exceptions raised by mpcdset."""

from __future__ import annotations

from collections.abc import Iterable


class MPCDSetError(Exception):
    """Base class for all mpcdset errors."""


class UnsupportedFormatVersionError(MPCDSetError):
    """Raised when a dataset's on-disk ``format_version`` has no registered reader."""

    def __init__(self, format_version: object, known_versions: Iterable[int]) -> None:
        self.format_version = format_version
        self.known_versions = sorted(known_versions)
        known = ", ".join(str(v) for v in self.known_versions) or "none"
        super().__init__(
            f"Dataset format_version={format_version!r} is not supported by this "
            f"version of mpcdset (known versions: {known}). Upgrade the mpcdset "
            "package to a version that supports this format."
        )


class IncompleteDatasetError(MPCDSetError):
    """Raised when a dataset is missing one or more samples."""

    def __init__(self, missing_indices: Iterable[int], n_samples: int) -> None:
        self.missing_indices = list(missing_indices)
        self.n_samples = n_samples
        preview = ", ".join(str(i) for i in self.missing_indices[:10])
        more = "" if len(self.missing_indices) <= 10 else f", ... ({len(self.missing_indices)} total)"
        super().__init__(
            f"Dataset is incomplete: {len(self.missing_indices)}/{n_samples} "
            f"samples missing (indices: {preview}{more})."
        )


class ShapeMismatchError(MPCDSetError):
    """Raised when a written sample's shape or dtype does not match the dataset metadata."""


class ShardAlignmentError(MPCDSetError):
    """Raised when a requested write range is not aligned to shard boundaries."""


class RangeNotAllowedError(MPCDSetError):
    """Raised when a write falls outside a writer's ``allowed_range``."""
