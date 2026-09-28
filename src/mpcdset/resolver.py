"""Turns a dataset "source" into a local filesystem path.

Only local paths are implemented today. This module exists as a seam:
readers (:mod:`mpcdset.readers`) and :class:`mpcdset.torch_dataset.PairedFieldDataset`
only ever see a local path, resolved through here -- so a resolver for a
remote source (e.g. downloading a pinned revision of a dataset to a local
cache, then handing back that cache path) can be added later without
touching either of them. See the README for where that would plug in.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Protocol, runtime_checkable


@runtime_checkable
class SourceResolver(Protocol):
    """Resolves a dataset "source" to a local filesystem path."""

    def resolve(self, source: str | os.PathLike) -> Path:
        """Return a local filesystem path for ``source``."""
        ...


class LocalPathResolver:
    """Resolves a source that is already a local path (the only kind supported today)."""

    def resolve(self, source: str | os.PathLike) -> Path:
        return Path(source)


#: The resolver used when none is given explicitly.
DEFAULT_RESOLVER: SourceResolver = LocalPathResolver()


def resolve_source(source: str | os.PathLike, resolver: SourceResolver | None = None) -> Path:
    """Resolve ``source`` to a local path using ``resolver`` (default: :data:`DEFAULT_RESOLVER`)."""
    resolver = resolver if resolver is not None else DEFAULT_RESOLVER
    return resolver.resolve(source)
