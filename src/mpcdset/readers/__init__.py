"""Reader registry: dispatches an on-disk dataset to the reader for its format_version.

Adding support for a new on-disk format means adding a new reader class
(decorated with :func:`register_reader`) in this package -- the public
entry point, :func:`open_dataset`, and the :class:`DatasetReader` contract
it returns never change.
"""

from __future__ import annotations

import abc
import os
from pathlib import Path

import numpy as np

from mpcdset.errors import UnsupportedFormatVersionError


class DatasetReader(abc.ABC):
    """Common read contract every format-version reader implements."""

    @abc.abstractmethod
    def __len__(self) -> int:
        """Number of samples in the dataset."""

    @abc.abstractmethod
    def get(self, index: int) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(input, target)`` for ``index`` as numpy arrays."""

    @property
    @abc.abstractmethod
    def attrs(self) -> dict:
        """The dataset's group attributes (metadata), as a plain dict."""


_REGISTRY: dict[int, type[DatasetReader]] = {}


def register_reader(format_version: int):
    """Class decorator registering a :class:`DatasetReader` for ``format_version``."""

    def _decorator(cls: type[DatasetReader]) -> type[DatasetReader]:
        _REGISTRY[format_version] = cls
        return cls

    return _decorator


def open_dataset(path: str | os.PathLike) -> DatasetReader:
    """Open the dataset at ``path`` and dispatch to the reader for its on-disk format_version.

    Raises:
        UnsupportedFormatVersionError: if the dataset's ``format_version``
            has no registered reader in this installed version of mpcdset.
    """
    import zarr

    path = Path(path)
    group = zarr.open_group(store=str(path), mode="r")
    format_version = group.attrs.get("format_version")
    try:
        reader_cls = _REGISTRY[format_version]
    except KeyError:
        raise UnsupportedFormatVersionError(format_version, _REGISTRY.keys()) from None
    return reader_cls(path, group=group)


# Import built-in readers so they self-register via `register_reader`.
from mpcdset.readers import v1 as _v1  # noqa: E402,F401
