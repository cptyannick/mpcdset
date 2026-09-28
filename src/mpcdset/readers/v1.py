"""Reader for on-disk format_version = 1.

See FORMAT.md at the repository root for the precise on-disk layout this
reads.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from mpcdset.readers import DatasetReader, register_reader


@register_reader(1)
class ReaderV1(DatasetReader):
    """Reads a format_version=1 dataset (a Zarr v3 group with ``input``/``target`` arrays)."""

    def __init__(self, path: str | os.PathLike, group=None) -> None:
        self._path = Path(path)
        if group is None:
            import zarr

            group = zarr.open_group(store=str(self._path), mode="r")
        self._group = group
        self._attrs = dict(group.attrs)
        self._input = group["input"]
        self._target = group["target"]

    def __len__(self) -> int:
        return int(self._attrs["n_samples"])

    def get(self, index: int) -> tuple[np.ndarray, np.ndarray]:
        n = len(self)
        if not -n <= index < n:
            raise IndexError(f"sample index {index} out of range for dataset of length {n}")
        input_arr = np.asarray(self._input[index])
        target_arr = np.asarray(self._target[index])
        return input_arr, target_arr

    @property
    def attrs(self) -> dict:
        return self._attrs
