"""Public PyTorch Dataset. Requires the ``torch`` extra (``pip install mpcdset[torch]``).

Kept out of ``mpcdset``'s core modules (writer, readers, sharding, resolver)
so a data-generation environment can depend on mpcdset without installing
torch at all.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from typing import Optional

import torch
from torch.utils.data import Dataset

from mpcdset.readers import DatasetReader, open_dataset
from mpcdset.resolver import SourceResolver, resolve_source

#: `(input, target) -> (input, target)`, applied to a sample's torch tensors.
Transform = Callable[[torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]]


class PairedFieldDataset(Dataset):
    """A ``torch.utils.data.Dataset`` over an mpcdset dataset.

    ``path`` is resolved to a local path (through ``resolver``) and opened
    lazily, the first time a sample is actually requested, in whichever
    process/worker makes that request -- so this works correctly with
    ``DataLoader(num_workers > 0)`` under both the "fork" and "spawn" start
    methods.

    Args:
        path: local path to a dataset directory (or any source accepted by
            ``resolver``).
        transform: optional callable applied to each sample's
            ``(input, target)`` tensors, returning the (possibly modified)
            pair.
        resolver: optional :class:`~mpcdset.resolver.SourceResolver`;
            defaults to resolving local paths as-is.
    """

    def __init__(
        self,
        path: str | os.PathLike,
        transform: Optional[Transform] = None,
        resolver: Optional[SourceResolver] = None,
    ) -> None:
        self._source = path
        self._resolver = resolver
        self.transform = transform
        self._reader: Optional[DatasetReader] = None
        self._reader_pid: Optional[int] = None

    def _get_reader(self) -> DatasetReader:
        pid = os.getpid()
        if self._reader is None or self._reader_pid != pid:
            local_path = resolve_source(self._source, self._resolver)
            self._reader = open_dataset(local_path)
            self._reader_pid = pid
        return self._reader

    def __len__(self) -> int:
        return len(self._get_reader())

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        input_arr, target_arr = self._get_reader().get(index)
        input_t = torch.from_numpy(input_arr)
        target_t = torch.from_numpy(target_arr)
        if self.transform is not None:
            input_t, target_t = self.transform(input_t, target_t)
        return input_t, target_t

    def __getstate__(self) -> dict:
        # Drop the open reader when pickled (e.g. for DataLoader's "spawn"
        # start method) so each destination process opens its own handles.
        state = self.__dict__.copy()
        state["_reader"] = None
        state["_reader_pid"] = None
        return state
