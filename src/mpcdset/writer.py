"""Writer for the current (latest) on-disk format.

This module only ever writes ``mpcdset.metadata.FORMAT_VERSION``. It never
reads or writes older formats -- those are handled read-only by
:mod:`mpcdset.readers`.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import zarr
from zarr.codecs import BloscCodec

from mpcdset.errors import (
    IncompleteDatasetError,
    RangeNotAllowedError,
    ShapeMismatchError,
    ShardAlignmentError,
    UnsupportedFormatVersionError,
)
from mpcdset.metadata import FORMAT_VERSION, CodecSpec, build_attrs
from mpcdset.sharding import SampleRange, is_shard_aligned

_INPUT_ARRAY_NAME = "input"
_TARGET_ARRAY_NAME = "target"
_WRITTEN_ARRAY_NAME = "_written"


def _blosc_codec(spec: CodecSpec) -> BloscCodec:
    return BloscCodec(cname=spec.cname, clevel=spec.clevel, shuffle=spec.shuffle)


class DatasetWriter:
    """Writes a dataset in the current on-disk format.

    Call :meth:`create` once, from a single process, to allocate a new
    dataset of known size. Samples can then be written from that same
    writer, or from several independent processes: split the index range
    with :func:`mpcdset.sharding.shard_aligned_ranges` and have each process
    call :meth:`open` with its own ``allowed_range``, so no two processes
    ever write into the same shard.
    """

    def __init__(self, path: Path, group: zarr.Group, allowed_range: SampleRange | None) -> None:
        self._path = Path(path)
        self._group = group
        self._attrs = dict(group.attrs)
        self._input = group[_INPUT_ARRAY_NAME]
        self._target = group[_TARGET_ARRAY_NAME]
        self._written = group[_WRITTEN_ARRAY_NAME]
        self._allowed_range = allowed_range

    # -- construction ---------------------------------------------------

    @classmethod
    def create(
        cls,
        path: str | os.PathLike,
        *,
        n_samples: int,
        in_height: int,
        in_width: int,
        out_height: int | None = None,
        out_width: int | None = None,
        dtype: Any = "float32",
        in_channels: int = 1,
        out_channels: int = 3,
        shard_size: int = 64,
        codec: CodecSpec | None = None,
        user_metadata: dict | None = None,
        overwrite: bool = False,
    ) -> "DatasetWriter":
        """Create a new, empty dataset on disk and return a writer for it.

        No samples are written yet; call :meth:`write_sample` /
        :meth:`write_batch` afterwards (from this writer, or from other
        processes via :meth:`open`).

        Args:
            path: directory to create the dataset in. Must not already
                exist unless ``overwrite=True``.
            n_samples: total number of samples ``N``.
            in_height: input sample height ``H_in``.
            in_width: input sample width ``W_in``.
            out_height: target sample height ``H_out``; defaults to
                ``in_height`` (pass explicitly when the target field is on a
                different-sized grid than the input).
            out_width: target sample width ``W_out``; defaults to
                ``in_width``.
            dtype: numpy dtype for both ``input`` and ``target`` (e.g.
                ``"float32"``, ``"float16"``, ``"float64"``). Fixed for the
                life of the dataset.
            in_channels: number of input channels (default 1).
            out_channels: number of target channels (default 3).
            shard_size: number of samples per Zarr shard (default 64).
            codec: :class:`mpcdset.metadata.CodecSpec` controlling
                compression; defaults to Blosc/zstd with bitshuffle.
            user_metadata: optional free-form dict stored verbatim in the
                group attributes under ``"user_metadata"``.
            overwrite: if ``True``, remove ``path`` first if it exists.
        """
        out_height = in_height if out_height is None else out_height
        out_width = in_width if out_width is None else out_width

        if n_samples <= 0:
            raise ValueError(f"n_samples must be positive, got {n_samples}")
        if in_height <= 0 or in_width <= 0:
            raise ValueError(f"in_height and in_width must be positive, got {in_height}x{in_width}")
        if out_height <= 0 or out_width <= 0:
            raise ValueError(f"out_height and out_width must be positive, got {out_height}x{out_width}")
        if in_channels <= 0 or out_channels <= 0:
            raise ValueError("in_channels and out_channels must be positive")
        if shard_size <= 0:
            raise ValueError(f"shard_size must be positive, got {shard_size}")

        dtype = np.dtype(dtype)
        codec = codec or CodecSpec()
        path = Path(path)

        if path.exists():
            if not overwrite:
                raise FileExistsError(f"{path} already exists; pass overwrite=True to replace it")
            shutil.rmtree(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        store = zarr.storage.LocalStore(str(path))
        group = zarr.open_group(store=store, mode="w", zarr_format=3)

        blosc = _blosc_codec(codec)
        group.create_array(
            name=_INPUT_ARRAY_NAME,
            shape=(n_samples, in_channels, in_height, in_width),
            chunks=(1, in_channels, in_height, in_width),
            shards=(shard_size, in_channels, in_height, in_width),
            dtype=dtype,
            compressors=[blosc],
        )
        group.create_array(
            name=_TARGET_ARRAY_NAME,
            shape=(n_samples, out_channels, out_height, out_width),
            chunks=(1, out_channels, out_height, out_width),
            shards=(shard_size, out_channels, out_height, out_width),
            dtype=dtype,
            compressors=[blosc],
        )
        group.create_array(
            name=_WRITTEN_ARRAY_NAME,
            shape=(n_samples,),
            chunks=(shard_size,),
            dtype=np.bool_,
            fill_value=False,
        )

        attrs = build_attrs(
            n_samples=n_samples,
            in_height=in_height,
            in_width=in_width,
            out_height=out_height,
            out_width=out_width,
            in_channels=in_channels,
            out_channels=out_channels,
            dtype=dtype,
            shard_size=shard_size,
            user_metadata=user_metadata,
        )
        group.attrs.update(attrs)

        return cls(path, group, allowed_range=None)

    @classmethod
    def open(
        cls,
        path: str | os.PathLike,
        *,
        allowed_range: SampleRange | None = None,
    ) -> "DatasetWriter":
        """Reopen an existing dataset (created with :meth:`create`) for writing.

        Args:
            path: the dataset directory.
            allowed_range: restrict this writer to a disjoint, shard-aligned
                slice of the dataset -- required when several processes
                write the same dataset concurrently. Obtain ranges from
                :func:`mpcdset.sharding.shard_aligned_ranges`.

        Raises:
            ShardAlignmentError: if ``allowed_range`` is not aligned to the
                dataset's ``shard_size``.
            UnsupportedFormatVersionError: if the on-disk data was not
                written by this format version.
        """
        path = Path(path)
        store = zarr.storage.LocalStore(str(path))
        group = zarr.open_group(store=store, mode="r+", zarr_format=3)
        attrs = dict(group.attrs)

        format_version = attrs.get("format_version")
        if format_version != FORMAT_VERSION:
            raise UnsupportedFormatVersionError(format_version, [FORMAT_VERSION])

        if allowed_range is not None:
            n_samples = attrs["n_samples"]
            shard_size = attrs["shard_size"]
            if not is_shard_aligned(allowed_range.start, allowed_range.stop, n_samples, shard_size):
                raise ShardAlignmentError(
                    f"allowed_range={allowed_range} is not aligned to shard_size="
                    f"{shard_size} for a dataset of {n_samples} samples; use "
                    "mpcdset.sharding.shard_aligned_ranges to compute safe ranges."
                )

        return cls(path, group, allowed_range=allowed_range)

    # -- metadata ---------------------------------------------------------

    @property
    def path(self) -> Path:
        return self._path

    @property
    def n_samples(self) -> int:
        return int(self._attrs["n_samples"])

    @property
    def shard_size(self) -> int:
        return int(self._attrs["shard_size"])

    @property
    def dtype(self) -> np.dtype:
        return np.dtype(self._attrs["dtype"])

    @property
    def in_channels(self) -> int:
        return int(self._attrs["in_channels"])

    @property
    def out_channels(self) -> int:
        return int(self._attrs["out_channels"])

    @property
    def in_height(self) -> int:
        return int(self._attrs["in_height"])

    @property
    def in_width(self) -> int:
        return int(self._attrs["in_width"])

    @property
    def out_height(self) -> int:
        return int(self._attrs["out_height"])

    @property
    def out_width(self) -> int:
        return int(self._attrs["out_width"])

    @property
    def allowed_range(self) -> SampleRange | None:
        return self._allowed_range

    # -- writing ------------------------------------------------------------

    def write_sample(self, index: int, input: np.ndarray, target: np.ndarray) -> None:
        """Write a single sample at ``index``.

        ``input`` must have shape ``(in_channels, in_height, in_width)`` and
        ``target`` ``(out_channels, out_height, out_width)``, both with this
        dataset's exact dtype (no implicit casting).
        """
        self._check_range_allowed(index, index + 1)
        self._validate_array("input", input, (self.in_channels, self.in_height, self.in_width))
        self._validate_array("target", target, (self.out_channels, self.out_height, self.out_width))
        self._input[index] = input
        self._target[index] = target
        self._written[index] = True

    def write_batch(self, start: int, input: np.ndarray, target: np.ndarray) -> None:
        """Write a contiguous batch of samples starting at ``start``.

        ``input`` must have shape ``(batch, in_channels, in_height, in_width)``
        and ``target`` ``(batch, out_channels, out_height, out_width)``.
        """
        batch = input.shape[0]
        stop = start + batch
        self._check_range_allowed(start, stop)
        self._validate_array("input", input, (batch, self.in_channels, self.in_height, self.in_width))
        self._validate_array("target", target, (batch, self.out_channels, self.out_height, self.out_width))
        self._input[start:stop] = input
        self._target[start:stop] = target
        self._written[start:stop] = True

    # -- completeness -------------------------------------------------------

    def missing_indices(self) -> list[int]:
        """Return the indices of samples that have not been written yet."""
        written = np.asarray(self._written[:])
        return np.flatnonzero(~written).tolist()

    def verify(self) -> None:
        """Raise :class:`~mpcdset.errors.IncompleteDatasetError` if any sample is missing."""
        missing = self.missing_indices()
        if missing:
            raise IncompleteDatasetError(missing, self.n_samples)

    def finalize(self) -> None:
        """Verify completeness, then mark the dataset ``finalized`` in its metadata.

        Raises:
            IncompleteDatasetError: if any sample is still missing.
        """
        self.verify()
        self._group.attrs.update({"finalized": True})
        self._attrs["finalized"] = True

    # -- lifecycle ------------------------------------------------------------

    def close(self) -> None:
        """No-op; present for symmetry/context-manager use. Zarr needs no explicit close."""

    def __enter__(self) -> "DatasetWriter":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    # -- internal -------------------------------------------------------------

    def _check_range_allowed(self, start: int, stop: int) -> None:
        if start < 0 or stop > self.n_samples:
            raise IndexError(
                f"range [{start}, {stop}) is out of bounds for a dataset of {self.n_samples} samples"
            )
        if self._allowed_range is not None and (
            start < self._allowed_range.start or stop > self._allowed_range.stop
        ):
            raise RangeNotAllowedError(
                f"write to [{start}, {stop}) is outside this writer's allowed "
                f"range {self._allowed_range}"
            )

    def _validate_array(self, name: str, array: np.ndarray, expected_shape: tuple[int, ...]) -> None:
        if tuple(array.shape) != expected_shape:
            raise ShapeMismatchError(f"{name} has shape {tuple(array.shape)}, expected {expected_shape}")
        if array.dtype != self.dtype:
            raise ShapeMismatchError(
                f"{name} has dtype {array.dtype}, expected {self.dtype} "
                "(this dataset's dtype is fixed at creation and is never cast)"
            )
