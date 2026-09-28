import multiprocessing as mp

import numpy as np
import pytest

from mpcdset import DatasetWriter, open_dataset
from mpcdset.errors import IncompleteDatasetError, RangeNotAllowedError, ShardAlignmentError
from mpcdset.sharding import SampleRange, shard_aligned_ranges


def _write_range(path: str, start: int, stop: int, seed: int) -> None:
    rng = np.random.default_rng(seed)
    writer = DatasetWriter.open(path, allowed_range=SampleRange(start, stop))
    h, w = writer.in_height, writer.in_width
    for i in range(start, stop):
        input_arr = rng.standard_normal((writer.in_channels, h, w)).astype(writer.dtype)
        target_arr = rng.standard_normal((writer.out_channels, h, w)).astype(writer.dtype)
        writer.write_sample(i, input_arr, target_arr)


def test_multiprocess_disjoint_shard_aligned_writes(tmp_path):
    n, h, w, shard_size = 16, 8, 8, 4
    path = tmp_path / "ds"
    DatasetWriter.create(path, n_samples=n, in_height=h, in_width=w, shard_size=shard_size)

    ranges = shard_aligned_ranges(n, shard_size, n_ranges=4)
    assert len(ranges) == 4

    ctx = mp.get_context("fork")
    procs = [
        ctx.Process(target=_write_range, args=(str(path), r.start, r.stop, i))
        for i, r in enumerate(ranges)
    ]
    for p in procs:
        p.start()
    for p in procs:
        p.join()
        assert p.exitcode == 0

    writer = DatasetWriter.open(path)
    writer.verify()  # must not raise: every sample was written

    reader = open_dataset(path)
    assert len(reader) == n
    for i in range(n):
        input_arr, target_arr = reader.get(i)
        assert input_arr.shape == (1, h, w)
        assert target_arr.shape == (3, h, w)


def test_verify_detects_missing_samples(tmp_path):
    n, h, w = 4, 8, 8
    path = tmp_path / "ds"
    writer = DatasetWriter.create(path, n_samples=n, in_height=h, in_width=w, shard_size=2)
    writer.write_sample(
        0, np.zeros((1, h, w), dtype="float32"), np.zeros((3, h, w), dtype="float32")
    )
    assert writer.missing_indices() == [1, 2, 3]
    with pytest.raises(IncompleteDatasetError):
        writer.verify()
    with pytest.raises(IncompleteDatasetError):
        writer.finalize()


def test_unaligned_allowed_range_rejected_at_open(tmp_path):
    n, h, w, shard_size = 16, 8, 8, 4
    path = tmp_path / "ds"
    DatasetWriter.create(path, n_samples=n, in_height=h, in_width=w, shard_size=shard_size)
    with pytest.raises(ShardAlignmentError):
        DatasetWriter.open(path, allowed_range=SampleRange(1, 5))


def test_write_outside_allowed_range_rejected(tmp_path):
    n, h, w, shard_size = 16, 8, 8, 4
    path = tmp_path / "ds"
    DatasetWriter.create(path, n_samples=n, in_height=h, in_width=w, shard_size=shard_size)
    writer = DatasetWriter.open(path, allowed_range=SampleRange(0, 4))
    with pytest.raises(RangeNotAllowedError):
        writer.write_sample(
            5, np.zeros((1, h, w), dtype="float32"), np.zeros((3, h, w), dtype="float32")
        )
