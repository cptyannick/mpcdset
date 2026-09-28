"""Helpers for splitting a dataset's samples into shard-aligned ranges.

Zarr's sharding codec packs many chunks into one shard file. If two
processes write into the same shard concurrently, the shard file can be
corrupted (writing it is a read-modify-write of the whole file, not an
atomic per-chunk operation). ``shard_aligned_ranges`` splits ``[0,
n_samples)`` into disjoint ranges whose boundaries always fall on shard
boundaries (except possibly the final one, which may be shorter), so that
independent writer processes can each own a range without ever touching
the same shard.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SampleRange:
    """A half-open range of sample indices: ``[start, stop)``."""

    start: int
    stop: int

    def __len__(self) -> int:
        return self.stop - self.start

    def __iter__(self):
        return iter(range(self.start, self.stop))


def shard_aligned_ranges(n_samples: int, shard_size: int, n_ranges: int) -> list[SampleRange]:
    """Split ``[0, n_samples)`` into up to ``n_ranges`` disjoint, shard-aligned ranges.

    Every range boundary (other than ``0`` and ``n_samples``) is a multiple
    of ``shard_size``, so no range ever splits a shard between two writers.
    If there are fewer whole shards than ``n_ranges``, fewer than
    ``n_ranges`` ranges are returned (each non-empty).

    Args:
        n_samples: total number of samples in the dataset.
        shard_size: number of samples per shard (must match the value the
            dataset was created with).
        n_ranges: desired number of ranges (e.g. number of worker processes).

    Returns:
        A list of :class:`SampleRange`, covering ``[0, n_samples)`` with no
        gaps or overlaps, in increasing order.

    Raises:
        ValueError: if ``n_samples``, ``shard_size`` or ``n_ranges`` is not positive.
    """
    if n_samples <= 0:
        raise ValueError(f"n_samples must be positive, got {n_samples}")
    if shard_size <= 0:
        raise ValueError(f"shard_size must be positive, got {shard_size}")
    if n_ranges <= 0:
        raise ValueError(f"n_ranges must be positive, got {n_ranges}")

    n_shards = -(-n_samples // shard_size)  # ceil division
    n_ranges = min(n_ranges, n_shards)

    base = n_shards // n_ranges
    extra = n_shards % n_ranges

    ranges: list[SampleRange] = []
    shard_cursor = 0
    for i in range(n_ranges):
        n_shards_for_range = base + (1 if i < extra else 0)
        start_shard = shard_cursor
        stop_shard = shard_cursor + n_shards_for_range
        shard_cursor = stop_shard

        start = start_shard * shard_size
        stop = min(stop_shard * shard_size, n_samples)
        if start < stop:
            ranges.append(SampleRange(start, stop))

    return ranges


def is_shard_aligned(start: int, stop: int, n_samples: int, shard_size: int) -> bool:
    """Return ``True`` if ``[start, stop)`` starts and ends on shard boundaries.

    The end of the dataset (``stop == n_samples``) always counts as
    aligned, even when ``n_samples`` is not a multiple of ``shard_size``
    (the final shard is then simply partial).
    """
    if start % shard_size != 0:
        return False
    if stop != n_samples and stop % shard_size != 0:
        return False
    return True
