import pytest

from mpcdset.sharding import SampleRange, is_shard_aligned, shard_aligned_ranges


def test_covers_full_range_with_no_gaps_or_overlap():
    ranges = shard_aligned_ranges(n_samples=1000, shard_size=64, n_ranges=4)
    covered: list[int] = []
    for r in ranges:
        covered.extend(range(r.start, r.stop))
    assert covered == list(range(1000))


def test_ranges_are_shard_aligned():
    n_samples, shard_size = 1000, 64
    ranges = shard_aligned_ranges(n_samples, shard_size, n_ranges=4)
    for r in ranges:
        assert is_shard_aligned(r.start, r.stop, n_samples, shard_size)


def test_fewer_shards_than_requested_ranges():
    ranges = shard_aligned_ranges(n_samples=10, shard_size=64, n_ranges=8)
    assert ranges == [SampleRange(0, 10)]


def test_exact_multiple_splits_evenly():
    ranges = shard_aligned_ranges(n_samples=256, shard_size=64, n_ranges=4)
    assert ranges == [
        SampleRange(0, 64),
        SampleRange(64, 128),
        SampleRange(128, 192),
        SampleRange(192, 256),
    ]


@pytest.mark.parametrize(
    "n_samples, shard_size, n_ranges",
    [(0, 64, 1), (10, 0, 1), (10, 64, 0), (-1, 64, 1)],
)
def test_invalid_arguments_raise(n_samples, shard_size, n_ranges):
    with pytest.raises(ValueError):
        shard_aligned_ranges(n_samples, shard_size, n_ranges)


def test_is_shard_aligned_end_of_dataset_is_always_aligned():
    # 10 is not a multiple of shard_size=64, but it's the dataset end.
    assert is_shard_aligned(0, 10, n_samples=10, shard_size=64)
    assert not is_shard_aligned(0, 5, n_samples=10, shard_size=64)
