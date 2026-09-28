import numpy as np
import pytest

from mpcdset import DatasetWriter, open_dataset
from mpcdset.errors import ShapeMismatchError


@pytest.mark.parametrize("dtype", ["float16", "float32", "float64"])
def test_roundtrip_write_sample(tmp_path, rng, dtype):
    n, h, w = 5, 32, 32
    path = tmp_path / "ds"
    writer = DatasetWriter.create(path, n_samples=n, height=h, width=w, dtype=dtype, shard_size=2)

    inputs = rng.standard_normal((n, 1, h, w)).astype(dtype)
    targets = rng.standard_normal((n, 3, h, w)).astype(dtype)
    for i in range(n):
        writer.write_sample(i, inputs[i], targets[i])
    writer.finalize()

    reader = open_dataset(path)
    assert len(reader) == n
    assert reader.attrs["dtype"] == str(np.dtype(dtype))
    assert reader.attrs["finalized"] is True
    for i in range(n):
        got_input, got_target = reader.get(i)
        np.testing.assert_array_equal(got_input, inputs[i])
        np.testing.assert_array_equal(got_target, targets[i])
        assert got_input.dtype == np.dtype(dtype)
        assert got_target.dtype == np.dtype(dtype)


def test_roundtrip_write_batch(tmp_path, rng):
    n, h, w = 6, 16, 16
    path = tmp_path / "ds"
    writer = DatasetWriter.create(path, n_samples=n, height=h, width=w, shard_size=3)
    inputs = rng.standard_normal((n, 1, h, w)).astype("float32")
    targets = rng.standard_normal((n, 3, h, w)).astype("float32")
    writer.write_batch(0, inputs[:3], targets[:3])
    writer.write_batch(3, inputs[3:], targets[3:])
    writer.finalize()

    reader = open_dataset(path)
    for i in range(n):
        got_input, got_target = reader.get(i)
        np.testing.assert_array_equal(got_input, inputs[i])
        np.testing.assert_array_equal(got_target, targets[i])


def test_default_channel_counts(tmp_path, rng):
    path = tmp_path / "ds"
    writer = DatasetWriter.create(path, n_samples=2, height=8, width=8, shard_size=2)
    assert writer.in_channels == 1
    assert writer.out_channels == 3


def test_shape_validation(tmp_path, rng):
    path = tmp_path / "ds"
    writer = DatasetWriter.create(path, n_samples=2, height=8, width=8, shard_size=2)
    with pytest.raises(ShapeMismatchError):
        writer.write_sample(
            0,
            rng.standard_normal((1, 4, 8)).astype("float32"),
            rng.standard_normal((3, 8, 8)).astype("float32"),
        )


def test_dtype_validation(tmp_path, rng):
    path = tmp_path / "ds"
    writer = DatasetWriter.create(path, n_samples=2, height=8, width=8, shard_size=2)
    with pytest.raises(ShapeMismatchError):
        writer.write_sample(
            0,
            rng.standard_normal((1, 8, 8)).astype("float64"),
            rng.standard_normal((3, 8, 8)).astype("float32"),
        )


def test_create_refuses_to_overwrite_by_default(tmp_path, rng):
    path = tmp_path / "ds"
    DatasetWriter.create(path, n_samples=2, height=8, width=8, shard_size=2)
    with pytest.raises(FileExistsError):
        DatasetWriter.create(path, n_samples=2, height=8, width=8, shard_size=2)
    # overwrite=True is allowed
    DatasetWriter.create(path, n_samples=2, height=8, width=8, shard_size=2, overwrite=True)


def test_user_metadata_is_preserved(tmp_path):
    path = tmp_path / "ds"
    DatasetWriter.create(
        path,
        n_samples=1,
        height=4,
        width=4,
        shard_size=1,
        user_metadata={"note": "small fixture"},
    )
    reader = open_dataset(path)
    assert reader.attrs["user_metadata"] == {"note": "small fixture"}
