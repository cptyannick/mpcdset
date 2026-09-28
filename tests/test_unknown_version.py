import zarr
import pytest

from mpcdset import DatasetWriter, open_dataset
from mpcdset.errors import UnsupportedFormatVersionError


def test_unknown_format_version_raises_with_helpful_message(tmp_path):
    path = tmp_path / "ds"
    DatasetWriter.create(path, n_samples=2, in_height=4, in_width=4, shard_size=2)

    group = zarr.open_group(store=str(path), mode="r+")
    group.attrs.update({"format_version": 999})

    with pytest.raises(UnsupportedFormatVersionError) as exc_info:
        open_dataset(path)

    message = str(exc_info.value)
    assert "999" in message
    assert "upgrade" in message.lower()
