"""Guarantees the format_version=1 on-disk layout stays readable forever.

The fixture under tests/fixtures/v1/dataset/ is a tiny, committed dataset
written once by scripts/make_fixture.py. This test must keep passing even
as the writer or reader implementation changes -- if it doesn't, the
on-disk format has silently changed and format_version must be bumped
instead.
"""

from pathlib import Path

import numpy as np

from mpcdset import open_dataset

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "v1" / "dataset"


def test_v1_fixture_is_still_readable():
    reader = open_dataset(FIXTURE_PATH)

    assert reader.attrs["format_version"] == 1
    n_samples = reader.attrs["n_samples"]
    assert len(reader) == n_samples

    expected_input_shape = (
        reader.attrs["in_channels"],
        reader.attrs["height"],
        reader.attrs["width"],
    )
    expected_target_shape = (
        reader.attrs["out_channels"],
        reader.attrs["height"],
        reader.attrs["width"],
    )
    expected_dtype = np.dtype(reader.attrs["dtype"])

    for i in range(n_samples):
        input_arr, target_arr = reader.get(i)
        assert input_arr.shape == expected_input_shape
        assert target_arr.shape == expected_target_shape
        assert input_arr.dtype == expected_dtype
        assert target_arr.dtype == expected_dtype
