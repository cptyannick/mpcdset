#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "numpy>=1.26",
#     "mpcdset",
# ]
#
# [tool.uv.sources]
# mpcdset = { path = "..", editable = true }
# ///
"""(re)generate the committed format_version=1 test fixture.

Run this once whenever the fixture needs to be (re)created -- normally that
should be *never* after the initial commit, since tests/test_fixture_v1.py
exists precisely to guarantee this exact on-disk data stays readable by
every future version of mpcdset. Only run this again if you are
deliberately introducing a new fixture for a new format_version (leave the
old one alone).

Usage:
    uv run scripts/make_fixture.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from mpcdset import DatasetWriter

FIXTURE_PATH = Path(__file__).parent.parent / "tests" / "fixtures" / "v1" / "dataset"


def main() -> None:
    n_samples, in_height, in_width, out_height, out_width = 3, 8, 8, 6, 6
    rng = np.random.default_rng(1234)

    writer = DatasetWriter.create(
        FIXTURE_PATH,
        n_samples=n_samples,
        in_height=in_height,
        in_width=in_width,
        out_height=out_height,
        out_width=out_width,
        dtype="float32",
        shard_size=2,
        user_metadata={"purpose": "committed test fixture, do not regenerate casually"},
        overwrite=True,
    )
    for i in range(n_samples):
        input_arr = rng.standard_normal((writer.in_channels, in_height, in_width)).astype("float32")
        target_arr = rng.standard_normal((writer.out_channels, out_height, out_width)).astype("float32")
        writer.write_sample(i, input_arr, target_arr)
    writer.finalize()

    print(f"Wrote fixture dataset to {FIXTURE_PATH}")


if __name__ == "__main__":
    main()
