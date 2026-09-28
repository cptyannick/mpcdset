#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "numpy>=1.26",
#     "numcodecs>=0.13",
#     "mpcdset",
# ]
#
# [tool.uv.sources]
# mpcdset = { path = "..", editable = true }
# ///
"""Compare lossless codec settings for encoding dataset samples.

Given a path to an existing mpcdset dataset, benchmarks a handful of lossless
codecs on a few of its samples. With no path, benchmarks the same codecs on
random float32 samples instead. Reports compression ratio and decode
throughput for each candidate.

Usage:
    uv run scripts/bench_codecs.py [PATH] [--n-samples N]
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass

import numpy as np
from numcodecs.abc import Codec


@dataclass
class CodecCase:
    name: str
    codec: Codec


def build_candidates() -> list[CodecCase]:
    from numcodecs import Blosc

    cases = []
    for shuffle_name, shuffle in [
        ("bitshuffle", Blosc.BITSHUFFLE),
        ("byteshuffle", Blosc.SHUFFLE),
        ("noshuffle", Blosc.NOSHUFFLE),
    ]:
        for clevel in (1, 5, 9):
            cases.append(
                CodecCase(
                    name=f"blosc-zstd-{shuffle_name}-{clevel}",
                    codec=Blosc(cname="zstd", clevel=clevel, shuffle=shuffle),
                )
            )
    for clevel in (1, 5, 9):
        cases.append(
            CodecCase(
                name=f"blosc-lz4-bitshuffle-{clevel}",
                codec=Blosc(cname="lz4", clevel=clevel, shuffle=Blosc.BITSHUFFLE),
            )
        )

    # A couple of other lossless codecs, if this numcodecs build has them.
    try:
        from numcodecs import Zstd

        cases.append(CodecCase(name="zstd-plain-5", codec=Zstd(level=5)))
    except ImportError:
        pass
    try:
        from numcodecs import LZ4

        cases.append(CodecCase(name="lz4-plain", codec=LZ4()))
    except ImportError:
        pass
    try:
        from numcodecs import Zlib

        cases.append(CodecCase(name="zlib-plain-6", codec=Zlib(level=6)))
    except ImportError:
        pass

    return cases


def load_samples(path: str | None, n_samples: int) -> list[np.ndarray]:
    if path is None:
        rng = np.random.default_rng(0)
        return [rng.standard_normal((1, 512, 512)).astype("float32") for _ in range(n_samples)]

    from mpcdset import open_dataset

    reader = open_dataset(path)
    n = min(n_samples, len(reader))
    samples: list[np.ndarray] = []
    for i in range(n):
        input_arr, target_arr = reader.get(i)
        samples.append(input_arr)
        samples.append(target_arr)
    return samples


def bench(codec: Codec, samples: list[np.ndarray]) -> tuple[float, float]:
    raw_bytes = sum(s.nbytes for s in samples)
    encoded = [codec.encode(s) for s in samples]
    encoded_bytes = sum(len(e) for e in encoded)
    ratio = raw_bytes / encoded_bytes if encoded_bytes else float("nan")

    start = time.perf_counter()
    for s, e in zip(samples, encoded):
        codec.decode(e)
    elapsed = time.perf_counter() - start
    throughput = (raw_bytes / (1024**2)) / elapsed if elapsed > 0 else float("inf")
    return ratio, throughput


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", nargs="?", default=None, help="mpcdset dataset directory (optional)")
    parser.add_argument("--n-samples", type=int, default=4)
    args = parser.parse_args()

    samples = load_samples(args.path, args.n_samples)
    print(f"Benchmarking on {len(samples)} arrays, shape {samples[0].shape}, dtype {samples[0].dtype}\n")

    header = f"{'codec':<28}{'ratio':>10}{'decode MB/s':>16}"
    print(header)
    print("-" * len(header))
    for case in build_candidates():
        ratio, throughput = bench(case.codec, samples)
        print(f"{case.name:<28}{ratio:>10.2f}{throughput:>16.1f}")


if __name__ == "__main__":
    main()
