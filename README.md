# mpcdset

Storage and loading for datasets of paired samples: an `input` array of
shape `(C_in, H, W)` and a `target` array of shape `(C_out, H, W)`, for
each of `N` samples in a dataset. Backed by [Zarr v3](https://zarr.dev/)
with sharding, so a dataset of a few hundred thousand samples stays a
handful of files on disk instead of one file per sample.

See [`FORMAT.md`](./FORMAT.md) for the precise on-disk layout.

## Install

Requires Python >= 3.11. With [uv](https://docs.astral.sh/uv/):

```sh
uv add mpcdset
# or, for the PyTorch Dataset too:
uv add "mpcdset[torch]"
```

The core package (writing and reading datasets) does not depend on
PyTorch. `PairedFieldDataset` (see below) needs the `torch` extra.

## Writing a dataset

```python
from mpcdset import DatasetWriter

writer = DatasetWriter.create(
    "/path/to/dataset",
    n_samples=1000,
    in_height=2048,
    in_width=2048,
    out_height=2048,        # defaults to in_height/in_width when omitted --
    out_width=2048,         # pass explicitly if target is on a different grid
    dtype="float32",       # also supports float16, float64
    in_channels=1,          # default
    out_channels=3,          # default
    shard_size=64,           # samples per Zarr shard, default 64
)

for i in range(1000):
    writer.write_sample(i, input_array, target_array)   # shapes (1, H_in, W_in), (3, H_out, W_out)

writer.finalize()  # raises IncompleteDatasetError if any sample is missing
```

`write_batch(start, input_batch, target_batch)` writes a contiguous batch
at once (shapes `(B, C_in, H_in, W_in)` / `(B, C_out, H_out, W_out)`).

### Writing from several processes

Generation is often split across independent processes. Because Zarr
shards pack several samples into one file, two processes must never write
into the same shard concurrently. Use
`mpcdset.sharding.shard_aligned_ranges` to split the dataset into disjoint,
shard-aligned ranges, one per process, and open each writer with its
`allowed_range` so any accidental out-of-range write raises immediately
instead of silently racing:

```python
from mpcdset import DatasetWriter
from mpcdset.sharding import shard_aligned_ranges

# from a coordinating process, once, after DatasetWriter.create(...):
ranges = shard_aligned_ranges(n_samples=1000, shard_size=64, n_ranges=8)

# in each worker process (given its own `r = ranges[worker_id]`):
writer = DatasetWriter.open("/path/to/dataset", allowed_range=r)
for i in r:
    writer.write_sample(i, ...)

# after all workers finish, from any process:
DatasetWriter.open("/path/to/dataset").verify()
```

## Reading a dataset

```python
from mpcdset import open_dataset

reader = open_dataset("/path/to/dataset")
len(reader)               # -> N
input, target = reader.get(0)   # numpy arrays, (C_in, H, W) / (C_out, H, W)
reader.attrs               # dict of the group's stored metadata
```

`open_dataset` dispatches on the dataset's stored `format_version` to the
matching reader. An unrecognized version raises
`UnsupportedFormatVersionError` telling you to upgrade the package, rather
than failing in some more confusing way.

## PyTorch Dataset

Requires the `torch` extra.

```python
from mpcdset.torch_dataset import PairedFieldDataset
from torch.utils.data import DataLoader

dataset = PairedFieldDataset("/path/to/dataset", transform=None)
loader = DataLoader(dataset, batch_size=8, num_workers=4, shuffle=True)
```

The underlying Zarr group is opened lazily, on first access, in whichever
process/worker requests it -- so `num_workers > 0` works correctly under
both the "fork" and "spawn" multiprocessing start methods. `transform`, if
given, is `(input_tensor, target_tensor) -> (input_tensor, target_tensor)`.

## Format-version policy

The on-disk format is versioned (`format_version` in the dataset's
metadata; see `FORMAT.md`). The writer always writes the current version.
Readers are a registry keyed by `format_version`: adding support for a new
version means adding a new reader class, never touching the public API.
Old formats are never dropped -- each is tested forever against a tiny
committed fixture dataset (`tests/fixtures/v1/...`), which is what
guarantees a dataset written today stays readable by every future version
of this package.

## Source resolution (extension point)

Dataset readers and `PairedFieldDataset` never open a path directly --
they go through `mpcdset.resolver.resolve_source`, which turns a "source"
into a local path via a `SourceResolver`. Today only
`LocalPathResolver` (the default) exists: it just returns the given path
unchanged. This seam is where a future resolver for a remote source (e.g.
downloading a pinned revision of a dataset from a hub to a local cache,
then returning that cache path) would plug in, without any change to the
readers or to `PairedFieldDataset` -- both only ever see the resolved
local path. No such resolver exists yet, and this package has no
dependency on any remote-hosting client.

## Development

```sh
uv sync --all-extras
uv run pytest
```

`scripts/bench_codecs.py` and `scripts/make_fixture.py` are standalone,
PEP 723 scripts (`uv run scripts/bench_codecs.py [path]`), each declaring
its own dependencies inline. `scripts/build.sh` runs the test suite and
builds an sdist + wheel into `dist/` (`uv build` under the hood; pass
`--no-test` to skip the suite).
