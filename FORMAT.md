# On-disk format

This document specifies the on-disk layout precisely enough to write an
independent reader, without depending on this package.

## format_version = 1

A dataset is a single directory holding one **Zarr v3 group** (a plain
directory store, as written by Zarr's `LocalStore` -- i.e. a `zarr.json`
file at the group root plus one subdirectory per array). Any standard Zarr
v3 implementation can open it.

### Group attributes

The group's `zarr.json` has an `attributes` object with these keys:

| key                     | type          | meaning                                                             |
|--------------------------|---------------|-----------------------------------------------------------------------|
| `format_version`         | int           | always `1` for this document                                          |
| `n_samples`               | int           | number of samples, `N`                                                |
| `height`                  | int           | sample height, `H`                                                    |
| `width`                   | int           | sample width, `W`                                                     |
| `in_channels`             | int           | number of input channels, `C_in` (default 1)                          |
| `out_channels`            | int           | number of target channels, `C_out` (default 3)                        |
| `dtype`                   | string        | numpy dtype string for both arrays, e.g. `"float32"`, `"float16"`, `"float64"` |
| `shard_size`              | int           | samples per shard (see below)                                         |
| `created_at`              | string        | ISO 8601 UTC timestamp                                                |
| `writer_package_version`  | string        | version of the mpcdset package that wrote the dataset                 |
| `user_metadata`           | object        | free-form, caller-supplied metadata; may be `{}`                      |
| `finalized`                | bool          | `true` once the writer has verified every sample was written          |

A reader must not assume any key not listed here.

### Arrays

Two arrays live directly under the group:

- **`input`**: shape `(N, C_in, H, W)`, dtype as given by `dtype` above.
- **`target`**: shape `(N, C_out, H, W)`, same dtype.

Both use the Zarr v3 **sharding codec**:

- Inner chunk shape: `(1, C, H, W)` -- exactly one sample per chunk.
- Shard shape: `(shard_size, C, H, W)` -- `shard_size` samples per shard.
  The last shard along the sample axis is partial when `N` is not a
  multiple of `shard_size`; Zarr v3 sharding handles this natively.
- Inside each shard, sample chunks are compressed with a codec chosen at
  write time (by default Blosc/zstd with bit-shuffle). **A reader must
  never assume or hardcode the codec** -- it is fully described in each
  array's own `zarr.json` (`codecs` key), and any spec-compliant Zarr v3
  reader decodes it transparently. This is what makes the compression
  setting configurable at write time without touching this document.

To read sample `i`: read `input[i]` and `target[i]` (either directly via
zarr's indexing, or a shard-aware Zarr v3 client) and get back a `(C, H,
W)` array of the stated dtype. Storage is lossless: decoded bytes are
bit-identical to what was written.

### `_written` array (internal bookkeeping)

A third array, **`_written`**, shape `(N,)`, dtype `bool`, with regular
(non-sharded) chunks of size `shard_size`, records which sample indices
have been written. It is not part of the logical dataset -- a reader
implementing only `input`/`target` access can ignore it entirely. It
exists so the writer can support multiple independent processes writing
disjoint index ranges and later verify that every sample was written (see
the README for why the chunking differs from `input`/`target`: unsharded
chunks are safe to write concurrently from different processes as long as
each process only touches whole, disjoint chunks, since each chunk is then
a separate file).

## Format version policy

- `format_version` is bumped whenever a change would break an existing
  reader (e.g. a different group layout, a new required array, a
  meaning change for an existing attribute).
- Old format versions are never deleted from this package; a reader for
  each historical version is kept and tested against a committed fixture
  dataset forever, so a dataset written years ago stays readable.
- Purely additive changes to `user_metadata` do not require a version
  bump, since `user_metadata` is free-form to begin with.
