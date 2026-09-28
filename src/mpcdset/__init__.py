"""mpcdset: paired-sample (input/target) dataset storage on Zarr v3.

Core modules (:mod:`mpcdset.writer`, :mod:`mpcdset.readers`,
:mod:`mpcdset.sharding`, :mod:`mpcdset.resolver`) have no torch dependency,
so datasets can be generated in an environment without torch installed. The
torch-dependent :class:`~mpcdset.torch_dataset.PairedFieldDataset` lives in
:mod:`mpcdset.torch_dataset` and requires the ``torch`` extra
(``pip install mpcdset[torch]``); it is intentionally not imported here.
"""

from mpcdset.errors import (
    IncompleteDatasetError,
    MPCDSetError,
    RangeNotAllowedError,
    ShapeMismatchError,
    ShardAlignmentError,
    UnsupportedFormatVersionError,
)
from mpcdset.metadata import FORMAT_VERSION, CodecSpec
from mpcdset.readers import DatasetReader, open_dataset, register_reader
from mpcdset.resolver import DEFAULT_RESOLVER, LocalPathResolver, SourceResolver, resolve_source
from mpcdset.sharding import SampleRange, is_shard_aligned, shard_aligned_ranges
from mpcdset.writer import DatasetWriter

__version__ = "0.1.0"

__all__ = [
    "CodecSpec",
    "DEFAULT_RESOLVER",
    "DatasetReader",
    "DatasetWriter",
    "FORMAT_VERSION",
    "IncompleteDatasetError",
    "LocalPathResolver",
    "MPCDSetError",
    "RangeNotAllowedError",
    "SampleRange",
    "ShapeMismatchError",
    "ShardAlignmentError",
    "SourceResolver",
    "UnsupportedFormatVersionError",
    "__version__",
    "is_shard_aligned",
    "open_dataset",
    "register_reader",
    "resolve_source",
    "shard_aligned_ranges",
]
