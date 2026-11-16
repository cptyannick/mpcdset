"""On-disk group metadata (attributes) for the current format version."""

from __future__ import annotations

import datetime as _dt
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _pkg_version
from typing import Any

import numpy as np

#: The on-disk format this writer produces. Readers are dispatched on this
#: value (see :mod:`mpcdset.readers`); it must be bumped whenever the on-disk
#: layout changes in a way existing readers can't handle.
FORMAT_VERSION = 1

try:
    WRITER_PACKAGE_VERSION = _pkg_version("mpcdset")
except PackageNotFoundError:  # running from a source checkout without install
    WRITER_PACKAGE_VERSION = "0+unknown"


@dataclass(frozen=True)
class CodecSpec:
    """Lossless Blosc codec configuration used for sample data.

    ``shuffle`` is one of ``"noshuffle"``, ``"shuffle"`` (byte shuffle) or
    ``"bitshuffle"``.
    """

    cname: str = "zstd"
    clevel: int = 5
    shuffle: str = "bitshuffle"


@dataclass(frozen=True)
class AugmentationSpec:
    """Provenance of an *input-side* augmentation baked into a dataset.

    Stored under ``user_metadata["augmentation"]`` (purely additive, so no
    ``format_version`` bump: readers that predate it just see an extra
    ``user_metadata`` key). It documents that the samples are not raw draws
    from the base recipe but were transformed -- e.g. a random rigid
    shift/rotation of the input field, with the target re-derived -- so a
    dataset of one augmentation is never mistaken for another.

    ``shift_px``/``rot_deg`` are the max uniform ``+-`` ranges, ``taper_px`` the
    cosine edge-taper width (all in input-grid pixels / degrees); ``kind`` names
    the transform family and ``extra`` carries anything else the producer wants
    to record.
    """

    kind: str = "rigid"
    shift_px: float = 0.0
    rot_deg: float = 0.0
    taper_px: float = 0.0
    extra: dict | None = None

    def to_dict(self) -> dict:
        return {
            "kind": self.kind,
            "shift_px": self.shift_px,
            "rot_deg": self.rot_deg,
            "taper_px": self.taper_px,
            "extra": dict(self.extra or {}),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "AugmentationSpec":
        return cls(
            kind=d.get("kind", "rigid"),
            shift_px=float(d.get("shift_px", 0.0)),
            rot_deg=float(d.get("rot_deg", 0.0)),
            taper_px=float(d.get("taper_px", 0.0)),
            extra=dict(d.get("extra") or {}),
        )


def build_attrs(
    *,
    n_samples: int,
    in_height: int,
    in_width: int,
    out_height: int,
    out_width: int,
    in_channels: int,
    out_channels: int,
    dtype: Any,
    shard_size: int,
    user_metadata: dict | None = None,
) -> dict:
    """Build the group attributes dict written at dataset creation time.

    ``input`` and ``target`` each have their own spatial size (``in_height``/
    ``in_width`` vs. ``out_height``/``out_width``) -- a paired sample need
    not have both fields on the same grid (e.g. one field sampled at a
    different resolution than the other)."""
    return {
        "format_version": FORMAT_VERSION,
        "n_samples": n_samples,
        "in_height": in_height,
        "in_width": in_width,
        "out_height": out_height,
        "out_width": out_width,
        "in_channels": in_channels,
        "out_channels": out_channels,
        "dtype": str(np.dtype(dtype)),
        "shard_size": shard_size,
        "created_at": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "writer_package_version": WRITER_PACKAGE_VERSION,
        "user_metadata": user_metadata or {},
        "finalized": False,
    }
