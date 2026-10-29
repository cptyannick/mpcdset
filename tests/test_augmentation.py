import numpy as np

from mpcdset import AugmentationSpec, DatasetWriter, open_dataset


def _make(tmp_path, **kw):
    w = DatasetWriter.create(tmp_path / "ds", n_samples=2, in_height=8, in_width=8, shard_size=2, **kw)
    for i in range(2):
        w.write_sample(i, np.zeros((1, 8, 8), "float32"), np.zeros((3, 8, 8), "float32"))
    w.finalize()
    return open_dataset(tmp_path / "ds")


def test_augmentation_roundtrips_and_keeps_user_metadata(tmp_path):
    spec = AugmentationSpec(kind="rigid", shift_px=15.0, rot_deg=2.0, taper_px=15.0, extra={"a": 1})
    reader = _make(tmp_path, user_metadata={"hash": "x"}, augmentation=spec)
    assert reader.augmentation == spec
    assert reader.attrs["user_metadata"]["hash"] == "x"
    assert reader.attrs["format_version"] == 1


def test_plain_dataset_has_no_augmentation(tmp_path):
    assert _make(tmp_path).augmentation is None
