import numpy as np
import pytest

torch = pytest.importorskip("torch")
from torch.utils.data import DataLoader  # noqa: E402

from mpcdset import DatasetWriter  # noqa: E402
from mpcdset.torch_dataset import PairedFieldDataset  # noqa: E402


def _make_dataset(tmp_path, n=8, h=8, w=8, shard_size=4):
    path = tmp_path / "ds"
    writer = DatasetWriter.create(path, n_samples=n, in_height=h, in_width=w, shard_size=shard_size)
    rng = np.random.default_rng(1)
    inputs = rng.standard_normal((n, 1, h, w)).astype("float32")
    targets = rng.standard_normal((n, 3, h, w)).astype("float32")
    for i in range(n):
        writer.write_sample(i, inputs[i], targets[i])
    writer.finalize()
    return path, inputs, targets


def test_dataloader_num_workers_returns_correct_data_in_order(tmp_path):
    path, inputs, targets = _make_dataset(tmp_path)
    dataset = PairedFieldDataset(path)
    loader = DataLoader(dataset, batch_size=1, num_workers=2, shuffle=False)

    seen = []
    for input_t, target_t in loader:
        seen.append((input_t[0].numpy(), target_t[0].numpy()))

    assert len(seen) == len(inputs)
    for i in range(len(inputs)):
        np.testing.assert_array_equal(seen[i][0], inputs[i])
        np.testing.assert_array_equal(seen[i][1], targets[i])


def test_transform_is_applied(tmp_path):
    _make_dataset_result = _make_dataset(tmp_path, n=2)
    path = _make_dataset_result[0]

    def transform(input_t, target_t):
        return input_t * 0 + 1, target_t

    dataset = PairedFieldDataset(path, transform=transform)
    input_t, _target_t = dataset[0]
    assert torch.all(input_t == 1)


def test_returns_torch_tensors(tmp_path):
    path, _inputs, _targets = _make_dataset(tmp_path, n=1)
    dataset = PairedFieldDataset(path)
    input_t, target_t = dataset[0]
    assert isinstance(input_t, torch.Tensor)
    assert isinstance(target_t, torch.Tensor)
