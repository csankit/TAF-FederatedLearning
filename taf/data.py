"""Datasets held as device tensors (no DataLoader: much faster on Colab GPUs)."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

MEANS = {"mnist": (0.1307,), "cifar10": (0.4914, 0.4822, 0.4465)}
STDS = {"mnist": (0.3081,), "cifar10": (0.2470, 0.2435, 0.2616)}


@dataclass
class TensorData:
    x: torch.Tensor  # normalised images, N x C x H x W
    y: torch.Tensor  # int64 labels

    def __len__(self):
        return len(self.y)

    def subset(self, idx):
        idx = torch.as_tensor(idx, dtype=torch.long, device=self.y.device)
        return TensorData(self.x[idx], self.y[idx])


def _normalise(x: torch.Tensor, name: str) -> torch.Tensor:
    m = torch.tensor(MEANS[name], device=x.device).view(1, -1, 1, 1)
    s = torch.tensor(STDS[name], device=x.device).view(1, -1, 1, 1)
    return (x - m) / s


def load_dataset(name: str, root: str, device: str):
    """Return (train, test) as normalised device tensors."""
    if name.startswith("fake_"):
        return _fake(name, device)
    import torchvision

    cls = {"mnist": torchvision.datasets.MNIST,
           "cifar10": torchvision.datasets.CIFAR10}[name]
    out = []
    for train in (True, False):
        ds = cls(root, train=train, download=True)
        x = torch.as_tensor(np.asarray(ds.data), dtype=torch.float32) / 255.0
        x = x.unsqueeze(1) if x.dim() == 3 else x.permute(0, 3, 1, 2)
        y = torch.as_tensor(np.asarray(ds.targets), dtype=torch.long)
        out.append(TensorData(_normalise(x.to(device), name), y.to(device)))
    return out[0], out[1]


def _fake(name: str, device: str):
    """Tiny learnable synthetic data for smoke tests (class-dependent mean)."""
    g = torch.Generator().manual_seed(0)
    shape = (1, 28, 28) if name == "fake_mnist" else (3, 32, 32)

    def make(n):
        y = torch.randint(0, 10, (n,), generator=g)
        proto = torch.randn(10, *shape, generator=torch.Generator().manual_seed(1))
        x = proto[y] + 0.8 * torch.randn(n, *shape, generator=g)
        return TensorData(x.to(device), y.to(device))

    return make(1200), make(400)


def dirichlet_partition(labels: np.ndarray, n_clients: int, kappa: float,
                        rng: np.random.Generator, min_size: int = 10):
    """Standard per-class Dirichlet label partition (smaller kappa = more skew)."""
    n_classes = int(labels.max()) + 1
    for _ in range(1000):
        parts = [[] for _ in range(n_clients)]
        for c in range(n_classes):
            idx = np.flatnonzero(labels == c)
            rng.shuffle(idx)
            p = rng.dirichlet(np.full(n_clients, kappa))
            cuts = (np.cumsum(p) * len(idx)).astype(int)[:-1]
            for k, chunk in enumerate(np.split(idx, cuts)):
                parts[k].extend(chunk.tolist())
        if min(len(p) for p in parts) >= min_size:
            return [np.array(p) for p in parts]
    raise RuntimeError("could not satisfy min_size; lower min_size or raise kappa")


def stamp_trigger(x: torch.Tensor, dataset: str) -> torch.Tensor:
    """3x3 white square in the bottom-right corner (in normalised units)."""
    x = x.clone()
    name = dataset.replace("fake_", "")
    white = torch.tensor([(1 - m) / s for m, s in zip(MEANS[name], STDS[name])],
                         device=x.device).view(-1, 1, 1)
    x[:, :, -4:-1, -4:-1] = white
    return x
