"""Local training. Returns the update u_i = w_i - w (flat vector)."""
from __future__ import annotations

import torch
import torch.nn.functional as F

from .utils import get_flat, set_flat


def local_update(model, global_flat, x, y, *, epochs=1, lr=0.01, bs=64,
                 momentum=0.0, gen=None):
    set_flat(model, global_flat.clone())
    model.train()
    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    n = len(y)
    for _ in range(epochs):
        perm = torch.randperm(n, generator=gen, device="cpu").to(y.device)
        for s in range(0, n, bs):
            b = perm[s:s + bs]
            opt.zero_grad(set_to_none=True)
            F.cross_entropy(model(x[b]), y[b]).backward()
            opt.step()
    return get_flat(model) - global_flat


@torch.no_grad()
def evaluate(model, flat, data, bs=2048):
    set_flat(model, flat)
    model.eval()
    correct = 0
    for s in range(0, len(data), bs):
        correct += (model(data.x[s:s + bs]).argmax(1) == data.y[s:s + bs]).sum().item()
    return correct / max(1, len(data))


@torch.no_grad()
def predict(model, flat, x, bs=2048):
    set_flat(model, flat)
    model.eval()
    return torch.cat([model(x[s:s + bs]).argmax(1) for s in range(0, len(x), bs)])
