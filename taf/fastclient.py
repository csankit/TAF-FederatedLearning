"""Local training for many clients.

Two interchangeable trainers, numerically equivalent for the same batch plan
(tests/test_fastclient.py):

  "seq"  : clients one after another (low memory, launch-bound on GPU)
  "vmap" : clients in parallel with torch.func.vmap. Clients are bucketed by
           data size so a bucket only runs as many steps as its largest
           member, and each step's batches are fetched with one gather.

Every client runs plain SGD over its own shuffled data for ``epochs`` epochs.
The batch plan for all clients is drawn up front from ``gen`` in client
order, so both trainers consume the generator identically.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch.func import functional_call, grad, vmap

from .utils import get_flat, set_flat


def _batch_plan(sizes, epochs, bs, gen):
    """List of (steps_i, bs) index tensors, -1 = padding."""
    plans = []
    for n in sizes:
        idx = []
        for _ in range(epochs):
            perm = torch.randperm(n, generator=gen)
            for s in range(0, n, bs):
                b = perm[s:s + bs]
                idx.append(torch.cat([b, torch.full((bs - len(b),), -1, dtype=torch.long)]))
        plans.append(torch.stack(idx))
    return plans


def _pad_plans(plans):
    steps = max(p.shape[0] for p in plans)
    out = torch.full((len(plans), steps, plans[0].shape[1]), -1, dtype=torch.long)
    for i, p in enumerate(plans):
        out[i, :p.shape[0]] = p
    return out


def _train_seq(model, global_flat, datasets, plans, lr):
    out = []
    for (x, y), plan in zip(datasets, plans):
        set_flat(model, global_flat.clone())
        model.train()
        opt = torch.optim.SGD(model.parameters(), lr=lr)
        for row in plan:
            idx = row[row >= 0].to(y.device)
            opt.zero_grad(set_to_none=True)
            F.cross_entropy(model(x[idx]), y[idx]).backward()
            opt.step()
        out.append(get_flat(model) - global_flat)
    return torch.stack(out)


def _train_vmap(model, global_flat, datasets, plans, lr, chunk):
    names = [n for n, _ in model.named_parameters()]
    shapes = [p.shape for _, p in model.named_parameters()]
    numels = [p.numel() for _, p in model.named_parameters()]
    dev = global_flat.device

    def loss_fn(params, x, y, m):
        out = functional_call(model, params, (x,))
        ce = F.cross_entropy(out, y, reduction="none")
        return (ce * m).sum() / m.sum().clamp(min=1.0)

    g = vmap(grad(loss_fn))
    K_all = len(datasets)
    order = sorted(range(K_all), key=lambda k: -plans[k].shape[0])   # bucket by steps
    result = torch.empty(K_all, global_flat.numel(), device=dev, dtype=global_flat.dtype)
    for c0 in range(0, K_all, chunk):
        members = order[c0:c0 + chunk]
        K = len(members)
        flat = global_flat.unsqueeze(0).repeat(K, 1)
        params, off = {}, 0
        for nme, shp, ne in zip(names, shapes, numels):
            params[nme] = flat[:, off:off + ne].view(K, *shp)
            off += ne
        # one concatenated data tensor per bucket; last row is a dummy for padding
        xs = [datasets[k][0] for k in members]
        ys = [datasets[k][1] for k in members]
        X_all = torch.cat(xs + [torch.zeros_like(xs[0][:1])])
        Y_all = torch.cat(ys + [torch.zeros_like(ys[0][:1])])
        dummy = X_all.shape[0] - 1
        offs = torch.tensor([0] + [len(y) for y in ys], device=dev).cumsum(0)[:-1]
        P = _pad_plans([plans[k] for k in members]).to(dev)        # (K, steps, bs)
        for s in range(P.shape[1]):
            idx = P[:, s]
            valid = idx >= 0
            gidx = torch.where(valid, idx + offs[:, None], torch.full_like(idx, dummy))
            grads = g(params, X_all[gidx], Y_all[gidx], valid.float())
            with torch.no_grad():
                for nme in names:
                    params[nme].sub_(lr * grads[nme])
        result[torch.as_tensor(members, device=dev)] = flat - global_flat
    return result


def train_many(model, global_flat, datasets, *, epochs=1, lr=0.01, bs=64, gen=None,
               chunk=32, trainer="vmap"):
    """datasets: list of (x, y) device tensors. Returns (K, d) updates."""
    plans = _batch_plan([len(y) for _, y in datasets], epochs, bs, gen)
    if trainer == "seq":
        return _train_seq(model, global_flat, datasets, plans, lr)
    return _train_vmap(model, global_flat, datasets, plans, lr, chunk)
