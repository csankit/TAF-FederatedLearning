"""Time one federated round of local training with each trainer, on the real
datasets and partitions, and report peak GPU memory. Takes a few minutes.

  python scripts/bench_trainer.py --data-root /content/data
"""
import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import argparse  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from configs.experiments import SCALE  # noqa: E402
from taf.data import dirichlet_partition, load_dataset  # noqa: E402
from taf.fastclient import train_many  # noqa: E402
from taf.models import build_model  # noqa: E402
from taf.utils import get_flat  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="data")
    ap.add_argument("--chunks", type=int, nargs="+", default=[16, 32, 64])
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", torch.cuda.get_device_name(0) if dev == "cuda" else "cpu")
    for ds in ("mnist", "cifar10"):
        train, _ = load_dataset(ds, a.data_root, dev)
        n = SCALE[ds]["n_clients"]
        parts = dirichlet_partition(train.y.cpu().numpy(), n, 0.5, np.random.default_rng(0))
        # the attacked setting trains ~1.6x as many local models (clean + poisoned)
        jobs = [(train.x[torch.as_tensor(p, device=dev)], train.y[torch.as_tensor(p, device=dev)])
                for p in parts]
        model = build_model(ds).to(dev)
        g = get_flat(model)
        configs = [("seq", None)] + [("vmap", c) for c in a.chunks]
        for trainer, chunk in configs:
            try:
                if dev == "cuda":
                    torch.cuda.empty_cache()
                    torch.cuda.reset_peak_memory_stats()
                train_many(model, g, jobs[:4], trainer=trainer, chunk=chunk or 32,
                           gen=torch.Generator().manual_seed(0))          # warm-up
                if dev == "cuda":
                    torch.cuda.synchronize()
                t = time.perf_counter()
                train_many(model, g, jobs, trainer=trainer, chunk=chunk or 32,
                           gen=torch.Generator().manual_seed(1))
                if dev == "cuda":
                    torch.cuda.synchronize()
                sec = time.perf_counter() - t
                mem = torch.cuda.max_memory_allocated() / 2**30 if dev == "cuda" else 0
                rounds = SCALE[ds]["rounds"]
                print(f"{ds:8} {trainer:4} chunk={str(chunk or '-'):>3}  {sec:6.2f} s/round  "
                      f"peak {mem:4.1f} GiB  -> ~{sec * rounds * 1.6 / 60:5.1f} min/run (attacked)")
            except torch.OutOfMemoryError:
                print(f"{ds:8} {trainer:4} chunk={chunk}  OUT OF MEMORY")
        del train


if __name__ == "__main__":
    main()
