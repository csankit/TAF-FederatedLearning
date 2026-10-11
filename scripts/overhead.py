"""Section 6.7: server-side aggregation time per round, every defence.

Times ``defense.aggregate`` on synthetic updates of the real model dimension,
so the measurement isolates aggregation cost from local training. FLTrust's
root-set training step is NOT included here (a stub returns a fixed vector);
the per-round ``agg_time_s`` logged by the runner does include it.

  python scripts/overhead.py --dataset cifar10 --n 50 100 200 --reps 10
"""
import argparse
import json
import platform
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from taf.defenses import DEFENSES, build_defense  # noqa: E402
from taf.models import build_model  # noqa: E402
from taf.utils import get_flat, layer_slices, set_flat  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="cifar10")
    ap.add_argument("--n", type=int, nargs="+", default=[50, 100, 200])
    ap.add_argument("--reps", type=int, default=10)
    ap.add_argument("--out", default="results/overhead.json")
    ap.add_argument("--defenses", nargs="+", default=list(DEFENSES))
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_model(a.dataset).to(dev)
    g = get_flat(model)
    shape = (3, 32, 32) if "cifar" in a.dataset else (1, 28, 28)

    def softmax(flat, x):
        set_flat(model, flat)
        with torch.no_grad():
            return F.softmax(model(x), 1)

    rows = []
    for n in a.n:
        torch.manual_seed(0)
        U = torch.randn(n, g.numel(), device=dev) * 1e-2
        for name in a.defenses:
            kw = {"TAF": dict(mode="v3"), "FLAME": dict(impl=2)}.get(name, {})
            d = build_defense(name, n, **kw)
            ctx = {"rho": 0.3, "slices": layer_slices(model), "round": 0, "global_flat": g,
                   "num_classes": 10, "input_shape": shape, "softmax": softmax,
                   "server_update": lambda: U[0].clone()}
            times = []
            for r in range(a.reps + 1):
                ctx["round"] = r
                if dev == "cuda":
                    torch.cuda.synchronize()
                t = time.perf_counter()
                d.aggregate(U, ctx)
                if dev == "cuda":
                    torch.cuda.synchronize()
                if r:  # drop warm-up
                    times.append(time.perf_counter() - t)
            rows.append(dict(defense=name, n=n, mean_s=sum(times) / len(times),
                             min_s=min(times), max_s=max(times)))
            print(f"N={n:4d} {name:12s} {rows[-1]['mean_s']*1e3:9.1f} ms", flush=True)
    meta = dict(device=dev, gpu=torch.cuda.get_device_name(0) if dev == "cuda" else None,
                cpu=platform.processor(), torch=torch.__version__, d=g.numel())
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(dict(meta=meta, rows=rows), indent=1))


if __name__ == "__main__":
    main()
