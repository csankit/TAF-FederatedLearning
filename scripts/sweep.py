"""Run an experiment group. Resumable: a run whose JSON already exists is
skipped, so a Colab disconnect only loses the run in progress.

  python scripts/sweep.py --group main --out /content/drive/MyDrive/taf_results
  python scripts/sweep.py --group main --workers 3 --filter dataset=cifar10
  python scripts/sweep.py --group main --dry-run       # count runs only
"""
import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import argparse
import hashlib
import json
import multiprocessing as mp
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from configs.experiments import GROUPS  # noqa: E402


def run_id(cfg):
    key = json.dumps(cfg, sort_keys=True)
    return f"{cfg['dataset']}_{cfg['defense']}_{cfg['attack']}_s{cfg['seed']}_" \
           + hashlib.sha1(key.encode()).hexdigest()[:10]


def _work(args):
    cfg, out_dir, data_root, runtime = args
    from taf.runner import run
    path = Path(out_dir) / f"{run_id(cfg)}.json"
    if path.exists():
        return "skip"
    try:
        # runtime options (trainer, chunk) do not change results, so they are
        # kept out of the run id
        run({**cfg, "data_root": data_root, **runtime}, out_path=str(path), verbose=False)
        (Path(out_dir) / f"{run_id(cfg)}.FAILED.txt").unlink(missing_ok=True)
        import torch
        torch.cuda.empty_cache()
        return "ok"
    except Exception:  # keep the sweep alive; record the failure
        (Path(out_dir) / f"{run_id(cfg)}.FAILED.txt").write_text(traceback.format_exc())
        return "fail"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", required=True, choices=GROUPS)
    ap.add_argument("--out", default=str(ROOT / "results"))
    ap.add_argument("--data-root", default=str(ROOT / "data"))
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--filter", action="append", default=[], help="key=value")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--trainer", default="vmap", choices=["vmap", "seq"])
    ap.add_argument("--chunk", type=int, default=32)
    a = ap.parse_args()

    cfgs = GROUPS[a.group]()
    for f in a.filter:
        k, v = f.split("=", 1)
        cfgs = [c for c in cfgs if str(c.get(k)) == v]
    out = Path(a.out) / a.group
    out.mkdir(parents=True, exist_ok=True)
    todo = [c for c in cfgs if not (out / f"{run_id(c)}.json").exists()]
    if a.limit:
        todo = todo[:a.limit]
    print(f"group={a.group}: {len(cfgs)} runs, {len(cfgs) - len(todo)} done, {len(todo)} to run")
    if a.dry_run or not todo:
        return
    # Download each dataset once, before workers start: parallel workers
    # downloading into the same folder corrupt each other's extraction.
    from taf.data import load_dataset
    for ds in sorted({c["dataset"] for c in todo}):
        if not ds.startswith("fake_"):
            print(f"preparing {ds} ...", flush=True)
            load_dataset(ds, a.data_root, "cpu")
    runtime = {"trainer": a.trainer, "chunk": a.chunk}
    jobs = [(c, str(out), a.data_root, runtime) for c in todo]
    if a.workers > 1:
        ctx = mp.get_context("spawn")
        with ctx.Pool(a.workers) as pool:
            for i, r in enumerate(pool.imap_unordered(_work, jobs), 1):
                print(f"  [{i}/{len(jobs)}] {r}", flush=True)
    else:
        for i, j in enumerate(jobs, 1):
            print(f"  [{i}/{len(jobs)}] {_work(j)}", flush=True)


if __name__ == "__main__":
    main()
