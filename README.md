# TAF: Trust Aware Aggregation for federated learning

Code, configurations and run logs for

> A. Agarwal, *Mitigating Model Poisoning in Federated Learning through Trust
> Aware Aggregation*, revised manuscript SNCS-D-26-03199, SN Computer Science
> (under review).

TAF is a server side aggregation rule. In each round it

1. bounds every update to the lower median norm of the round,
2. admits clients through a majority density cluster gate (HDBSCAN on the
   local models, minimum cluster size floor(N/2)+1),
3. multiplies that gate by an anomaly penalty (robust z scores of layer log
   norms and layer cosines) and a collusion penalty (robust z score of the mean
   similarity to the closest 10% of peers),
4. keeps a reputation per client, an asymmetric moving average of the gate
   (up rate 0.1, down rate 0.5), and
5. takes the mean of the bounded updates weighted by gate times reputation.

It needs no clean data on the server and changes nothing on the client side.

## Layout

```
taf/
  runner.py            one experiment -> one JSON log
  attacks.py           LF, BD, GS, CO (ALIE), AD (adaptive), MI (mimicry),
                       CN (noisy collusion), IM (intermittent)
  defenses/            FedAvg, Krum, Median, TrimmedMean, FLTrust, FLAME,
                       DeepSight, FoolsGold (optionally with norm bound), TAF
  data.py models.py client.py fastclient.py utils.py
configs/experiments.py experiment groups, one per table or figure
scripts/sweep.py       resumable runner for a group
scripts/tables.py      run logs -> tables/all_runs.csv, aggregated.csv, significance.csv
scripts/latex_tables.py  CSV -> the LaTeX tables of the paper
scripts/figures.py     CSV and logs -> the figures of the paper
scripts/overhead.py    server time per round on synthetic updates
tests/                 unit, smoke and theory checks (pytest)
docs/                  implementation notes and experiment lists
results_logs.zip       JSON logs of every run reported in the paper
```

## Install and test

```bash
pip install -r requirements.txt
python -m pytest -q
```

## Reproduce the tables from the logs

```bash
unzip results_logs.zip            # creates results/<group>/*.json
python scripts/tables.py --results results --out tables
python scripts/latex_tables.py tables latex results
python scripts/figures.py --tables tables --out figures
```

## Rerun an experiment group

```bash
python scripts/sweep.py --group main --out results --data-root data --workers 3
```

Groups are defined in `configs/experiments.py` (for example `main`,
`main_deepsight`, `strength`, `rho_sweep`, `noniid`, `ablation`,
`sensitivity`, the `review_*` groups and `r2_fg_clip`). Finished runs are
skipped, so a sweep can be resumed. A seed fixes the data partition, the
malicious set and the initial model, so runs with the same seed are paired
across defences. The paper reports 778 runs; Appendix B of the paper lists
them by experiment. The archive also holds four debugging runs of DeepSight
on the validation split (`deepsight_diag`), which the paper does not report.

Settings used in the paper: MNIST with 50 clients and 50 rounds, CIFAR-10
with 100 clients and 100 rounds, Dirichlet label skew 0.5, 30% attackers,
client learning rate 0.1, one local epoch, batch size 64. The experiments
were run on one NVIDIA A100 GPU.

## Notes on the baselines

FLAME, DeepSight and FoolsGold are reimplementations from the published
descriptions; `docs/BASELINES.md` lists the known differences. FoolsGold
divides its weighted sum by the sum of weights and does not use feature
importance weighting; `FoolsGold(clip=True)` adds TAF's norm bound to every
update and leaves its weights unchanged.

## License

MIT, see `LICENSE`.
