# Compute budget for the main experiments (frozen 6 Oct 2026)

Colab Pro, A100-80GB, 3 workers, vmap chunk 16: about 2 effective minutes per run
(pilot 3/4 wall times ~5.5 min per run, three in parallel; DeepSight ~8 min).
Balance on 6 Oct 2026 after pilot 4: 161.83 units at ~6.77 units/hour (~23.9 A100-hours).

| Group           | Runs | Est. A100-h |
|-----------------|-----:|------------:|
| deepsight_diag  |    4 |        0.2  |
| main            |  252 |        8.4  |
| main_deepsight  |   36 |        1.6  |
| strength        |   84 |        2.8  |
| rho_sweep       |   72 |        2.4  |
| noniid          |   72 |        2.4  |
| ablation        |   63 |        2.1  |
| sensitivity     |   24 |        0.8  |
| **total**       |  607 |     ~20.7   |

Estimates; Krum and FLTrust have not been timed on the A100. If the balance runs short,
run `main --filter dataset=mnist` on the second account.
