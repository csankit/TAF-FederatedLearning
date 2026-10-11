# Experiments for the second round of feedback (9 Oct 2026)

CIFAR-10, lr 0.1, test split, rho 0.3, kappa 0.5, N = 100, 100 rounds, unless stated.
New code: attack `CN` (coordinated nonidentical collusion), attack `IM`
(intermittent attacker), `FoolsGold(clip=True)` (TAF's lower median norm bound
on every update; weights unchanged), reputation logging (`R_honest_mean`,
`R_mal_mean`, and per client `R`, `M_all`, `w_all` when `log_rep=True`),
`n_mal_part` (attackers among the participants of each round).

| Tier | Group | Runs | Feedback item | Question |
|---|---|---:|---|---|
| 1 | r2_fg_clip | 21 | 1 | Does FoolsGold with TAF's norm bound survive the scaled attacks, and does TAF keep its worst case accuracy lead? |
| 1 | r2_fg_mimic | 30 | 2 | Does FoolsGold's history test catch mimicry, eps 0.1 to 0.5? |
| 1 | r2_intermittent | 24 | 4 | Can attackers that poison one round in four exploit reputation? TAF with and without reputation |
| 2 | r2_noisy_co | 36 | 3 | Collusion with pairwise cosine 0.9 or matched to honest peers |
| 2 | r2_seeds | 48 | 5 | Seeds 5 to 7 (and 3, 4 for FoolsGold) so TAF, FLAME, FoolsGold, FoolsGold+clip have seeds 1 to 7 under BD, CO, AD |
| 3 | r2_partial | 54 | 6 | Participation 25% and 75% |

Total 213 runs. At the A100 rate of the earlier groups (about 2 effective
minutes per run with 3 workers) this is about 7 A100 hours.

Pre registered test family for item 5: TAF minus each of FLAME, FoolsGold and
FoolsGold+clip, under BD (ASR), CO (accuracy) and matched AD (ASR), seeds 1 to
7, paired two sided t tests, Holm over the 9 tests.
