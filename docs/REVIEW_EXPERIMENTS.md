# Experiments added after the internal review (8 Oct 2026)

All CIFAR-10 unless stated, lr 0.1, test split, rho 0.3, kappa 0.5.

| Group | Runs | Question it answers |
|---|---:|---|
| review_ablation | 18 | Are the anomaly and collusion penalties needed when both are removed? MNIST collusion, where FLAME failed |
| review_mimic | 12 | Mimicry at eps 0.15, between 0.1 and 0.2 |
| review_thresholds | 36 | Anomaly threshold (2, 4), collusion threshold (1, 3), peer set (5%, 20%) under CO and AD |
| review_partial | 27 | Half of the clients per round (FedAvg, FLAME, TAF) |
| review_clients | 12 | Per client acceptance without attack, kappa 0.5 and 0.1 |
| review_seeds | 24 | Seeds 3 and 4 for TAF, FLAME, FLTrust, DeepSight under BD, CO, AD |
| review_foolsgold | 21 | FoolsGold (Fung et al., RAID 2020) under the main attacks, plus MNIST CO |

Timing at larger client counts: `python scripts/overhead.py --dataset cifar10 --n 100 200 500 1000 --reps 5 --defenses FedAvg Krum FLAME FoolsGold TAF`.
Every run now logs `client_stats` (participation, acceptance, weight per client, label divergence).
