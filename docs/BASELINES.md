# Baseline implementation notes

Every baseline is implemented in this repository. Where an implementation is a
reimplementation from a paper rather than a port of the authors' code, the
departures we know of are listed here. **FLAME and DeepSight must be validated
before their numbers are reported** (see "Validation" below).

## FedAvg
Unweighted mean of client updates (not data-size weighted), so that TAF with
equal trust reduces to it exactly. State this in Section 5.3.

## Krum
Single-Krum, `f = rho * N`, score = sum of squared distances to the
`N - f - 2` nearest neighbours. The defence is given the true `rho`.

## Median / Trimmed mean
Coordinate-wise. Trimmed mean removes `floor(rho * N)` values from each end,
i.e. it is given the true `rho` (favours the baseline).

## FLTrust
Root set: 100 examples held out from the **training** split (disjoint from
client data and from the validation split). Server update = one local epoch
on the root set with the same optimiser settings as clients. Trust score =
ReLU(cosine); client updates rescaled to the server update's norm.

## FLAME (reimplementation)
1. Pairwise cosine distances between local **models** W_i = G + u_i, as in
   Algorithm 1 of the paper. (An earlier version of this code used the updates
   u_i instead; that made FLAME blind to boosted updates and it failed the
   CIFAR-10 backdoor in pilot 2. Fixed, and verified against the paper text.)
2. HDBSCAN, `min_cluster_size = N/2 + 1`, `min_samples = 1`,
   `allow_single_cluster = True`; keep the largest cluster; if everything is
   noise, keep all.
3. Clip kept updates to the median update norm.
4. Average; add Gaussian noise with `sigma = 0.001 * median_norm`
   (the paper's image-classification setting).

## DeepSight (reimplementation)
1. NEUPs from output-layer weight and bias updates; threshold exceedings with
   `xi = max(0.01, 1/C) * max NEUP`; a model is labelled suspicious if its
   exceedings are at most half the median.
2. DDifs on `n_probe = 512` uniform random inputs per seed, 3 seeds (the paper
   uses far more random inputs — **deviation**, for cost).
3. HDBSCAN clusterings on each DDif seed, on NEUPs and on output-layer bias
   updates (cosine distance); ensemble distance = fraction of clusterings that
   separate a pair; final HDBSCAN on the ensemble distance. Noise points are
   singleton clusters.
4. Reject clusters in which at least 1/3 of members are suspicious.
5. Clip accepted updates to the median norm and average.

The paper's exact ensemble/merging procedure is not reproduced line by line;
treat this as "DeepSight-style" until validated.

## BayBFed
Not implemented. Decide before submission whether to add it or to state in
Section 5.3 why it is omitted.

## Validation (do before reporting)
For FLAME and DeepSight, run the BD attack at rho = 0.1 and 0.3 on CIFAR-10
and confirm each one reduces backdoor ASR well below FedAvg's with a modest
clean-accuracy cost, which is the qualitative behaviour both papers report.
If not, fix the implementation before anything goes in a table — a weak
baseline would make TAF look better than it is, which reviewers will catch.
