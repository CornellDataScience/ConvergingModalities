# Anisotropy-Control Sensitivity: Frozen Protocol

Written before running this extension. Existing baseline, held-out layer, and robustness results remain untouched. This is an exploratory check on an already inspected dataset.

## Question

Centering and centered PCA128 raised measured alignment (ROBUSTNESS_PROTOCOL.md; PAIRED_PROTOCOL.md). Does the measured alignment also change under controls aimed specifically at anisotropy (shared dominant directions, rogue dimensions), and do the measured anisotropy diagnostics move with those changes? No direction is assumed: removing anisotropy may raise, lower, or not change overlap. Do not tune any control toward an expected curve.

## Reuse

This study lives in its own repository and shares no code with any other. Its inputs are produced by `prepare_data.py`, an independent implementation of the published measurement recipe: WIT-1024 pairs, BLOOMZ 560M/1.1B/1.7B and DINOv2 small/base, per-layer features, 95th-percentile clipping fitted on selection rows, L2 normalization, k=10, and held-out layer selection on three 512/512 splits (seeds 20261004-20261006). Absolute scores may differ slightly from other implementations because of hardware and precision, so compare trends, not decimals. `anisotropy.py` asserts that its baseline scores and null draws match the held-out file it reads.

- All six cached BLOOMZ/DINOv2 pairs; no new inference.
- The three existing 512/512 splits, selection-fitted clipping thresholds, and selection-chosen layers in the held-out results file produced by `prepare_data.py`.
- N=512, k=10, neighbors rebuilt within the evaluation half, and the 1,000 evaluation permutations with seed+10000, so every condition shares the same permutation draws (as in ROBUSTNESS_PROTOCOL.md).
- Start from clipped, L2-normalized vectors. Fit every control on the 512 selection rows only, independently per modality, then apply to the evaluation rows and L2-normalize.
- Layers stay frozen at the baseline-selected choice. Do not claim these are each control's best layers.

## Conditions (parameters fixed here, not searched)

1. `baseline`, `center`, `pca128`: same definitions as the original robustness study (selection-fitted mean subtraction; exact-SVD PCA without whitening; L2 renormalization). Baseline must reproduce the earlier held-out scores and null draws exactly.
2. `standardize`: subtract the selection mean, divide each dimension by its selection standard deviation (floored at 1e-8). Deflates rogue dimensions.
3. `remove3`: subtract the selection mean, project out the top 3 principal directions of the selection rows (all-but-the-top). The count 3 is fixed in advance and is not tuned.
4. `pca128_whiten`: centered PCA128, then divide each retained component by its selection singular value / sqrt(n-1). Whitening happens only inside the 128-dimensional subspace: the selection covariance has rank at most 511 while BLOOMZ widths reach 2048, so full-dimensional whitening (including ZCA) is ill-posed here and is excluded.

Matched contrasts of interest: each control minus baseline; `pca128_whiten` minus `pca128` (isolates whitening); `remove3` minus `center` (isolates top-direction removal).

## Diagnostics

For every model, split, layer, and condition, compute on the evaluation vectors after the control: mean pairwise cosine (off-diagonal), effective rank (exp of the entropy of the normalized singular values of the centered vectors), top-dimension variance share, and top-principal-component variance share. Report them next to scores. They describe geometry; they are not shown to cause score changes.

## Reporting and verification

Report raw overlap, null mean and 95th percentile, excess over null, and (score-null_mean)/(1-null_mean) for each condition. Report matched changes with the number of increases out of 18 and strict BLOOMZ size increases out of 6. Save exact source hashes, inherited split/layer provenance, all null draws and scores, diagnostics, a detailed CSV, a summary report, and a plot.

Check that baseline scores and all 1,000 null draws match the earlier held-out run, and compare centered and PCA128 scores with the original robustness results where available. Unit-test the control fitting and the diagnostics on synthetic anisotropic features.

## Limits

Repeated splits and model pairs are dependent; ranges are sensitivity ranges, not confidence intervals. No significance claims. Equal output dimension does not control architecture or training differences. Controls are fit on 512 pairs, so high-dimensional estimates (especially per-dimension scales) are noisy. A control that changes scores is a geometry intervention, not a validated correction. If a result looks worth reporting, confirm it with a conditional query/reference bootstrap (as in the original project's paired-uncertainty protocol) rather than from these ranges. Not a test of cross-dataset generalization, and not a test of the sample-size critique.