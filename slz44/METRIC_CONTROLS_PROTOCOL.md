# Metric sensitivity to anisotropy controls: protocol

Written and committed before running `metric_controls.py`. Exploratory; same already-inspected WIT-1024 data.

## Question
The mutual k-NN study found that centering helps and other controls do not. Is that a property of the *geometry* or of the *metric*? Re-score the identical inputs with two further metrics from the design document:

1. **Linear CKA** (centered Gram, normalized HSIC), with a shuffled-pairing null.
2. **Gromov-Wasserstein structure cost** between within-model cosine-distance matrices (each scaled to mean off-diagonal 1), with a shuffled-pairing null; reported as reduction = (null mean - observed) / null mean. Higher = more shared structure.
3. Optional (`--gw-match`, needs POT): **blind matching**, i.e. GW coupling argmax top-1/top-5 accuracy against chance (1/512 and 5/512). No correspondence is given to the method.

## Fixed inputs (nothing re-selected)
- 18 rows: 3 splits x 6 BLOOMZ/DINOv2 pairs, layers frozen from `data/heldout_layers/results.json`, 512 evaluation rows.
- Six conditions as in `ANISOTROPY_PROTOCOL.md`: baseline, center, standardize, pca128, remove3, pca128_whiten. Controls are fitted on the selection half only.
- 200 shuffled pairings per metric (fewer than the k-NN study because GW is costly); seeds derived from split seed.

## Reported
- Per condition: mean [range] of each metric's excess/reduction, and how many of 18 rows rise vs baseline.
- Agreement between metrics: Pearson r of changes vs baseline over the 90 non-baseline condition-rows (descriptive; rows are dependent).

## Predictions (stated in advance, so they can be wrong)
- CKA ignores a constant offset, so `center` should change CKA little; `standardize`/`remove3`/`pca128_whiten` may change it more because CKA is dominated by high-variance directions.
- If a control helps k-NN but not CKA/GW, the "centering helps" result is metric-specific.
- GW blind matching is expected to be near chance at N=512 for cross-family pairs; a result near chance is reported as such, not tuned.

## Limits
Repeated splits and pairs are dependent; ranges are sensitivity ranges, not confidence intervals. No multiplicity-adjusted claims. Layers were selected under k-NN, which may favour k-NN.
