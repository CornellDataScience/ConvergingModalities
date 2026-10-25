# Metric x control x sample-size grid: protocol

Written and committed before running `scale_grid.py`. Exploratory; same already-inspected WIT-1024 data.

## Question
The leads flagged that mutual k-NN agreement can be inflated at small N (neighbors are drawn from a tiny pool) and can degrade as the pool grows. Do our two earlier findings survive a change in N, for every metric?
1. Centering never hurts and helps k-NN; pca128 is the best control under the global metrics (CKA, GW).
2. top-PC removal and whitening look neutral under k-NN but collapse CKA and GW.

## Design
- Same 18 rows (3 splits x 6 BLOOMZ/DINOv2 pairs), frozen layers and selection-fitted controls as `metric_controls.py`. No re-selection.
- Sample sizes N = 128, 256, 512: nested subsets of the 512 evaluation rows, taken from a seeded random permutation per split (seed + 777), so the same rows are used for every model and condition.
- k-NN is reported in two regimes: fixed k = 10, and k scaled with pool size, k = round(10 (N-1)/511) = 2, 5, 10. The regimes coincide at N = 512 and are not independent evidence.
- Metrics: k-NN excess, linear CKA excess, GW cost reduction, each against a condition-matched shuffled-pairing null (200 shuffles; same seeds across conditions).
- Blind matching is omitted: it was at chance at N = 512 and cannot improve at smaller N.

## Reported
Per metric, condition and N: mean [range] across the 18 rows. For each non-baseline condition: change vs baseline and the number of rows (of 18) that rise.

## Predictions (stated in advance)
- Raw k-NN overlap and its null both rise as N falls under fixed k, so excess over null is the quantity to read.
- Centering's k-NN gain should persist at all N. If it vanishes at large N, it was a small-pool effect.
- remove3 and whitening should keep hurting CKA and GW at every N; if the damage shrinks with N it is noise-driven.
- CKA and GW excess are expected to be biased upward at small N (few samples, many dimensions); the null should absorb most of this, and any residual trend is itself a finding.

## Limits
N = 512 is the ceiling given a 512-row selection half, so this cannot test the 15M-sample regime. Subsets nest and splits overlap, so ranges are sensitivity ranges, not confidence intervals. No multiplicity-adjusted claims.
