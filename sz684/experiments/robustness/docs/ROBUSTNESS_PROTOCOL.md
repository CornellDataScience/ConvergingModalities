# Sample-size and preprocessing sensitivity: frozen protocol

Written before running these two extensions. Reuse the three existing 512/512 splits and selection-only layer choices in results/heldout_layers/results.json. These are exploratory checks on an already inspected dataset.

## Sample size
- All six model pairs, three shared splits. Nested prefixes of the saved evaluation rows at N=128, 256, 512; no use of selection rows for evaluation.
- Fixed k=10 versus approximately fixed neighborhood fraction: k=round(10*(N-1)/511), giving k=2,5,10. Report actual fractions. At N=512 the regimes coincide; do not count them as independent evidence.
- Freeze clipping and selected layers from the previous experiment. Recompute neighbors within each evaluation subset. This isolates pool-size effects at those layers; it does not recreate a full layer-maximization study.
- 1,000 condition-matched correspondence permutations; same permutation seed across models within a condition.
- Report raw overlap, null mean and 95th percentile, excess over null, and (score-null_mean)/(1-null_mean). The latter is a descriptive rescaling, not a correction that makes different neighborhood regimes equivalent.
- No N=1024 held-out condition is possible with the available 1,024 pairs and a separate 512-pair selection set. Larger-N conclusions remain out of scope.

## Preprocessing
- All six pairs, same three splits, N=512, k=10, fixed selection-chosen layers.
- Start from clipped and L2-normalized baseline vectors. Compare unchanged baseline, selection-fitted mean subtraction followed by L2 normalization, and selection-fitted centered PCA with 128 components followed by L2 normalization. PCA uses exact SVD, without whitening.
- Fit each modality independently. Freeze the layer choice across transforms to isolate preprocessing at baseline-selected layers; do not claim these are each transform's best layers.
- Record training variance retained by PCA. Equal output dimension does not control architecture or training differences.
- Same 1,000 evaluation permutations in each condition. Interpret changes as geometry sensitivity, not automatic improvement.

## Reporting and verification
Save exact source hashes, inherited split/layer provenance, all null draws and scores, a detailed CSV, summary report, and plots. Check scoring against the earlier held-out baseline and test PCA fitting/application and matched null calculations. Repeated splits and model pairs are dependent; ranges are sensitivity ranges, not confidence intervals. No unadjusted multiple-testing significance claims. Preserve previous results.
