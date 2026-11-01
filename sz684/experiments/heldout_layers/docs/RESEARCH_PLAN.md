> Historical WIT/alignment-stage record. The completed natural-state experiment (including raw → cooked) is documented in [README.md](../../../README.md) and [STATE_TRANSFER_PAPER.md](../../state_transfer/docs/STATE_TRANSFER_PAPER.md). Pending work described below reflects this earlier stage.

# Work-session research plan

Primary question: Does the observed model-capability/alignment trend survive evaluation choices?

Selected directions, in order:
1. Held-out layer selection: select layers on one half of the cached WIT pairs, evaluate on the other half. Run now.
2. Sample-size sensitivity: repeated matched subsets, fixed k and separately fixed k/N, with condition-matched nulls. Run next.
3. Preprocessing sensitivity: baseline versus train-fitted centering and dimensionality reduction. Run after sample-size analysis.

Defer audio, new model training, additional families, GW, and the visualizer until these checks inform the scope.

## Frozen protocol for experiment 1

Written before running this extension. Existing baseline remains untouched.
- All six cached BLOOMZ/DINOv2 pairs; no new inference.
- Three prespecified random 512/512 selection/evaluation splits, seeds 20261004, 20261005, 20261006. Same splits for every model.
- Fit the original clipping threshold on selection rows only; apply it to both halves and L2-normalize. k=10; rebuild neighborhoods within each half.
- Select the maximum layer pair on selection rows, then freeze it for evaluation.
- For each fixed layer pair, run 1,000 evaluation correspondence permutations. No evaluation layer search in this null, because layers were selected independently.
- Also calculate the evaluation-set maximum as a diagnostic of selection optimism; do not attach the fixed-layer null p-value to that maximum.
- Report fixed-layer scores, shuffled mean/95th percentile, scores above null, and size ordering separately for each vision model and split.
- Repeated splits overlap and are sensitivity checks, not independent replications or confidence intervals. Tests are exploratory; no multiplicity-adjusted significance claim.
- N=512 scores are not directly comparable with the N=1024 baseline. Selection versus evaluation score differences also reflect sample composition; evaluation maximum minus frozen-layer score isolates the within-evaluation search advantage.
- This is held-out layer selection within an already inspected dataset, not a fresh confirmatory dataset.

Deliverables: reproducible script, exact split indices and source hashes, results JSON, report, plot.

## Completion update

All three selected pilot directions are now complete. Experiments 2 and 3 are specified in ROBUSTNESS_PROTOCOL.md and implemented by robustness.py. Results, scores, plots, and interpretation are in results/robustness/. Local agreement persists; numerical magnitudes and model-size ordering remain sensitive to evaluation choices. A larger independent paired evaluation set is the recommended next phase, not yet run.

## Paired uncertainty and transfer follow-up

Conditional paired inference completed with separate reference/query banks (PAIRED_PROTOCOL.md; results/paired_uncertainty/). Centering and PCA versus baseline have adjusted intervals above zero in all three splits; a largest-versus-smallest model advantage remains inconclusive. This is a changed conditional estimand, not a retrospective CI for the original shared-pool metric.

A supervised text→image difference-map pilot is complete (TRANSFER_PROTOCOL.md; results/difference_transfer/). Related semantic concept transfer remains a separate, unrun stage requiring curated repeated-relation examples; the concrete proposed benchmark is in CONCEPT_TRANSFER_NEXT.md.
