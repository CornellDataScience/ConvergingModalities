# State relations: future work and rerun checklist

Status: planning only. This document implements neither track. No new experiment package, training method, evaluation stage, CLI command, dataset or result is created by the documentation cleanup.

The two tracks follow the chat “Choose directions and split work.” They form one research project with shared data and evaluation. Paths below are relative to `sz684/`. New filenames are proposed, not existing commands or modules.

## Research questions and team ownership

| Person | Track | Ownership | Future deliverable |
| --- | --- | --- | --- |
| 1 | 1: understand transfer | Estimate text/visual directions; compare corresponding versus mismatched relations; measure stability across training-object subsets | Alignment matrix and stability analysis |
| 2 | 1: understand transfer | Shuffled-correspondence bridges, wrong/reversed controls, held-out retrieval and its association with alignment | Controlled retrieval/alignment comparisons |
| 3 | 2: preserve identity | Coordinate independent label checks, category-family/source groups, duplicate checks and error analysis; everyone annotates | Versioned data and frozen splits |
| 4 | 2: preserve identity | Small regularized predictor of source-dependent strength; train-only fitting and validation-only model selection | One adaptive-strength method and ablations |
| 5 | 2 and integration | Integrate baselines, preservation follow-ups, uncertainty, protocol lock and reporting | Common evaluator and final comparison |

Track 1 asks whether an estimated text change corresponds to a visual change on new object families. Compare the correct direction against wrong/reversed directions, shuffled bridge correspondences and an image-derived reference. Visual differences between image groups must not be described as physical before/after transitions.

Track 2 starts with `normalize(source image + predicted strength × state direction)`. Compare the global shift, existing preservation follow-ups, direct target text, and source-image-plus-target-text retrieval. Explicitly label methods that receive test-noun information. Adapting the direction itself is deferred until the strength-only method is assessed.

Joint state-and-object success and object retention are headline outcomes, with AP as a supporting ranking measure. Predeclare the preservation requirement and uncertainty procedure. Smaller shifts alone do not establish improvement.

## Proposed package boundary

Future implementation should use `experiments/state_relations/`, keeping the completed `state_transfer/results/` immutable. No directory is created until research implementation is separately authorized.

```text
experiments/state_relations/        PROPOSED; does not exist
  README.md
  run.py                          explicit input/output run directories
  configs/                        relations, splits, fit/selection settings
  data/                           review, grouping, manifest generation
  extraction/                     frozen encoders and compatible feature reuse
  training/                       preprocessing, bridge, directions, predictor
  evaluation/                     shared metrics, baselines, controls, inference
  reporting/                      reports generated from saved scores
  docs/                           annotation instructions and locked protocol
  tests/                          leakage, selection, metrics and run isolation
  results/<run_id>/                immutable data identities, fits, locks, scores
  output/<run_id>/                 generated reports and figures
```

## File-level implementation outline

| Area | Existing files to reuse or carefully refactor later | Proposed files within the new package |
| --- | --- | --- |
| Data and family/source holdout | `experiments/state_transfer/data/state_data.py`, `mit_states_data.py` | `data/annotations.py`, `families.py`, `prepare.py`; annotation schema and frozen splits |
| Model revisions and feature reuse | `experiments/state_transfer/data/state_extract.py`, `shared/cache.py`, `shared/data.py`; baseline manifest dependency | `extraction/extract.py`, explicit pinned model specification |
| Shared fitting | `experiments/state_transfer/training/bridge.py`, `shared/training.py`, train-only transforms in `shared/data.py` | `training/build_fits.py`, `training/directions.py` |
| Shared metrics and baselines | Pure scoring helpers in `experiments/state_transfer/evaluation/state_analysis.py`; preservation logic in `state_identity_followup.py` | `evaluation/retrieval.py`, `baselines.py` |
| Track 1 | Existing text offsets, image reference directions and bridge fitting | `evaluation/relation_alignment.py`, `bridge_controls.py` |
| Track 2 | Existing global shift and preservation baselines | `training/adaptive_strength.py`, `evaluation/method_comparison.py` |
| Inference | Existing macro aggregation and noun-cluster bootstrap; decide explicitly whether to refit within resampling | `evaluation/statistics.py` |
| Run isolation | `run.py`, `shared/paths.py` only where necessary | New package `run.py` with explicit paths; optional top-level registration later |
| Reporting | Existing saved-result reporting patterns; preserve historical generators | `reporting/report.py`, alignment/stability and state/identity tradeoff plots |
| Tests | Existing retrieval, identity and layout checks | Tests of family leakage, train-only fitting, shuffled pairing, validation selection, run isolation |

Do not fit preprocessing, directions, bridges or the predictor using test families. Estimate visual reference directions independently of map-fitting images. Use the same query/gallery definitions across comparisons. The already inspected MIT-States sample is development evidence; fresh verified testing supports new claims.

## Order and dependencies

1. Agree on annotation schema, family/source grouping, split manifest and metric interfaces.
2. Reproduce the existing state baseline into a new run directory before changing methods. Record the distinction between code reproduction and fresh testing.
3. Person 3 builds the verified benchmark while the other people develop on existing data.
4. People 1–2 develop Track 1; people 4–5 develop Track 2 and the shared evaluator.
5. Lock methods, settings, baselines, inference and preservation criteria before final test scoring.
6. Run both tracks against the same frozen benchmark, then generate tables and writing.

If time is short, finish verified data, the baseline and Track 1 first. Broader model families, adaptive directions and a separate 3D viewer remain deferred.

## What needs rerunning

| Change | Required work after future authorization |
| --- | --- |
| Documentation, file inventory, disposable cache removal | Check links and inventory coverage; no research rerun |
| Pure code/import reorganization | Relevant tests and CLI checks; small saved-score equivalence check |
| Family/source splits only | Refit preprocessing, bridges, directions, predictor and validation settings; rerun scores and inference. Raw embeddings may be reused only after explicit identity/row verification |
| Labels, sample selection or captions | Rebuild manifest; validate unaffected feature rows or re-extract affected inputs; rerun all downstream stages |
| Encoder, weights, layer, pooling, prompts or input processing | Re-extract affected features; rerun downstream fitting, selection, scoring and inference |
| Track 1 | Fit direction estimates and bridge controls; evaluate relation alignment, stability, held-out retrieval and uncertainty |
| Track 2 | Train strength predictor; select on validation; freeze; evaluate baselines/ablations and uncertainty |
| Presentation only | Regenerate reports from saved scores |
| State follow-up work | No multimodal-study rerun required |

A changed manifest hash is not compatible merely because feature rows have the same count. Source hashes and extraction settings also constrain reuse. The two historical extractor migrations are pinned to exact source bytes; do not delete locks or broaden hash acceptance to force compatibility.

The existing top-level `state-*` commands target historical directories and may replace outputs. They are not safe substitutes for the proposed run-isolated implementation. Exact new commands can be documented only once that implementation exists.
