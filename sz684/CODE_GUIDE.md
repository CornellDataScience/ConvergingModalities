# Reading the state-transfer code

For the complete repository map, including the separate multimodal study and historical files, start with the [newcomer guide and data-flow diagrams](docs/NEWCOMER_GUIDE.md) and [file inventory](docs/FILE_INVENTORY.md). The two follow-up tracks are [planning only](docs/planning/STATE_RELATIONS_PLAN.md).

Use `run.py --list` for commands. The [repository README](README.md) and each experiment README map the folder roles. Start with `experiments/state_transfer/configs/state_transfer.json`, then `experiments/state_transfer/evaluation/state_analysis.py`. The reference state-transfer experiment is natural-image retrieval across five state changes; the color and WIT scripts are earlier stages.

## Pipeline and file map

| File | Responsibility |
| --- | --- |
| `experiments/state_transfer/configs/state_transfer.json` | Five relations, seed, models/layers, five folds, PCA64, ridge penalty, scales, bootstrap settings |
| `experiments/state_transfer/data/state_data.py` | Select MIT-States compositions, remove exact duplicates, assign global noun groups, sample and verify images, write manifest/contact sheets |
| `experiments/state_transfer/data/mit_states_data.py` | Read the original ZIP catalog and download selected archive members with integrity checks |
| `experiments/state_transfer/data/state_extract.py` | Extract frozen BLOOMZ/DINOv2 features into resumable, identity-checked batches |
| `experiments/state_transfer/evaluation/state_analysis.py` | Fit per-fold preprocessing/maps, estimate relations, select validation scales, score held-out queries, compute primary inference |
| `shared/training.py::fit_ridge` | Centered ridge solver used to train the translator |
| `shared/data.py` | Shared preprocessing, cache loading, and gallery-mask helpers |
| `experiments/state_transfer/evaluation/state_identity_followup.py` | Post-primary constrained scale selection and noun-centroid-subspace projection |
| `experiments/state_transfer/reporting/state_report.py` | Figures, companion report, and archival generated manuscript from saved scores |
| `experiments/state_transfer/reporting/paper_pdf.py` | Render the separately edited manuscripts to PDF |
| `experiments/state_transfer/tests/test_state_transfer.py` | Independent AP rank calculation, gallery exclusions, shared-noun bootstrap, fold coverage |
| `experiments/state_transfer/tests/test_state_identity.py` | Identity-proxy projection and constrained selection checks |

Launch stages through `run.py` (see the repository README). Project paths resolve relative to `sz684/`, regardless of the shell working directory. The extractor's `--stage` value labels progress output; it does not select a different dataset directory.

## Where the translator is trained

Within `experiments/state_transfer/evaluation/state_analysis.py::analyze()`, each fold holds out noun group `fold` for test and `(fold + 1) % 5` for validation. The other three groups supply fitting data across all five relations. Each training composition has eight images: the first four fit the map, the other four estimate the image-derived relation reference.

`fit_preprocessing` fits the clipping threshold and centered PCA on fitting rows only, normalizing endpoints before and after projection. Each modality gets its own transform. `load_selected` reads the cached selected-layer features as float64 tensors; extraction itself used float32. The image's assigned caption template supplies the corresponding map-fitting text row, so repeated captions are retained as fitting observations.

The translator is fitted by `experiments/state_transfer/training/bridge.py::fit_bridge`, called from the evaluation loop. Its matrix `W` is returned by `shared/training.py::fit_ridge(x, y, alpha=1)`, where rows of `x` and `y` are corresponding preprocessed text and image embeddings. The solve is:

```text
xc = x - mean(x)
yc = y - mean(y)
W  = solve(xc.T @ xc + alpha * I, xc.T @ yc)
```

An endpoint prediction is `(text - mean(x)) @ W + mean(y)`. A difference prediction is simply `text_difference @ W`, since the intercept cancels. No BLOOMZ or DINOv2 weights are trained. There is one bridge per fold shared by its five relations, rather than one translator per state change. A centered orthogonal Procrustes bridge is a secondary comparison.

## From raw → cooked to a retrieval query

For each fitting noun, average the four template embeddings for “cooked noun” and subtract the corresponding average for “raw noun”. Average those contrasts equally across fitting nouns, then multiply by `W`. This yields one shared cooking direction in image coordinates, without test-noun wording.

`select_scale` tries `[0, .25, .5, 1, 2, 4]` using validation-noun AP. The first maximum wins, favoring the smaller scale on ties. The held-out source-image query becomes `normalize(source + scale * direction)`. Candidate image embeddings stay fixed. Image-derived and Procrustes directions select their own validation scales; random, reversed, and wrong-relation controls use the text method's scale.

Direct target-text and noun-specific text references deliberately receive the test noun. They have more information than the universal direction and help judge whether transfer offers a retrieval advantage.

## Arrays and retrieval metrics

`endpoints` has shape `[noun, image_index, state, coordinate]`, with state 0 for source and 1 for target. In this study there are eight images per state and 64 coordinates. Source/target indices refer to independent photographs, never matched physical transitions.

`score_queries` flattens queries in noun/image order and candidates in noun/image/state order. For query index `i`, `bank_allowed` excludes source index `i` and target index `(i + 1) % 8` in **every noun**. Each noun therefore contributes seven source and seven target candidates. The exclusions balance the bank and remove the query itself.

A relevant candidate has both the requested target state and the query noun. AP averages precision at every relevant rank. The returned arrays have shape `[noun, source_query]`:

| Key | Meaning |
| --- | --- |
| `ap` | Target-state-and-noun AP over the full allowed gallery; original primary metric |
| `within_ap` | Target-state AP restricted to the query noun |
| `state` | First result has the requested state |
| `noun` | First result has the query noun |
| `joint` | First result has both; foregrounded in the revised manuscripts |
| `within_auc` | Within-noun positive/negative score comparison, with half credit for ties |

`macro` averages queries within noun, nouns within relation, then relations equally. Large relations do not receive extra weight. Random and wrong-relation controls average metrics after scoring each direction separately.

## Uncertainty and decision

`clustered_bootstrap` receives `[contrast, relation/noun_cell, query]` effects. The outer draw samples global noun labels and reuses their multiplicities across every relation containing that noun. Draws leaving any relation empty are retried. Inner draws resample queries within each cell. The returned category-only variant omits inner query resampling. Fitted maps, scales, folds, and galleries stay fixed, so these intervals do not include training or gallery-selection uncertainty.

The four planned AP contrasts compare the text direction with no shift, random, reversed, and wrong-relation directions. Each gets a Bonferroni-adjusted 98.75% interval. Full success also requires the one-sided 95% lower bound on noun retention relative to no shift to exceed −0.05. The saved primary result fails this preservation guardrail despite positive AP contrasts.

`experiments/state_transfer/evaluation/state_identity_followup.py` tries a validation noun-retention constraint and removes components in a training-noun centroid subspace. That subspace is an identity proxy, not a proven disentangled representation. These analyses reused the inspected test set and remain exploratory.

## Historical modules and provenance

- `experiments/baseline/data/experiment.py`, `experiments/baseline/evaluation/alignment.py`, `experiments/heldout_layers/evaluation/heldout_layers.py`, `experiments/robustness/evaluation/robustness.py`, `experiments/paired_uncertainty/evaluation/paired_uncertainty.py`, `experiments/baseline/evaluation/language_eval.py`, `experiments/baseline/evaluation/verify_extraction.py`, and `experiments/baseline/reporting/report.py`: WIT alignment/replication and robustness work. Its feature loader is used by historical analysis modules; state data preparation separately reads the baseline manifest for model revisions.
- `experiments/difference_transfer/evaluation/difference_transfer.py` and `experiments/endpoint_audit/evaluation/endpoint_audit.py`: arbitrary endpoint-difference transfer and its audit. Their results do not by themselves establish semantic relation transfer.
- `experiments/color_transfer/data/quickdraw_data.py`, `experiments/color_transfer/data/semantic_extract.py`, and the executable part of `experiments/color_transfer/evaluation/semantic_analysis.py`: retired color experiment. `experiments/color_transfer/evaluation/pilot_planning.py` belongs to that earlier planning stage.
- Earlier protocols and research reports retain their original claims and timing, with status notices linking to the active experiment.

Keep the frozen `experiments/state_transfer/results/source_snapshot/`, design locks, and result artifacts together when reproducing the original run. Working-source comments and formatting now differ from historical hashes. Re-running analysis writes current hashes and replaces its outputs; it does not recreate the original source identity. The relocated extractors explicitly validate the known pre-move source identity. Acceptance is pinned to the reorganized source bytes and all other cache settings must match; future source edits or unknown hashes are rejected.
