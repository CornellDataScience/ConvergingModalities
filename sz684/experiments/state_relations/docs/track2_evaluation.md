# Track 2 Week 1 handoff

Status: candidate split prepared. The fresh MIT-States image bytes and independent label review are pending.

From `sz684/`:

```sh
.venv/bin/python -m experiments.state_relations.data.prepare
.venv/bin/python -m experiments.state_relations.data.prepare --archive /absolute/path/to/release_dataset.zip
```

The first command regenerates the review queue from the existing catalog and historical manifest. The second also creates contact sheets under `results/week1/review_images/`. Image labels in the queue come from the archive. The current split includes 312 training, 276 validation and 96 test candidate images. All five transitions have source and target candidates in every split. Test covers one family each for cleaning, cooking, ripening and thawing, and two for drying. Treat transition-level comparisons as descriptive for this roster. Six candidates per composition keep review bounded.

After annotation and duplicate review, export a versioned manifest containing accepted image IDs, file references and SHA-256 hashes, noun/state labels, family and split, and query IDs with requested state. Use all accepted images in a split as its fixed gallery and exclude the source image from its own ranking. Retain only queries with a verified same-noun target-state image in that gallery. Recheck family and duplicate disjointness and transition coverage before hashing and freezing the manifests. Share split counts with M4/M5 and give M4 access to train and validation for fitting and model selection.

## Locked scoring for M4/M5

Use the same queries and gallery for every run. Top-1 joint success requires the target state and source noun. Top-1 object retention requires the source noun. Report paired percentage-point deltas, with denominators by transition and family. The locked preservation rule is higher joint success than the global shift and object retention within 5 percentage points of the no-shift control. M5 should recompute both baselines for this split.

Fit the bridge and all learned transforms on train. Use validation to choose edit settings. M5 should run the locked methods on test and export per-query predictions with stable query and image IDs. The example export is provisional pending M1 schema review. M5 can rehearse the pipeline on historical data while the candidate images are reviewed.

## Image-backed pilot

The [MIT-States* release](https://huggingface.co/datasets/BJTUYXD/MIT-States_star) is available locally at `results/mit_states_star/source.zip`. The pilot uses only ZIP members matching an original catalog entry by CRC32 and byte length and excludes the old sample. It keeps the original catalog labels for visual review. It uses three images per composition and recomputes family splits within the available subset.

```sh
.venv/bin/python -m experiments.state_relations.data.prepare \
  --output experiments/state_relations/results/mit_states_star \
  --archive experiments/state_relations/results/mit_states_star/source.zip \
  --per-composition 3
```

The pilot contains 192 decodable images and 54 test candidates. `first_pass.jsonl` records the initial visual review of those 54. Some thawing labels are visually indeterminate and all three ripe-pear images were rejected. The pilot still needs independent review, near-duplicate adjudication, and coverage checks after exclusions before a test set can be frozen.

## Provisional test roster and second review

Run `.venv/bin/python -m experiments.state_relations.data.audit_pilot` from `sz684/` after generating the pilot. It checks the 54 test candidates against the other pilot images and the historical sample, and writes review and provisional manifest files under `results/mit_states_star/`. A grayscale 64-bit difference-hash screen at Hamming distance 6 found no flags among the 1,184 image files checked. Crops and other transformations can escape this screen.

The follow-up visual check moved three previously accepted labels to uncertain, leaving 17 provisional gallery images. Six query images have an accepted same-noun target: two for cleaning, two for drying, and two for ripening. Cooking has no accepted raw-pasta source and thawing has no visually verified pair. The six queries cover two families and can exercise integration. A larger reviewed roster is needed for the planned benchmark. `independent_review_queue.jsonl` lists all 54 test candidates and their sheet positions without first-pass decisions.

The user gave a group-level assessment in chat that the labels looked good for all 54 test candidates. The assessment was recorded in `user_label_assessment.jsonl`. It agrees with 17 first-pass accepts. Eight first-pass rejects and 29 uncertain cases remain for adjudication. The provisional manifests still reflect images both assessments support.

## Frozen smoke-test roster

`results/mit_states_star/smoke_test_v1/` contains the six agreed-label queries, 17 gallery images, and 138 train/validation candidates from disjoint families. The lock records the manifest SHA-256 hashes. M5 can fit on train, exercise selection on validation and check the end-to-end result export on this roster. The development candidates carry archive labels. Report the smoke test as an integration check. The five-transition research benchmark still needs enough visually verifiable examples, especially for cooking and thawing. The good-view JEPA validation run requires usable weights or a CUDA training run.

The expansion pass rendered 50 additional images from the same test families. The cooking and thawing review is saved in `expansion_cooking_thawing_review.jsonl`. The additional cooked-pasta photo was clear. Nine additional raw-pasta labels show prepared dishes and the tenth is ambiguous. None of the 12 additional bread and pie examples show enough evidence to establish the frozen/thawed label. This batch adds no verified source-target pairs for the missing transitions.

The retention floor is the no-shift top-1 object retention minus 5 percentage points, inclusive. Apply it to the same queries and gallery and require strictly higher top-1 joint success than the global-shift baseline.

The historical fallback is `results/historical_fallback_v1/images.jsonl`, with image paths relative to `experiments/state_transfer/results/`. Use the original noun groups and gallery from the historical protocol. Scores from those previously inspected, archive-labeled images should be reported as historical replication.

The fresh cooking supplement is `results/changeit_cooking/`. The pinned ChangeIt frame release yielded 32 candidate pancake frames across 16 source videos. First-pass visual review selected four clear raw/cooked pairs. The four query records exclude every gallery frame from the query's source video. This supplies a fresh cooking check on one object category. Measure object retention only in a gallery with other object categories. Use the historical MIT-States fallback for thawing. [Frame release](https://github.com/brown-palm/object-states) · [ChangeIt label map](https://github.com/soCzech/ChangeIt/blob/main/labels.csv)

## Expanded smoke test

`results/mit_states_star/smoke_test_v2/` has 13 queries and 35 gallery images, with 138 development candidates. The test queries span three families. Cleaning, drying, and ripening have three queries each. Cooking has four ChangeIt pancake queries. The gallery includes other categories, including cooked pasta as a wrong-object cooking distractor. Exclude all entries from a pancake query's source video before ranking. Resolve gallery entries with `local_member` inside the archive named in the lock and entries with `image` relative to the manifest directory.

The expansion uses first-pass visual review. The difference-hash screen flagged zero of the 18 added images against historical images, development candidates, the original gallery, and each previously accepted addition. Thawing remains historical replication. Use the expanded set for integration and descriptive scores. Regenerate the manifests with `.venv/bin/python -m experiments.state_relations.data.freeze_expansion` from `sz684/`.
