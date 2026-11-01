# Current research direction

The reference study for the next research phase is **natural-image object-state transfer**: cleaning (dirty → clean), cooking (raw → cooked), ripening (unripe → ripe), thawing (frozen → thawed), and drying (wet → dry). See [README.md](README.md) for commands and [CODE_GUIDE.md](CODE_GUIDE.md) for implementation details.

## Completed study

The two editorial manuscripts are [STATE_TRANSFER_PAPER.md](experiments/state_transfer/docs/STATE_TRANSFER_PAPER.md) and [STATE_TRANSFER_PAPER_V2.md](experiments/state_transfer/docs/STATE_TRANSFER_PAPER_V2.md). Five globally noun-held-out folds cover 992 images, 51 noun labels, and 496 queries. Frozen BLOOMZ/DINOv2 embeddings are connected by a supervised ridge translator fitted separately in each fold.

First-result state-and-noun success is 38.7%, versus 20.9% for source-only similarity and 42.1% for direct target-text retrieval. The primary macro AP gain over no shift is +0.08516 with adjusted 98.75% interval [+0.02936, +0.14707]; all four planned AP contrasts are positive. Noun retention falls from 86.6% to 79.4%, so the full success criterion fails. Post-primary constrained scaling and identity-subspace projection do not fix the preservation guardrail.

The historical primary sources and locks remain under `experiments/state_transfer/results/`. Follow-ups are exploratory. Weak released labels, related noun categories, and unknown image-source dependence limit the claims; these are independent photographs, not matched state transitions.

## Planned follow-up: two tracks, one shared evaluation

These tracks are plans only. Neither is implemented, and no `experiments/state_relations/` package has been created. See the [file-level work plan and rerun checklist](docs/planning/STATE_RELATIONS_PLAN.md).

1. **Understand transfer (two people):** compare mapped text state directions with independently estimated visual state directions; measure relation specificity, object-subset stability, shuffled-bridge controls, and held-out retrieval.
2. **Preserve object identity (three people):** develop a small regularized predictor for a source-dependent shift strength, compared with the global shift and existing preservation follow-ups. Source-conditioned direction changes are a later extension.

Both tracks require independently checked labels, source/category-family separation, train-only fitting, validation-only selection, a common evaluator, and frozen methods before fresh test scoring. Joint state-and-object success and object retention are headline outcomes; AP supports ranking analysis. Everyone contributes to annotation.

A separate 3D vector viewer remains deferred. The documentation diagrams describe data flow; they do not implement that research visualization.

## Separate exploratory study: multimodal structure

The newer [multimodal structure experiment](experiments/multimodal_structure/README.md) already contains six frozen encoders, 80 concepts and 1,200 audiovisual recordings. It evaluates RSA, CKA, neighborhoods and category CAV transfer. Read its [week-8 results](experiments/multimodal_structure/docs/week8/RESULTS.md) and [validation limits](experiments/multimodal_structure/docs/week8/VALIDATION.md).

This is a separate released-label exploratory study. Category-axis transfer is not physical state-change transfer. Refit/family-cluster uncertainty, multiplicity decisions and fresh-source replication remain subsequent work. The original expansion proposal is [historical](docs/history/MULTIMODAL_EXPANSION_PLAN.md), not a current implementation checklist.

## Retired and historical work

The WIT replication, robustness analyses, and arbitrary endpoint-difference pilot are earlier completed stages. Their reports and protocols remain as historical records.

The user retired red → blue as the main task on October 15, 2026. Main Quick, Draw! extraction was interrupted before test scoring. Preserve partial caches and pilot results; do not resume or present that benchmark as the main study without a new instruction. SEMANTIC_PROTOCOL.md, QUICKDRAW_PROTOCOL.md, and CONCEPT_TRANSFER_NEXT.md describe earlier proposals.
