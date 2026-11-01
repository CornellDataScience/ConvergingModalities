# Current research direction

The latest completed experiment is **natural-image object-state transfer**: cleaning (dirty → clean), cooking (raw → cooked), ripening (unripe → ripe), thawing (frozen → thawed), and drying (wet → dry). See [README.md](README.md) for commands and [CODE_GUIDE.md](CODE_GUIDE.md) for implementation details.

## Completed study

The two editorial manuscripts are [STATE_TRANSFER_PAPER.md](experiments/state_transfer/docs/STATE_TRANSFER_PAPER.md) and [STATE_TRANSFER_PAPER_V2.md](experiments/state_transfer/docs/STATE_TRANSFER_PAPER_V2.md). Five globally noun-held-out folds cover 992 images, 51 noun labels, and 496 queries. Frozen BLOOMZ/DINOv2 embeddings are connected by a supervised ridge translator fitted separately in each fold.

First-result state-and-noun success is 38.7%, versus 20.9% for source-only similarity and 42.1% for direct target-text retrieval. The primary macro AP gain over no shift is +0.08516 with adjusted 98.75% interval [+0.02936, +0.14707]; all four planned AP contrasts are positive. Noun retention falls from 86.6% to 79.4%, so the full success criterion fails. Post-primary constrained scaling and identity-subspace projection do not fix the preservation guardrail.

The historical primary sources and locks remain under `experiments/state_transfer/results/`. Follow-ups are exploratory. Weak released labels, related noun categories, and unknown image-source dependence limit the claims; these are independent photographs, not matched state transitions.

## Next research questions

Investigate state transfer that preserves noun information on independently verified labels, with source/taxonomy separation and broader galleries. These are follow-up directions, not completed experiments. A separate 3D visualization remains deferred; high-dimensional natural-state vectors are saved for it.

## Retired and historical work

The WIT replication, robustness analyses, and arbitrary endpoint-difference pilot are earlier completed stages. Their reports and protocols remain as historical records.

The user retired red → blue as the main task on October 15, 2026. Main Quick, Draw! extraction was interrupted before test scoring. Preserve partial caches and pilot results; do not resume or present that benchmark as the main study without a new instruction. SEMANTIC_PROTOCOL.md, QUICKDRAW_PROTOCOL.md, and CONCEPT_TRANSFER_NEXT.md describe earlier proposals.
