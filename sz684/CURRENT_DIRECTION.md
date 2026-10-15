# Current research direction

October 15, 2026: user rejected red→blue as the main research task and asked for substantially more useful relations. The Quick, Draw! color experiment is **retired as a sanity check**. Its main GPU extraction was interrupted on request before test scoring. Preserve partial caches and pilot results for provenance; do not resume it or present it as the paper's main result without a new instruction.

The active direction is transferable physical object-state relations in natural images, with multiple meaningful state changes and object/source-held-out evaluation. Candidate relations include opening/closing, filling/emptying, cutting/peeling, cleaning, cooking and melting, depending on verified dataset coverage. No availability or result is assumed before inspecting annotations. Compare universal relation offsets against source-only retrieval, direct target text, random/reversed/wrong relations, image-derived directions and supervised endpoint maps. Separate useful target-state retrieval from mere object recognition.

The requested stronger paper remains the objective. Earlier SEMANTIC_PROTOCOL.md and QUICKDRAW_PROTOCOL.md are historical proposals, not the active primary experiment. A separate3D visualization remains a later request, not part of the current run.

## Completed natural-state study

The active paper is `STATE_TRANSFER_PAPER.md`, with a PDF at `output/pdf/state_transfer_paper.pdf`. Five named state relations were run on992 natural images across51 noun labels using globally noun-held-out folds. The primary macro AP gain is+.08516 with adjusted98.75% interval[+.02936,+.14707], and all four planned AP contrasts are positive. Noun-top1 falls by7.20 percentage points, so the full success criterion fails. Post-primary constrained-scaling and identity-subspace projection do not fix the preservation guardrail. Primary analysis sources match their pre-extraction lock; these follow-ups are clearly exploratory.

Next substantive direction: state transfer that preserves noun information, evaluated on independently verified states with source/taxonomy separation and broader galleries. Do not expand the abandoned color benchmark. A separate3D visualization remains deferred; high-dimensional natural-state vectors are saved for it.
