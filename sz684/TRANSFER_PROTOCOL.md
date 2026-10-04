# Cross-modal difference-vector transfer pilot

User intent: related concept differences transferable across text and image spaces.

## Stage A: executable prerequisite, not yet a semantic relation benchmark

Use BLOOMZ 1.7B and DINOv2 base, first saved split (20261004), with selection-chosen layers and clipping. Model choice is a fixed pilot choice, not selected by transfer results. Fit centered PCA128 independently per modality on the 512 selection pairs, then normalize projected sample vectors. Fit a supervised ridge map W from paired text to image vectors on those 512 pairs, with intercept through centering and fixed ridge alpha=1. No hyperparameter or layer search on evaluation data.

Arrange the 512 evaluation rows into 256 disjoint consecutive endpoint pairs (a,b), using saved random row order. Compute predicted image difference (text_b-text_a) W and true image difference image_b-image_a. Measure mean cosine of corresponding directions, and top-1/top-5 retrieval of the matching true difference among 256 directed target differences. Null controls: 199 permutations of training image correspondence, each refitting W; and 999 permutations of evaluation target-difference correspondence. Same PCA and layers across controls. For the two primary statistics (mean cosine, top-1 retrieval), report Bonferroni-adjusted p-values within the two-statistic training-permutation family; evaluation shuffles are secondary diagnostics. Top-5 is descriptive.

This is supervised correspondence-assisted transfer, not blind alignment. Endpoints within the evaluation set are disjoint, but this is still a previously inspected dataset. Training-layer selection also used correct selection correspondences; the training shuffle control is conditional on those layers and representations, not an end-to-end shuffled selection null. A successful map may transfer arbitrary endpoint differences without learning a compositional relation. A significant mean direction score alone does not prove semantic analogy or useful retrieval. Report all results, including failure.

## Stage B: related-concept transfer design

Need a separately curated benchmark of recurring relation families, with multiple source/target concepts per family and multiple independent images per concept. Caption keyword matching in WIT is inadequate (e.g. 'old' appears in place names; 'young' in personal names). Do not call arbitrary held-out endpoints 'related concepts'.

For example, color change red→blue across held-out object categories can supply training contrasts on cars and test contrasts on chairs, with image style/background balanced. Fit the space map on a separate paired alignment set excluding evaluation concepts. Estimate a text relation direction from training relation examples. Transfer that direction through W and apply it to a held-out image source embedding; rank target images among same-object distractors. Keep relation-training subjects, map-fitting concepts, and evaluation subjects separate. Learn any direction scale on validation only.

Controls: no translation, shuffled relation direction, reversed direction, shuffled map-training correspondence, and within-image relation direction as a diagnostic upper reference (not guaranteed upper bound). Include target-word-only text retrieval to distinguish analogy from direct target recognition. Score target retrieval and same-object specificity; paired query differences against no-translation control, with source object as the resampling unit. Human-check visual labels, ambiguity, and repeated sources before extraction.

The cached corpus has no verified repeated-relation annotations. Stage B is designed but not executed; Stage A must not be reported as evidence of named concept transfer.

## Post-run inferential correction

On reviewing the first run, the training-map permutation statistic was demoted to a diagnostic: preserving layers selected with true training correspondence while shuffling only map-fitting correspondence does not repeat the full data-dependent selection procedure. Do not interpret its tail fraction as a valid p-value. Use the precomputed 999 evaluation-correspondence permutations for the two-statistic adjusted reference instead, explicitly exploratory. This change was made after inspecting the first run and is not presented as preregistered; no model, hyperparameter, direction, score, or permutation draw was changed. The evaluation null tests correspondence of held-out directions, not whether a named relation composes.
