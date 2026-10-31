# State directions: does a text-derived change match the visual change in new object families?

Written and committed before running `state_directions.py`. Exploratory; reuses the cached MIT-States
features and the fold structure of `STATE_CONTROLS_PROTOCOL.md` (seed 20261016, five category groups).

## Question
Retrieval gains from a text-derived state direction do not show that the direction corresponds to the
visual change. Here the correspondence is measured directly: is the bridged text direction for a relation
closer to the visual difference between its states, in held-out object categories, than to the visual
differences of the other four relations?

## Quantities (per fold, relation and preprocessing condition; same preprocessing as the retrieval study)
- Text direction: mean over training categories of (target text - source text), mapped through the ridge
  bridge without its intercept (alpha 1).
- Held-out visual direction: mean over test categories of (mean target image - mean source image), all
  eight images per state. These are differences between groups of images, not before-and-after pairs.
- Alignment matrix: cosine between each relation's text direction (rows) and each relation's held-out
  visual direction (columns). Diagonal = matching relation.
- Primary: margin = mean diagonal cosine - mean off-diagonal cosine, and top-1 match rate (the matching
  relation has the largest cosine in its row; chance 0.2). Averaged over relations, then over the five folds.

## Controls
1. Shuffled bridge: pair text rows with image targets in shuffled order, refit, recompute the same statistics
   (200 shuffles per fold; the null is the fold-averaged statistic).
2. Image-derived reference: direction from training categories' images (not used for the bridge), compared with
   the same held-out visual directions. This is how much agreement is possible without crossing modalities.
3. Stability: 20 random splits of the training categories into halves per relation; cosine between the two
   halves' directions (text and image), averaged.
4. Off-diagonal cells serve as the wrong-relation control; reversing a direction only flips the sign.

## Conditions
The six preprocessing conditions of the retrieval study, to check that conclusions do not depend on them.

## Predictions (stated in advance)
- Reference condition: positive margin and top-1 above 0.2 for the text direction, shuffled-bridge margin near
  zero, and the image-derived reference above the text direction.
- Directions of different relations may share a component (an "improving" state in general), so off-diagonal
  cosines are not expected to be zero.
- Text-direction stability is lower than image-direction stability.
- A negative result (margin not above the shuffled bridge) would mean retrieval gains occur without a stable
  shared state direction, and is reported as such.

## Limits
Five relations, 8 images per state and category, categories repeat across folds, folds are dependent. Intervals
are not claimed; fold ranges are shown. Exploratory. Retrieval-gain linkage is the second Track 1 member's part.
