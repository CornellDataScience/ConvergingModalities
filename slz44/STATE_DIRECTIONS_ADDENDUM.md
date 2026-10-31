# State directions: addendum (written before running the additions)

Adds to `STATE_DIRECTIONS_PROTOCOL.md`. The first run is already done; its numbers stay in the report unchanged.

## 1. Per-category alignment (export, not a new test)
For every held-out (relation, category): cosine between that relation's bridged text direction and the
category's own visual direction (all eight images per state), the mean cosine of the other four relations'
text directions with the same visual direction, their difference, and whether the matching relation has the
largest cosine. Written to `categories.csv` so retrieval-gain analysis can join on `relation, noun`. The
retrieval link itself is another team member's analysis and is not performed here.

## 2. Uncertainty
Category-label bootstrap (2,000 draws; a category resampled as a unit across all relations it appears in),
conditional on the fitted directions. Statistics: mean per-category margin and mean per-category top-1,
averaged within relation and then equally across relations. Per-category directions come from eight images per
state and are noisier than the test-group means used in the matrix statistics, so these are separate
quantities and are reported separately. Intervals are percentile, descriptive, and unadjusted for six conditions.

## 3. Shared-component check (post-hoc)
Suggested by the first run, where the cooking and thawing rows overlapped. Subtract the mean of the five text
directions from each text direction, and the mean of the five held-out visual directions from each visual
direction, then recompute the matrix margin and top-1. Labelled post-hoc; the original statistics remain the
primary results. If top-1 rises and the cooking-thawing overlap shrinks, a shared "state change in general"
component is a plausible explanation; no change would argue against it.

## Not done here
Linking alignment to retrieval gain; more relations; other model pairs.
