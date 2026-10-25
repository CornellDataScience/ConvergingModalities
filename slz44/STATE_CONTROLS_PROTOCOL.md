State-transfer preprocessing sensitivity: protocol (commit before extracting features)
Question

A text-derived state relation (for example raw -> cooked, estimated from some object categories) can be mapped into image space with a ridge bridge and added to an image of a held-out category to retrieve images in the requested state. Reported results for this kind of method show a ranking gain over plain image search but a loss of object-category information in the top result. The published pipeline preprocesses features by clipping, L2 normalization, mean subtraction and a 64-dimension PCA. My anisotropy study found that mean subtraction raised mutual k-NN alignment while removing top directions and whitening did not.

Does the choice of preprocessing change state-transfer retrieval, in particular category retention, and do anisotropy diagnostics track the changes? No direction is assumed.

Design

An independent implementation (no shared code, own image sample) of the published method:

Images: MIT-States release archive. Five relations: cleaning (dirty -> clean), cooking (raw -> cooked), ripening (unripe -> ripe), thawing (frozen -> thawed), drying (wet -> dry). A category is eligible for a relation if both states have at least 10 distinct images. Eight images per state/category are sampled with a fixed seed (20261016) after skipping exact duplicates. The sample is a fresh draw and will differ from any earlier one.
Models: BLOOMZ 1.7B (hidden-state index 19, mean-pooled, left-padded) and DINOv2 base (block 10 CLS token), float32. Layers come from an earlier unrelated study and are not searched.
Text: four prompt templates per state/category.
Folds: category labels are split into five groups (seed 20261016; every relation must have a category in every group). Round f tests group f, validates on group f+1, trains on the rest.
Preprocessing (the only thing that varies): clip at the mean per-example 95th percentile of absolute feature values over all layers (training examples), L2-normalize, then a condition step fitted on training rows of each modality, then renormalize.
Bridge: ridge regression (alpha 1, intercept) from text vectors to the mean of the first four training images of the same state/category. Shared direction: mean over training categories and templates of (target text - source text), mapped through the bridge without the intercept. Image-derived direction: from the remaining four training images.
Query: normalize(image + s * direction); s is chosen per round and relation on validation AP from {0, 0.25, 0.5, 1, 2, 4}, ties favoring the smaller value.
Candidates: both states of the test categories supporting the relation, excluding the query image and one target-state image per category (7 per state per category). Relevant means target state and the query's category.
Comparisons: no change, shared text direction, image-derived direction, reversed direction, 100 random directions of equal norm, the other four relations' directions rescaled to equal norm, and direct target text. Random, reversed and wrong-change use the shared direction's chosen step size.
Conditions
reference: subtract training mean, PCA to 64 dimensions.
uncentered_pca64: PCA basis fitted on centered data, applied without subtracting the mean.
center_only: subtract mean, no PCA.
standardize_pca64: subtract mean, divide each dimension by its training standard deviation, PCA to 64.
remove3_pca64: subtract mean, project out the top 3 principal directions, PCA to 64.
whiten_pca64: PCA to 64, scale each component to unit variance.

Parameters (64 dimensions, 3 directions) are fixed here and not tuned.

Sanity gate (before interpreting other conditions)

Under reference the pattern should match the published description: shared text direction has higher average precision and first-result state-and-category success than no change; reversed is below no change; random and wrong-change directions are near no change; and correct-category first-result rate for the shared direction is below the no-change rate. Published numbers for orientation only (different sample, so not expected to match): AP 0.368 -> 0.453, joint first result 20.9% -> 38.7%, category retention 86.6% -> 79.4%. If the pattern fails, debug extraction and data selection first and do not report other conditions.

Outcomes and analysis

Per condition and method: average precision, first-result state-and-category success (joint), state success, and category retention, averaged within category, then relation, then equally across relations. Primary contrasts: shared direction minus no change on AP and on category retention for every condition. Intervals are descriptive: a category-label bootstrap (2,000 draws) conditional on per-category means, unadjusted for six conditions. Diagnostics (mean pairwise cosine, effective rank, top-dimension and top-PC variance share) are logged on test-category vectors per fold and modality; association with outcomes is descriptive only.

Limits
Exploratory; the same data are used for every condition, and categories repeat across folds, so conditions and folds are not independent replications.
Related category labels can fall into different groups, so held-out categories are not fully novel.
MIT-States labels are noisy; scores measure agreement with released labels.
Centering cannot change a difference of two vectors and the ridge intercept cancels in differences, so centering effects should arise in the retrieval step, not the direction itself.
A condition that looks better is a lead for a fresh evaluation with independently verified labels, not a result. Do not pick a condition from these test results and present it as the method.
Features come from a CPU run; compare patterns, not decimals, with other implementations.