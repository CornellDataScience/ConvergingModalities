> Historical proposal/report. The active natural-state study and results are in [STATE_TRANSFER_PAPER.md](../../state_transfer/docs/STATE_TRANSFER_PAPER.md) and [CURRENT_DIRECTION.md](../../../CURRENT_DIRECTION.md). The red-to-blue main experiment was retired at the user's request.

# Proposed confirmatory experiment: reusable red→blue transfer

Draft, October 15, 2026. Not run or preregistered. Finalize category/asset manifests, licenses, hashes and nuisance assignments before final evaluation; use separate retired pilot categories for debugging. Do not make confirmatory claims from the inspected WIT sample.

## Question and intervention

Does one text-derived red→blue vector, learned without final test categories, shift an unseen red object's image embedding toward blue instances of the same category, while preserving category identity?

Use a controlled rendered benchmark first. Each counterfactual pair holds asset identity, camera, lighting and background fixed and changes only foreground color. This distinguishes a color intervention from unrelated-image displacement. Randomize nuisances independently across instances and balance them across categories. Verify that red/blue pixel masks change as intended, backgrounds do not change, and each object is recognizable. This tests controlled visual attributes, not natural-language semantics in general.

## Data budget and split

- 24 recognizable object categories: 8 training, 4 validation, 12 test. Lock the exact named categories and their partition in a manifest before embedding the final test set.
- 40 independent asset identities per category, each rendered in red and blue: 1,920 images. Multiple views of one asset are a single cluster, not independent examples. If only a few assets are available, reduce the claimed sample size and expand the asset collection rather than counting camera views as new objects.
- Training categories: 20 identities per category for fitting the modality map, 20 disjoint identities for image-relation diagnostic estimation. Category overlap within training is allowed; test category/identity overlap is forbidden. This yields 320 paired caption/image map-fitting rows, which is a proposed budget, not available data.
- Four fixed matched text templates per training category; e.g. 'a photo of a red {object}' / 'a photo of a blue {object}'. Treat templates as repeated measurements, not independent samples. The photographic wording must be replaced by rendering wording consistently if pilot validation indicates a domain mismatch, then frozen.
- Validation and test identities and sources must be disjoint from training. Check duplicates at image and asset level. Save category, asset_id, source, license, split, color, nuisance seed, caption, image SHA256, and verification status.
- Acquire licensed assets and inspect labels; these assets are not presently supplied by the WIT cache. No benchmark labels or natural-image results are fabricated.

## Fixed model and representation pipeline

Primary pair: current BLOOMZ 1.7B / DINOv2 base. Freeze existing pilot layers 19 / 10 rather than reselecting on final test data. Fit clipping thresholds, centered PCA128 and ridge alpha=1 only on new map-fitting rows, separately per modality. The new representation distribution differs from WIT; record this design choice. Normalize projected endpoints as in the existing pilot. Use an affine endpoint map, whose intercept cancels for directions.

For training categories c and templates t, compute d_text = mean(x_blue(c,t) − x_red(c,t)). Map once: d = d_text W. No test-category text contrasts may enter d. Choose translation magnitude s from {0, .25, .5, 1, 2, 4} on validation categories only using the primary metric; break ties toward smaller s. If validation chooses zero, retain and report that outcome. Score translated image queries normalize(y_red + s d). Targets remain unmodified normalized image features. Do not reapply the PCA transform to vectors already in PCA coordinates.

## Primary evaluation and controls

Primary outcome: per-category mean of [cos(translated query, matched blue counterfactual) − cos(translated query, original red query)]. Compare its category-macro value against the same margin with no translation. The original red vector is intentionally the source reference for this paired margin, not a retrieved duplicate. This measures moving toward the intervention, with object identity exactly matched.

Report also retrieval in two banks: (a) red/blue images within the correct category and (b) all test categories. Exclude the exact red query from retrieval candidates; retain matched blue and use equal numbers of red and blue independent distractors. Report target-color top-1, correct-category top-1, joint target-color-and-category top-1, and matched-blue rank. Keep candidate counts fixed across controls; publish exact bank indices. A pure blue attractor can improve color and destroy identity, so color alone cannot establish successful transfer.

Controls, all with the same train-only preprocessing and banks:
1. No shift; reversed d at the same magnitude; category pairing shuffled in text contrasts (preserve marginals, permute blue-category labels).
2. Random directions matched to ||d||, generated from 100 fixed seeds; no best-seed reporting. Shuffled-map fits with the same fixed layers, but treat these as diagnostics unless the complete adaptive procedure is repeated.
3. Source-anchored direct target-text retrieval: map the caption of the requested blue test object and retrieve. This intentionally uses target text and is a stronger-information recognition reference; it is not the same task as a universal vector without target wording.
4. Image-derived relation averaged over training-only counterfactuals; freeze its scale on validation. Diagnostic reference, not a guaranteed upper bound.
5. Ridge versus orthogonal Procrustes after matched PCA128. Fixed baseline fit only on map-training rows; validate scale separately. Add an explicitly cross-modally pretrained encoder only as a labeled reference.

## Inference and success criteria

Primary inference unit is held-out category (12), with asset identities nested within category. Compute paired category-macro effects and a two-stage bootstrap that resamples categories then assets within each sampled category, using 20,000 saved draws. Report sensitivity to category-only resampling and leave-one-category-out estimates. These intervals depend on representativeness/exchangeability of chosen categories; more views do not increase the number of independent categories.

Two planned primary contrasts: transferred direction versus no shift, and versus the mean random-direction control. Use Bonferroni simultaneous 97.5% intervals for these two contrasts (approximate bootstrap coverage), while reporting ordinary 95% intervals descriptively. Predeclare an object-preservation tolerance of at most 5 percentage points loss in correct-category retrieval versus no shift, assessed by a one-sided 95% lower confidence bound above −.05. This is an additional required guardrail, not a claim of equivalent performance. If the interval is too wide, call preservation inconclusive.

Claim a controlled reusable relation only if both adjusted primary margin contrasts exclude zero positively and the object-preservation guardrail passes. Also require the result to exceed the reversed-direction control descriptively; investigate any reversed-direction benefit. Exact paired intervention retrieval provides a stricter secondary measure. A failed significance threshold is inconclusive unless intervals exclude a preregistered meaningful effect; choose that practical-effect threshold after pilot variance estimation and before locking test data.

Power planning: estimate between-category and between-asset variance on retired pilot categories; simulate the planned hierarchical procedure and publish a minimum-detectable-effect curve before final evaluation. Add categories if sensitivity is insufficient. The counts above are a starting resource budget, not a demonstrated adequate sample size.

## Interpretable outcomes and expansion

- Text and image relations transfer: replicate in reverse, on a second relation, and with a second model family using fresh test data.
- Only image relation transfers: investigate text direction stability and map mismatch; endpoint fit may discard intervention information.
- Only target-text retrieval succeeds: recognition exists but a universal additive direction has not been shown.
- Neither relation transfers: assess whether category-specific directions outperform one global vector, with all comparison choices made on validation data.

After a positive rendered result, curate independent natural photographs (at least 10 verified independent source objects per color per test category as an initial budget). Background/source imbalance and color ambiguity must be audited. Natural photos lack exact counterfactual pairs, so use a separately specified retrieval endpoint rather than reusing the paired intervention margin. Success on renders alone supports only the controlled domain.
