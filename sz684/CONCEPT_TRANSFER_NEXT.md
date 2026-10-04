# Next experiment: repeated semantic relations across modalities

The completed pilot in results/difference_transfer/ tests arbitrary held-out endpoint differences. It does not yet test the user's stronger related-concept hypothesis.

Concrete target: transfer red→blue across object categories. A text relation estimated from some categories should shift an image of a held-out red object toward blue images of that same category, without losing object identity.

Required benchmark:
- At least 8 visually unambiguous object categories, e.g. car, chair, mug, bicycle, bag, boat, umbrella, shoe.
- Multiple independent red and blue images for every category, with balanced style/background and visible color labels checked by a human. These are proposed categories, not claims about available examples.
- Separate categories for direction fitting, validation of magnitude, and final evaluation; split image sources to avoid near-duplicate leakage.
- Paired mapping data must exclude final evaluation concepts/sources. Freeze map, layers, and direction magnitude before evaluation.
- Use matched text templates to minimize wording confounds. Average multiple training contrasts rather than selecting a favorable single vector.

Evaluation:
1. Form a relation vector from text training contrasts, then map it into image coordinates.
2. Add a validation-fixed multiple to each held-out source image representation.
3. Retrieve among target-color and source-color images of the same category, plus different-category distractors. Exclude the exact query image.
4. Compare against no translation, reversed relation, shuffled relation/map, and direct target-text retrieval. Report preservation of object identity separately.
5. Compute paired retrieval differences using independent source objects as sampling units; do not count near-duplicate images as independent trials.

Use source↔target reversal and a second relation only after the first protocol is fixed. Do not infer compositional semantics from successful endpoint mapping alone.

Current blocker for this specific semantic experiment: the cached WIT pairs have no verified repeated-relation annotations. Keyword inspection produced ambiguous names and captions rather than a controlled relation dataset. No images or relation labels have been fabricated. Benchmark curation is the next substantive task.
