# Locked held-out study: October 23, 2026

The user approved released labels for this stage rather than waiting for new human
reannotation. This changes the interpretation: the study evaluates released
VGGSounder event labels, not independently verified physical transformations.

## Before test scoring

Select 80 concepts with 15 distinct original recording IDs per concept. Exclude
all first-pilot concepts and recording IDs. Pin annotation and media revisions.
Save label selection and backup order before downloading or extracting features.
Use broad provisional family groups to keep obvious related classes together.
Require at least 12 concepts and at least four examples of each category-axis
class per split. Choose the first seeded family assignment meeting coverage;
never inspect model scores to choose an assignment.

An initial whole-word matching error was caught before extraction. The initial
sampling plan is preserved. The corrected grouping and deterministic coverage
assignment are recorded as a pre-extraction amendment. Three silent recordings
were replaced by the next usable prespecified backups. No model result informed
those changes. Same-split exact-image duplicates are reported; no source ID or
exact decoded image crosses splits. A unique source ID does not guarantee a
completely independent scene, uploader or pretraining example.

The resulting design has 50 training, 14 validation and 16 test concepts, and
four label-derived text descriptions per concept. Fifteen audiovisual sources
support descriptive repeatability checks; this is a feasibility sample-size
choice, not a claim of formally demonstrated statistical power. Larger inference
and replication remain scheduled after week 8.

## Models and preprocessing

Freeze BLOOMZ-560m, Pythia-410m, original DINOv2-base, original MAE-base,
BEATs iter1, and pretrained AudioMAE. The two masked-autoencoder adapters are
new. All have unimodal pretraining; do not use CLIP/CLAP, a supervised AudioMAE
checkpoint, or a language-aligned DINO head. Model specs include immutable
revisions, local file/weight hashes, processor/pooling and conversion provenance.
AudioMAE is a published timm conversion of the pretraining backbone; exact
byte equivalence to the authors' checkpoint is not independently asserted.

Use documented native processors, last-layer CLS features for image transformers
and AudioMAE, token-mean BEATs, and masked-mean text features. MAE uses every image
patch with fixed patch order. No layer/pooling is selected using test scores.
Geometry uses unaligned native concept centroids. Average sources equally.

## Category CAVs

1. Instrument performance versus other events: positive families begin with
   `instrument_`; other events are negative. This is instrumental performance,
   not all music (singing is not automatically positive).
2. Biological versus manufactured sound sources: positive and negative families
   are explicitly enumerated in `data/axes.py`. Exclude unsupported natural
   phenomena or ambiguous categories from this particular contrast.

These are researcher-derived category labels based on released event names.
Export the full mapping for inspection. They are not released binary relation
annotations, and do not define changes preserving the same object. Consequently,
category-classifier transfer has no meaningful noun-retention retrieval metric.
The historical image–text state-transfer arm remains separate and exploratory;
its endpoint retrieval and identity metrics are not substituted for AV results.

Within each training concept, sort source IDs and put the first half into map
anchors and the rest into probe training. Anchor sources and probe sources are
disjoint. Use matching training-concept centroids to fit separate PCA transforms
and Procrustes. Test families do not enter either operation.

Choose probe alpha from 0.1, 1, 10 by validation balanced accuracy. Choose common
PCA width from 8, 16, 32 (effective rank permitting) per map direction by mean
validation transferred balanced accuracy over both category axes. Ties choose
smaller values. Save all transforms, probes, curves and training source lists.
Development repeatability uses train/validation sources only.

## Lock and evaluate

Separate `week7-fit`, `week8-lock`, and `week8-score` commands. The lock records
fit/data/feature/software hashes, axes, k=3, and 999 correspondence shuffles.
Test scoring refuses changed inputs or software and never overwrites a finished
result. A held-out test is defined relative to this fitting/selection pipeline;
pretraining-set independence is not established.

Evaluate all 15 unordered model pairs for native-space RSA, CKA and neighbor
Jaccard/intersection. Image–text and vision–audio pairs are primary; text–audio
and within-modality pairs are secondary. Save per-concept distances and neighbors.
Evaluate 30 ordered maps × two axes for signed CAV cosine, transported balanced
accuracy, target-probe accuracy, shuffled-anchor and reversed-normal controls.
Save per-example predictions and original source/family IDs.

Report fixed-fit scores, class/sample support, validation choices and limitations.
No significance declaration or semantic-equivalence claim follows from raw CKA,
RSA, an unadjusted permutation p-value, or a favorable direction. Refit uncertainty,
family-block inference, multiplicity adjustments and independent-source
replication belong to the subsequent inference stage.

## Model lineage references

- [Pythia base model](https://huggingface.co/EleutherAI/pythia-410m).
- [Image MAE pretraining model](https://huggingface.co/facebook/vit-mae-base).
- [AudioMAE paper](https://arxiv.org/html/2207.06405v3), particularly its audio-only from-scratch pretraining and ImageNet-initialization ablations.
- [AudioMAE pretraining conversion](https://huggingface.co/gaunernst/vit_base_patch16_1024_128.audiomae_as2m).

The published audio-only lineage is the eligibility basis. Weight conversion is
not a new training stage; conversion provenance remains distinct from an
independent byte-for-byte verification of the original artifact.
