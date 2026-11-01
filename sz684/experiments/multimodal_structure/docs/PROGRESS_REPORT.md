# Cross-modal representation study: progress report

**Report date:** October 24, 2026. **Latest completed evaluation:** October 23, 2026.

We built and ran a study asking whether models trained separately on text, images, and audio represent shared concepts similarly, and whether a linear concept classifier learned in one model can transfer to another. The primary comparisons are image–text and vision–audio. All encoders remained frozen: we did not pretrain or fine-tune their neural-network weights.

The main completed study contains **80 concepts, 1,200 paired audiovisual recordings, six models, 15 geometry comparisons, and 60 ordered category-axis transfer tests**. It uses released event labels, as approved for this stage. Its audiovisual CAV results concern category distinctions, not verified physical changes to an object.

The sections below describe the implementation and saved artifacts, rather than treating planned work as completed work. The original score report is [week-8 RESULTS.md](/Users/sz/Programming/CDS/conv-mod/sz684/experiments/multimodal_structure/docs/week8/RESULTS.md); the fixed protocol is [WEEK8_PROTOCOL.md](/Users/sz/Programming/CDS/conv-mod/sz684/experiments/multimodal_structure/docs/WEEK8_PROTOCOL.md).

## 1. Work completed and reused

| Stage | What we used or completed | Role in the current study |
| --- | --- | --- |
| Earlier object-state study | MIT-States images; frozen BLOOMZ-1.7B and DINOv2; five state relations such as raw → cooked | Provided feature caches, splitting/retrieval ideas, and a supervised-bridge baseline. Its data were not pooled with the new audiovisual dataset. |
| First multimodal pilot | 24 concepts, 72 recordings, five frozen models; geometry evaluation and reuse of the older image–text state features | Checked acquisition, extraction, metrics, and CAV/map fitting. Results were exploratory. |
| Expanded study through week 8 | 80 new concept labels, 15 recordings each, two models per modality; validation-selected fits and a locked test evaluation | Supplies the main results described here. All first-pilot concept labels and original recording IDs were excluded. |

The earlier image–text state pipeline fitted a **ridge translator** between text and image embeddings. The new study instead uses **orthogonal Procrustes** to align spaces and **ridge binary classifiers** to define CAVs; these are different fitted components. Historical work and its outputs remain preserved.

Files: [data/legacy.py](/Users/sz/Programming/CDS/conv-mod/sz684/experiments/multimodal_structure/data/legacy.py) imports saved state features; [pilot results](/Users/sz/Programming/CDS/conv-mod/sz684/experiments/multimodal_structure/docs/RESULTS.md) records the earlier pilot; [state-transfer bridge.py](/Users/sz/Programming/CDS/conv-mod/sz684/experiments/state_transfer/training/bridge.py) implements the older translator.

## 2. Data used

### 2.1 Source and selection

We used VGGSounder’s released event annotations and the underlying public VGGSound audiovisual media. A concept is an event label, such as a particular instrument being played or an animal vocalization. Each selected recording contributes an audio input and one image input. Text inputs are descriptions constructed from the same released concept name.

Selection retained recordings with:

- Exactly one event label annotated as both audible and visible.
- No released background-music, voice-over, or static-image flag.
- A distinct original recording ID within the selected sample.
- A concept with sufficient available recordings to support the requested sample.

Other audio-only or visual-only labels can still occur in a retained recording. This filter does not guarantee that the recording contains only one event.

The annotation revision was `5a91bf6851d8cbf830d85b00ba0953f1befaffa4`; the media revision was `dc7815aa65132c48f070686e0004180f03b5abcf`. Sampling order and backup recordings were saved before feature extraction. Dataset size was a feasibility choice, not a formally demonstrated power calculation.

Files: [data/expanded.py](/Users/sz/Programming/CDS/conv-mod/sz684/experiments/multimodal_structure/data/expanded.py) defines sampling and acquisition; [data/download_vggsounder.py](/Users/sz/Programming/CDS/conv-mod/sz684/experiments/multimodal_structure/data/download_vggsounder.py) records source revisions; [sampling_plan.json](/Users/sz/Programming/CDS/conv-mod/sz684/experiments/multimodal_structure/results/week8_study/sampling_plan.json) contains the actual selection, exclusions, families, and splits.

### 2.2 Split and sample counts

Concepts were assigned to provisional semantic families, such as instrument groups or animal groups. Whole families were kept in one split. Assignment was seeded and required both category-axis classes to be sufficiently represented in training, validation, and test data.

| Split | Concepts | Images | Audio clips | Text descriptions |
| --- | ---: | ---: | ---: | ---: |
| Training | 50 | 750 | 750 | 200 |
| Validation | 14 | 210 | 210 | 56 |
| Test | 16 | 240 | 240 | 64 |
| **Total** | **80** | **1,200** | **1,200** | **320** |

The image and audio columns are two views of the same recordings, not 2,400 independent situations. Text descriptions are generated formulations, not additional observed recordings. A recording ID is a useful grouping unit, but does not establish uploader independence, scene independence, or absence from pretraining data.

A provisional taxonomy-matching error was corrected before extraction, and the original sampling plan was preserved. These groups are practical safeguards against obvious related-label leakage, not an independently validated taxonomy.

Files: [data/split.py](/Users/sz/Programming/CDS/conv-mod/sz684/experiments/multimodal_structure/data/split.py), [final manifest.json](/Users/sz/Programming/CDS/conv-mod/sz684/experiments/multimodal_structure/results/week8_study/manifest.json), and [concepts_and_axes.csv](/Users/sz/Programming/CDS/conv-mod/sz684/experiments/multimodal_structure/docs/week8/concepts_and_axes.csv).

### 2.3 Quality checks

We checked required metadata, unique sample IDs, source/family separation, file hashes, image decoding, audio decoding, and silence. Three silent recordings were replaced using the prespecified backup order, without consulting model scores. The final audit found no original source ID or exact decoded image crossing splits. It recorded two within-split exact-image duplicate groups, containing four and two images respectively; therefore, distinct recording IDs do not imply distinct visual content.

These checks establish basic input integrity. They do **not** independently verify that the event is visible in every selected still or audible in every clip.

Files: [data/prepare.py](/Users/sz/Programming/CDS/conv-mod/sz684/experiments/multimodal_structure/data/prepare.py), [data/audit.py](/Users/sz/Programming/CDS/conv-mod/sz684/experiments/multimodal_structure/data/audit.py), [media_audit_final.json](/Users/sz/Programming/CDS/conv-mod/sz684/experiments/multimodal_structure/results/week8_study/media_audit_final.json), and [media_replacements.json](/Users/sz/Programming/CDS/conv-mod/sz684/experiments/multimodal_structure/results/week8_study/media_replacements.json).

## 3. Models used

| Modality | Model/checkpoint | Pretraining basis used for eligibility | Saved feature width |
| --- | --- | --- | ---: |
| Text | `bigscience/bloomz-560m` | Text-only language modeling and text instruction tuning | 1,024 |
| Text | `EleutherAI/pythia-410m` | Text-only autoregressive language modeling | 1,024 |
| Vision | Original DINOv2-base backbone | Image-only self-distillation; no language-aligned head | 768 |
| Vision | `facebook/vit-mae-base` | Image-only masked-autoencoder pretraining | 768 |
| Audio | BEATs iter1 pretraining checkpoint | Audio-only pretraining with a random-projection tokenizer | 768 |
| Audio | AudioMAE AudioSet-2M pretraining backbone | Audio-only masked-autoencoder pretraining; no supervised classification head | 768 |

We used immutable revisions and recorded checkpoint/source hashes. BEATs came from an original-format checkpoint mirror; AudioMAE used a documented timm conversion of its pretraining weights. Their provenance is recorded, but we did not independently establish byte-for-byte equivalence to the authors’ original download artifacts. An incompatible converted BEATs checkpoint was rejected during the pilot.

The modality-only requirement applies to the encoders’ training history. Our later alignment maps and probes do use shared concept correspondences and category labels. Their success cannot establish that alignment arose without supervision.

Files: [run_summary.json](/Users/sz/Programming/CDS/conv-mod/sz684/experiments/multimodal_structure/docs/week8/run_summary.json) records revisions and provenance; [models.week8.local.json](/Users/sz/Programming/CDS/conv-mod/sz684/experiments/multimodal_structure/results/models.week8.local.json) contains the actual local model settings; [setup_week8_models.py](/Users/sz/Programming/CDS/conv-mod/sz684/experiments/multimodal_structure/data/setup_week8_models.py) downloads the added backbones. Published lineage references are collected in [the protocol](/Users/sz/Programming/CDS/conv-mod/sz684/experiments/multimodal_structure/docs/WEEK8_PROTOCOL.md).

## 4. Preprocessing and feature extraction

An embedding is the fixed-length vector returned for an input. Different models need not produce vectors of the same width.

### 4.1 Text

Each concept received four formulations: the label itself, “an example of …”, “the sound of …”, and “an event involving …”. Each model tokenized its own input. We padded batches, excluded padding from pooling, and averaged the last hidden-layer vectors over non-padding tokens. The configured token limits were 64 for BLOOMZ and 128 for Pythia.

These descriptions contain the event name explicitly. This gives text a different information source from recognizing an event in pixels or sound.

File: [extraction/text.py](/Users/sz/Programming/CDS/conv-mod/sz684/experiments/multimodal_structure/extraction/text.py).

### 4.2 Images

For the main audiovisual sample, we extracted the frame at five seconds, intended as the midpoint of the ten-second recording, and converted it to RGB. Each encoder retained its own image processor rather than sharing an arbitrary resize or normalization rule.

DINOv2 used its configured 518 × 518 input and the CLS token from its last transformer block. MAE used its processor’s 224 × 224 input and image normalization, disabled patch masking, fixed the patch order, and returned the output CLS token. These extraction points have architecture-specific normalization conventions; they are not guaranteed to be equivalent layers.

Files: [frame extraction in expanded.py](/Users/sz/Programming/CDS/conv-mod/sz684/experiments/multimodal_structure/data/expanded.py), [DINOv2 extraction](/Users/sz/Programming/CDS/conv-mod/sz684/experiments/multimodal_structure/extraction/vision.py), and [MAE extraction](/Users/sz/Programming/CDS/conv-mod/sz684/experiments/multimodal_structure/extraction/hf_vision.py).

### 4.3 Audio

We decoded waveforms, averaged channels to mono, resampled to 16 kHz, retained at most the first ten seconds, and rejected empty, nonfinite, or silent inputs. The main data already consisted of short event recordings.

BEATs processed one unpadded waveform at a time through its own acoustic processor; we averaged its output token features. AudioMAE used 128 filterbank bins, a fixed 1,024-frame window, its published normalization constants, and its CLS token. Short AudioMAE filterbanks were padded before normalization as required by that processor. Those padded values are not extra observations.

Files: [data/audio.py](/Users/sz/Programming/CDS/conv-mod/sz684/experiments/multimodal_structure/data/audio.py), [BEATs extraction](/Users/sz/Programming/CDS/conv-mod/sz684/experiments/multimodal_structure/extraction/audio.py), and [AudioMAE extraction](/Users/sz/Programming/CDS/conv-mod/sz684/experiments/multimodal_structure/extraction/audiomae.py).

### 4.4 Numerical precision and caches

Encoders ran in evaluation/inference mode. Text, vision, and AudioMAE extraction used the Mac MPS device; BEATs used CPU. Saved features were float32. Analysis loaded them as float64 for numerical fitting and scoring.

Every cache records sample IDs, dataset identity, model revision, processing settings, and extraction-source identity. Batches were saved atomically and could resume only with matching settings. Each text model produced a 320 × 1,024 matrix; each image/audio model produced a 1,200 × 768 matrix.

Files: [extraction/extract.py](/Users/sz/Programming/CDS/conv-mod/sz684/experiments/multimodal_structure/extraction/extract.py), [extraction/cache.py](/Users/sz/Programming/CDS/conv-mod/sz684/experiments/multimodal_structure/extraction/cache.py), and [saved feature caches](/Users/sz/Programming/CDS/conv-mod/sz684/experiments/multimodal_structure/results/week8_study/features).

## 5. What we trained

### 5.1 Separate map anchors and probe examples

Within each training concept, sources were sorted and divided into disjoint halves. For image/audio models, seven recordings per concept supplied map anchors and the other eight supplied probe training. For text, two formulations supplied anchors and two supplied probe examples. Validation and test families never supplied fitting rows.

The anchor recordings were averaged into one training centroid per concept. These shared concept centroids supplied correspondences between models; the map was not trained to match an arbitrary image to an unrelated audio example.

File: [source_partition and build_fits in training/build_fits.py](/Users/sz/Programming/CDS/conv-mod/sz684/experiments/multimodal_structure/training/build_fits.py).

### 5.2 PCA and orthogonal alignment

For each model, we fitted centering and PCA on training-anchor centroids only. Candidate shared widths were 8, 16, and 32, subject to effective rank. PCA was not whitened, and projected endpoints were not renormalized.

For each ordered model pair, centered orthogonal Procrustes fitted a rotation/reflection matrix R between the corresponding projected training centroids. With row vectors, endpoint transport is:

`target ≈ (source − source_mean) @ R + target_mean`.

No nonlinear translator, encoder update, or isotropic scale was fitted in this stage.

Files: [training/preprocessing.py](/Users/sz/Programming/CDS/conv-mod/sz684/experiments/multimodal_structure/training/preprocessing.py) and [training/alignment.py](/Users/sz/Programming/CDS/conv-mod/sz684/experiments/multimodal_structure/training/alignment.py).

### 5.3 Category probes and CAVs

We defined two category axes from released event names and the saved family mapping:

| Axis | Positive class | Negative class |
| --- | --- | --- |
| Instrument performance | Instrument families | Other retained events; this is not a definition of all music |
| Biological versus manufactured | Explicitly listed biological-source families | Explicitly listed manufactured-source families, including instruments |

Unsupported categories were excluded from the second probe task. They could still contribute unlabeled concept anchors. The concept-level class counts were:

| Axis | Training: positive / negative / excluded | Validation | Test |
| --- | --- | --- | --- |
| Instrument performance | 12 / 38 / 0 | 4 / 10 / 0 | 4 / 12 / 0 |
| Biological versus manufactured | 9 / 30 / 11 | 7 / 6 / 1 | 11 / 4 / 1 |

A class-balanced ridge binary classifier was fitted on the probe-training sources. Its coefficient vector, perpendicular to the decision boundary, is the CAV. This is least-squares classification, not logistic regression and not full TCAV output-derivative analysis.

We chose ridge regularization from 0.1, 1, and 10 using validation balanced accuracy. We then selected the PCA width separately for each map direction using mean validation transferred accuracy across the two axes. Ties preferred smaller settings. Of the 30 ordered maps, 16 selected width 32, ten selected width 8, and four selected width 16.

For orthogonal R, a source CAV transfers as `source_CAV @ R`. The classifier’s decision-plane origin was transported too; copying the target probe’s origin would introduce extra target supervision. These rules apply to the declared projected Euclidean spaces and do not justify transporting a classifier normal through an arbitrary ridge map as if it were a displacement.

Files: [category definitions in data/axes.py](/Users/sz/Programming/CDS/conv-mod/sz684/experiments/multimodal_structure/data/axes.py), [training/probes.py](/Users/sz/Programming/CDS/conv-mod/sz684/experiments/multimodal_structure/training/probes.py), [validation selection in build_fits.py](/Users/sz/Programming/CDS/conv-mod/sz684/experiments/multimodal_structure/training/build_fits.py), and [saved week7_fits.json](/Users/sz/Programming/CDS/conv-mod/sz684/experiments/multimodal_structure/results/week8_study/week7_fits.json).

## 6. How we measured results

### 6.1 Native representation geometry

Geometry metrics used native, unaligned features—not the learned Procrustes maps or PCA spaces. Within each model we averaged within source first, then averaged sources equally into concept centroids. Every comparison used the same 16 held-out concepts.

| Measure | Calculation | What it tests |
| --- | --- | --- |
| RSA | Cosine-distance matrix, followed by Spearman correlation between matching strict upper triangles | Agreement in relative distance ordering; 16 concepts give 120 unique pairwise distances |
| Top-3 overlap | Fraction of a concept’s three neighbors shared across models | Local labeled-neighborhood preservation |
| Jaccard | Neighbor intersection divided by neighbor union | A second neighborhood measure with a different denominator |
| Linear CKA | Normalized agreement between centered Gram matrices of unnormalized features | Agreement in centered geometry, allowing different feature widths |

Neighbors excluded self and used deterministic tie handling. This was labeled graph comparison, not graph isomorphism or graph edit distance. CKA and RSA have different scales; neither alone proves semantic identity.

We evaluated all 15 unordered model pairs: four image–text and four vision–audio primary pairs, four text–audio secondary pairs, and three within-modality comparisons. We saved 999 whole-concept correspondence shuffles per pair. Their one-sided permutation p-values are unadjusted and rely on exchangeability; related concept families complicate that assumption. They are not final significance decisions.

Files: [rsa.py](/Users/sz/Programming/CDS/conv-mod/sz684/experiments/multimodal_structure/evaluation/rsa.py), [neighbors.py](/Users/sz/Programming/CDS/conv-mod/sz684/experiments/multimodal_structure/evaluation/neighbors.py), [cka.py](/Users/sz/Programming/CDS/conv-mod/sz684/experiments/multimodal_structure/evaluation/cka.py), [controls.py](/Users/sz/Programming/CDS/conv-mod/sz684/experiments/multimodal_structure/evaluation/controls.py), and [evaluate.py](/Users/sz/Programming/CDS/conv-mod/sz684/experiments/multimodal_structure/evaluation/evaluate.py).

### 6.2 CAV alignment and classifier transfer

For all 30 ordered maps and both axes, we measured signed cosine agreement between transferred and target CAVs, and **balanced accuracy** of the transported classifier on target-model test features. Balanced accuracy averages recall for the two classes, so predicting only the majority class scores 0.5 rather than receiving credit for class imbalance.

Controls were the target model’s separately trained probe, a map fitted with shuffled training-anchor correspondences, and a reversed source classifier normal. The shuffled map is a descriptive control, not a full refit null distribution. Target probe accuracy is a reference, not a theoretical oracle; its width can differ across map directions.

The 60 tests produced **10,540 per-example prediction rows**. Those rows reuse the same held-out examples across models and axes; they are not 10,540 independent observations.

Files: [directions.py](/Users/sz/Programming/CDS/conv-mod/sz684/experiments/multimodal_structure/evaluation/directions.py), [locked.py](/Users/sz/Programming/CDS/conv-mod/sz684/experiments/multimodal_structure/evaluation/locked.py), [week8_test.json](/Users/sz/Programming/CDS/conv-mod/sz684/experiments/multimodal_structure/results/week8_study/week8_test.json), and [per-example predictions](/Users/sz/Programming/CDS/conv-mod/sz684/experiments/multimodal_structure/results/week8_study/week8_test_predictions.csv).

### 6.3 Repeatability and evaluation locks

Before test scoring, we compared concept RDMs from disjoint source halves using training and validation data. For text, this measured formulation stability rather than independent-event reliability. It was descriptive, not a corrected noise ceiling or confidence interval. Validation RDM correlations ranged from −0.0405 for BLOOMZ to 0.8983 for BEATs. The low BLOOMZ value limits interpretation of its weak cross-modal scores in this particular extraction configuration.

Fitting, locking, and scoring were separate stages. The lock recorded dataset, feature, fitted-parameter, protocol, and code hashes. Scoring refused changed inputs/software and did not overwrite a completed evaluation. Saved source snapshots match the lock. Software checks included tests that changing test features does not alter fitted parameters, and that modified caches cannot be locked against old fits. The recorded suite passed 58 tests.

Files: [reliability.py](/Users/sz/Programming/CDS/conv-mod/sz684/experiments/multimodal_structure/evaluation/reliability.py), [repeatability results](/Users/sz/Programming/CDS/conv-mod/sz684/experiments/multimodal_structure/results/week8_study/week7_reliability.json), [evaluation_lock.json](/Users/sz/Programming/CDS/conv-mod/sz684/experiments/multimodal_structure/results/week8_study/evaluation_lock.json), and [lock/separation tests](/Users/sz/Programming/CDS/conv-mod/sz684/experiments/multimodal_structure/tests/test_locked_study.py).

## 7. Results so far

For worked calculations and metric-specific random baselines, see [Results explained](/Users/sz/Programming/CDS/conv-mod/sz684/experiments/multimodal_structure/docs/week8/RESULTS_EXPLAINED.md) and [CHANCE_BASELINES.csv](/Users/sz/Programming/CDS/conv-mod/sz684/experiments/multimodal_structure/docs/week8/CHANCE_BASELINES.csv).

The complete per-pair geometry table is in [geometry_scores.csv](/Users/sz/Programming/CDS/conv-mod/sz684/experiments/multimodal_structure/docs/week8/geometry_scores.csv). Representative primary comparisons are:

| Pair | RSA | CKA | Top-3 overlap |
| --- | ---: | ---: | ---: |
| Pythia / DINOv2 | 0.4653 | 0.7334 | 0.2292 |
| Pythia / image MAE | 0.3107 | 0.6944 | 0.3958 |
| DINOv2 / BEATs | 0.3467 | 0.4092 | 0.3542 |
| DINOv2 / AudioMAE | 0.4048 | 0.5746 | 0.3125 |

The random top-3 overlap expectation for 16 concepts is 3/15 = 0.2. Within audio, BEATs/AudioMAE had RSA 0.7094, the highest RSA in this roster. Shared data/objectives and sampling conditions must be considered before interpreting these comparisons as general convergence.

Mean category-transfer scores across the eight map directions in each primary comparison were:

| Comparison | Axis | Transferred balanced accuracy | Target probe | Shuffled-anchor map |
| --- | --- | ---: | ---: | ---: |
| Image–text | Instrument performance | 65.31% | 78.87% | 59.24% |
| Image–text | Biological/manufactured | 71.49% | 76.48% | 48.88% |
| Vision–audio | Instrument performance | 65.28% | 77.33% | 50.90% |
| Vision–audio | Biological/manufactured | 65.99% | 78.67% | 50.56% |

These are descriptive averages of dependent, fixed-fit comparisons—not independent replications or uncertainty-adjusted effects. Transfer is asymmetric, and several directions stayed at chance. The unscaled orthogonal map operates in native-unit PCA spaces; source/target scale and decision-threshold calibration can affect classification even when geometry or CAV angles agree. No post-test calibration was added to improve results.

Files: [all category scores](/Users/sz/Programming/CDS/conv-mod/sz684/experiments/multimodal_structure/docs/week8/category_scores.csv), [geometry heatmaps](/Users/sz/Programming/CDS/conv-mod/sz684/experiments/multimodal_structure/docs/week8/geometry_heatmaps.png), [category-transfer plots](/Users/sz/Programming/CDS/conv-mod/sz684/experiments/multimodal_structure/docs/week8/category_transfer.png), and [validation and interpretation notes](/Users/sz/Programming/CDS/conv-mod/sz684/experiments/multimodal_structure/docs/week8/VALIDATION.md).

## 8. What these results do and do not establish

The completed work demonstrates an executable, documented pipeline and gives held-out point estimates of concept-geometry agreement and category-classifier transfer across independently pretrained modalities. Encoder weights stayed fixed throughout.

It does not establish that representations are identical, that all semantic relations transfer, or that audiovisual state changes preserve object identity. Category CAVs are classifier normals; they are not automatically meaningful object-editing displacements. The older image–text state-retrieval work remains separate and includes its own identity-retention assessment.

The remaining work is refit-inclusive uncertainty, family-aware inference and multiplicity adjustment, independent-source replication, and verified audiovisual state-change data for stronger relational claims. Released video-level labels, provisional families, still-frame sampling, explicit text cues, low template stability in some configurations, and uncertain pretraining overlap remain material limitations.

For commands and file organization, use [the experiment README](/Users/sz/Programming/CDS/conv-mod/sz684/experiments/multimodal_structure/README.md) and [WEEK8_RUN_GUIDE.md](/Users/sz/Programming/CDS/conv-mod/sz684/experiments/multimodal_structure/docs/WEEK8_RUN_GUIDE.md). Large data, features, fits and locks are stored under the Git-ignored [results/week8_study folder](/Users/sz/Programming/CDS/conv-mod/sz684/experiments/multimodal_structure/results/week8_study); compact reports, plots, and score exports are under [docs/week8](/Users/sz/Programming/CDS/conv-mod/sz684/experiments/multimodal_structure/docs/week8).
