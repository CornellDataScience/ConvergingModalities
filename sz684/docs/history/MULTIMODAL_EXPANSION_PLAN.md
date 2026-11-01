> Historical planning document from October 18, 2026. Repository findings and proposed dates below describe that time, not the current implementation. The implemented study is indexed in [the multimodal README](../../experiments/multimodal_structure/README.md), with [week-8 results](../../experiments/multimodal_structure/docs/week8/RESULTS.md). Current follow-up planning is in [STATE_RELATIONS_PLAN.md](../planning/STATE_RELATIONS_PLAN.md).

# Independently trained unimodal models: cross-modal research plan

Planning date: October 18, 2026. Proposed start: October 19. This is a proposed study, not a preregistration or an implemented pipeline.

## Objective and scope

Test whether independently trained text-only, image-only, and audio-only encoders share concept geometry and transferable relations. Image–text and vision–audio are the primary comparisons. Within-modality comparisons are calibration controls; text–audio is a useful secondary comparison.

The requirement concerns encoder training history, not whether researchers can use labels to identify corresponding concepts. Text-only language modeling is eligible. Image encoders trained with captions, audio encoders trained with transcripts or text contrastive losses, and encoders distilled from multimodal teachers are ineligible for the primary roster. Check every stage, including initialization, tokenizer/teacher training, and fine-tuning. Use the original unimodal backbones, not CLIP, SigLIP, CLAP, ImageBind, audiovisual-pretrained models, or language-aligned DINO heads.

The main study freezes pretrained encoders. Only preprocessing, linear probes, and alignment maps are fitted. This tests structure in independently pretrained representations. Supervised maps cannot establish that cross-modal alignment arose without correspondence supervision. Training foundation models from scratch is a separate, substantially larger project.

## Repository findings

- The active MIT-States experiment uses frozen BLOOMZ-1.7B and DINOv2-base, five state relations, five global noun-label folds, 992 images, and 496 source queries. Only the ridge bridge is trained. The recorded result improves AP over no shift but fails the noun-preservation guardrail; preserve that guardrail in follow-ups.
- `experiments/state_transfer/data/state_data.py` provides sampling, exact deduplication, contact sheets, hashes, and global noun groups. It inherits model revisions from a historical baseline manifest and couples data creation to that model pair.
- `experiments/state_transfer/data/state_extract.py` selects the first model for a modality, writes `features_language`/`features_vision`, assumes transformer vision blocks and CLS pooling, and saves one selected layer. It has no audio path. Current state-layer indices must not become defaults for all new architectures.
- `experiments/baseline/data/experiment.py` already has a multi-model loop, model-specific caches, and layer extraction. Reuse the design, rather than starting from scratch.
- `experiments/baseline/evaluation/alignment.py` already measures neighbor intersection divided by k with self-exclusion and correspondence permutations. This is not Jaccard and is not a graph-isomorphism test.
- `experiments/heldout_layers/` already separates layer selection from scoring. `shared/training.py`, `shared/data.py`, and `experiments/state_transfer/training/bridge.py` provide ridge, PCA, and Procrustes building blocks.
- `state_analysis.py` provides held-out retrieval, no-shift/reversed/random/wrong-relation controls, and noun-cluster bootstrap machinery. Its uncertainty conditions on fixed fits and galleries; the new study must explicitly decide whether to refit these within resampling.
- There is no general RSA, linear CKA, or CAV evaluation pipeline. Existing state differences are centroid contrasts, not classifier-normal CAVs.
- The existing data audit reports ambiguous state labels, unpaired source/target photos, and related noun labels across folds. Current MIT-States is development/legacy replication data, not a fresh confirmatory test set.
- Local verification: `.venv/bin/python -m pytest -q` from `sz684/` passed all 38 tests. This does not verify full extraction. `shared/cache.py` references a missing `shared/extractor_migrations.json`; test and repair that migration path before relying on resumed historical extraction. README/CODE_GUIDE commands still include pre-reorganization script paths; use `run.py` as the entry point and update documentation during implementation.

The working tree already contains a substantial folder reorganization. This plan adds a document only; do not undo that work or overwrite historical results.

## Model roster and eligibility gate

Start with one model per modality for a three-model pilot; expand to two per modality after extraction and data QA pass.

| Modality | Proposed main checkpoints | Required provenance check |
| --- | --- | --- |
| Text | BLOOM-560m base and Pythia-410m base | Text-only autoregressive training; pin revision, tokenizer, and corpus documentation. Existing BLOOMZ is a historical comparison, not a fresh training replicate. |
| Vision | Original DINOv2-base backbone and MAE ViT-base **pretraining** checkpoint | Image-only teacher/initialization/objective; exclude language heads and supervised fine-tuned checkpoint variants. |
| Audio | AudioMAE ViT-base **pretraining** checkpoint and BEATs **iter1** pretraining checkpoint | Audio-only initialization and objectives. BEATs iter1 uses a random-projection tokenizer; later variants require extra teacher-lineage scrutiny. Exclude fine-tuned checkpoints. |

These are proposed candidates, subject to checkpoint access, provenance, and runtime validation in week 1. The original AudioMAE repository relies on an old patched runtime; isolate its extractor or port and verify numerical agreement, rather than changing the existing analysis environment globally. Alternative audio candidates must pass the same full-lineage audit. Speech encoders such as WavLM are a separate speech-focused extension, not an automatic substitute for environmental-sound encoders.

Six models give four image–text and four vision–audio pairs: eight primary unordered pairs. Directional transfer evaluates both directions, giving sixteen primary map directions. Same-modality pairs and four text–audio pairs are secondary. Pair counts do not equal independent model-training replications. DINOv2 distilled variants in particular share lineage. Avoid model-population or scaling-law claims from this roster.

## Data: two connected benchmarks

### A. Shared concept geometry

Use a concept registry with stable IDs and modality-specific exemplars. Candidate concepts include a dog barking, a bell ringing, an engine running, or water flowing. Text gets several natural descriptions; vision gets independently verified frames/images; audio gets independently verified sound clips. These are shared event concepts, not arbitrary dictionary words.

For vision–audio, audit VGGSound/VGGSounder as candidate sources. Verify current media access and actual visible/audible correspondence before committing. Paired audiovisual evaluation data are allowed: the frozen encoders still receive only their own input channel. Include independently sourced image/audio exemplars or different-video concept matching as a replication to separate concept structure from shared recording context.

Provisional full target: 80–120 jointly observable concepts with 15–25 independent recordings/exemplars per concept per non-text modality, plus 4–8 text formulations. With 100 concepts and 20 shared audiovisual clips each, this is roughly 2,000 clips, not 2,000 independent frames per modality. Multiple frames/crops/windows from one recording are one source cluster. Candidate pilot: 20–30 concepts and 5–10 sources each; its role is feasibility, not statistical confirmation.

Store `concept_id`, modality, exemplar ID, source-recording ID, uploader/session ID if known, timestamp, hash, annotation confidence, and split group. Detect exact and perceptual duplicates. Group by original recording and, where possible, uploader/session. Keep an explicit policy for multi-label clips. Balance sources and exemplars across concepts; report missingness rather than silently changing the concept roster for each model pair.

Two annotators independently judge images and sounds using separate channels, with adjudication and an ambiguous/absent category. Check speech that names the class, subtitles, overlays, watermarks, background music, and offscreen sound. Handle these using rules frozen before model scoring. Human concept names are metadata and probe targets; they never enter a non-text encoder. Text templates are repeated descriptions, not independent real-world samples.

Primary cross-modal RSA/CKA/neighbors use one balanced concept centroid per modality. Separately report matched-recording vision–audio geometry as an instance-level secondary analysis. Never pair the nth unrelated image with the nth sound simply because they have the same label.

### B. Directional relations

Keep the five current state relations as an image–text development benchmark. Build a fresh verified image–text evaluation sample with taxonomy/source grouping; independently sampled endpoints must not be described as physical before/after pairs.

Build a distinct vision–audio relation panel. Candidate axes include slow→fast repeated motion, sparse→dense events, or single→multiple active sources. These are hypotheses to audit: a static image often cannot reveal speed, and event multiplicity can be confounded by framing. If speed is not observable in sampled frames, require a vision-only video encoder or drop that axis; do not feed time-averaged images and assert motion semantics. Prefer independently verified static cues for the first release.

Pilot target: two viable axes over 10–15 independent object/event families, both endpoints present in both modalities, with about 10 sources per endpoint/family. Expand to 20–30 families where possible. Freeze final relation count and sample size after pilot precision/power simulations. If no axes meet observability and support criteria, release cross-modal geometry results first and keep vision–audio directional claims open. Do not force raw→cooked or wet→dry into an audio task.

Use separate development and untouched confirmatory pools. Split whole concept families for geometry generalization and whole object/event families for CAV/map generalization. Within development, use grouped train/validation folds. Keep final-test sources and families out of preprocessing, layer selection, probes, maps, and scale tuning. Choose the exact grouping and feasibility constraints before splitting. The small legacy sample cannot automatically support the larger study's nominal five-fold design.

## What each technique requires

| Technique | Fitted components | Evaluation and intended conclusion |
| --- | --- | --- |
| RSA | No alignment. Select distance/layer using development data only. | Spearman correlation of matching strict-upper-triangle cosine RDM entries. Report overall, within-family, and relation-focused structure to distinguish broad category agreement from relation agreement. Evidence of shared distance ordering, not identical semantics. |
| CAV + relative translation | Regularized linear binary probes per relation/model; separate train-only alignment anchors; train-fitted PCA if equal dimensions are needed. | First show probe generalization and sign stability on held-out families. Then signed mapped-CAV cosine, transferred probe performance, and held-out relation retrieval. Compare to mean endpoint differences as a separate method. |
| Neighbor graphs | No alignment. Prespecify k=3 primary, with k=5/10 sensitivity if concept count permits. | Per-concept Jaccard, intersection/k, and labeled edge overlap. Specify directed k-NN versus reciprocal-edge graphs and deterministic tie handling. Report hubness and a correspondence-permutation null. |
| Linear CKA | Centering; layer choice on development only. No cross-modal map. | Compute `||Xc.T @ Yc||_F² / (||Xc.T @ Xc||_F * ||Yc.T @ Yc||_F)` on corresponding concepts, with arbitrary feature widths. Report sample-size sensitivity and shuffled/random-network controls. Evidence of similar centered Gram geometry. |

RSA/CKA/neighbors should primarily describe native frozen features, without learned cross-modal alignment. Use each checkpoint's documented preprocessing and a declared pooling policy. RSA/graph cosine analyses normalize rows; standard primary linear CKA uses centered, unnormalized pooled features, with row-normalized CKA explicitly labeled as a sensitivity analysis. Avoid automatically applying legacy clipping/PCA64 to every method: those operations can change the geometry being tested. Report native and projected spaces separately.

For aligned CAVs, fit a train-only common rank d bounded by both widths and effective anchor rank. Choose d on validation, not on the final test. Procrustes estimates a square orthogonal map between the projected coordinate systems. Fit anchors on separate training families where possible, and test axes on unseen families; also hold out a relation as a stronger secondary test if there are enough relations.

The repository uses row vectors: with `Y ≈ X @ R`, orthogonality permits `cav_X @ R` to be compared with `cav_Y` in the common projected Euclidean spaces. Fix target-minus-source signs. CAVs are classifier normals; mean differences are displacement vectors. Under a general nonorthogonal map these transform differently (normals require inverse-transpose in column convention), so do not reuse ridge displacement translation as a CAV-normal translation. PCA truncation and whitening likewise change the question; evaluate normals in the declared projected space and document it.

Full TCAV also measures how a model output changes along a CAV. A frozen embedding/probe study should call its measure CAV alignment or transfer, not TCAV, unless an output function and derivative protocol are added.

High RSA does not establish each proposed triplet ordering; add preregistered triplet accuracy for specific A/B/C hypotheses. High CKA does not by itself prove representations or semantic axes are identical. Graph isomorphism disregards concept identities; because correspondence is known, labeled graph overlap is the useful primary measure. General graph edit distance adds computational cost without resolving these scientific issues.

## Training, inference, and decision rules

1. Fit balanced regularized probes on training examples, validating regularization on held-out families. Labels supervise the probes, not the frozen encoders. Require useful held-out discrimination and stable orientation before interpreting CAV agreement.
2. Fit orthogonal maps on training correspondences; ridge remains a secondary displacement/endpoint baseline. Record anchor supervision granularity: same recording versus shared concept centroid. Test both transfer directions separately.
3. For relation retrieval, tune scale on validation families; report target-state AP and joint identity+state success, identity retention, no shift, target-model oracle probe/direction, reverse, norm-matched random, wrong relation, and shuffled anchors. Adapt gallery/source exclusions to real source IDs rather than legacy cyclic indices.
4. Estimate within-model split-half reliability across independent exemplars. Low reliability limits interpretable cross-model agreement. Inspect low-level visual/acoustic features and coarse category RDMs as confound controls; a shared background or event superclass can drive agreement.
5. Resample independent source/concept-family clusters and recompute centroids, RDMs, and graphs. RDM entries/edges are dependent; do not bootstrap their cells as independent observations or use ordinary Spearman p-values. Permute correspondence IDs jointly on matrix rows and columns under a prespecified exchangeable/block-aware null. Relation-specific nulls must preserve broad category structure where relevant.
6. Report confidence intervals and permutation null distributions. Separate fixed-fit uncertainty from refit-inclusive resampling; use refit-inclusive analysis for key transfer claims. Repeated folds/seeds are sensitivity analyses, not independent datasets or encoder seeds.
7. Preregister a small primary hypothesis family covering the eight cross-modal pairs and chosen endpoints, with Holm adjustment or a permutation maximum statistic. Keep exploratory layers/k/relations separate. Use pilot simulations over concept/family counts to choose precision and minimal meaningful effects; no universal CKA/RSA cutoff.
8. Treat geometry agreement, usable relation axes, and identity-preserving transfer as distinct findings. Agreement on one metric need not imply agreement on the others. A mixed or null result is a valid completion of the study.

## Proposed implementation changes

Create `experiments/multimodal_structure/` with its own configs, protocol, tests, results, and reporting. Preserve current state-transfer outputs and extraction identity. Share generic infrastructure, not hard-coded state-specific assumptions.

| Path/change | Work |
| --- | --- |
| `shared/models/registry.py` and modality adapters | Typed model specs: checkpoint/revision/hash, modality, full training lineage, objective, adapter, processor, layers, pooling, feature width. Text, image, waveform/spectrogram adapters; a video adapter only if justified. |
| `multimodal_structure/data/{registry,prepare,av_data,qa}.py` | Concept/relation registry; downloads and access audit; audio decode/resampling; source grouping; annotation records; disjoint data/model/split manifests. |
| `multimodal_structure/data/extract.py` | Arbitrary model IDs, explicit modality dispatch, selected layers, deterministic inference, resumable atomic chunks, finite/shape/row checks. Cache by data hash/model hash/layer/pooling/processor version, not modality alone. |
| `multimodal_structure/training/{preprocess,probes,alignment}.py` | Serializable train-only transforms, regularized CAV probes, Procrustes and ridge, distinct normal/displacement APIs, grouped validation and scale selection. |
| `multimodal_structure/evaluation/{rsa,cka,graphs,cav_transfer,retrieval,inference}.py` | Pure metric functions plus orchestration, reliability, block-aware nulls/cluster resampling, all-pair and directional result tables. Reuse existing neighbor/bridge/retrieval implementations where assumptions hold. |
| `multimodal_structure/reporting/report.py` | Model-pair heatmaps per benchmark/metric, layer sensitivity, per-concept neighborhoods, relation transfer versus identity retention, uncertainty and provenance exports. |
| `run.py`, requirements, docs | Add prepare/extract/fit/evaluate/report commands, separate optional audio dependencies/runtime, consistent actual paths, and an execution budget estimator. Extend device synchronization for CUDA if used. |

Data manifests must not require a historical baseline run. Model changes should invalidate that model's cache, not require resampling the dataset or reextracting all other models. Store independent data, model, fit, and evaluation identities. Serialize actual fitted arrays, not just transform summaries. Save exclusions and missing media, checkpoint hashes, source snapshots, sample IDs, seed streams, configuration, software versions, and evaluation locks under unique run IDs.

Tests should catch meaningful failures: row/concept mismatch; source/family leakage; cache collision across two same-modality models; silent partial-cache completion; incorrect audio padding/sample rates; analytic RSA/CKA invariances and degenerate features; neighbor self-exclusion/ties; recovery of known rotations; normal-versus-displacement handling; and null calibration on synthetic independent data. Add a tiny three-modality end-to-end fixture plus one real-checkpoint smoke test per adapter. Keep the current 38-test suite green.

## Timeline and delivery gates

Planning estimate: **12 working weeks**, assuming one full-time research engineer/researcher, part-time annotation help, available media/checkpoints, and access to one suitable GPU for extraction. Dates are nominal; add 1–2 calendar weeks for holidays/access delays. A Mac can support development and small sequential extractions; benchmark throughput before promising full extraction time.

| Week | Nominal dates | Deliverable and exit gate |
| --- | --- | --- |
| 1 | Oct 19–23 | Audit six checkpoint lineages and media access; freeze candidate concept/axis definitions; document hypotheses and proposed primary tests. Smoke-test one model per modality and repair migration/docs prerequisites. |
| 2 | Oct 26–30 | Implement concept/source manifests, audio ingest, model adapters and cache schema. Annotate the 20–30-concept pilot and begin relation-observability audit. Gate: source-disjoint data and reliable modality-specific labels. |
| 3 | Nov 2–6 | Run three-model pilot extraction; implement and validate RSA, graphs, CKA, reliability and nulls. Deliver image–text and vision–audio geometry pilot with resource measurements. |
| 4 | Nov 9–13 | Implement probes/maps/transfer pilot and finish axis audit. Estimate annotation effort and sample-size precision. Freeze final protocol, roster, split grouping, metrics and test family before final-test extraction/scoring. Gate: viable axes; otherwise explicitly defer unsupported directional claims. |
| 5–6 | Nov 16–27 | Acquire/reannotate full data and independent confirmatory sources; finish remaining adapters; extract six models with resumable QA. Data acquisition/annotation may overlap implementation but test scores remain hidden. |
| 7 | Nov 30–Dec 4 | Fit probes/transforms/maps on development folds; validate ranks, regularization, layers and scales; lock final configurations. Deliver all-pair development tables and transfer sanity checks. |
| 8 | Dec 7–11 | Run locked final RSA/CKA/graphs and directional transfer on unseen families/sources. Save per-concept/per-source outputs and identity-retention contrasts. |
| 9 | Dec 14–18 | Run planned nulls, cluster/refit uncertainty and multiplicity adjustment; finish native-space/projected-space and background/coarse-category sensitivities. |
| 10 | Dec 21–25 | Independent-source/dataset replication where accessible; compare matched-recording and concept-only vision–audio results. No test-driven method tuning. |
| 11 | Dec 28–Jan 1 | Reproduce from locked manifests in a clean run; verify reports, exclusion counts, checkpoint provenance and reproducibility package. |
| 12 | Jan 4–8, 2027 | Final report/manuscript with pairwise geometry, transfer and preservation conclusions, limitations, artifacts and next-stage decision. |

Expected milestones: cross-modal geometry pilot in **3 weeks**, all-four-method feasibility in **4 weeks**, full frozen-encoder study in **12 weeks** (likely **13–14 calendar weeks** around holidays). Without annotation help or reliable audiovisual access, budget **16–20 weeks** or reduce the roster/data scope while retaining both primary cross-modal arms.

## Compute and training scope

Feature extraction is per model and can be reused for every pair/method. Fit probes/maps on cached embeddings on CPU; reserve accelerators for frozen extraction. Extract sequentially on unified-memory Macs. Use short measured runs to estimate audio decoding, model latency, peak memory, disk, and total cost. No defensible dollar or GPU-hour estimate exists yet for the selected data and adapters.

For N examples, L retained layers, width d, float32 features cost approximately `4*N*L*d` bytes per model before metadata. At N=10,000, L=4, d=768, that is about 123 MB/model; source media and checkpoints will usually dominate. A dense float32 N×N Gram matrix is about 400 MB at N=10,000, so compute concept-centroid geometry and block larger instance analyses. Permutation/refit inference may cost more than the core metrics; implement chunking and save reusable sufficient statistics where valid.

If training new encoders is required, first specify corpus size, architecture, objective, seeds, GPU allocation, and target training budget. A controlled small-model study can match data/computation and use three or more seeds, but answers a different question from comparing existing foundation models. Budget an additional **6–12+ weeks after infrastructure/data readiness**, contingent on throughput and cluster access; full-scale foundation pretraining is outside this estimate. Modality-only fine-tuning is another explicitly separate arm because it can change the geometry under study.

## Primary references and checkpoint sources

- [RSA: Kriegeskorte et al.](https://doi.org/10.3389/neuro.06.004.2008) and [inference on representational geometries](https://elifesciences.org/articles/82566).
- [Linear CKA: Kornblith et al.](https://proceedings.mlr.press/v97/kornblith19a.html).
- [Concept activation vectors / TCAV: Kim et al.](https://proceedings.mlr.press/v80/kim18d.html).
- [DINOv2](https://arxiv.org/abs/2304.07193), [original repository including separate language-aligned heads](https://github.com/facebookresearch/dinov2), and [MAE pretraining checkpoints](https://github.com/facebookresearch/mae).
- [BLOOM base model card](https://huggingface.co/bigscience/bloom-560m) and [Pythia model card](https://huggingface.co/EleutherAI/pythia-410m).
- [AudioMAE pretraining versus fine-tuned checkpoints](https://github.com/facebookresearch/AudioMAE), [BEATs checkpoint variants](https://github.com/microsoft/unilm/blob/master/beats/beats_README.md), and [BEATs tokenizer lineage](https://arxiv.org/abs/2212.09058).
- [VGGSound](https://arxiv.org/abs/2004.14368) and [VGGSounder independent audiovisual annotation effort](https://vggsounder.github.io/). These are candidates, not a claim of completed access or adequate relation coverage.
