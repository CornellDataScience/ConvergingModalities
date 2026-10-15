# Do State-Change Directions Transfer from Text to Images?

Working research manuscript | October 15, 2026 | Not submitted or peer reviewed

## Abstract

Can a change represented in language transfer to images of object categories that did not supply the learning examples? We estimate a shared state-change direction from text contrasts such as "cooked beef" minus "raw beef," map it into the representation space of a separately pretrained image model, and add it to a held-out source image before retrieval. The direction is shared across objects: a test category supplies neither text contrasts nor images for fitting or validation. In an actual cooking fold, training contrasts involve beef and six other categories, while chicken and meat are held out. The experiment therefore tests both text-to-image relation transfer and transfer across object labels; the final search itself is image-to-image. Across five changes, 992 MIT-States images and 496 queries, the macro rate of retrieving the requested state AND object category first rises from 20.9% for image similarity search without a state instruction to 38.7%. This comparison alone is weak because only the latter receives the requested change. Direct target-text search, supplied with the requested state and test noun, reaches 42.1%; the shared direction does not establish retrieval superiority. The originally specified average-precision analysis favors the transferred direction over random, reversed and wrong-relation controls, but correct-category retention falls from 86.6% to 79.4% and the preservation criterion fails. The evidence supports limited cross-modal relation transfer with category drift, rather than a general shared semantic geometry or a superior retrieval system. Released-label noise further limits physical-state claims.

## 1. Research question and claim

The central question concerns a relation between modalities: does the text-space difference between "raw" and "cooked," estimated using some foods, specify a useful direction in image space for other foods? We keep a language encoder and a vision encoder fixed, learn a supervised bridge between their representations, and test whether a language-derived relation transfers through that bridge. We do not train an image model to turn raw chicken into cooked chicken.

There are two distinct transfers. First, the state direction originates in text and is applied in vision. Second, the exact object labels used for testing are excluded from fitting the bridge, constructing the direction and selecting its strength. In a chicken-held-out fold, success means that applying a direction learned from other foods helps retrieve pictures labeled cooked chicken from an image labeled raw chicken. It does not mean generating an image or finding the same physical chicken after cooking.

The final ranking takes place entirely in image space. That is how we measure whether the cross-modal transfer was useful; it is not where the direction was estimated. A direction learned directly from training images is a separate within-modality reference. Comparing that reference with the text-derived direction helps ask whether language supplies a useful relation, although similar retrieval scores do not prove identical geometry across modalities.

The distinction from an ordinary supervised retrieval result matters. Giving a method the requested change should help relative to a search that receives no such instruction. Our stronger evidence is that a shared direction constructed without test-noun examples improves the planned ranking score over random, reversed and wrong-change directions on held-out nouns. Our limits are equally central: direct target-text search has higher first-result success, category retention declines, and the complete predeclared success criterion fails. The paper presents evidence of transferable relation information, not a claim of state-of-the-art retrieval or a universal concept algebra.

Earlier work in this project measured local agreement between language and vision representations. The present experiment asks whether that agreement carries a specific relation across modalities. For an affine map f(x)=xW+b, f(x_b)-f(x_a)=(x_b-x_a)W identically; successful retrieval of arbitrary endpoint differences can follow from endpoint prediction alone. A named relation shared across excluded nouns is a more informative test. It still does not establish that the two pretrained spaces independently organize all semantic changes in the same way.

## 2. Related work

Huh et al. [1] motivates cross-modal geometric agreement. Fournier et al. [2] and Linzen [3] explain why analogy scores can combine different sources of regularity. Visalogy [4] learns representations for visual analogy questions, while StyleCLIP [5] uses cross-modally pretrained CLIP guidance for image manipulation. These works establish substantial precedent; we do not claim that vector arithmetic or input-independent edit directions are new.

MIT-States [6] provides the natural state/noun image associations used here. BLOOMZ [7] and DINOv2 [8] supply frozen independently pretrained encoders; the downstream ridge bridge is explicitly supervised. Recent object-state studies [9,10] motivate evaluation beyond object recognition. We investigate a specific held-out-noun transfer protocol, not a state-of-the-art claim against those benchmarks. A broader novelty assessment and direct comparisons remain necessary.

## 3. Data and experimental design

We selected five physical-state relations before inspecting transfer scores: cleaning (9 noun labels), cooking (10 noun labels), ripening (11 noun labels), thawing (17 noun labels), drying (15 noun labels). Every supported state/noun composition had at least 10 images in the original catalog; after deterministic archive duplicate screening we sampled 8 per composition with fixed seeds. The final set contains 124 compositions and 992 images. All selected SHA256 hashes are distinct. The original dataset derives labels from image-search queries and human cleaning; its README explicitly notes residual ambiguity and mislabeled images. We performed visual QA but did not reannotate the frozen sample.

![Natural-state examples](/Users/sz/Programming/CDS/conv-mod/sz684/output/figures/state_examples.png)

Figure 1. First sampled source/target examples from five specified noun labels. Images are independent photographs; rows do not depict paired physical instances or controlled interventions. Labels are from the original MIT-States release [6].

A seeded, coverage-balanced assignment partitions noun labels into five groups. In each fold one group supplies test queries, the next supplies validation, and the remaining three fit the representation transforms, map and universal relations. Each eligible noun is tested once per relation. All appearances of an exact test noun across relations are absent from fitting. Related labels such as fish/salmon/seafood or fruit/apple remain possible across groups, so the design establishes lexical noun-label separation, not taxonomic novelty.

For each relation/fold, the gallery contains source-state and target-state pictures from its test nouns. For source query index i we remove source index i and target index (i+1) mod 8 from every noun. Thus the query is excluded and each noun contributes 7 candidates in each state. Bank sizes range from 14 to 56 images; the same banks are used for every control. With few test nouns, noun recognition can be easy, so state-sensitive metrics and per-relation reporting are required.

## 4. Representation mapping and relation transfer

The information flow is: training text contrasts -> shared text direction -> supervised text-to-image map -> shared image-space direction -> held-out image query -> image retrieval. Text supplies the relation; the source image supplies the test object's appearance. The universal method is told which of the five changes to apply, but receives no test-category text description. Both pretrained encoders remain fixed. Only the bridge and preprocessing are fit, and the translation strength is selected on validation nouns.

### An actual cooking split

The saved fold 2 uses the following categories for cooking:

| Role | Categories | Use |
|---|---|---|
| Fit cooking direction | Beef, eggs, fish, pasta, salad, seafood, vegetable | Average their raw-to-cooked text contrasts; training images also support the supervised bridge |
| Validate strength | Salmon | Select how far to move; no chicken or meat examples enter selection |
| Test transfer | Chicken, meat | Apply the frozen shared direction to raw-image queries and score retrieval |

The bridge uses training categories across all five relations, not only the seven cooking categories. No chicken or meat example from any relation enters that fold's bridge fitting. Test photographs in both states enter only the evaluation query/gallery. The table specifies a real split, not a claim that beef alone taught a chicken-specific operation or that every chicken query succeeded. Related labels remain: excluding chicken and meat is not excluding all foods or all meat-related concepts.

### Exact construction

We fix BLOOMZ 1.7B hidden-state index 19 (including the embedding output as index 0) and DINOv2 base block 10 (zero-based), using layers chosen earlier on the unrelated WIT study. Four matched prompt templates describe each state/noun pair. Float32 MPS extraction uses pinned cached model revisions; CPU float64 analysis fits 95th-percentile clipping, L2 normalization, centered PCA64 and projected endpoint normalization on map-fitting rows only.

The first 4 sampled images per fitting composition fit a centered ridge endpoint map with alpha=1. The remaining 4 supply an image-derived relation reference. For a relation r, average normalized text differences across templates within a fitting noun, then average fitting nouns equally. The transferred direction is d_r=mean(x_target-x_source)W. A source image is translated as normalize(y_source+s*d_r). Select s from {0,.25,.5,1,2,4} using validation macro noun AP, with ascending-scale tie break. No test-noun text contrast enters this universal direction.

Controls are source-only retrieval, reversed correct direction, 100 fixed equal-norm isotropic directions, the other four state relations evaluated separately after norm matching, an image-derived relation, and orthogonal Procrustes transfer. Direct target-text endpoint retrieval and a source-anchored noun-specific text contrast deliberately use test-noun wording; they are stronger-information recognition references. Every map or scale fit is confined to fitting/validation groups.

## 5. Outcomes and uncertainty

For exposition we lead with **first-result success** (joint top-1): the top retrieved picture must have both the requested state and the source object's noun label. For raw chicken with a cooking request, cooked chicken succeeds; raw chicken and cooked beef both fail. Correct-category retention (noun top-1) checks whether the first result is chicken regardless of state. Target-state top-1 checks the state regardless of category. These are different questions.

The original protocol selected average precision (AP) as the primary outcome and used validation AP to choose translation strength. We retain that analysis. AP evaluates the full ranking, counting only requested-state-and-noun images as relevant; it is not an accuracy percentage. Promoting first-result success in this rewrite is a post-results presentation choice, not a new preregistered test. The confidence intervals below concern AP and category retention, not the headline top-1 differences.

For every outcome we average queries within noun, nouns within relation, then the five fixed relations. Thus headline percentages are macro success rates, not a pooled count out of 496 queries. No exact matching target photograph is assumed.

Four primary paired contrasts compare the text direction with source-only, random, reversed and wrong-relation shifts. A 20,000-draw paired bootstrap resamples unique noun labels jointly across all their relation appearances and source queries within relation/noun cells. Two-sided 98.75% percentile intervals implement Bonferroni adjustment over these four aggregate contrasts; ordinary 95% and category-only intervals are sensitivity summaries. Noun-top-1 preservation uses a one-sided 95% lower bound above -.05. Gallery banks, fitted maps and validation-selected scales stay fixed; their uncertainty is omitted. The relation roster is fixed and is not resampled as a population of semantic transformations.

## 6. Results

### What happened on held-out categories?

| Search method | Requested state AND category first | Correct category first |
|---|---:|---:|
| Image similarity search (no state change) | 20.9% | 86.6% |
| Shared text-to-image direction | 38.7% | 79.4% |
| Direction learned from training images | 39.5% | 71.3% |
| Direct target-text search | 42.1% | 56.8% |

Image similarity search ranks candidates by similarity to the unmodified source-image embedding. It receives no instruction to prefer the new state. The shared text-to-image direction uses the same source image and candidate gallery, but adds the requested relation learned from training nouns. Its 17.8-percentage-point descriptive gain shows the effect of that added information, not superiority over an equally instructed retrieval method.

Direct target-text search maps a description such as "cooked chicken" into image space and uses it as the whole query. It receives both the requested state and an explicit test-category noun; the shared direction receives the source image and relation instead. This is an informative task-level comparator despite the different inputs. It succeeds first more often (42.1% versus 38.7%), while retaining the category less often (56.8% versus 79.4%). We do not demonstrate that the shared direction is the better retrieval method. The image-derived direction provides a within-modality reference: it has similar first-result success, so the text-derived direction is useful without showing an advantage over learning the change from images. These comparisons are descriptive; no new top-1 significance claim is attached.

### Was the text-derived relation more useful than an arbitrary shift?

| Primary paired AP contrast | Gain | Adjusted 98.75% interval |
|---|---:|---|
| Text minus no shift | +0.0852 | [+0.0294, +0.1471] |
| Text minus random shifts | +0.0870 | [+0.0334, +0.1476] |
| Text minus reverse shift | +0.1516 | [+0.0833, +0.2252] |
| Text minus wrong relations | +0.0795 | [+0.0278, +0.1364] |

All four adjusted AP contrasts are positive, but the noun-preservation guardrail fails. The complete predeclared success criterion is therefore not met; retrieval gains do not establish a state edit that preserves object category.

Noun-top-1 changes by -0.0720; its one-sided 95% lower bound is -0.1356, compared with the -.05 guardrail. These are aggregate criteria. A positive result for the fixed suite would not establish improvement for every individual relation.

![Paired retrieval effects](/Users/sz/Programming/CDS/conv-mod/sz684/output/figures/state_transfer_effects.png)

Figure 2. Left: four primary paired macro AP contrasts with multiplicity-adjusted intervals. Right: relation-specific gains over source-only retrieval, with descriptive 95% intervals. Per-relation intervals are not a second family of significance claims.

| Relation | Nouns | Source-only AP | Text AP | AP gain |
|---|---:|---:|---:|---:|
| Cleaning | 9 | 0.4182 | 0.5037 | +0.0855 |
| Cooking | 10 | 0.3833 | 0.4922 | +0.1089 |
| Ripening | 11 | 0.3359 | 0.5360 | +0.2001 |
| Thawing | 17 | 0.3526 | 0.3629 | +0.0103 |
| Drying | 15 | 0.3501 | 0.3710 | +0.0209 |

The observed AP gain over image similarity search is positive in all five relation means. This is a descriptive count, not an independent-replication test.

| Condition | Macro AP | Target-state top-1 | Noun top-1 | Joint top-1 |
|---|---:|---:|---:|---:|
| Image similarity (no change) | 0.3680 | 26.6% | 86.6% | 20.9% |
| Universal text relation | 0.4532 | 54.9% | 79.4% | 38.7% |
| Reverse relation | 0.3016 | 18.4% | 83.8% | 14.6% |
| Mean random shift | 0.3662 | 29.2% | 84.0% | 22.2% |
| Mean wrong relation | 0.3736 | 29.2% | 83.0% | 21.6% |
| Image relation reference | 0.4557 | 61.0% | 71.3% | 39.5% |
| Procrustes text relation | 0.4671 | 67.8% | 70.5% | 44.8% |
| Direct target text | 0.4354 | 77.1% | 56.8% | 42.1% |
| Noun-specific text offset | 0.4508 | 56.5% | 75.7% | 38.3% |

### Validation choices and robustness

Text translation scales selected on validation only:

| Relation | Fold 0 | Fold 1 | Fold 2 | Fold 3 | Fold 4 |
|---|---:|---:|---:|---:|---:|
| Cleaning | 2 | 4 | 4 | 4 | 4 |
| Cooking | 1 | 4 | 4 | 2 | 1 |
| Ripening | 4 | 4 | 4 | 2 | 4 |
| Thawing | 4 | 4 | 2 | 2 | 0.5 |
| Drying | 1 | 4 | 4 | 4 | 2 |

A zero scale is retained when selected and is not replaced using test performance. Saved results include every validation curve, train/validation/test noun list, category-only interval and leave-one-noun-out estimate.

### Connection to the endpoint pilot

The earlier WIT audit obtains19.5-25.8% arbitrary difference top-1 among 256 candidates across three overlapping splits, with single-endpoint ablations at3.9-8.6%. That correspondence result alone cannot establish semantic transfer. The present task instead transfers one named relation to excluded noun labels without using their text endpoints to form the universal direction. The two experiments measure different capabilities and should not be pooled statistically.

### Exploratory preservation experiments

After inspecting the primary result, we tested two remedies using the same test set. These are explicitly post-primary exploratory analyses. The first constrains scale selection to a validation noun-top-1 loss no worse than five percentage points. The second estimates a training-noun centroid subspace retaining 80% of between-centroid variance, removes 0%,50% or100% of the relation component in that subspace, norm-matches the direction, and selects strength and scale under the same validation guardrail. Training centroids are an identity proxy, not a proven disentangled factor.

| Exploratory method | Macro AP | Noun top-1 | AP gain versus source only (95%) |
|---|---:|---:|---|
| Constrained scale | 0.4212 | 81.2% | +0.0532 [+0.0192, +0.0916] |
| Identity-subspace projection | 0.4338 | 80.6% | +0.0658 [+0.0254, +0.1131] |

Neither remedy demonstrates the predeclared noun-preservation tolerance on the test queries. Both trade away some of the original AP gain; a validation-time guardrail did not ensure a test-time guarantee. These ordinary 95% intervals are descriptive, unadjusted follow-up summaries. Primary results are unchanged.

![State/category tradeoff](/Users/sz/Programming/CDS/conv-mod/sz684/output/figures/state_category_tradeoff.png)

Figure 3. Descriptive state/noun AP versus noun top-1 for selected primary methods. Higher is better on both axes. More aggressive state targeting can sacrifice noun retention; no confidence intervals are attached to these method coordinates.


## 7. Limitations and interpretation

Released state labels are the central validity limitation. Visual QA found some “thawed” examples depicting cooked dishes, some unripe labels on ripe-looking or unrelated fruit, and states not visually decidable from product or generic object photographs. These observations do not estimate a label-error rate. Our numerical endpoint therefore measures retrieval against released labels, not verified physical states. We retain every frozen sample and report this limitation rather than repairing labels after seeing outcomes.

Exact noun labels are separated, but taxonomic, photographer, web-source and near-duplicate dependence can remain. The sampled galleries are small and use eight images per composition; source URLs and photographer IDs are not available in the archive. The bootstrap conditions on overlapping fitted folds and banks and uses approximate coverage. Five selected relation families and one encoder pair cannot establish a general semantic algebra or cross-family scaling. Hyperparameters and research direction were selected through prior exploratory work; the saved local design lock is not external preregistration.

Earlier exploratory alignment and color-sketch work informed the research protocol. Neither supplies independent replication evidence for the natural-state result.

## 8. Next experiments and paper scope

The next study should test the cross-modal relation itself more directly. On a fresh split, compare text-derived and image-derived directions for the same named change using training objects only, evaluate their alignment against wrong-relation controls, and test both on held-out object families. Such alignment would still be evaluated after a supervised bridge; it would not imply that the original coordinate systems coincide. A shuffled text/image correspondence bridge can test whether meaningful cross-modal supervision is necessary. All settings and relation-matching tests must be selected before inspecting the new test outcomes.

For a practical retrieval claim, give competing methods the same intended task: a source image and requested change, with explicit noun wording supplied consistently where it is used. Compare against a source-image-plus-target-text baseline and a suitably trained vision-language retrieval model. Predeclare first-result state-and-category success and category retention, retain AP as a supporting ranking metric, and calculate paired uncertainty for those new outcomes. Beating unmodified image similarity alone is insufficient. These baselines have not yet been evaluated here.

The immediate next milestone is a verified-label replication, with source-video or photographer separation and broader galleries. The ChangeIt-derived object-state golden subset [9] provides an alternative with existing human-verified state annotations; STATUS Bench [10] provides paired state descriptions but requires access to its underlying Ego4D images. Neither is claimed as evaluated here. We should also compare a second language/vision model family, test source/taxonomy-blocked splits, and estimate whether improvements persist with hundreds of same-noun and wrong-noun distractors.

If only the image-derived direction works, investigate whether the endpoint map discards relation information. If direct target text works but the universal direction does not, recognition is present without demonstrated reusable transfer. If effects vary by relation, investigate category-conditioned or nonlinear transformations on validation data and evaluate those on a fresh untouched split. Those alternatives are future experiments, not retrospective replacements for the current primary criterion.

The current deliverable is a results-bearing working manuscript. Submission readiness depends on label validation, broader baselines and a fuller novelty review. A separate 3D viewer can later display the saved high-dimensional source, target and relation vectors; projection retention and full-space metrics must remain visible.

## Reproducibility

Run state_data.py, then state_extract.py for language and vision, and state_analysis.py from sz684/. state_report.py regenerates figures and the companion report; the two manuscripts are maintained editorially and rendered with paper_pdf.py. The exact configuration is configs/state_transfer.json. The analysis lock and source snapshot precede natural-image feature extraction. Source image hashes, archive checksums, noun groups, sampled image indices, model revisions, library versions, controls, bootstrap draws and per-query scores are saved under results/state_transfer/. That directory is Git-ignored and must accompany any shared research archive. Tests are restricted to tests/ so that archived test snapshots are not collected twice.

## References

[1] Huh et al. (2024). Position: The Platonic Representation Hypothesis. ICML. https://proceedings.mlr.press/v235/huh24a.html

[2] Fournier, Dupoux and Dunbar (2020). Analogies minus analogy test: measuring regularities in word embeddings. CoNLL. https://aclanthology.org/2020.conll-1.29/

[3] Linzen (2016). Issues in evaluating semantic spaces using word analogies. https://arxiv.org/abs/1606.07736

[4] Sadeghi, Zitnick and Farhadi (2015). Visalogy: Answering Visual Analogy Questions. NeurIPS. https://papers.neurips.cc/paper_files/paper/2015/hash/45f31d16b1058d586fc3be7207b58053-Abstract.html

[5] Patashnik et al. (2021). StyleCLIP: Text-Driven Manipulation of StyleGAN Imagery. ICCV. https://arxiv.org/abs/2103.17249

[6] Isola, Lim and Adelson (2015). Discovering States and Transformations in Image Collections. CVPR. https://web.mit.edu/phillipi/Public/states_and_transformations/index.html

[7] Muennighoff et al. (2023). Crosslingual Generalization through Multitask Finetuning. ACL. https://arxiv.org/abs/2211.01786

[8] Oquab et al. DINOv2: Learning Robust Visual Features without Supervision. https://arxiv.org/abs/2304.07193

[9] Newman et al. (2024). Do Pre-trained Vision-Language Models Encode Object States? https://arxiv.org/abs/2409.10488

[10] Ukai, Kurita and Inoue (2025). STATUS Bench: A Rigorous Benchmark for Evaluating Object State Understanding in Vision-Language Models. ACM Multimedia. https://arxiv.org/abs/2510.22571
