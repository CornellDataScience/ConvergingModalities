> Historical WIT/alignment-stage record. The completed natural-state experiment (including raw → cooked) is documented in [README.md](../../README.md) and [STATE_TRANSFER_PAPER.md](../../experiments/state_transfer/docs/STATE_TRANSFER_PAPER.md). Pending work described below reflects this earlier stage.

# Cross-modal representation alignment: robustness and difference-vector transfer

**Date:** October 4, 2026  
**Status:** Exploratory empirical report; reduced-model study on an existing dataset  
**Project:** Vision–language representational convergence

## Executive summary

This study investigates whether independently trained language and vision models share local representation geometry, whether the measured relationship with model size survives evaluation changes, and whether differences between text representations can be mapped to differences between image representations.

The results support non-random local cross-modal alignment in the tested setup. They provide less support for a consistent increase in alignment with model size. Centering and centered PCA increase measured alignment; in a separate query/reference evaluation, their improvements over baseline have positive confidence intervals after adjustment for multiple comparisons. A larger-language-model advantage is not established by those intervals.

A supervised text-to-image mapping also transfers differences between held-out example pairs: top-1 matching accuracy is 25.8% among 256 target differences, compared with approximately 0.4% under shuffled correspondence. This is a preliminary direction-transfer result, not evidence that a semantic relation such as “red → blue” generalizes across object categories.

| Research question | Current finding |
|---|---|
| Is local alignment above chance? | Yes, under correspondence-permutation tests for the held-out study. |
| Does alignment consistently increase with language model size? | Not established; ordering varies across splits and conditions. |
| Does evaluation-pool size affect the result? | Descriptively yes; even the direction depends on the neighborhood rule. Conditional difference tests do not establish a consistent pool-size effect. |
| Does preprocessing affect alignment? | Yes. Centering and PCA exceed baseline in all three splits under adjusted conditional confidence intervals. |
| Do text differences transfer into image space? | A supervised map transfers arbitrary held-out endpoint differences above shuffled correspondence. |
| Do reusable semantic relations transfer across modalities? | Not yet tested; a curated relation benchmark is required. |

## 1. Research scope

The project began with interest in the Platonic Representation Hypothesis: the possibility that models trained on different modalities develop related representation structures. The completed work focuses on vision–language models, building on a reduced baseline implementation of the original study. Audio–vision alignment and Gromov–Wasserstein blind matching have not been executed.

Three distinct claims are evaluated separately:

1. **Alignment exists:** corresponding image and caption representations have overlapping neighborhoods more often than random correspondence predicts.
2. **Alignment scales:** more capable or larger models exhibit greater alignment.
3. **Directions transfer:** a relation expressed as a difference between text vectors predicts a corresponding difference in image space.

Evidence for one claim does not establish the others. In particular, above-chance alignment does not imply a monotonic scaling law, and successful supervised endpoint mapping does not establish semantic analogy.

## 2. Data, models, and baseline

The dataset contains 1,024 ordered WIT image–caption pairs from the pinned `minhuh/prh` release. The cached model roster comprises BLOOMZ 560M, 1.1B, and 1.7B, and DINOv2 small and base: six language–vision model pairs. These pretrained models were evaluated, not retrained.

Extraction used float32 inference on Apple MPS; scoring used CPU float32. The original baseline clips features using the released default 95th-percentile procedure and L2-normalizes each layer's vectors. Exact model and dataset revisions, sample ordering, and cache identities are recorded in the baseline manifest.

For corresponding sample i, neighborhood overlap is:

`overlap_i = |neighbors_language(i) ∩ neighbors_vision(i)| / k`

The reported alignment is the mean over samples. The shared-pool experiments exclude each point from its own neighbor candidates. At a fixed layer pair, random correspondence has expected overlap `k/(N−1)`. Searching over layer pairs changes the null distribution, so the baseline repeats the full layer search under each permutation.

The 1,024-pair baseline takes the maximum over layer pairs with k=10. Scores increase across the three BLOOMZ checkpoints for both vision models:

| BLOOMZ | DINOv2 small | DINOv2 base |
|---|---:|---:|
| 560M | 0.086914 | 0.099609 |
| 1.1B | 0.093750 | 0.101855 |
| 1.7B | 0.097266 | 0.108105 |

These are descriptive trends for three related checkpoints. The language-performance evaluation uses a reduced OpenWebText protocol, and matched author reference arrays are unavailable. Exact reproduction of the original paper is not established. [Baseline report](../../experiments/baseline/results/baseline/REPORT.md)

## 3. Held-out layer selection

To separate layer selection from evaluation, each of three fixed random splits divides the dataset into 512 selection pairs and 512 evaluation pairs. The same splits are used for all models. Clipping thresholds are fitted on selection rows, layers are chosen by selection-set alignment, and both are frozen before evaluation. Neighbor sets are rebuilt within the evaluation half, with k=10.

Across the 18 model-pair/split evaluations:

- Alignment ranges from approximately 0.111 to 0.147.
- Shuffled means are approximately 0.0195–0.0196.
- None of 1,000 correspondence permutations reaches the observed score in any evaluation.
- Using the finite-permutation correction, each unadjusted p-value is `1/1001 ≈ 0.001`.
- Bonferroni correction within this family of 18 tests gives adjusted p-values of approximately 0.018.

This supports non-random alignment under the tested null. The multiplicity statement is scoped to these 18 tests, not every exploratory comparison subsequently performed.

Strictly increasing alignment across BLOOMZ sizes occurs in only two of six split/vision-model comparisons: zero of three for DINOv2 small and two of three for DINOv2 base. Several differences are small, so reversals alone do not demonstrate meaningful negative scaling. The evaluation pool is also smaller than the original baseline; changes cannot be attributed solely to held-out layer selection.

![Held-out alignment across model sizes](../../experiments/heldout_layers/results/heldout.png)

The three splits overlap and are not independent replications. [Detailed results](../../experiments/heldout_layers/results/REPORT.md)

## 4. Evaluation-pool size

The selected layers and preprocessing are held fixed while evaluation uses nested subsets of 128, 256, and 512 pairs. Two neighborhood rules are compared:

- Fixed k=10.
- Approximately fixed neighborhood fraction, with k=2, 5, and 10 respectively.

The following values average across six model pairs and three splits. They are descriptive averages of dependent measurements, not independent-replication estimates.

| Rule | N=128: excess over shuffled mean | N=256 | N=512 |
|---|---:|---:|---:|
| Fixed k=10 | 0.12224 | 0.11695 | 0.10474 |
| Approximately fixed k/(N−1) | 0.08907 | 0.09634 | 0.10474 |

At fixed k, raw mean overlap falls from 0.20078 to 0.12432. Some of that decline accompanies the decreasing chance expectation, but excess over chance also decreases. With approximately fixed neighborhood fraction, excess instead increases.

Therefore, “alignment decreases with larger samples” is not an adequate description of these results without specifying the neighborhood rule. The rules probe different neighborhood scales. Integer rounding also makes the N=128 fraction somewhat smaller than the other fractions. These experiments do not address N greater than 512.

![Sample-size sensitivity](../../experiments/robustness/results/sample_size.png)

Shading shows ranges across model sizes and splits, not confidence intervals.

## 5. Representation preprocessing

At fixed selected layers and N=512, the study compares:

1. Baseline clipped, L2-normalized vectors.
2. Selection-fitted mean subtraction, followed by L2 normalization.
3. Selection-fitted centered PCA to 128 dimensions, followed by L2 normalization, without whitening.

Transforms are fitted separately for each model on selection rows only. Layers are not reselected after transforming features.

| Comparison with baseline | Mean raw overlap change | Comparisons with an increase |
|---|---:|---:|
| Centering | +0.01217 | 18/18 |
| Centered PCA128 | +0.00978 | 17/18 |

PCA retains approximately 73–81% of selection-set variance for language representations and 81–97% for vision representations. Equal dimensionality therefore does not imply equal retained information or control all model differences.

Strict model-size ordering occurs in two of six split/vision comparisons under each of baseline, centering, and PCA. Increasing the alignment score does not make the scaling trend uniformly more stable. These transformations change the geometry; higher scores do not by themselves validate an “anisotropy correction.”

![Preprocessing sensitivity](../../experiments/robustness/results/preprocessing.png)

[Robustness results](../../experiments/robustness/results/REPORT.md)

## 6. Are the differences statistically supported?

### A separate conditional evaluation

The earlier shared-pool metric makes each evaluation item both a query and a possible neighbor. Simply bootstrapping its saved per-item scores would ignore this dependence. The uncertainty study instead divides each evaluation half into 256 reference pairs and 256 query pairs. Selection and transform fitting still use the separate 512 selection pairs.

For each condition, neighborhoods are recomputed from queries to references. The bank then remains fixed while query pairs are resampled. A paired difference compares two conditions on the same query, averaging across the specified fixed model roster before inference. This is a different evaluation target from the earlier shared-pool metric.

Intervals use 20,000 BCa bootstrap resamples. Each of six contrasts is evaluated on three splits, giving a family of 18. Alongside ordinary 95% intervals, individual intervals at confidence `1−0.05/18 ≈ 99.7222%` provide a Bonferroni family adjustment, subject to approximate bootstrap coverage. The paired bootstrap implementation follows the behavior documented by [SciPy](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.bootstrap.html).

### Results

Effects below are absolute overlap differences. “Adjusted exclusions” counts splits whose family-adjusted interval excludes zero; the splits are dependent.

| Contrast | Effects across the three splits | Adjusted exclusions | Interpretation |
|---|---|---:|---|
| Centering − baseline | +0.02435, +0.01595, +0.01393 | 3/3, positive | Consistent conditional improvement |
| PCA − baseline | +0.02611, +0.02467, +0.01406 | 3/3, positive | Consistent conditional improvement |
| PCA − centering | +0.00176, +0.00872, +0.00013 | 1/3, positive | Additional benefit not consistent |
| BLOOMZ 1.7B − 560M | +0.00625, +0.00996, +0.00059 | 0/3 | Larger-model advantage not established |
| Reference pool 256 − 128, fixed k, above chance | −0.01842, −0.01426, −0.00137 | 1/3, negative | Pool-size effect not consistent |
| Reference pool 256 − 128, fixed fraction, above chance | +0.00007, +0.00918, +0.01191 | 0/3 | Pool-size effect not established |

For example, centering's adjusted intervals are [0.01163, 0.03890], [0.00245, 0.03074], and [0.00371, 0.02485]. All are positive. Conversely, all three adjusted intervals for the largest-versus-smallest language model include zero. This is not evidence that the model sizes are equivalent.

Pool comparisons subtract the analytical chance shift, using chance `k/R` for R reference candidates. Their setup differs from the nested shared-pool experiment in Section 4.

![Paired confidence intervals](../../experiments/paired_uncertainty/results/paired.png)

These intervals condition on reference banks, selected layers, and fitted transformations. They do not include uncertainty from bank selection or model fitting. Exchangeable query pairs are assumed; source or topic clustering is not modeled. The dataset was already inspected, so these are exploratory results rather than prospective confirmation. [All confidence intervals](../../experiments/paired_uncertainty/results/REPORT.md)

## 7. Cross-modal difference-vector transfer

### Method

The pilot uses BLOOMZ 1.7B and DINOv2 base on the first saved split. Centered PCA128 is fitted separately in each modality using the 512 selection pairs. A supervised ridge map W, with fixed regularization alpha=1, is fitted from paired text to image vectors on those rows.

The 512 evaluation examples are organized into 256 disjoint endpoint pairs (a,b). For each pair:

`predicted image difference = (text_b − text_a) W`

`actual image difference = image_b − image_a`

Evaluation measures cosine similarity between corresponding directions and retrieval of the matching actual difference among 256 target differences. No evaluation pair is used to fit the map. The mapping is supervised, not a blind geometric alignment.

### Results

| Metric | Observed | Evaluation-shuffle mean | Adjusted permutation p-value |
|---|---:|---:|---:|
| Mean direction cosine | 0.45030 | 0.00051 | 0.002 |
| Top-1 difference retrieval | 25.78% | 0.394% | 0.002 |
| Top-5 difference retrieval | 53.13% | 1.933% | Descriptive only |

The inferential reference uses 999 permutations of evaluation difference correspondence. Mean cosine and top-1 retrieval form the two-metric adjustment family. Their unadjusted finite-permutation p-values are 0.001. Top-5 accuracy is descriptive.

Training-map shuffles were also evaluated. Their tail fractions are diagnostic only: layers were selected using the true training correspondence, and the shuffle did not rerun that selection. After the first run, this inferential issue was identified and reporting was corrected to use evaluation-correspondence permutations. No embeddings, mapping choices, scores, or permutation draws changed. This correction was made after inspecting results and is documented rather than presented as preregistered.

### Meaning and boundary of the finding

The learned map transfers arbitrary held-out endpoint differences above shuffled correspondence. However, for a linear map, mapping a difference equals subtracting mapped endpoints. Successful endpoint mapping can therefore explain this result without a separately reusable relation representation.

The pilot has no repeated semantic relation labels. It does not establish that a direction learned from “red car → blue car” will transform an unseen red chair toward a blue chair. That stronger claim requires an explicitly controlled relation benchmark. [Transfer results](../../experiments/difference_transfer/results/REPORT.md)

## 8. Limitations and claim boundaries

- **Restricted roster:** three checkpoints from one language family and two vision models do not establish cross-family scaling.
- **Restricted data:** one 1,024-pair dataset, with overlapping exploratory splits; no larger independent evaluation set has been tested.
- **Alignment versus scale:** rejecting random correspondence does not establish a significant model-size effect.
- **Changed statistical target:** fixed-reference conditional intervals do not retroactively quantify uncertainty in the shared-pool scores.
- **Preprocessing changes geometry:** improved alignment is not proof of improved semantic accuracy, and PCA combines centering with dimension reduction.
- **Approximate inference:** bootstrap coverage and query exchangeability are assumptions. Model pairs and splits are not independent replicates.
- **Possible source dependence:** no duplicate exact image hashes were found, but there is one duplicate exact caption. Neither check excludes near duplicates or shared sources/topics.
- **Scoped multiplicity:** corrections apply to explicitly named test families, not the entire adaptive research process.
- **Supervised transfer:** the direction pilot uses paired map-fitting data and does not demonstrate blind matching or semantic compositionality.
- **Replication limits:** the baseline differs from the original paper in model roster, environment, language evaluation, and unresolved implementation details; numerical reproduction remains unassessed.

## 9. Recommended next experiment

The strongest supported findings are persistent local cross-modal agreement, conditional preprocessing effects, and supervised endpoint-difference transfer. The next semantic experiment should test a recurring relation across held-out object categories.

A concrete first relation is **red → blue**. Fit the direction using some object categories and evaluate on different categories, with independent images, balanced backgrounds, and verified labels. Keep map-fitting examples separate from final evaluation concepts. Choose translation magnitude on validation data only.

Compare transferred directions against no translation, reversed direction, shuffled relation/map controls, and direct target-text retrieval. Measure both target-color retrieval and preservation of object identity. Resample independent source objects for paired uncertainty, rather than treating near-duplicate images as independent observations.

The current cache lacks verified repeated-relation annotations. Benchmark curation is therefore the next substantive step. In parallel with that new research direction, a larger independent image–caption dataset would be needed to assess whether the alignment findings generalize beyond the present sample. [Semantic benchmark plan](../../experiments/color_transfer/docs/CONCEPT_TRANSFER_NEXT.md)

## 10. Reproducibility and artifacts

The latest recorded test run passed **24 tests**, covering alignment calculations, permutation relabeling, train-only transformations, PCA against an independent eigendecomposition, paired bootstrap equivalence, reference-bank scoring, and ridge-map/difference calculations. The unchanged robustness conditions reproduced all 18 previous held-out scores and their 1,000 null draws exactly. Earlier result directories were preserved.

The project stores exact split indices, selected layers, transformation thresholds, scores, permutation/bootstrap draws, and source hashes in experiment result files. SciPy 1.18.1 was added for bootstrap inference and recorded in the analysis dependency file; runtime versions are stored with the new experiments.

| Artifact | Location |
|---|---|
| Research plan | [RESEARCH_PLAN.md](../../experiments/heldout_layers/docs/RESEARCH_PLAN.md) |
| Baseline protocol | [PROTOCOL.md](../../experiments/baseline/docs/PROTOCOL.md) |
| Robustness protocol | [ROBUSTNESS_PROTOCOL.md](../../experiments/robustness/docs/ROBUSTNESS_PROTOCOL.md) |
| Paired inference protocol | [PAIRED_PROTOCOL.md](../../experiments/paired_uncertainty/docs/PAIRED_PROTOCOL.md) |
| Transfer protocol and correction | [TRANSFER_PROTOCOL.md](../../experiments/difference_transfer/docs/TRANSFER_PROTOCOL.md) |
| Held-out layer script | [heldout_layers.py](../../experiments/heldout_layers/evaluation/heldout_layers.py) |
| Robustness script | [robustness.py](../../experiments/robustness/evaluation/robustness.py) |
| Paired inference script | [paired_uncertainty.py](../../experiments/paired_uncertainty/evaluation/paired_uncertainty.py) |
| Transfer script | [difference_transfer.py](../../experiments/difference_transfer/evaluation/difference_transfer.py) |
| Analysis dependencies | [requirements-analysis.txt](../../requirements-analysis.txt) |

Run the scripts from `/Users/sz/Programming/CDS/conv-mod` using the project virtual environment. Each script regenerates its own output directory's result files, so preserve a copy before a deliberate rerun if exact existing artifacts must be retained.
