# Results explained: chance baselines and worked calculations

**October 24, 2026.** This expands the [progress report](/Users/sz/Programming/CDS/conv-mod/sz684/experiments/multimodal_structure/docs/PROGRESS_REPORT.md) using the original locked outputs. No models were retrained, no hyperparameters were changed, and no new held-out examples were introduced. The supplementary shuffled-CAV angles are descriptive calculations from already saved fit coefficients.

## 1. What should random have produced?

There is no single chance value for every metric. We also did not train randomly initialized control encoders. For the geometry controls, we scrambled concept correspondence between the same pretrained representations.

| Measure | Random reference | Qualification |
| --- | --- | --- |
| RSA | Expected 0 under uniform concept-label permutation | Individual shuffled correlations can be considerably above or below zero. |
| Top-3 overlap | Expected 0.2000 for uniform three-neighbor sets among 15 candidates | The 16 node scores are dependent. |
| Neighbor Jaccard | Expected approximately 0.1288 under the same uniform-set model | This has a different denominator from overlap. |
| Implemented linear CKA | Positive, model-pair-specific | Saved shuffle means ranged from 0.2358 to 0.6414. Zero is not the correct generic baseline. |
| Binary balanced accuracy | Expected 0.5 for predictions independent of true class | Constant predictors also score 0.5, despite potentially high ordinary accuracy. |
| Signed cosine of independent uniform random directions | Mean 0; standard deviation 1/√d | A mathematical illustration, not the actual null for trained and selected CAVs. |

Full observed scores, analytical null means, empirical null means, percentiles, and unadjusted tail values are in [CHANCE_BASELINES.csv](/Users/sz/Programming/CDS/conv-mod/sz684/experiments/multimodal_structure/docs/week8/CHANCE_BASELINES.csv). The original 999 draws per model pair remain in [week8_test_geometry.json](/Users/sz/Programming/CDS/conv-mod/sz684/experiments/multimodal_structure/results/week8_study/week8_test_geometry.json).

## 2. How the geometry results were calculated

For each of the 16 held-out concepts, every image/audio model supplied 15 recording vectors; each text model supplied four description vectors. We averaged raw vectors within a source, then equally across sources. We did not normalize individual examples before averaging.

This produced matrices with the same concept-ID row order: **16 × 1,024** for text and **16 × 768** for vision/audio. Geometry used these native features. No learned alignment, PCA transform, probe, or test-selected layer entered these scores.

Files: [concept_centroids in evaluate.py](/Users/sz/Programming/CDS/conv-mod/sz684/experiments/multimodal_structure/evaluation/evaluate.py), [cache loading](/Users/sz/Programming/CDS/conv-mod/sz684/experiments/multimodal_structure/extraction/cache.py), and [saved distance matrices/neighbors](/Users/sz/Programming/CDS/conv-mod/sz684/experiments/multimodal_structure/results/week8_study/week8_test_geometry_details.json).

### RSA: all pairwise distance orderings

We normalized each concept centroid to unit length, calculated `D[i,j] = 1 − dot(unit_i, unit_j)`, extracted the strict upper triangle, and computed the Spearman rank correlation between models.

Sixteen concepts give **120 unique distances**. For Pythia–DINOv2, both distance vectors had 120 distinct entries, and the sum of squared differences between their ranks was **153,974**. The no-ties Spearman formula therefore gives:

`rho = 1 − (6 × 153,974) / [120 × (120² − 1)] = 0.4653309`.

That is the reported **RSA 0.4653**. It describes agreement in relative distance ordering, not 46.53% classification accuracy.

Uniformly relabeling one fixed RDM gives expected correlation zero, but the distances share concepts. We therefore use whole-concept shuffles to characterize the null spread rather than treating 120 cells as independent observations.

Code: [rsa.py](/Users/sz/Programming/CDS/conv-mod/sz684/experiments/multimodal_structure/evaluation/rsa.py).

### Neighbors: shared nearest concepts

For each concept we selected the three largest cosine similarities, excluding itself. If J is the number of shared neighbors, node overlap is `J/3` and node Jaccard is `J/(6−J)`. Each reported metric is its average across the 16 concepts.

For Pythia–DINOv2, seven concepts shared zero neighbors, seven shared one, and two shared two:

`overlap = (7×0 + 7×1 + 2×2) / (16×3) = 11/48 = 0.2292`.

`mean Jaccard = [7×0 + 7×(1/5) + 2×(2/4)] / 16 = 0.1500`.

A concrete example from the saved neighbor lists:

| Query concept | Model | Three neighbors |
| --- | --- | --- |
| baby crying | Pythia | baby laughter; dog growling; child speech, kid speaking |
| baby crying | DINOv2 | baby laughter; female speech, woman speaking; child speech, kid speaking |

For this concept, two neighbors match: overlap 2/3, Jaccard 2/4. The final scores also include every other test concept.

For independent uniform random sets, each of one model’s three neighbors has probability 3/15 of also appearing in the other set. The expected intersection is `3×3/15 = 0.6`, so expected overlap is **0.2**. Across 16 nodes, that is **9.6 expected shared directed edges among 48**, compared with the observed 11.

Jaccard needs a different calculation. The probabilities of sharing 0, 1, 2, or 3 neighbors are approximately 0.4835, 0.4352, 0.0791, and 0.0022. Weighting Jaccard values 0, 0.2, 0.5, and 1 gives **0.1288**. Taking a ratio of expected intersection and expected union would not give the expectation of Jaccard.

These analytical values describe uniform random sets. For the actual graph, including dependent nodes and deterministic tie handling, we use the saved correspondence-shuffle distribution.

Code: [neighbors.py](/Users/sz/Programming/CDS/conv-mod/sz684/experiments/multimodal_structure/evaluation/neighbors.py).

### CKA: centered inner-product structure

CKA uses raw centroids, not the unit-normalized centroids used for cosine RSA/neighbors. We subtract feature-column means to obtain Xc and Yc and form two 16 × 16 Gram matrices:

`K = Xc @ Xc.T`, `L = Yc @ Yc.T`.

The implemented score is:

`CKA = sum(K * L) / [Frobenius_norm(K) × Frobenius_norm(L)]`.

This gave **0.7334 for Pythia–DINOv2**. It weights inner-product magnitudes rather than just distance ranks, so it need not agree with RSA or local neighbors.

The standard biased CKA implementation has a positive finite-sample correspondence baseline. Under uniform row-and-column permutation of one centered Gram matrix:

`E[CKA] = trace(K) × trace(L) / [(n−1) × norm(K) × norm(L)]`.

For the saved Pythia–DINOv2 features, this analytical expectation is **0.6399**; the 999 shuffles averaged **0.6414**. The observed score’s excess over the empirical null is therefore **0.0920**, not 0.7334 above zero.

Intuitively, unrelated high-dimensional representations can share strong diagonal Gram structure. If both centered Gram matrices approach scaled copies of the centering matrix, CKA approaches one even without meaningful concept correspondence. The pair-specific baseline matters.

Code: [cka.py](/Users/sz/Programming/CDS/conv-mod/sz684/experiments/multimodal_structure/evaluation/cka.py). Analytical and empirical means are both exported in [the baseline table](/Users/sz/Programming/CDS/conv-mod/sz684/experiments/multimodal_structure/docs/week8/CHANCE_BASELINES.csv).

## 3. What the shuffles actually tested

For every model pair, the evaluation calculated the correct-correspondence scores, randomly permuted model B’s 16 centroid rows, recalculated every metric, and repeated that process 999 times.

The vectors were not replaced with Gaussian noise. The question was: **does matching the correct concepts produce more agreement than arbitrary matching between these same representations?** This does not compare trained networks with untrained networks, or establish independent pretraining exposure.

The saved one-sided calculation was:

`p = (1 + number of null scores >= observed) / 1000`.

The smallest reportable value was 0.001, not zero. A p-value is not the probability that the semantic interpretation is true.

A 95th null percentile describes scores under the shuffle construction. It is not a 95% confidence interval for the observed effect. Graph scores are discrete and can tie at a percentile; the saved tail count includes ties.

The p-values are unadjusted. We tested multiple model pairs/metrics, and related concepts complicate unrestricted exchangeability. Family-aware inference and multiplicity adjustment remain unfinished; we do not make final significance declarations here.

Code: [controls.py](/Users/sz/Programming/CDS/conv-mod/sz684/experiments/multimodal_structure/evaluation/controls.py).

![Observed scores versus saved shuffle draws](/Users/sz/Programming/CDS/conv-mod/sz684/experiments/multimodal_structure/docs/week8/chance_distributions.png)

## 4. What the geometry comparisons mean against chance

Each cell is **observed / shuffled mean**:

| Pair | RSA | CKA | Top-3 overlap |
| --- | --- | --- | --- |
| BLOOMZ / DINOv2 | 0.0607 / 0.0026 | 0.4751 / 0.4587 | 0.2083 / 0.1992 |
| Pythia / DINOv2 | 0.4653 / 0.0090 | 0.7334 / 0.6414 | 0.2292 / 0.2006 |
| Pythia / image MAE | 0.3107 / 0.0055 | 0.6944 / 0.5024 | 0.3958 / 0.2006 |
| DINOv2 / BEATs | 0.3467 / −0.0014 | 0.4092 / 0.3556 | 0.3542 / 0.2015 |
| DINOv2 / AudioMAE | 0.4048 / 0.0000 | 0.5746 / 0.4404 | 0.3125 / 0.1997 |
| BEATs / AudioMAE | 0.7094 / 0.0052 | 0.6848 / 0.2358 | 0.6458 / 0.2009 |

- **BLOOMZ–DINOv2:** none of the four observed metrics exceeds its saved 95th null percentile. This configuration shows little separation from arbitrary correspondence. It does not establish that text and images lack shared structure; BLOOMZ’s low template stability is also relevant.
- **Pythia–DINOv2:** RSA exceeds its 95th null percentile 0.3151 and CKA exceeds 0.6925; their unadjusted tail values are 0.004 and 0.003. But top-3 overlap is near chance and below its null 95th percentile 0.3750. Broad distance/inner-product agreement does not automatically preserve exact nearest neighbors.
- **DINOv2–AudioMAE:** RSA and CKA exceed their saved 95th null percentiles, but top-3 overlap has unadjusted tail value 0.080. The global and local pictures again differ.
- **BEATs–AudioMAE:** the within-audio comparison separates substantially from its shuffle controls across all metrics. It is a within-modality reference, not independent evidence of cross-modal convergence; common training exposure and objectives matter.

## 5. How CAV transfer was calculated

This was a separate supervised branch:

1. Split training sources into map anchors and separate probe examples.
2. Fit PCA on training-anchor concept means, then fit orthogonal Procrustes between corresponding training concepts.
3. Fit class-balanced ridge binary probes on the other training sources. Each probe’s coefficient vector defines its CAV.
4. Choose probe regularization and the map direction’s PCA width using validation performance only.
5. Rotate the source normal into target coordinates and also transport its decision-plane origin.
6. Classify held-out target-model examples and compare the transferred normal with the target probe’s normal.

For row vectors, the transported normal is `source_normal @ R`. The classifier center is `(source_probe_center − map_source_mean) @ R + map_target_mean`. Classification uses the sign of `(target_example − transported_center) @ transported_normal`.

**CAV cosine compares fitted training-derived vectors; it is not itself an independent held-out outcome.** Held-out target classification checks whether those fitted objects generalize to new examples/families.

Files: [build_fits.py](/Users/sz/Programming/CDS/conv-mod/sz684/experiments/multimodal_structure/training/build_fits.py), [probes.py](/Users/sz/Programming/CDS/conv-mod/sz684/experiments/multimodal_structure/training/probes.py), [alignment.py](/Users/sz/Programming/CDS/conv-mod/sz684/experiments/multimodal_structure/training/alignment.py), and [directions.py](/Users/sz/Programming/CDS/conv-mod/sz684/experiments/multimodal_structure/evaluation/directions.py).

### Chance for classification

Balanced accuracy is the mean of positive recall and negative recall. If a prediction is positive independently of truth with probability q, expected recalls are q and 1−q, so expected balanced accuracy is **0.5**, regardless of class imbalance. Individual finite random runs need not equal 0.5.

The instrument test had 60 positive and 180 negative recording examples per perception model. Always predicting negative gives **75% ordinary accuracy**, but **50% balanced accuracy**. This is why ordinary accuracy would be misleading.

Our empirical control shuffled training-anchor correspondences before fitting Procrustes, retaining the fitted probes and selected width. It was one saved shuffled map per pair/width, not 999 refitted classifier controls or a rerun of hyperparameter selection under each randomization. Its observed accuracy can differ from 0.5.

### Chance for CAV angles

Independent uniform random directions in d dimensions have mean signed cosine 0 and standard deviation 1/√d. Their hypothetical central 95% ranges are:

| Width | Standard deviation | Central 95% range |
| --- | ---: | --- |
| 8 | 0.354 | −0.666 to +0.666 |
| 16 | 0.250 | −0.482 to +0.482 |
| 32 | 0.177 | −0.344 to +0.344 |

These are mathematical illustrations, not confidence intervals or valid p-values for trained, validation-selected CAVs. Learned spaces and shared class definitions violate the independent-uniform model.

We also reconstructed angles from the already saved shuffled-anchor maps—using saved coefficients only, without refitting. The primary means were:

| Comparison | Axis | Correct-anchor cosine | Shuffled-anchor cosine |
| --- | --- | ---: | ---: |
| Image–text | Instrument performance | 0.7391 | 0.2244 |
| Image–text | Biological/manufactured | 0.5234 | 0.0925 |
| Vision–audio | Instrument performance | 0.5898 | 0.0628 |
| Vision–audio | Biological/manufactured | 0.5682 | −0.0296 |

These nonzero control means illustrate why zero alone is not the proper empirical comparison. A single shuffled map per pair is still not an angular significance test. All values and explicitly labeled hypothetical references are in [CAV_CHANCE_COMPARISONS.csv](/Users/sz/Programming/CDS/conv-mod/sz684/experiments/multimodal_structure/docs/week8/CAV_CHANCE_COMPARISONS.csv).

## 6. Worked classification examples

### AudioMAE → DINOv2: instrument performance

This direction selected 16 dimensions. The saved predictions were:

| True class | Predicted instrument | Predicted other |
| --- | ---: | ---: |
| Instrument: 60 | 54 | 6 |
| Other: 180 | 23 | 157 |

Positive recall = 54/60 = 0.9000; negative recall = 157/180 = 0.8722. Their mean is **0.8861 balanced accuracy**. CAV cosine was 0.7822; the separately trained target probe scored 0.7833 and the shuffled-anchor control scored 0.5806.

This one transferred classifier outperformed its target-probe reference on this test. It does not establish general superiority or a reliable improvement across datasets.

### DINOv2 → AudioMAE: the reverse failed

The reverse selected eight dimensions, so it was not simply the inverse of the previous experiment. It predicted negative for every example:

| True class | Predicted instrument | Predicted other |
| --- | ---: | ---: |
| Instrument: 60 | 0 | 60 |
| Other: 180 | 0 | 180 |

Positive recall = 0; negative recall = 1; balanced accuracy = **0.5**. Ordinary accuracy would be 75%. Yet CAV cosine was positive, **0.6008**. Angular agreement did not place the transported decision boundary usefully among target examples.

A scale/centering/threshold mismatch is a possible explanation, not a demonstrated cause without an ablation. No post-test calibration was added. The target AudioMAE probe scored 0.7139, showing usable category information in the target features.

Both tables were counted from [week8_test_predictions.csv](/Users/sz/Programming/CDS/conv-mod/sz684/experiments/multimodal_structure/results/week8_study/week8_test_predictions.csv).

## 7. How an average of about 65% arose

The vision–audio instrument summary averaged eight ordered map scores equally:

| Source → target | Width | Balanced accuracy |
| --- | ---: | ---: |
| DINOv2 → BEATs | 32 | 49.72% |
| DINOv2 → AudioMAE | 8 | 50.00% |
| Image MAE → BEATs | 32 | 68.61% |
| Image MAE → AudioMAE | 32 | 50.00% |
| BEATs → DINOv2 | 8 | 65.00% |
| BEATs → image MAE | 32 | 74.72% |
| AudioMAE → DINOv2 | 16 | 88.61% |
| AudioMAE → image MAE | 32 | 75.56% |

The mean is **65.28%**, not an ordinary accuracy pooled over all prediction rows. It hides substantial directional differences.

| Comparison | Axis | Transferred BA | Ideal random-label expectation | Saved shuffled-anchor BA | Target probe BA |
| --- | --- | ---: | ---: | ---: | ---: |
| Image–text | Instrument | 65.31% | 50% | 59.24% | 78.87% |
| Image–text | Biological/manufactured | 71.49% | 50% | 48.88% | 76.48% |
| Vision–audio | Instrument | 65.28% | 50% | 50.90% | 77.33% |
| Vision–audio | Biological/manufactured | 65.99% | 50% | 50.56% | 78.67% |

Image–text instrument transfer exceeds its saved shuffled-map control by about **6.07 percentage points**, whereas vision–audio instrument transfer exceeds its control by about **14.38 points**. These are descriptive fixed-fit contrasts, not confidence intervals or proven improvements. Directions share models/data and are not independent replications.

## 8. What we can conclude now

The evidence is **partial and metric-dependent in this particular concept roster**. Some global geometry comparisons separate from their saved wrong-correspondence controls, while exact local neighbors agree less consistently. Category transfer can work, but is asymmetric; CAV angle alone does not guarantee useful predictions.

The geometric sample is 16 concepts, not 120 independent distances. Likewise, the 10,540 prediction rows reuse examples across models/axes and are not 10,540 independent samples. Unrestricted null percentiles are not effect confidence intervals, and their unadjusted tail values do not account for all comparisons or family structure.

BLOOMZ’s validation template-stability correlation was −0.0405. Weak correspondence in that configuration is not solely evidence of missing semantic structure. Text descriptions explicitly name events, and evaluation/pretraining overlap is not excluded.

The axes are researcher groupings of released event labels, not verified physical state changes. No identity-preserving audiovisual editing claim follows. Refit/cluster uncertainty, multiplicity handling and independent-source replication remain necessary for stronger conclusions.

The original locked evaluation, fits, protocol and code remain unchanged. This document explains those results; it is not a new confirmatory run.
