> Historical proposal/report. The active natural-state study and results are in [STATE_TRANSFER_PAPER.md](STATE_TRANSFER_PAPER.md) and [CURRENT_DIRECTION.md](CURRENT_DIRECTION.md). The red-to-blue main experiment was retired at the user's request.

# From Endpoint Alignment to Relation Transfer: Evaluating Cross-Modal Vector Arithmetic

Working manuscript, October 15, 2026. Exploratory technical report; no submission, peer review, novelty claim, or completed semantic intervention benchmark. Author information to be supplied.

## Abstract

We examine which conclusions a small cross-modal representation study can support. On 1,024 paired WIT examples represented by three BLOOMZ checkpoints and two DINOv2 models, local neighborhood agreement exceeds correspondence-permutation references under held-out layer selection. A separate fixed-reference analysis supports conditional improvements from centering and centered PCA, but does not establish a largest-versus-smallest language-model advantage. We then fit a supervised ridge map between language and image representations. Across three overlapping splits, arbitrary held-out difference retrieval achieves 19.5–25.8% top-1 accuracy among 256 candidates, compared with 0.39% uniform chance. Single-endpoint ablations reach 3.9–8.6%. We show why this evaluation cannot identify reusable semantic relations: any affine endpoint predictor produces the same difference predictions algebraically. We propose a category-held-out, controlled attribute-intervention experiment that separates endpoint correspondence from relation transfer. Our findings are limited to the inspected corpus and model roster; the semantic experiment remains unrun.

## 1. Motivation and research question

Cross-modal geometric agreement motivates asking whether independently trained representations also share reusable transformations. Huh et al. (2024) studies convergence across representation spaces. That question is distinct from whether a fixed relation direction learned on some objects transfers to new objects. Analogy-evaluation work has already emphasized separating different sources of apparent regularity (Fournier et al., 2020; Linzen, 2016). We apply that distinction to a reduced vision–language experiment. Our aim is a discriminating evaluation, not a new algebraic identity or a general scaling-law claim.

## 2. Experimental setting

We reuse the pinned 1,024-example WIT cache, BLOOMZ 560M/1.1B/1.7B and DINOv2 small/base. Three fixed overlapping splits use 512 selection/map-training and 512 evaluation rows. Layers and clipping are selected/fitted on selection rows. Model revisions, extraction settings and sample hashes are in the baseline manifest. Float32 CPU scoring follows the existing extraction cache; we did not independently re-extract the pretrained model outputs in this review.

The earlier alignment work uses neighborhood overlap, held-out layer selection and correspondence permutations. A separate query/reference design conditions on a fixed reference bank for paired bootstrap analysis. These are different estimands. We preserve the prior protocols and reports rather than presenting their results as a single prospective experiment.

For transfer, BLOOMZ 1.7B and DINOv2 base use selection-chosen layers, separately fitted centered PCA128 with projected endpoint normalization, and ridge regularization alpha=1. Each evaluation half forms 256 disjoint ordered endpoint pairs. The ridge map minimizes centered endpoint squared error plus alpha times squared Frobenius norm. No evaluation hyperparameter search is performed.

## 3. What difference transfer identifies

Let f(x)=xW+b be a supervised endpoint predictor. For any endpoints a,b,

    f(x_b) − f(x_a) = (x_b − x_a)W.

Consequently, every prediction in the arbitrary-difference task can be generated entirely by subtracting endpoint predictions. Moreover, for endpoint errors e_i=f(x_i)−y_i, the difference error is e_b−e_a. Shared endpoint bias can cancel. Difference success need not be accompanied by accurate endpoints individually and does not require a separately represented named relation.

This is an identity, not a claim that relations cannot exist or that supervised mapping is uninformative. It shows an identification limitation: the task has no repeated relation variable or held-out relation instances, so it cannot distinguish an endpoint explanation from reusable relational structure.

## 4. Results

### 4.1 Existing alignment findings

In all 18 held-out model-pair/split evaluations, observed alignment exceeds all 1,000 correspondence permutations. Bonferroni-adjusted finite-permutation p-values within this family are approximately .018. This tests correspondence, not a scaling trend. In the fixed-reference conditional analysis, centering-versus-baseline effects range from .01393 to .02435, with positive family-adjusted intervals in all three splits. The largest-versus-smallest BLOOMZ contrast has adjusted intervals containing zero in every split. See RESEARCH_REPORT.md for the complete design, dependence assumptions and multiplicity scope.

### 4.2 New endpoint contribution audit

| Evaluation split | Full difference cosine | Full difference top-1 | Target endpoint only top-1 | Negated source endpoint top-1 |
|---|---:|---:|---:|---:|
| 20261004 | .4503 | 25.78% | 5.47% | 8.59% |
| 20261005 | .4305 | 19.53% | 7.42% | 3.91% |
| 20261006 | .4916 | 20.31% | 6.64% | 5.47% |

All three predictors rank the identical 256 actual difference candidates. The single-endpoint ablations use affine predictions, so their values depend on coordinate origin. They are diagnostics of partial information, not an additive decomposition of accuracy. The full difference performs better descriptively than either endpoint alone. This supports useful held-out correspondence information from both endpoints, while the identity in Section 3 still holds.

The first split reproduces saved pilot vectors within absolute 1e-6 / relative 1e-5 tolerance. The maximum absolute discrepancy between independently computed mapped differences and differenced affine endpoint predictions is at most 1.20e-6 across splits. The original first split has an exploratory evaluation-correspondence permutation reference, with adjusted p=.002 for mean cosine and top-1. The two newly evaluated splits are descriptive extensions; no new inferential claim or pooling across dependent splits is made.

### 4.3 Visualization and its limits

The interactive explorer displays corresponding mapped and actual directions, selected-pair residuals, full-dimensional target ranks and nearest candidates. The first two components of a joint uncentered direction SVD retain 28.61% of total squared norm. This is a display fitted on evaluation vectors, not a transformation used for scoring. Apparent 2D agreement cannot establish high-dimensional agreement. The optional pair mean has no named semantic interpretation.

## 5. Next experiment: semantic intervention transfer

The decisive experiment estimates one red→blue text relation on training categories, maps it into image coordinates, and applies a validation-fixed translation to red images from disjoint test categories. Controlled counterfactual renders hold asset, lighting, background and camera fixed while changing foreground color. A matched intervention margin, target-color retrieval and category preservation distinguish attribute transfer from a generic target-color attractor. No-shift, reversed, random, category-shuffled, image-relation and direct-target-text controls isolate different explanations. Training map and relation examples exclude final test categories; uncertainty is computed with categories and asset identities as clusters. SEMANTIC_PROTOCOL.md provides the proposed data budget, guardrail and primary contrasts. No results for this experiment are available.

## 6. Limitations and conclusion

One small inspected corpus, one language family, two vision models and overlapping splits restrict external validity. Conditional bootstrap inference does not include selection/bank/fitting uncertainty. PCA, normalization and neighborhood choices change geometry. The transfer stage uses supervision and cannot establish blind alignment. A documented post-run inferential correction makes existing transfer inference exploratory. Feature-cache checks and 26 passing analysis tests do not establish annotation quality or independently verify extraction.

The present result is that endpoint correspondence, local geometric agreement and reusable semantic intervention transfer must be evaluated separately. The first two show measurable signal in this reduced setup; the third remains an explicit empirical question. A stronger paper contribution would be a controlled benchmark and evidence showing when endpoint alignment does or does not predict semantic relation transfer across unseen categories.

## Reproducibility

From `sz684/`, run `.venv/bin/python endpoint_audit.py` and `.venv/bin/python -m pytest -q`. The new script writes only `results/endpoint_audit/`, preserving prior outputs, and saves input hashes, per-split metrics, display coordinates and a self-contained explorer. Dependencies are the existing analysis environment. Generated results directories are ignored by Git; archive them separately for sharing. Consult ENDPOINT_AUDIT_PROTOCOL.md and the original protocols before rerunning other studies.

## References

Huh, M., Cheung, B., Wang, T., and Isola, P. (2024). Position: The Platonic Representation Hypothesis. ICML. https://proceedings.mlr.press/v235/huh24a.html

Fournier, L., Dupoux, E., and Dunbar, E. (2020). Analogies minus analogy test: measuring regularities in word embeddings. CoNLL. https://aclanthology.org/2020.conll-1.29/

Linzen, T. (2016). Issues in evaluating semantic spaces using word analogies. https://arxiv.org/abs/1606.07736
