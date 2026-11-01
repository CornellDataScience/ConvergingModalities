> Historical WIT/alignment-stage record. The completed natural-state experiment (including raw → cooked) is documented in [README.md](../../README.md) and [STATE_TRANSFER_PAPER.md](../../experiments/state_transfer/docs/STATE_TRANSFER_PAPER.md). Pending work described below reflects this earlier stage.

> Historical proposal/report. The active natural-state study and results are in [STATE_TRANSFER_PAPER.md](../../experiments/state_transfer/docs/STATE_TRANSFER_PAPER.md) and [CURRENT_DIRECTION.md](../../CURRENT_DIRECTION.md). The red-to-blue main experiment was retired at the user's request.

# Research assessment and decision

October 15, 2026. Based on saved protocols, executable analysis, result manifests, tests and the new endpoint audit. This is an assessment of this reduced study, not a claim to have independently reproduced feature extraction or the original PRH paper.

## Decision

Continue with **cross-category transfer of a repeated semantic intervention**. The question is whether a text-derived relation can predict how an image representation changes when a controlled attribute changes on an unseen object category. Keep alignment robustness as context and an appendix. Stop expanding the scaling story until there is an independent dataset and broader model roster.

Working paper title: **From Endpoint Alignment to Relation Transfer: Evaluating Cross-Modal Vector Arithmetic**. The current evidence supports a presentable exploratory technical report. A convincing empirical paper needs the controlled relation benchmark; paper acceptance and novelty are not established.

## What is good

- The study separates existence of alignment from scaling. Frozen held-out layer selection, condition-specific correspondence nulls, train-only transforms and saved sample identities are meaningful safeguards.
- Centering and PCA effects survive a separate fixed-reference paired evaluation. That is a solid conditional finding, with its scope stated.
- The transfer pilot has a measurable signal with modest local compute and reusable cached embeddings. It is a useful engineering prerequisite for a semantic intervention experiment.
- The report acknowledges the adaptive inferential correction, overlapping splits and changed estimand. This honesty is a strength, although it does not undo adaptivity.
- The new audit reproduces saved transfer vectors and checks every split's endpoint/difference identity. There are now 26 passing analysis tests.

## What is weak or missing

- **The core semantic claim is untested.** Arbitrary WIT endpoints have no repeated relation label. For an affine predictor f(x)=xW+b, f(x_b)-f(x_a)=(x_b-x_a)W identically. Difference retrieval is not a separate test of compositionality.
- **The scaling result is fragile and low-powered.** Three related language checkpoints and two vision models cannot support a broad law; lack of an interval exclusion does not show equivalence or refute PRH.
- **No independent replication data.** The three splits reuse one inspected 1,024-example corpus. Repeated splits cannot be counted as independent studies. Exact caption checks do not address source/topic clusters.
- **The metric changes the question.** Fixed k and fixed neighborhood fraction vary different scales; centering/PCA changes representation geometry. Higher neighbor agreement is not necessarily better semantics.
- **Selection consumes scientific flexibility.** The displayed model pair, representation, map and research direction are exploratory choices. Future claims need a frozen primary analysis and fresh test concepts.
- **Supervision has no adequate competitor yet.** Compare ridge to orthogonal Procrustes and other simple fixed maps, and repeat beyond this single model-family pair only after the main benchmark works. A cross-modal model such as CLIP is a useful supervised reference, not a fair independent-training control.
- **No claimed novelty yet.** Huh et al. (2024) motivates cross-modal geometric agreement; Fournier et al. (2020) distinguishes regularities behind analogy scores, and Linzen (2016) describes evaluation pitfalls. The endpoint algebra is elementary, not a novel theorem. A broader related-work review remains necessary.

## New evidence obtained in this review

Fixed original pipeline, all three saved splits; no tuning. Each difference task retrieves among the same 256 candidates.

| Split | Full difference top-1 | Target endpoint only | Negated source only | Mean direction cosine |
|---|---:|---:|---:|---:|
| 20261004 | 25.78% | 5.47% | 8.59% | 0.4503 |
| 20261005 | 19.53% | 7.42% | 3.91% | 0.4305 |
| 20261006 | 20.31% | 6.64% | 5.47% | 0.4916 |

Uniform candidate chance is 0.39%. These are descriptive results on dependent splits; no new p-values. Endpoint-only ablations use affine endpoint predictions, so depend on coordinate origin. Both endpoints together perform better descriptively, which indicates useful information from both. It does not imply a relation mechanism beyond endpoint prediction, nor quantify the fraction of performance attributable to either endpoint. The new full difference results extend the original pilot beyond its first split.

The explorer's first two joint direction components retain **28.61%** of squared vector norm. This makes projection distortion a substantial concern; its displayed scores and ranks always use 128D. The mean of random endpoint differences has no named semantic interpretation.

## A concrete result to present now

A 10–15 minute talk / exploratory report can make three defensible claims: local agreement exists in this reduced setup; preprocessing affects the measured agreement more consistently than this small model-size contrast; supervised mappings transfer arbitrary endpoint differences across the three saved splits, but that task does not identify reusable semantic relations. Show the endpoint ablations and interactive errors. Do not title this a demonstration of semantic analogy or a refutation of PRH.

The accompanying PAPER_DRAFT.md is a working manuscript for this current result, with the missing experiment explicitly separated. SEMANTIC_PROTOCOL.md defines the next milestone and failure cases.

## Next milestones and stopping rules

1. Curate and validate a small red→blue pilot with category and source separation, then freeze the full protocol before looking at final test embeddings. A pilot is for debugging and validation; retire its categories from the final test.
2. Run the controlled 24-category benchmark, report category-macro effects against no shift and shuffled/reversed relation controls, and report object preservation. Hold target images of test categories out of direction/map fitting and scale selection.
3. If transfer fails while an image-derived training direction works, investigate modality/map mismatch. If both fail, investigate whether a global linear relation exists at all. If only direct target-text retrieval succeeds, report recognition without relation transfer. All are interpretable outcomes.
4. If the main relation succeeds, confirm blue→red, a second attribute relation, and a second independently trained language/vision family on untouched test data. These are additional tests with an explicitly expanded multiplicity family.
5. Validate a successful controlled effect on natural photographs with independent sources before claiming natural-image semantic generalization. Publish a methods/negative-result paper if relation transfer consistently fails despite strong endpoint alignment, subject to adequate sensitivity and broader baselines.

Do not spend the next phase on blind GW alignment, audio, more model-size plots or a large visualization platform. The present bottleneck is intervention data and a discriminating evaluation.

## References checked for this assessment

- Huh et al. (2024), Position: The Platonic Representation Hypothesis: https://proceedings.mlr.press/v235/huh24a.html
- Fournier, Dupoux and Dunbar (2020), Analogies minus analogy test: measuring regularities in word embeddings: https://aclanthology.org/2020.conll-1.29/
- Linzen (2016), Issues in evaluating semantic spaces using word analogies: https://arxiv.org/abs/1606.07736
