# Validation and interpretation notes

These checks were computed on train/validation data before the evaluation lock. The locked test was not used to choose layers, width or regularization.

| Model | Train split-half RDM correlation | Validation split-half RDM correlation |
| --- | ---: | ---: |
| bloomz-560m | 0.1179 | -0.0405 |
| pythia410 | 0.3506 | 0.4657 |
| dinov2_base | 0.7477 | 0.8827 |
| mae_base | 0.7240 | 0.6064 |
| beats_iter1 | 0.7388 | 0.8983 |
| audiomae | 0.6677 | 0.6272 |

Image/audio halves use disjoint original recording IDs. Text halves use disjoint template formulations, not independent situations. These descriptive correlations are not corrected noise ceilings or confidence intervals.
BLOOMZ template stability is low in this fixed last-layer masked-mean configuration. Its low cross-modal geometry scores cannot be interpreted solely as evidence that text and perception lack shared structure.

## Classifier transport and calibration

Transfer is asymmetric, and some map directions predict only one class (balanced accuracy 0.5). The primary fit uses a centered orthogonal map in native-unit PCA spaces without an isotropic scale or target calibration. Geometry agreement, CAV angle and the transported decision threshold can therefore disagree. No post-test rescaling was added to improve these results.
CAV cosine concerns the projected spaces, not the discarded native coordinates. Selected widths are per direction, so a target-model probe baseline can differ across source models. The target baseline is a separately trained target probe, not a theoretical oracle.

## Scope and provenance

The six encoders cover two text model families, two image learning objectives and two audio learning objectives. All stayed frozen. This is one checkpoint per chosen model, not independent encoder-training seed replication.
Released AV annotations identify event presence across a video. Automatic media QA verifies decoding/silence and source/exact-image leakage, not whether every midpoint still displays its event. Same-split image duplicates are reported; unique recording IDs do not rule out shared scenes or uploaders.
The category labels in concepts_and_axes.csv are researcher-derived from released event names. Instrument performance excludes human singing by definition; the second axis excludes unsupported natural phenomena. Neither axis is a physical state-change operation on the same object.
The historical state-transfer retrieval/identity guardrail remains separately documented in ../RESULTS.md. No new audiovisual identity-preserving state transfer is claimed.

## Completed stages

- Weeks 5–6: 80-concept released-label data acquisition and QA; six-model frozen extraction.
- Week 7: disjoint map-anchor/probe sources, train-only transforms/maps, validation-selected widths/regularization, descriptive repeatability and a saved evaluation lock.
- Week 8: 15 native geometry pairs and 60 ordered category-axis tests; controls, per-example predictions, per-concept neighborhoods, plots and exports.
- Subsequent stages: refit/family-cluster uncertainty, multiplicity decisions, fresh-source replication and any stronger physical-state claims.

Per-example prediction rows: 10540. All 41 code files and both protocol files were copied to an ignored analysis snapshot and verified against the lock.
