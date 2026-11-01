# Completed through week 8: released-label scope

Latest outputs: [held-out results](docs/week8/RESULTS.md),
[validation notes](docs/week8/VALIDATION.md),
[locked protocol](docs/WEEK8_PROTOCOL.md), and
[reproduction guide](docs/WEEK8_RUN_GUIDE.md).

The user approved released labels for this stage. Independent human reannotation
is not claimed. Audiovisual CAVs use explicit category contrasts, not physical
state-change labels. The earlier pilot and historical state-transfer results
remain preserved separately.

| Stage | Delivered |
| --- | --- |
| Weeks 2–4 | Data formats, adapters/caches, metrics, probes/maps and pilot runs |
| Weeks 5–6 | 80 new concept labels; 1,200 paired recordings; 320 text descriptions; six frozen encoders; media/source QA |
| Week 7 | Separate anchor/probe training sources; train-only PCA/maps; validation selection; repeatability checks; fit/feature/software/protocol lock |
| Week 8 | 15 geometry model pairs; 60 ordered category-axis tests; controls; per-example predictions; per-concept distances/neighbors; plots and CSV exports |

Tests: 58 passed. Real extraction and evaluation were executed separately from
synthetic software tests. The first locked test output was preserved, with no
post-test tuning. Code, configuration and protocol snapshots match their hashes.

## Remaining beyond this stage

- Family-cluster/refit uncertainty and multiplicity-adjusted inference (week 9).
- Independent-source replication (week 10).
- Independently verified audiovisual state-change axes before making stronger
  identity-preserving relational claims.
- Calibration/scale and layer/pooling sensitivities, explicitly exploratory after
  this inspected test. Low text-template repeatability and asymmetric transport
  limit semantic interpretation of the current fixed-feature configuration.

Model architectures/objectives are now more diverse: BLOOMZ/Pythia,
DINOv2/MAE, and BEATs/AudioMAE. This remains one checkpoint per model, not a
multi-seed encoder-training study.
