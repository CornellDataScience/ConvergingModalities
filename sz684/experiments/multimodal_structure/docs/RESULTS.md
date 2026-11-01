# Actual execution results

This is the October 22 pilot. The expanded held-out study is documented in
[the week-8 report](week8/RESULTS.md).

October 22, 2026. All findings below are exploratory.

## Implemented and run

- Real public audiovisual data: 24 event concepts, 72 recordings, 72 midpoint images, 72 audio clips and 72 label-derived text descriptions.
- Pinned VGGSounder annotations and media revisions; all 72 selected recordings downloaded successfully. Download logs retain failures and SHA256s.
- Frozen extraction: BLOOMZ-560m, BLOOMZ-1.7B, DINOv2-small, DINOv2-base, BEATs iter1. No encoder fine-tuning.
- Geometry: all ten model pairs, 199 correspondence permutations per pair. Both full-roster exploratory (24 concepts) and test-split (5 concepts) reports are saved.
- Reused image–text study: 30 held-out concepts, five relations evaluated in both directions (10 directional results, zero skipped).
- Tests: 54 passed, including unchanged historical tests, analytic metrics, real WAV resampling and a synthetic three-modality integration test.

## Real AV pilot: full 24-concept exploratory roster

| Model pair | RSA | CKA | Jaccard | Top-3 overlap |
| --- | ---: | ---: | ---: | ---: |
| bloomz-560m / bloomz-1b7 | 0.6601 | 0.6867 | 0.4083 | 0.5417 |
| bloomz-560m / dinov2_small | 0.1075 | 0.5426 | 0.1625 | 0.2500 |
| bloomz-560m / dinov2_base | 0.0520 | 0.5026 | 0.1625 | 0.2500 |
| bloomz-560m / beats_iter1 | 0.0047 | 0.2208 | 0.1417 | 0.2222 |
| bloomz-1b7 / dinov2_small | 0.1819 | 0.4512 | 0.1375 | 0.2083 |
| bloomz-1b7 / dinov2_base | 0.0269 | 0.4340 | 0.1500 | 0.2222 |
| bloomz-1b7 / beats_iter1 | -0.0688 | 0.1793 | 0.2042 | 0.2917 |
| dinov2_small / dinov2_base | 0.6020 | 0.9354 | 0.4000 | 0.5278 |
| dinov2_small / beats_iter1 | 0.0764 | 0.3636 | 0.1667 | 0.2500 |
| dinov2_base / beats_iter1 | 0.1438 | 0.3406 | 0.1542 | 0.2361 |

The full roster intentionally pools development splits for a geometry feasibility analysis; it is not a held-out confirmatory result. Random top-3 intersection/k expectation is 3/23 ≈ 0.1304. The five-concept test split has expectation 3/4 = 0.75, so superficially large overlap there is not strong evidence. CKA has its own finite-sample baseline; inspect correspondence nulls rather than comparing its numeric scale to RSA.

## Reused image–text study

Legacy selected-layer features yield RSA 0.4183, CKA 0.1957, Jaccard 0.3567 and overlap 0.4778. These use different concepts/layers from the new AV pilot and cannot be read as a controlled benchmark improvement.

| Transfer | Relation | Signed CAV cosine | Transferred balanced accuracy | Target probe accuracy |
| --- | --- | ---: | ---: | ---: |
| bloomz → dinov2 | cleaning | -0.3849 | 0.5625 | 0.5000 |
| bloomz → dinov2 | cooking | 0.6931 | 0.5833 | 0.6667 |
| bloomz → dinov2 | drying | -0.2116 | 0.5469 | 0.4219 |
| bloomz → dinov2 | ripening | 0.0888 | 0.5000 | 0.7188 |
| bloomz → dinov2 | thawing | 0.0059 | 0.5000 | 0.6094 |
| dinov2 → bloomz | cleaning | -0.3849 | 0.5000 | 0.9375 |
| dinov2 → bloomz | cooking | 0.6931 | 0.6250 | 1.0000 |
| dinov2 → bloomz | drying | -0.2116 | 0.1250 | 1.0000 |
| dinov2 → bloomz | ripening | 0.0888 | 0.5000 | 0.9375 |
| dinov2 → bloomz | thawing | 0.0059 | 0.0000 | 1.0000 |

Signed CAV agreement and transported classification are mixed. Do not infer a universal relation algebra. Source CAV stepping is evaluated separately from classifier transfer; saved directional JSON includes validation-selected scales, AP, identity retention, joint success, no-shift/reverse/random controls, fitted arrays and all split lists.

## Limitations and remaining work

- AV recordings have released video-level labels; midpoint frames were not independently reviewed. Exact event labels stand in for families, not a verified taxonomy.
- Three examples per event and a small test roster support feasibility work, not stable population estimates.
- No vision–audio directional results: the AV data has no verified binary relation axes. Labeling event categories as arbitrary endpoints would answer a different question.
- The five-model roster reuses sizes within two families; it is not six models from independent training objectives. MAE/AudioMAE adapters and objective-diverse replication remain.
- No refit confidence intervals, reviewed family-block nulls, multiplicity-adjusted claims, or fresh-source replication.
- BEATs source is pinned official code. Its checkpoint is an original-format iter1 mirror; configuration rejects fine-tuned models. Byte equivalence to the official OneDrive artifact is not independently established.
- One converted checkpoint mirror was rejected because its architecture differed. Its URL/revision/hash and reason are logged in ignored assets; its unused 361 MB weights were removed.
- Runtime additions were installed only in the existing repo virtual environment: torchaudio 2.5.1, soundfile 0.14.0, imageio-ffmpeg 0.6.0 and their dependencies.

## Artifact locations

All paths below are relative to this experiment folder. `results/` is ignored by Git; small score summaries in this docs folder are tracked candidates.

- `results/av_pilot/manifest.json`, `records.csv`, `annotations.csv`, `download_log.json`: actual data and provenance.
- `results/models.local.json`: pinned model specs, revisions, weight and source hashes, pooling and eligibility notes.
- `results/av_pilot/features/`: five model caches and resumable chunks. The working audio file is `beats_iter1_original.npz`.
- `results/av_pilot/geometry_all.json`, `geometry_test.json`, `REPORT_all.md`, `REPORT_test.md`: actual AV scores and null draws.
- `results/legacy_pilot/`: imported historical data/features, geometry, direction fits and report.
- `docs/PILOT_SCORES.csv`: compact export of actual geometry scores.
- `docs/run_summary.json`: dataset identities, model revisions and source hashes.

Original experiment results and the pre-existing folder reorganization were preserved.
