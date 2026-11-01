# Reproducing the expanded study

Run commands from `sz684/`, using `.venv/bin/python`. The completed run lives in
`experiments/multimodal_structure/results/week8_study/`. Results are ignored by Git;
the compact report and exported score tables live in `docs/week8/`.

Read `WEEK8_PROTOCOL.md` before changing a setting. New intentional settings need
a new run folder, fit file and evaluation lock. Do not rewrite historical locks.
The model registry retains local absolute paths: use setup to regenerate them on
another machine. The prior pilot's three existing checkpoint specs are reused.

```sh
.venv/bin/python -m experiments.multimodal_structure.run setup-week8-models \
  experiments/multimodal_structure/results/models.local.json \
  experiments/multimodal_structure/results/models.week8.local.json

.venv/bin/python -m experiments.multimodal_structure.run study-plan \
  experiments/multimodal_structure/results/av_pilot/annotations.csv \
  experiments/multimodal_structure/results/av_pilot/download_log.json \
  experiments/multimodal_structure/results/week8_study

.venv/bin/python -m experiments.multimodal_structure.run study-acquire \
  experiments/multimodal_structure/results/week8_study

.venv/bin/python -m experiments.multimodal_structure.run study-finalize \
  experiments/multimodal_structure/results/week8_study
```

`study-finalize` records provisional families and category-coverage split assignment
before extraction. It runs once and preserves the initial plan. Our actual run
also called `data.audit.audit_media` to decode inputs and check exact-image/source
leakage, and `replace_unusable_audio` to substitute three silent recordings from
the frozen backup order. Preserve both the pre-QA and final audits. These functions
are deliberately separate from model extraction and do not consult model scores.

Extract each of the six models with the existing `extract` command. Example:

```sh
.venv/bin/python -m experiments.multimodal_structure.run extract \
  experiments/multimodal_structure/results/week8_study/manifest.json \
  experiments/multimodal_structure/results/models.week8.local.json mae_base \
  experiments/multimodal_structure/results/week8_study/features/mae_base.npz \
  --device mps --batch-size 8
```

Repeat for BLOOMZ/Pythia/DINOv2 (MPS, batch 8), BEATs (CPU, batch 1), and AudioMAE
(MPS, batch 4). The CPU audio processor is part of each adapter. Do not change
batch size or extraction code while resuming chunks. On machines without MPS,
choose CPU explicitly and use a new cache path.

Each following command accepts six repeated `--cache NAME=PATH` arguments. The
names are `bloomz-560m`, `pythia410`, `dinov2_base`, `mae_base`, `beats_iter1`,
`audiomae`, and paths are `results/week8_study/features/NAME.npz` under this
experiment folder. All six must refer to the same prepared manifest.

```text
week7-fit MANIFEST week7_fits.json --cache NAME=PATH ...
week8-lock MANIFEST week7_fits.json evaluation_lock.json --cache NAME=PATH ...
week8-score MANIFEST week7_fits.json evaluation_lock.json week8_test.json --cache NAME=PATH ...
```

The separate `evaluation.reliability.development_reliability` function reads the
same caches but only computes train/validation split-half scores. The fit artifact
records feature and software hashes. The lock also records protocol hashes;
changing source, inputs or protocol stops scoring. A finished test output cannot
be overwritten. Save and inspect the first evaluation even when results are null.

```sh
.venv/bin/python -m experiments.multimodal_structure.run week8-report \
  experiments/multimodal_structure/results/week8_study \
  experiments/multimodal_structure/docs/week8
```

Reporting reads saved scores; it does not refit or rerun evaluation. The synthetic
integration tests check train/test separation and lock behavior. Real extraction
and scoring are separately recorded in the final report.
