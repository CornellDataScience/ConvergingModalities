> Historical WIT replication instructions. Start with [README.md](../../../README.md) for the completed natural-state experiment. The milestones below describe the earlier study, not current pending work.

# Mac-scale Platonic Representation Hypothesis replication

Target: Figure 3 of https://arxiv.org/pdf/2405.07987v5, using a reduced original model roster on an Apple M3 with 36 GB unified memory. This project starts with pipeline validation, not a replication verdict.

## Environment

Python 3.12; PyTorch 2.5.1; MPS inference in float32, CPU scoring. No quantization. This differs from the authors' Python 3.11 / PyTorch 2.2 environment and hardware-dependent precision. `requirements.lock.txt` records the complete installed environment.

```sh
uv venv --python python3.12 .venv
uv pip install --python .venv/bin/python -r requirements.lock.txt
.venv/bin/python -m pytest -q
```

MPS may be hidden by the Codex sandbox. Run inference with GPU access; explicitly request `--device mps` so it fails clearly instead of silently using CPU. Downloads remain in the project `.cache/huggingface` directory.

## Pilot

```sh
.venv/bin/python run.py baseline init --run experiments/baseline/results/pilot --samples 32
.venv/bin/python run.py baseline extract --run experiments/baseline/results/pilot --model bigscience/bloomz-560m --device mps
.venv/bin/python run.py baseline extract --run experiments/baseline/results/pilot --model timm/vit_small_patch14_dinov2.lvd142m --device mps
.venv/bin/python run.py baseline align --run experiments/baseline/results/pilot --permutations 100
```

Initialization pins dataset/model revisions and upstream commit, and saves an ordered image/caption manifest. Extraction runs one model at a time, saving atomic per-batch feature chunks. Repeating the same command resumes an interrupted extraction. Cache settings must match; use a new run directory for changed settings. Scores include the full layer-pair matrix, selected indices, and a shuffled correspondence null that repeats the maximum over all layers.

The pilot uses the first 32 original pairs. Its scores **cannot be compared directly with N=1024 scores**: neighborhood size relative to dataset size is much larger. Its purpose is correctness and resource profiling.

## Full reduced-model baseline

After pilot validation:

```sh
.venv/bin/python run.py baseline init --run experiments/baseline/results/baseline --samples 1024
.venv/bin/python run.py baseline extract --run experiments/baseline/results/baseline --device mps
.venv/bin/python run.py baseline align --run experiments/baseline/results/baseline --permutations 100
```

The roster is BLOOMZ 560M/1.1B/1.7B and DINOv2 small/base. All five models use the same sample order. This is six model-pair comparisons, not the full paper's roster. Larger models or more vision families are deferred until measured cost justifies them.

## Frozen measurement choices and discrepancies

- Follow the released repository's **BLOOMZ** checkpoints, despite the paper's BLOOM labels.
- Use k=10, masked token-average language features including the embedding hidden state, and each vision block's CLS output before final model normalization.
- Clip raw `[N,L,D]` features using the released default: average of per-sample absolute-value 95th percentiles. This differs from the optional global order-statistic `--precise` path. Then L2-normalize each layer's feature vector.
- Compare every individual layer pair and take the maximum. The paper mentions concatenated features; the current released extraction/scoring path does not explicitly construct them. Concatenation is not silently added here.
- Language inference uses `AutoModel` to avoid vocabulary logits, with the same backbone hidden states, and batch-local padding. No caption truncation. Vision weights are pinned Hugging Face safetensors, with the original timm architecture and preprocessing.
- Full inference uses float32 on MPS. Scoring uses float32 CPU, with self-neighbors excluded. Record these departures before comparison.

Source files: [model roster](https://github.com/minyoungg/platonic-rep/blob/main/tasks.py), [extraction](https://github.com/minyoungg/platonic-rep/blob/main/extract_features.py), [alignment](https://github.com/minyoungg/platonic-rep/blob/main/measure_alignment.py), [metric](https://github.com/minyoungg/platonic-rep/blob/main/metrics.py). Actual upstream commit is recorded in each manifest.

## What remains before a scientific verdict

1. Complete the pilot and compare feature extraction with the original implementation on the same small inputs.
2. Run the 1024-pair reduced roster and obtain matching author reference scores if available.
3. Resolve the original OpenWebText sample/context protocol for an exact performance-axis comparison. A separately labeled reduced evaluation is implemented below. Caption losses and model sizes are not substitutes for Figure 3's performance axis.
4. Freeze score tolerances based on repeatability and reference precision before inspecting the full comparison. Report numerical agreement separately from trend agreement.
5. With only three language checkpoints from one family, treat correlations as descriptive and do not claim independent evidence from six model pairs.
6. Preserve the baseline before any k/sample-size/clipping/pooling/held-out layer-selection experiments.

A passing pilot does not establish replication. Missing references, protocol ambiguities, or resource limits are reported as incomplete/inconclusive, not as a failed scientific result.

## Reduced language-performance evaluation

`language_eval.py` defines a separate, explicitly downscaled protocol before baseline analysis: first 64 documents from the OpenWebText-10k archive in sorted archive order, at most 1024 Unicode characters per document. The exact texts and archive checksum are persisted. All text tokens are scored once, conditioned on an initial EOS, with at most 256 preceding context tokens and 256 target tokens per window. BPB is total negative log likelihood in bits divided by total UTF-8 text bytes. This uses the original corpus family but does not recreate the paper's sample or context settings.

```sh
.venv/bin/python run.py baseline-language --run experiments/baseline/results/baseline --device mps
.venv/bin/python run.py baseline-report --run experiments/baseline/results/baseline
```

Evaluation uses the already downloaded pinned checkpoints, sequentially. Do not run it concurrently with feature extraction on a memory-constrained Mac. The output report labels the protocol differences and shows descriptive trends without treating related checkpoints as independent trials.

Verify the first pilot pair against the upstream-style causal language model and torchvision vision extraction paths:

```sh
.venv/bin/python run.py baseline-verify --run experiments/baseline/results/pilot --device mps
```

