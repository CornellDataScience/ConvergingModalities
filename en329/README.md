# Blind matching across modalities (audio ↔ vision)

Tests whether two independently trained models, one audio and one vision, arrange concepts in similar geometric structures, using no paired supervision between them. This follows the Platonic Representation Hypothesis (Huh et al. 2024) and the blind-matching approach of Schnaus et al. 2025.

> **Status:** the code has not been run end to end. It passes a syntax check, but POT could not be installed in the environment where it was written. Run the synthetic test (below) first.

## The idea in one paragraph

Raw embedding coordinates are arbitrary: dimension 17 of CLIP means nothing to DINOv2 or CLAP, so you can't compare spaces directly. What *is* comparable is the **internal pattern of distances**. If "dog" is close to "wolf" and far from "violin" in both spaces, that pattern is a fingerprint. **Gromov-Wasserstein (GW)** optimal transport finds the correspondence between two point sets that best preserves those internal distance patterns, without ever comparing coordinates across spaces. If GW can recover the correct concept-to-concept matching with no labels, the two spaces share structure.

## Pipeline

```
paired clips (frame + audio, same clip)
        │
        ▼
extract_embeddings.py ──► vision.npz   audio.npz     (X: N×D, labels: N)
        │
        ▼                                  blind_match.py
 1. aggregate        class centroids (--level class) or N samples (--level sample)
 2. preprocess       center → standardize per dim → PCA to common dim (→ optional whiten)
 3. distances        cosine distance matrix per model, scaled to [0, 1]
 4. GW matching      ot.gromov.gromov_wasserstein, several restarts, keep lowest GW cost
 5. evaluate         compare the plan to ground-truth correspondence (row i ↔ column i)
```

### Why each step exists

| Step | Reason |
|---|---|
| Standardize per dimension | Transformer embeddings are anisotropic: a few "rogue" dimensions dominate variance. Dividing by each dimension's std deflates them so semantic dimensions count. |
| PCA to a common dim | Bigger models have wider embeddings. Projecting everything to the same dimension keeps larger models from getting a dimensionality advantage. |
| Optional whitening (`--whiten`) | Stronger fix for anisotropy: equalizes variance across the retained components. |
| Distance matrices | Make the representation coordinate-free, which is what GW needs. |
| Multiple restarts | GW is non-convex, so different initializations can land in different local minima. The spread across restarts is reported. |
| Class-level vs sample-level | Class centroids give a clean, cheap test (~300 points). Sample-level is harder and tests whether structure survives at finer grain, but GW cost grows as O(N²). |

## Files

- `extract_embeddings.py`: pulls embeddings from HuggingFace models for a manifest of paired clips.
- `blind_match.py`: preprocessing, GW matching, metrics.
- `requirements.txt`: dependencies.

## Setup

```bash
pip install -r requirements.txt
```

Sanity check on synthetic data (two "models" viewing the same latent structure through different maps):

```bash
python blind_match.py --synthetic --restarts 5
```

Top-1 here should be far above chance (1/N = 0.5% for N=200). If it isn't, something is wrong with the GW setup, so don't proceed to real data.

## Running on real data

**1. Build a manifest CSV** with one row per clip:

```
clip_id,label,frame_path,audio_path
abc123,dog barking,frames/abc123.jpg,audio/abc123.wav
```

Use natural audio-visual clips (e.g. VGGSound or AudioSet): one frame plus its audio track. Labels are used **only as ground truth**, never as input to the models or matcher. Restrict to sound-producing objects and animals; classes like "speech" or "music" have no distinctive visual referent.

**2. Extract embeddings** (one run per model):

```bash
python extract_embeddings.py --manifest m.csv --model dinov2-base --out dino_b.npz
python extract_embeddings.py --manifest m.csv --model clip-b16    --out clip_b16.npz
python extract_embeddings.py --manifest m.csv --model clap        --out clap.npz
python extract_embeddings.py --manifest m.csv --model ast         --out ast.npz
```

Available models:

- Vision: `dinov2-{small,base,large}`, `clip-{b32,b16,l14}`, `mae-base`
- Audio: `clap`, `ast`, `wav2vec2-{base,large}`, `hubert-{base,large}`

BEATs and AudioMAE aren't in `transformers`. Extract with their official repos and save in the same format: `np.savez(out, X=embeddings, labels=labels)`.

**3. Match:**

```bash
# class level: match class centroids
python blind_match.py --a dino_b.npz --b clap.npz --level class

# sample level: match 1000 individual clips
python blind_match.py --a dino_b.npz --b clap.npz --level sample --n 1000 --out result.json
```

Both files must contain the same samples in the same order (the script asserts this).

Options:

| Flag | Default | Meaning |
|---|---|---|
| `--dim` | 64 | Common PCA dimension |
| `--whiten` | off | Whiten after PCA |
| `--restarts` | 10 | GW initializations |
| `--eps` | None | If set, uses entropic GW with this epsilon |
| `--k` | 10 | k for mutual k-NN |
| `--n` | 1000 | Samples at sample level |
| `--seed` | 0 | Subsampling/restart seed |

## Reading the output

| Field | Meaning |
|---|---|
| `top1`, `top5`, `top10` | Fraction of items whose true partner is in the top-k of its row of the GW plan |
| `hungarian_top1` | Accuracy after forcing a one-to-one assignment (Hungarian algorithm on the plan) |
| `chance_top1` | 1/N, the accuracy of random guessing |
| `mutual_knn` | Average overlap of k-nearest-neighbor sets between the two spaces (0 to 1) |
| `gw_dist_best` | Lowest GW cost across restarts |
| `gw_dist_restart_std` | Spread across restarts; large values mean unstable matching |
| `null_top1_mean/std` | GW top-1 when one space is shuffled |

GW returns a soft probability matrix (the "plan"), not a hard matching. Top-k metrics read ranks directly from the plan; Hungarian converts it to a discrete one-to-one assignment.

## Experiment design

Run these conditions so the result is interpretable:

1. **Controls first.** Vision↔vision (DINOv2 vs CLIP) is an upper-bound sanity check. If GW can't match two vision models, audio-vision results mean nothing.
2. **Language-free pair.** DINOv2 or MAE with a self-supervised audio model (BEATs, AudioMAE). This is the cleanest test of convergence without language.
3. **Language-supervised pair.** CLIP with CLAP. Both embedding spaces are shaped by text supervision, so any convergence here could be inherited from language rather than from shared structure in the world.
4. **Mixed pairs.** DINOv2 with CLAP, and CLIP with a self-supervised audio model.
5. **Scale sweep.** Repeat across model sizes (e.g. CLIP B/32, B/16, L/14) and plot accuracy against size.
6. **Sample-size sweep.** Vary `--n` (1k, 5k, 10k, ...). Prior work found that apparent alignment from mutual k-NN shrank substantially at large sample sizes, so conclusions drawn from small N may not hold.

## Known limitations

- **Pooling:** non-CLIP/CLAP models use mean-pooled last-layer hidden states. The best layer may differ, which is a confound worth sweeping.
- **Speech-oriented audio models:** wav2vec2 and HuBERT are trained on speech and may represent environmental sound poorly. BEATs/AudioMAE are better suited.
- **The shuffle baseline** (`null_top1_*`) shows what GW achieves when the structure is destroyed. For pure chance level, use 1/N.
- **Scaling:** sample-level GW needs O(N²) memory and time. For large N, use minibatching or stay at class level.
- **GW is non-convex.** Always inspect `gw_dist_restart_std` before trusting a single run.
- **Entropic GW** (`--eps`) is more stable but blurs the plan. Start unregularized and add it if restarts disagree.

## References

- Huh et al. 2024, *The Platonic Representation Hypothesis*
- Schnaus et al. 2025, *It's a (Blind) Match!*
- POT: Python Optimal Transport (`ot.gromov`)