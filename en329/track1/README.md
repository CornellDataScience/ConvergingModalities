# Track 1 / M2: bridge controls, held-out retrieval, alignment vs retrieval gain

Answers: *when a text state direction ("cooked X" − "raw X") is mapped into image space with the existing bridge and added to a held-out image, does it retrieve the same object in the new state, beyond what controls achieve? And does M1's alignment score predict that gain?*

- [PROTOCOL.md](PROTOCOL.md): frozen design (conditions, λ selection, metrics, inference, decision rule, claim boundaries)
- [bridge_retrieval.py](bridge_retrieval.py): the whole analysis (numpy + scipy only)
- [tests/](tests/): unit tests, a leakage test, and end-to-end signal vs. planted-null tests

## Run

```bash
python -m pytest -q tests                                                    # ~5 s
python bridge_retrieval.py --synthetic --out results/synthetic               # controls on a world with signal
python bridge_retrieval.py --synthetic --null-text --out results/null        # text carries nothing: must say NOT supported
python bridge_retrieval.py --bridge bridge.npz --states states.npz --split split.json --out results/mit_states
```

Each run writes `results.json` (draft schema `convmod.track1.m2/draft-1`, with per-query rows for the visualizer and error analysis) and `REPORT.md`.

## Inputs still needed

| File | What | From |
|---|---|---|
| `bridge.npz` | `text`, `image`: the 512 WIT selection pairs (split 20261004), BLOOMZ-1.7B / DINOv2-base at sz684's selected layers, clipped. This is the same data `sz684/difference_transfer.py` fits its ridge map on. | sz684's local `results/` cache (not in git) |
| `states.npz` | MIT-States image embeddings + prompt embeddings ("cooked chicken") with object/state labels, extracted with the **same** layers and clipping | M1 extraction on M3's verified data |
| `split.json` | family-level `train`/`val`/`test` object lists + the five transitions | M3's frozen test set |

The bridge's PCA-128 and ridge are refit inside the script from `bridge.npz`, so the shuffled-bridge control can refit them too. Nothing is fit on MIT-States test objects.

## What the synthetic checks showed (and what changed because of them)

1. **Wrong-direction controls must reuse the bridge's step.** When tuned on their own, `reversed_bridge` and `other_transition` picked the smallest step and acted like no change, so beating them proved nothing.
2. **Alignment predicts retrieval even when language carries nothing** (r = +0.57, p = 0.0002 with random text). The alignment→gain question is now answered with a calibration curve: random directions at a fixed cosine to the visual direction. The output says whether the bridge gains more than its alignment alone predicts.
3. **The query bootstrap can't see that the bridge is one random draw.** A positive verdict needs the adjusted CIs **and** a shuffled-bridge tail ≤ 0.05. The rule passed 9/9 seeds with signal and 9/9 null seeds failed.
