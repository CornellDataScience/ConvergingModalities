"""Prespecified held-out layer-selection sensitivity study; see RESEARCH_PLAN.md."""

from shared.paths import PROJECT_ROOT, project_path
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from experiments.baseline.evaluation.alignment import neighbors, overlap
from experiments.baseline.data.experiment import load_features
from shared.io import load_manifest, digest, write_json

OUT = project_path("experiments/heldout_layers/results")
SEEDS = [20261004, 20261005, 20261006]


def fit_threshold(raw, rows, q):
    return torch.quantile(raw[rows].abs().flatten(1), q, dim=1).mean()


def layer_matrix(left, right):
    rows = np.arange(len(left[0]))[:, None]
    result = np.empty((len(left), len(right)))
    for j, nb in enumerate(right):
        membership = np.zeros((len(nb), len(nb)), dtype=bool)
        membership[rows, nb] = True
        for i, na in enumerate(left):
            result[i, j] = membership[rows, na].mean()
    return result


def main():
    torch.set_num_threads(4)
    OUT.mkdir(parents=True, exist_ok=True)
    run = project_path("experiments/baseline/results/baseline")
    manifest = load_manifest(run)
    assert manifest["samples"] == 1024
    config = manifest["config"]
    assert config["clip_mode"] == "upstream_default"
    splits = []
    for seed in SEEDS:
        p = np.random.default_rng(seed).permutation(1024)
        splits.append(
            {
                "seed": seed,
                "selection": p[:512].tolist(),
                "evaluation": p[512:].tolist(),
            }
        )
    graphs, thresholds = {}, {}
    for spec in config["models"]:
        name = spec["id"]
        raw = load_features(run, spec, manifest)
        graphs[name], thresholds[name] = [], []
        for split in splits:
            threshold = fit_threshold(raw, split["selection"], config["clip_quantile"])
            thresholds[name].append(float(threshold))
            sides = {}
            for side in ["selection", "evaluation"]:
                x = F.normalize(raw[split[side]].clamp(-threshold, threshold), dim=-1)
                sides[side] = [
                    neighbors(x[:, layer], config["k"]) for layer in range(x.shape[1])
                ]
            graphs[name].append(sides)
        del raw, x
        print("Prepared " + name, flush=True)
    results = []
    for s, split in enumerate(splits):
        for language in [
            m["id"] for m in config["models"] if m["modality"] == "language"
        ]:
            for vision in [
                m["id"] for m in config["models"] if m["modality"] == "vision"
            ]:
                a, b = graphs[language][s], graphs[vision][s]
                train = layer_matrix(a["selection"], b["selection"])
                i, j = np.unravel_index(train.argmax(), train.shape)
                evaluation = layer_matrix(a["evaluation"], b["evaluation"])
                left, right = a["evaluation"][i], b["evaluation"][j]
                observed = overlap(left, right)
                assert np.isclose(observed, evaluation[i, j])
                rng = np.random.default_rng(split["seed"] + 10000)
                null = []
                for _ in range(1000):
                    p = rng.permutation(len(right))
                    null.append(overlap(left, np.argsort(p)[right[p]]))
                results.append(
                    dict(
                        seed=split["seed"],
                        language=language,
                        vision=vision,
                        layers=[int(i), int(j)],
                        selection_score=float(train[i, j]),
                        evaluation_score=observed,
                        evaluation_max=float(evaluation.max()),
                        null_mean=float(np.mean(null)),
                        null_q95=float(np.quantile(null, 0.95)),
                        excess=observed - float(np.mean(null)),
                        p=(1 + sum(v >= observed for v in null)) / 1001,
                        null=null,
                        selection_matrix=train.tolist(),
                        evaluation_matrix=evaluation.tolist(),
                    )
                )
                print(
                    f"{split['seed']} {language} {vision}: {observed:.6f}", flush=True
                )
    hashes = {
        p: hashlib.sha256(project_path(p).read_bytes()).hexdigest()
        for p in [
            "experiments/heldout_layers/evaluation/heldout_layers.py",
            "experiments/baseline/evaluation/alignment.py",
            "experiments/baseline/data/experiment.py",
            "experiments/heldout_layers/docs/RESEARCH_PLAN.md",
            "shared/io.py",
            "shared/data.py",
            "shared/training.py",
            "shared/transforms.py",
            "shared/paths.py",
        ]
    }
    write_json(
        OUT / "results.json",
        dict(
            manifest_hash=digest(manifest),
            source_hashes=hashes,
            splits=splits,
            thresholds=thresholds,
            k=config["k"],
            permutations=1000,
            results=results,
        ),
    )
    lines = [
        "# Held-out layer selection",
        "",
        "Exploratory sensitivity study: three overlapping 512/512 splits; k=10. Clipping fitted on selection rows only. Layers chosen on selection rows and frozen for evaluation.",
        "",
        "| Seed | BLOOMZ | DINOv2 | Layers | Held-out | Null mean | Excess | Evaluation max |",
        "|---|---|---|---|---:|---:|---:|---:|",
    ]
    for r in results:
        lines.append(
            f"| {r['seed']} | {r['language'].split('-')[-1]} | {'small' if 'small' in r['vision'] else 'base'} | {r['layers']} | {r['evaluation_score']:.6f} | {r['null_mean']:.6f} | {r['excess']:.6f} | {r['evaluation_max']:.6f} |"
        )
    lines += ["", "## Size ordering", ""]
    for seed in SEEDS:
        for size in ["small", "base"]:
            rr = [r for r in results if r["seed"] == seed and size in r["vision"]]
            increasing = all(
                a["evaluation_score"] < b["evaluation_score"]
                for a, b in zip(rr, rr[1:])
            )
            lines.append(
                f"- Seed {seed}, DINOv2 {size}: strictly increasing held-out alignment across 560M, 1.1B, 1.7B: {increasing}."
            )
    lines += [
        "",
        "## Limits",
        "",
        "Repeated splits are not independent replications. No confidence intervals or universal scaling claim. Fixed-layer permutation p-values in JSON are exploratory and unadjusted; they do not apply to evaluation maxima. Raw scores cannot be compared directly to N=1024. The dataset was already inspected in the baseline. This checks layer-selection generalization, not cross-dataset generalization.",
        "",
        "Reproduce: `.venv/bin/python run.py heldout-layers` from the project root. Protocol: RESEARCH_PLAN.md. Exact splits, thresholds, matrices, null draws, and source hashes: results.json.",
    ]
    (OUT / "REPORT.md").write_text("\n".join(lines) + "\n")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for ax, size in zip(axes, ["small", "base"]):
        for seed in SEEDS:
            rr = [r for r in results if r["seed"] == seed and size in r["vision"]]
            ax.plot(
                ["560M", "1.1B", "1.7B"],
                [r["excess"] for r in rr],
                marker="o",
                label=str(seed),
            )
        ax.axhline(0, color="gray", linewidth=0.8)
        ax.set_title("DINOv2 " + size)
        ax.set_xlabel("BLOOMZ size")
        ax.legend(title="Split seed", fontsize=8)
    axes[0].set_ylabel("Held-out k-NN overlap minus shuffled mean")
    fig.suptitle("Layers selected on 512 pairs; evaluated on 512 other pairs")
    fig.tight_layout()
    fig.savefig(OUT / "heldout.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
