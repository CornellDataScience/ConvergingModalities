"""Diagnose endpoint contributions; export a self-contained vector explorer."""

from shared.paths import PROJECT_ROOT, project_path
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from experiments.baseline.data.experiment import load_features
from shared.io import load_manifest, digest, write_json
from shared.transforms import fit_transform, apply_transform
from shared.training import fit_ridge
from experiments.difference_transfer.evaluation.difference_transfer import (
    difference_scores,
)

OUT = project_path("experiments/endpoint_audit/results")


def direction_details(predicted, target):
    similarities = F.normalize(predicted, dim=-1) @ F.normalize(target, dim=-1).T
    # Strict-greater rank avoids arbitrary tie order; real cached vectors have no ties.
    ranks = 1 + (similarities > similarities.diag()[:, None]).sum(1)
    return similarities, ranks


def project_directions(predicted, target):
    combined = torch.cat([predicted, target])
    _, _, vh = torch.linalg.svd(combined, full_matrices=False)
    basis = vh[:2].T
    projected = combined @ basis
    retention = float(projected.square().sum() / combined.square().sum())
    return projected, retention


def main():
    torch.set_num_threads(4)
    OUT.mkdir(parents=True, exist_ok=True)
    old = json.loads(
        project_path("experiments/heldout_layers/results/results.json").read_text()
    )
    pilot = json.loads(
        project_path("experiments/difference_transfer/results/results.json").read_text()
    )
    manifest = load_manifest(project_path("experiments/baseline/results/baseline"))
    assert digest(manifest) == old["manifest_hash"] == pilot["manifest_hash"]
    names = [pilot["language"], pilot["vision"]]
    prepared = {}
    for modality, name in enumerate(names):
        spec = next(m for m in manifest["config"]["models"] if m["id"] == name)
        raw = load_features(
            project_path("experiments/baseline/results/baseline"), spec, manifest
        )
        for s, split in enumerate(old["splits"]):
            row = next(
                r
                for r in old["results"]
                if r["seed"] == split["seed"]
                and r["language"] == names[0]
                and r["vision"] == names[1]
            )
            layer = row["layers"][modality]
            threshold = old["thresholds"][name][s]
            train = F.normalize(
                raw[split["selection"], layer].clamp(-threshold, threshold), dim=-1
            )
            ev = F.normalize(
                raw[split["evaluation"], layer].clamp(-threshold, threshold), dim=-1
            )
            fitted = fit_transform(train, "pca128")
            prepared[s, modality] = (
                apply_transform(train, fitted),
                apply_transform(ev, fitted),
            )
        del raw
        print("Prepared " + name, flush=True)
    rows = []
    for s, split in enumerate(old["splits"]):
        (x, xe), (y, ye) = prepared[s, 0], prepared[s, 1]
        w = fit_ridge(x, y)
        endpoints = (xe - x.mean(0)) @ w + y.mean(0)
        predicted = (xe[1::2] - xe[::2]) @ w
        target = ye[1::2] - ye[::2]
        error = float((predicted - (endpoints[1::2] - endpoints[::2])).abs().max())
        torch.testing.assert_close(
            predicted, endpoints[1::2] - endpoints[::2], atol=1e-6, rtol=1e-5
        )
        controls = {
            "full_difference": predicted,
            "target_endpoint_only": endpoints[1::2],
            "negative_source_endpoint_only": -endpoints[::2],
        }
        metrics = {name: difference_scores(p, target) for name, p in controls.items()}
        metrics["endpoint_retrieval_512"] = difference_scores(endpoints, ye)
        rows.append(dict(seed=split["seed"], metrics=metrics, algebra_max_error=error))
        if s == 0:
            saved = np.load(
                project_path("experiments/difference_transfer/results/directions.npz")
            )
            np.testing.assert_allclose(
                predicted.numpy(), saved["predicted"], atol=1e-6, rtol=1e-5
            )
            np.testing.assert_allclose(
                target.numpy(), saved["target"], atol=1e-6, rtol=1e-5
            )
            projected, retention = project_directions(predicted, target)
            similarities, ranks = direction_details(predicted, target)
            top = similarities.topk(5, dim=1).indices.tolist()
            payload = dict(
                predicted=projected[:256].tolist(),
                target=projected[256:].tolist(),
                retention=retention,
                cosine=similarities.diag().tolist(),
                rank=ranks.tolist(),
                top=top,
                pairs=np.array(split["evaluation"]).reshape(-1, 2).tolist(),
                metrics=metrics,
                seed=split["seed"],
            )
            template = project_path(
                "experiments/endpoint_audit/reporting/vector_explorer.html"
            ).read_text()
            (OUT / "explorer.html").write_text(
                template.replace("__VECTOR_DATA__", json.dumps(payload))
            )
            write_json(OUT / "display_data.json", payload)
    sources = [
        "experiments/endpoint_audit/evaluation/endpoint_audit.py",
        "experiments/endpoint_audit/docs/ENDPOINT_AUDIT_PROTOCOL.md",
        "experiments/endpoint_audit/reporting/vector_explorer.html",
        "experiments/difference_transfer/evaluation/difference_transfer.py",
        "experiments/robustness/evaluation/robustness.py",
        "experiments/heldout_layers/results/results.json",
        "shared/io.py",
        "shared/data.py",
        "shared/training.py",
        "shared/transforms.py",
        "shared/paths.py",
    ]
    write_json(
        OUT / "results.json",
        dict(
            manifest_hash=digest(manifest),
            splits=rows,
            source_hashes={
                p: hashlib.sha256(project_path(p).read_bytes()).hexdigest()
                for p in sources
            },
        ),
    )
    lines = [
        "# Endpoint contribution audit",
        "",
        "Exploratory; three overlapping splits, fixed pipeline, no significance claims.",
        "",
        "| Split | Predictor | Cosine | Top-1 | Top-5 |",
        "|---|---|---:|---:|---:|",
    ]
    for row in rows:
        for name, metric in row["metrics"].items():
            lines.append(
                f"| {row['seed']} | {name} | {metric['mean_cosine']:.4f} | {metric['top1']:.2%} | {metric['top5']:.2%} |"
            )
    lines += [
        "",
        "Difference controls use 256 identical candidates; endpoint retrieval uses 512 endpoints and is not directly comparable. The endpoint-only ablations use affine predictions in the original PCA coordinate origin; their values depend on that origin. The full difference cancels the intercept.",
        "",
        "The first split reproduces the saved pilot vectors within 1e-6 absolute / 1e-5 relative tolerance. The algebraic identity holds on every split. Endpoint-only performance diagnoses partial information; it does not isolate a semantic relation.",
        "",
        f"Display: joint uncentered SVD of predicted and target directions retains {retention:.2%} of squared vector norm in 2D. All displayed cosine and retrieval ranks use the full 128 dimensions. The displayed mean is an average of arbitrary pairs, not an estimated named semantic relation.",
        "",
        "Reproduce from sz684: `.venv/bin/python run.py endpoint-audit`. Open `results/endpoint_audit/explorer.html` in a browser.",
    ]
    (OUT / "REPORT.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
