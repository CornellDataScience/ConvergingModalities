"""Mac-sized PRH replication. See README.md for protocol and limitations."""

from shared.io import digest, write_json, device_for, synchronize, load_manifest, pooled
from shared.paths import PROJECT_ROOT, project_path
import argparse
import gc
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import resource
import time

ROOT = PROJECT_ROOT
os.environ.setdefault("HF_HOME", str(ROOT / ".cache/huggingface"))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache/matplotlib"))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
import numpy as np
import torch
from experiments.baseline.evaluation.alignment import prepare, score_layers


def initialize(args):
    from huggingface_hub import HfApi
    from datasets import load_dataset

    config = json.loads(project_path(args.config).read_text())
    if config["dtype"] != "float32" or config["concatenate_layers"]:
        raise ValueError(
            "This baseline implements float32 and individual-layer comparisons only"
        )
    if (args.run / "manifest.json").exists():
        raise ValueError(
            "Run already initialized; use extract/align or a new directory"
        )
    api = HfApi()
    revision = api.dataset_info(
        config["dataset"], revision=config["dataset_revision"]
    ).sha
    data = load_dataset(config["dataset"], revision=revision, split="train")
    if not config["k"] < args.samples <= len(data):
        raise ValueError("Require k < samples <= dataset size")
    # Prefix for the pilot; full sample order is unchanged for the baseline.
    data = data.select(range(args.samples))
    records = []
    for i, row in enumerate(data):
        image = row["image"].convert("RGB")
        image_path = args.run / "data" / f"{i:04d}.png"
        image_path.parent.mkdir(parents=True, exist_ok=True)
        image.save(image_path)
        captions = row["text"]
        text = (
            captions[config["caption_index"]]
            if isinstance(captions, (list, tuple))
            else captions
        )
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"Invalid caption at row {i}")
        records.append(
            {
                "source_row": i,
                "text": text,
                "image": str(image_path.relative_to(args.run)),
                "image_sha256": hashlib.sha256(image_path.read_bytes()).hexdigest(),
            }
        )
    for model in config["models"]:
        model["revision"] = api.model_info(model["id"]).sha
    import requests

    response = requests.get(
        "https://api.github.com/repos/minyoungg/platonic-rep/commits/main", timeout=30
    )
    response.raise_for_status()
    upstream = response.json()["sha"]
    manifest = {
        "config": config,
        "dataset_revision": revision,
        "records": records,
        "sample_hash": digest(records),
        "samples": len(records),
        "upstream_commit": upstream,
        "platform": platform.platform(),
        "packages": {
            name: importlib.metadata.version(name)
            for name in [
                "torch",
                "torchvision",
                "transformers",
                "timm",
                "datasets",
                "numpy",
            ]
        },
        "scope": "pilot" if args.samples != 1024 else "reduced-model baseline",
        "protocol": "BLOOMZ; upstream default clipping; individual layers; float32; no quantization",
    }
    write_json(args.run / "manifest.json", manifest)
    print(f"Initialized {len(records)} pairs; upstream {upstream}", flush=True)


def extract(args):
    from PIL import Image
    from transformers import AutoModel, AutoTokenizer
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file
    import timm
    from timm.data import resolve_data_config, create_transform

    manifest = load_manifest(args.run)
    config = manifest["config"]
    torch.manual_seed(config["seed"])
    torch.set_num_threads(4)
    device = device_for(args.device)
    models = [m for m in config["models"] if args.model in ("all", m["id"])]
    if not models:
        raise ValueError(f"Unknown model: {args.model}")
    records = manifest["records"]
    for record in records:
        if (
            hashlib.sha256((args.run / record["image"]).read_bytes()).hexdigest()
            != record["image_sha256"]
        ):
            raise ValueError("Image contents differ from the pinned manifest")
    for spec in models:
        name = spec["id"].replace("/", "__")
        target = args.run / "features" / name
        target.mkdir(parents=True, exist_ok=True)
        identity = {
            "model": spec,
            "sample_hash": manifest["sample_hash"],
            "dtype": config["dtype"],
            "batch_size": args.batch_size,
            "device": device,
            "packages": manifest["packages"],
            "extractor_version": 1,
        }
        meta_path = target / "metadata.json"
        if meta_path.exists():
            if json.loads(meta_path.read_text())["identity"] != identity:
                raise ValueError(
                    "Extraction settings differ from cache; use another run"
                )
        else:
            write_json(meta_path, {"identity": identity, "complete": False})
        chunks = list(range(0, len(records), args.batch_size))
        if all((target / f"{start:04d}.npz").exists() for start in chunks):
            # Recover if interrupted after the final chunk but before the completion marker.
            metadata = json.loads(meta_path.read_text())
            if not metadata["complete"]:
                metadata.update(complete=True, recovered_completion=True)
                write_json(meta_path, metadata)
            load_features(args.run, spec, manifest)
            print(f"Cached: {spec['id']}", flush=True)
            continue
        started = time.monotonic()
        print(f"Loading {spec['id']} on {device} in float32", flush=True)
        hooks = []
        if spec["modality"] == "language":
            tokenizer = AutoTokenizer.from_pretrained(
                spec["id"], revision=spec["revision"]
            )
            if tokenizer.pad_token_id is None:
                tokenizer.pad_token = tokenizer.eos_token
            tokenizer.padding_side = "left"
            # AutoModel avoids computing huge vocabulary logits; hidden states are unchanged.
            model = (
                AutoModel.from_pretrained(
                    spec["id"], revision=spec["revision"], torch_dtype=torch.float32
                )
                .eval()
                .to(device)
            )
        else:
            model = timm.create_model(spec["id"].split("/", 1)[1], pretrained=False)
            checkpoint = hf_hub_download(
                spec["id"], "model.safetensors", revision=spec["revision"]
            )
            model.load_state_dict(load_file(checkpoint), strict=True)
            transform = create_transform(
                **resolve_data_config(model.pretrained_cfg, model=model)
            )
            model = model.eval().to(device)
            captured = []

            def hook(module, inputs, output):
                captured.append(output[:, 0, :].detach().float().cpu())

            hooks = [block.register_forward_hook(hook) for block in model.blocks]
        for start in chunks:
            output = target / f"{start:04d}.npz"
            if output.exists():
                continue
            batch = records[start : start + args.batch_size]
            synchronize(device)
            tick = time.monotonic()
            with torch.inference_mode():
                if spec["modality"] == "language":
                    tokens = tokenizer(
                        [r["text"] for r in batch], padding=True, return_tensors="pt"
                    )
                    tokens = {k: v.to(device) for k, v in tokens.items()}
                    result = model(**tokens, output_hidden_states=True, use_cache=False)
                    features = torch.stack(
                        [
                            pooled(h, tokens["attention_mask"]).cpu()
                            for h in result.hidden_states
                        ],
                        dim=1,
                    )
                    del result, tokens
                else:
                    captured.clear()
                    images = []
                    for r in batch:
                        with Image.open(args.run / r["image"]) as image:
                            images.append(transform(image.convert("RGB")))
                    model.forward_features(torch.stack(images).to(device))
                    features = torch.stack(captured, dim=1)
                if not torch.isfinite(features).all():
                    raise ValueError("Nonfinite model features")
                temp = output.with_suffix(".tmp")
                with temp.open("wb") as stream:
                    np.savez(
                        stream,
                        feats=features.numpy(),
                        rows=np.arange(start, start + len(batch)),
                    )
                temp.replace(output)
            synchronize(device)
            print(
                f"{name}: {start + len(batch)}/{len(records)} ({time.monotonic()-tick:.1f}s)",
                flush=True,
            )
        for handle in hooks:
            handle.remove()
        write_json(
            meta_path,
            {
                "identity": identity,
                "complete": True,
                "elapsed_seconds_this_invocation": time.monotonic() - started,
                "process_peak_rss_bytes": resource.getrusage(
                    resource.RUSAGE_SELF
                ).ru_maxrss,
                "feature_shape": list(features.shape[1:]),
            },
        )
        del model, features
        gc.collect()
        if device == "mps":
            torch.mps.empty_cache()


def load_features(run, spec, manifest):
    folder = run / "features" / spec["id"].replace("/", "__")
    metadata = json.loads((folder / "metadata.json").read_text())
    if (
        not metadata["complete"]
        or metadata["identity"]["sample_hash"] != manifest["sample_hash"]
    ):
        raise ValueError("Incomplete or mismatched feature cache")
    arrays, rows = [], []
    for file in sorted(folder.glob("*.npz")):
        with np.load(file) as chunk:
            arrays.append(chunk["feats"])
            rows.extend(chunk["rows"].tolist())
    if rows != list(range(manifest["samples"])):
        raise ValueError("Missing, duplicated, or reordered sample rows")
    return torch.from_numpy(np.concatenate(arrays))


def align(args):
    manifest = load_manifest(args.run)
    config = manifest["config"]
    torch.set_num_threads(4)
    available = [
        m
        for m in config["models"]
        if (
            args.run / "features" / m["id"].replace("/", "__") / "metadata.json"
        ).exists()
        and json.loads(
            (
                args.run / "features" / m["id"].replace("/", "__") / "metadata.json"
            ).read_text()
        )["complete"]
    ]
    results = []
    for language in [m for m in available if m["modality"] == "language"]:
        x = prepare(
            load_features(args.run, language, manifest),
            config["clip_quantile"],
            config["clip_mode"],
        )
        for vision in [m for m in available if m["modality"] == "vision"]:
            y = prepare(
                load_features(args.run, vision, manifest),
                config["clip_quantile"],
                config["clip_mode"],
            )
            result = score_layers(x, y, config["k"], args.permutations, config["seed"])
            result.update(language=language["id"], vision=vision["id"])
            results.append(result)
            print(
                f"{language['id']} / {vision['id']}: {result['best_score']:.6f}",
                flush=True,
            )
    if not results:
        raise ValueError("Need at least one complete language and vision model")
    write_json(
        args.run / "alignment.json",
        {
            "manifest_hash": digest(manifest),
            "results": results,
            "complete_roster": len(available) == len(config["models"]),
            "note": "No language-performance axis evaluated; this is not a Figure 3 replication verdict.",
        },
    )
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, max(3, len(results) * 0.6)))
    ax.barh(
        range(len(results)),
        [r["best_score"] for r in results],
        label="Observed layer maximum",
    )
    ax.scatter(
        [np.mean(r["permuted_maxima"]) if r["permuted_maxima"] else 0 for r in results],
        range(len(results)),
        color="black",
        marker="x",
        label="Mean shuffled layer maximum",
    )
    ax.set_yticks(
        range(len(results)),
        [
            r["language"].split("/")[-1]
            + " / "
            + r["vision"].split("/")[-1].split(".")[0]
            for r in results
        ],
        fontsize=8,
    )
    ax.set_xlabel("Mutual k-NN overlap")
    ax.set_title(f"{manifest['scope']} — N={manifest['samples']}, k={config['k']}")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(args.run / "alignment.png", dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["init", "extract", "align"])
    parser.add_argument(
        "--run",
        type=project_path,
        default=project_path("experiments/baseline/results/pilot"),
    )
    parser.add_argument("--config", default="experiments/baseline/configs/mac.json")
    parser.add_argument("--samples", type=int, default=32)
    parser.add_argument("--model", default="all")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--device", choices=["auto", "cpu", "mps"], default="auto")
    parser.add_argument("--permutations", type=int, default=100)
    args = parser.parse_args()
    if args.batch_size < 1 or args.permutations < 0:
        parser.error("batch-size must be positive and permutations nonnegative")
    {"init": initialize, "extract": extract, "align": align}[args.command](args)


if __name__ == "__main__":
    main()
