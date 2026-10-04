"""Compare saved pilot features with upstream-style extraction on the first pair."""
from experiment import load_manifest, load_features, pooled, write_json, device_for
from pathlib import Path
import argparse
import gc
import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=Path("results/pilot"))
    parser.add_argument("--device", default="mps")
    args = parser.parse_args()
    device = device_for(args.device)
    manifest = load_manifest(args.run)
    record = manifest["records"][0]
    checks = []
    for spec in [manifest["config"]["models"][0], manifest["config"]["models"][3]]:
        expected = load_features(args.run, spec, manifest)[:1]
        if spec["modality"] == "language":
            from transformers import AutoTokenizer, AutoModelForCausalLM
            tokenizer = AutoTokenizer.from_pretrained(spec["id"], revision=spec["revision"], local_files_only=True)
            tokenizer.padding_side = "left"
            model = AutoModelForCausalLM.from_pretrained(spec["id"], revision=spec["revision"],
                        torch_dtype=torch.float32, local_files_only=True).eval().to(device)
            tokens = tokenizer(record["text"], return_tensors="pt")
            tokens = {k: v.to(device) for k, v in tokens.items()}
            with torch.inference_mode():
                output = model(**tokens, output_hidden_states=True, use_cache=False)
                actual = torch.stack([pooled(h, tokens["attention_mask"]).cpu() for h in output.hidden_states], dim=1)
            del output, tokens
        else:
            import timm
            from PIL import Image
            from timm.data import resolve_data_config, create_transform
            from torchvision.models.feature_extraction import create_feature_extractor
            from safetensors.torch import load_file
            from huggingface_hub import hf_hub_download
            model = timm.create_model(spec["id"].split("/", 1)[1], pretrained=False)
            model.load_state_dict(load_file(hf_hub_download(spec["id"], "model.safetensors", revision=spec["revision"], local_files_only=True)))
            transform = create_transform(**resolve_data_config(model.pretrained_cfg, model=model))
            nodes = [f"blocks.{i}.add_1" for i in range(len(model.blocks))]
            model = create_feature_extractor(model.eval(), return_nodes=nodes).to(device)
            with Image.open(args.run / record["image"]) as image:
                batch = transform(image.convert("RGB")).unsqueeze(0).to(device)
            with torch.inference_mode():
                actual = torch.stack([v[:, 0, :].cpu() for v in model(batch).values()], dim=1)
        error = float((actual - expected).abs().max())
        passed = torch.allclose(actual, expected, atol=1e-5, rtol=1e-5)
        checks.append({"model": spec["id"], "max_absolute_error": error, "passed": passed,
                       "scope": "first pair, same pinned weights/device/dtype; upstream-style extraction"})
        print(checks[-1], flush=True)
        del model
        gc.collect()
        if device == "mps":
            torch.mps.empty_cache()
    write_json(args.run / "extraction_verification.json", checks)
    if not all(c["passed"] for c in checks):
        raise SystemExit("Feature verification failed")


if __name__ == "__main__":
    main()
