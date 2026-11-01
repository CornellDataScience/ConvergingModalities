"""Small, explicitly non-paper OpenWebText evaluation for the Mac baseline.

64 documents, first 1024 Unicode characters each, original OpenWebText-10k
archive. EOS prefixes each document; non-overlapping 256-target windows have
up to 256 preceding tokens of context. Every text token is scored once.
"""

from shared.paths import PROJECT_ROOT, project_path
from shared.io import load_manifest, write_json, digest, device_for, ROOT
from pathlib import Path
import argparse
import gc
import hashlib
import io
import json
import math
import re
import tarfile
import time
import torch
import torch.nn.functional as F

URL = "https://cdn-datasets.huggingface.co/nlp/datasets/openwebtext/openwebtext-10k.tar.xz"


def prepare_corpus(path, count=64, chars=1024):
    import requests

    archive = ROOT / ".cache/openwebtext-10k.tar.xz"
    archive.parent.mkdir(parents=True, exist_ok=True)
    if not archive.exists():
        response = requests.get(URL, timeout=120)
        response.raise_for_status()
        temp = archive.with_suffix(".tmp")
        temp.write_bytes(response.content)
        temp.replace(archive)
    docs = []
    with tarfile.open(archive) as outer:
        for member in sorted(outer.getmembers(), key=lambda m: m.name):
            if not member.isfile() or not member.name.endswith(".xz"):
                continue
            # Read nested archives in memory, never extract paths onto the filesystem.
            with tarfile.open(
                fileobj=io.BytesIO(outer.extractfile(member).read())
            ) as inner:
                for text_file in sorted(inner.getmembers(), key=lambda m: m.name):
                    if not text_file.isfile() or not text_file.name.endswith(".txt"):
                        continue
                    text = inner.extractfile(text_file).read().decode("utf-8")
                    text = re.sub(r"\n\n\n+", "\n\n", text).strip()[:chars]
                    if text:
                        docs.append(
                            {"source": member.name + "/" + text_file.name, "text": text}
                        )
                    if len(docs) == count:
                        break
            if len(docs) == count:
                break
    if len(docs) != count:
        raise ValueError("Insufficient documents in archive")
    corpus = {
        "url": URL,
        "archive_sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
        "docs": docs,
        "documents": count,
        "max_characters": chars,
        "protocol": __doc__,
        "target_window": 256,
        "previous_context": 256,
    }
    write_json(path, corpus)
    return corpus


def windows(token_count, target_window=256, previous_context=256):
    # Position zero is the conditioning EOS; positions 1..T are target text tokens.
    for start in range(1, token_count + 1, target_window):
        end = min(start + target_window, token_count + 1)
        begin = max(0, start - previous_context)
        yield begin, start, end


def main():
    from transformers import AutoTokenizer, AutoModelForCausalLM

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run",
        type=project_path,
        default=project_path("experiments/baseline/results/baseline"),
    )
    parser.add_argument("--device", default="mps", choices=["mps", "cpu", "auto"])
    args = parser.parse_args()
    device = device_for(args.device)
    torch.set_num_threads(4)
    manifest = load_manifest(args.run)
    corpus_path = args.run / "language_corpus.json"
    corpus = (
        json.loads(corpus_path.read_text())
        if corpus_path.exists()
        else prepare_corpus(corpus_path)
    )
    outputs = []
    for spec in manifest["config"]["models"]:
        if spec["modality"] != "language":
            continue
        target = args.run / "language" / (spec["id"].replace("/", "__") + ".json")
        identity = {
            "model": spec,
            "corpus_hash": digest(corpus),
            "device": device,
            "dtype": "float32",
            "version": 1,
        }
        if target.exists():
            result = json.loads(target.read_text())
            if result["identity"] != identity:
                raise ValueError("Evaluation cache mismatch")
            outputs.append(result)
            continue
        tokenizer = AutoTokenizer.from_pretrained(
            spec["id"], revision=spec["revision"], local_files_only=True
        )
        model = (
            AutoModelForCausalLM.from_pretrained(
                spec["id"],
                revision=spec["revision"],
                torch_dtype=torch.float32,
                local_files_only=True,
            )
            .eval()
            .to(device)
        )
        if tokenizer.eos_token_id is None:
            raise ValueError("Protocol requires EOS token")
        totals = []
        tick = time.monotonic()
        for index, doc in enumerate(corpus["docs"]):
            text = doc["text"]
            tokens = tokenizer(text, add_special_tokens=False)["input_ids"]
            ids = torch.tensor([[tokenizer.eos_token_id] + tokens], device=device)
            nats = 0.0
            with torch.inference_mode():
                for begin, start, end in windows(
                    len(tokens), corpus["target_window"], corpus["previous_context"]
                ):
                    output = model(input_ids=ids[:, begin:end], use_cache=False)
                    logits = output.logits[
                        :, start - begin - 1 : end - begin - 1, :
                    ].float()
                    labels = ids[:, start:end]
                    nats += F.cross_entropy(
                        logits.reshape(-1, logits.shape[-1]),
                        labels.reshape(-1),
                        reduction="sum",
                    ).item()
                    del output, logits, labels
            totals.append(
                {
                    "source": doc["source"],
                    "nats": nats,
                    "bytes": len(text.encode("utf-8")),
                    "tokens": len(tokens),
                }
            )
            if (index + 1) % 8 == 0:
                print(
                    f"{spec['id']}: language evaluation {index+1}/{len(corpus['docs'])}",
                    flush=True,
                )
        bpb = (
            sum(t["nats"] for t in totals)
            / math.log(2)
            / sum(t["bytes"] for t in totals)
        )
        result = {
            "identity": identity,
            "bits_per_byte": bpb,
            "one_minus_bpb": 1 - bpb,
            "documents": totals,
            "elapsed_seconds": time.monotonic() - tick,
            "scope": "Reduced OpenWebText protocol; not the paper's 4M-token evaluation",
        }
        write_json(target, result)
        outputs.append(result)
        print(f"{spec['id']}: {bpb:.6f} bits/byte", flush=True)
        del model
        gc.collect()
        if device == "mps":
            torch.mps.empty_cache()
    write_json(args.run / "language_scores.json", outputs)


if __name__ == "__main__":
    main()
