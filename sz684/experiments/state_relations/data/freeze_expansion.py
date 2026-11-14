"""Freeze the visually reviewed v2 integration roster."""

import hashlib
import io
import json
from collections import Counter
from pathlib import Path
from zipfile import ZipFile

import numpy as np
from PIL import Image, ImageOps

from shared.paths import project_path

RESULTS = project_path("experiments/state_relations/results")
PILOT = RESULTS / "mit_states_star"
HISTORICAL = project_path("experiments/state_transfer/results")
# Positions on expansion_review.jpg, selected before looking at model scores.
ACCEPTED = {8, 9, 10, 15, 17, 18, 31, 33, 46, 49}
TRANSITIONS = {
    "dirty": ("cleaning", "clean"),
    "wet": ("drying", "dry"),
    "unripe": ("ripening", "ripe"),
    "raw": ("cooking", "cooked"),
}


def read_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def signature(payload):
    with Image.open(io.BytesIO(payload)) as image:
        gray = ImageOps.exif_transpose(image).convert("L")
        pixels = np.asarray(gray.resize((9, 8), Image.Resampling.LANCZOS))
    return (pixels[:, 1:] > pixels[:, :-1]).ravel()


def freeze():
    v1 = PILOT / "smoke_test_v1"
    development = read_rows(v1 / "development_candidates.jsonl")
    gallery = read_rows(v1 / "gallery.jsonl")
    families = {row["archive"]["noun"]: row["family"] for row in read_rows(PILOT / "review_queue.jsonl")}
    history = json.loads((HISTORICAL / "manifest.json").read_text())["records"]
    comparisons = np.array([signature((HISTORICAL / row["image"]).read_bytes()) for row in history])
    candidates = json.loads((PILOT / "expansion_candidates.json").read_text())
    flags = []
    added = []

    with ZipFile(PILOT / "source.zip") as archive:
        development_hashes = [signature(archive.read(row["local_member"])) for row in development]
        existing_hashes = [signature(archive.read(row["local_member"])) for row in gallery]
        comparisons = np.vstack([comparisons, development_hashes, existing_hashes])

        def add(row, payload):
            nonlocal comparisons
            features = signature(payload)
            distance = int(np.count_nonzero(comparisons != features, axis=1).min())
            if distance <= 6:
                flags.append({"image_id": row["image_id"], "minimum_hamming_distance": distance})
                return
            row["sha256"] = hashlib.sha256(payload).hexdigest()
            comparisons = np.vstack([comparisons, features])
            added.append(row)

        for index in sorted(ACCEPTED):
            candidate = candidates[index]
            noun = candidate["noun"]
            add({
                "image_id": hashlib.sha256(candidate["original"].encode()).hexdigest()[:16],
                "noun": noun,
                "state": candidate["state"],
                "family": families[noun],
                "local_member": candidate["member"],
                "review_sheet_position": index,
            }, archive.read(candidate["member"]))

        cooking = RESULTS / "changeit_cooking"
        for frame in read_rows(cooking / "reviewed_frames.jsonl"):
            payload = (cooking / frame["image"]).read_bytes()
            if hashlib.sha256(payload).hexdigest() != frame["sha256"]:
                raise ValueError(f"Frame checksum mismatch: {frame['image']}")
            add({
                "image_id": f"changeit-{frame['image']}",
                "noun": "pancake",
                "state": frame["state"],
                "family": "pancake",
                "source_video": frame["source_video"],
                "source": frame["source"],
                "image": f"../../changeit_cooking/{frame['image']}",
            }, payload)

    gallery.extend(added)
    if {row["family"] for row in gallery} & {row["family"] for row in development}:
        raise ValueError("A family appears in development and test")
    queries = []
    for row in gallery:
        if row["state"] not in TRANSITIONS:
            continue
        transition, target_state = TRANSITIONS[row["state"]]
        video = row.get("source_video")
        targets = [
            other["image_id"] for other in gallery
            if other["noun"] == row["noun"] and other["state"] == target_state
            and (video is None or other.get("source_video") != video)
        ]
        if targets:
            queries.append({
                "query_id": f"{transition}-{row['image_id']}",
                "source_image_id": row["image_id"],
                "noun": row["noun"],
                "family": row["family"],
                "transition": transition,
                "source_state": row["state"],
                "target_state": target_state,
                "target_image_ids": targets,
                **({"excluded_source_video": video} if video else {}),
            })

    out = PILOT / "smoke_test_v2"
    out.mkdir(exist_ok=True)
    hashes = {}
    for name, rows in {
        "gallery.jsonl": gallery, "queries.jsonl": queries,
        "development_candidates.jsonl": development, "duplicate_flags.jsonl": flags,
    }.items():
        payload = "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows).encode()
        path = out / name
        if path.exists() and path.read_bytes() != payload:
            raise ValueError(f"Frozen manifest changed: {name}")
        path.write_bytes(payload)
        hashes[name] = hashlib.sha256(payload).hexdigest()
    lock = {
        "version": out.name, "purpose": "pipeline integration",
        "source_archive": "../source.zip", "files": hashes,
        "gallery_count": len(gallery), "query_count": len(queries),
        "transition_counts": dict(Counter(row["transition"] for row in queries)),
        "review": "v1 assent plus first-pass visual review of added images",
    }
    (out / "lock.json").write_text(json.dumps(lock, indent=2, sort_keys=True) + "\n")
    print(lock["query_count"], lock["gallery_count"], lock["transition_counts"], "flags", len(flags))


if __name__ == "__main__":
    freeze()
