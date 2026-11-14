"""Screen the MIT-States* pilot and write its provisional review manifests."""

import hashlib
import io
import json
from collections import Counter
from pathlib import Path
from zipfile import ZipFile

import numpy as np
from PIL import Image, ImageOps

root = Path(__file__).resolve().parents[2]
pilot = root / "state_relations/results/mit_states_star"
old = root / "state_transfer/results"
def read_rows(name):
    return [json.loads(line) for line in name.read_text().splitlines()]
def write_rows(name, rows):
    (pilot / name).write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))

def signature(payload):
    with Image.open(io.BytesIO(payload)) as image:
        pixels = np.asarray(ImageOps.exif_transpose(image).convert("L").resize((9, 8), Image.Resampling.LANCZOS))
        return pixels[:, 1:] > pixels[:, :-1]

def audit():
    queue = read_rows(pilot / "review_queue.jsonl")
    review = {r["image_id"]: r for r in read_rows(pilot / "first_pass.jsonl")}
    history = json.loads((old / "manifest.json").read_text())["records"]
    items = [(f"historical:{r['image']}", "historical", (old / r["image"]).read_bytes()) for r in history]
    with ZipFile(pilot / "source.zip") as archive:
        items.extend((r["image_id"], r["split"], archive.read(r["local_member"])) for r in queue)

    hashes = [hashlib.sha256(payload).hexdigest() for _, _, payload in items]
    signatures = np.array([signature(payload) for _, _, payload in items])
    flags = []
    flag_ids = set()
    for i, (image_id, split, _) in enumerate(items):
        if split != "test":
            continue
        differences = np.count_nonzero(signatures != signatures[i], axis=(1, 2))
        for j in np.flatnonzero(differences <= 6):
            other_id, other_split, _ = items[j]
            if i == j or (other_split == "test" and j < i):
                continue
            flags.append(dict(image_id=image_id, other_id=other_id, other_split=other_split,
                              distance=int(differences[j]), exact=hashes[i] == hashes[j]))
            flag_ids.update([image_id, other_id])
    write_rows("duplicate_screen.jsonl", flags)

    corrections = {
        "1fd1dc8d006bd903": "The raw-pasta thumbnail shows a dressed dish. The bowl of penne is in the cooked row.",
        "cdcb7ce8cfffefed": "The bowl of plain penne could contain cooked or dried pasta.",
        "76037ac6f8ebe002": "The pool looks clear in this view and visible evidence of dirt is weak.",
    }
    write_rows("review_corrections.jsonl", [
        dict(image_id=image_id, reviewer="codex", decision="uncertain", reason=reason)
        for image_id, reason in corrections.items()
    ])
    test = [r for r in queue if r["split"] == "test"]
    sha = {item[0]: digest for item, digest in zip(items, hashes)}
    accepted = [r for r in test if review[r["image_id"]]["decision"] == "accept"
                and r["image_id"] not in corrections and r["image_id"] not in flag_ids]
    queries = []
    for r in accepted:
        for relation, role in r["roles"].items():
            if role != "source":
                continue
            targets = [x["image_id"] for x in accepted
                       if x["archive"]["noun"] == r["archive"]["noun"] and x["roles"].get(relation) == "target"]
            if targets:
                queries.append(dict(query_id=f"{relation}-{r['image_id']}", source_image_id=r["image_id"],
                                    family=r["family"], noun=r["archive"]["noun"], transition=relation,
                                    target_image_ids=targets))
    write_rows("provisional_queries.jsonl", queries)
    write_rows("provisional_gallery.jsonl", [
        dict(image_id=r["image_id"], sha256=sha[r["image_id"]], local_member=r["local_member"],
             noun=r["archive"]["noun"], state=r["archive"]["state"], family=r["family"])
        for r in accepted
    ])
    write_rows("independent_review_queue.jsonl", [
        dict(image_id=r["image_id"], noun=r["archive"]["noun"], state=r["archive"]["state"],
             local_member=r["local_member"], sheet=f"sheet_{index // 30:02d}.jpg",
             position=index % 30 + 1)
        for index, r in enumerate(test)
    ])
    print(len(items), len(flags), len(accepted), len(queries), Counter(q["transition"] for q in queries))


if __name__ == "__main__":
    audit()
