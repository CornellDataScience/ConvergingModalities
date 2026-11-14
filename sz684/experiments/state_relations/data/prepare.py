"""Build a deterministic review queue from the cached MIT-States catalog."""

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from random import Random
from zipfile import ZipFile

from experiments.state_transfer.data.mit_states_data import catalog
from shared.paths import project_path

ROOT = Path(__file__).resolve().parents[1]
HISTORICAL = project_path("experiments/state_transfer/results")
SPLITS = ("train", "validation", "test")


def prepare(output, archive_path=None, per_composition=6):
    old = json.loads((HISTORICAL / "manifest.json").read_text())
    family_config = json.loads((ROOT / "configs/families.json").read_text())
    families = {noun: group for group, nouns in family_config.items() for noun in nouns}
    if len(families) != sum(map(len, family_config.values())):
        raise ValueError("A noun appears in multiple families")
    group = {noun: families.get(noun, noun) for nouns in old["support"].values() for noun in nouns}
    source = catalog()
    local_members = None
    if archive_path:
        with ZipFile(archive_path) as archive:
            local_members = {
                (item.CRC, item.file_size): item.filename
                for item in archive.infolist() if not item.is_dir()
            }
    seen = {(r["archive"]["crc32"], r["archive"]["size"]) for r in old["records"]}
    inspected = {r["archive"]["path"] for r in old["records"]}
    available = defaultdict(list)
    for entry in sorted(source["files"], key=lambda item: item["path"]):
        key = entry["crc32"], entry["size"]
        if local_members is not None and key not in local_members:
            continue
        if key not in seen and entry["path"] not in inspected:
            available[entry["state"], entry["noun"]].append(entry)
            seen.add(key)

    relations = old["config"]["relations"]
    support = {
        r["name"]: [
            noun for noun in old["support"][r["name"]]
            if min(len(available[r[state], noun]) for state in ("source", "target")) >= per_composition
        ]
        for r in relations
    }
    groups = sorted({group[n] for nouns in support.values() for n in nouns})
    rng = Random(old["config"]["seed"])
    for _ in range(10_000):
        rng.shuffle(groups)
        holdout_size = max(1, len(groups) // (3 if archive_path else 4))
        assignments = {
            name: "validation" if index < holdout_size
            else "test" if index < 2 * holdout_size else "train"
            for index, name in enumerate(groups)
        }
        if all({assignments[group[n]] for n in nouns} == set(SPLITS) for nouns in support.values()):
            break
    else:
        raise ValueError("Could not cover all relations in every split")

    roles = defaultdict(dict)
    for relation in relations:
        name = relation["name"]
        for noun in support[name]:
            for role in ("source", "target"):
                roles[relation[role], noun][name] = role
    records = []
    for (state, noun), uses in sorted(roles.items()):
        for entry in rng.sample(available[state, noun], per_composition):
            records.append({
                "image_id": hashlib.sha256(entry["path"].encode()).hexdigest()[:16],
                "archive": entry,
                "local_member": local_members[(entry["crc32"], entry["size"])] if local_members is not None else entry["path"],
                "family": group[noun],
                "split": assignments[group[noun]],
                "roles": uses,
            })
    records.sort(key=lambda row: (("test", "validation", "train").index(row["split"]), row["archive"]["noun"], row["archive"]["state"]))
    output.mkdir(parents=True, exist_ok=True)
    with (output / "review_queue.jsonl").open("w") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")


def contact_sheets(output, archive_path):
    """Read only selected members from a local copy of the original archive."""
    from PIL import Image, ImageDraw, ImageOps

    records = [json.loads(line) for line in (output / "review_queue.jsonl").read_text().splitlines()]
    images = output / "review_images"
    images.mkdir(exist_ok=True)
    with ZipFile(archive_path) as archive:
        for start in range(0, len(records), 30):
            sheet = Image.new("RGB", (1000, 6 * 125), "white")
            draw = ImageDraw.Draw(sheet)
            for index, row in enumerate(records[start:start + 30]):
                entry = row["archive"]
                info = archive.getinfo(row["local_member"])
                if (info.CRC, info.file_size) != (entry["crc32"], entry["size"]):
                    raise ValueError(f"Archive member changed: {entry['path']}")
                with archive.open(info) as handle, Image.open(handle) as image:
                    thumb = ImageOps.contain(image.convert("RGB"), (192, 96))
                    x, y = (index % 5) * 200, (index // 5) * 125
                    sheet.paste(thumb, (x, y))
                    draw.text((x, y + 96), f"{entry['state']} {entry['noun']}", fill="black")
                    draw.text((x, y + 109), row["image_id"], fill="black")
            sheet.save(images / f"sheet_{start // 30:02d}.jpg")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "results/week1")
    parser.add_argument("--archive", type=Path, help="Original MIT-States release ZIP")
    parser.add_argument("--per-composition", type=int, default=6)
    args = parser.parse_args()
    if args.per_composition < 1:
        parser.error("--per-composition must be positive")
    prepare(args.output, args.archive, args.per_composition)
    if args.archive:
        contact_sheets(args.output, args.archive)
