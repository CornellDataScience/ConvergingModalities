"""Generate figures, a companion report and an archival draft from saved state scores."""

from shared.paths import PROJECT_ROOT, project_path

import json
import os
import re
from pathlib import Path
import numpy as np

os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".cache/matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(PROJECT_ROOT / ".cache"))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = PROJECT_ROOT
RUN = project_path("experiments/state_transfer/results")
OUT = project_path("output/figures")


def build():
    r = json.loads((RUN / "results.json").read_text())
    manifest = json.loads((RUN / "manifest.json").read_text())
    OUT.mkdir(parents=True, exist_ok=True)
    inf = r["inference"]
    conditions = r["conditions"]
    rels = r["relations"]
    labels = ["No shift", "Random shifts", "Reverse shift", "Wrong relations"]
    keys = [
        "text_minus_no_shift",
        "text_minus_mean_random",
        "text_minus_reverse",
        "text_minus_mean_wrong",
    ]
    fig, (a, b) = plt.subplots(
        1, 2, figsize=(10, 3.8), gridspec_kw={"width_ratios": [1.15, 1]}
    )
    values = np.array([inf[k]["effect"] for k in keys])
    ci = np.array([inf[k]["ci_adjusted"] for k in keys])
    a.errorbar(
        values,
        np.arange(4),
        xerr=np.stack([values - ci[:, 0], ci[:, 1] - values]),
        fmt="o",
        color="#246078",
        capsize=4,
    )
    a.axvline(0, color="#777777", lw=1)
    a.set_yticks(np.arange(4), labels)
    a.invert_yaxis()
    a.set_xlabel("Paired AP gain of transferred text direction")
    a.set_title("Primary contrasts · 98.75% intervals", fontsize=10)
    names = list(rels)
    effects = [rels[n]["contrasts"]["text_minus_no_shift"]["effect"] for n in names]
    intervals = np.array(
        [rels[n]["contrasts"]["text_minus_no_shift"]["ci95_descriptive"] for n in names]
    )
    effects = np.array(effects)
    b.errorbar(
        effects,
        np.arange(len(names)),
        xerr=np.stack([effects - intervals[:, 0], intervals[:, 1] - effects]),
        fmt="o",
        color="#8a4f27",
        capsize=4,
    )
    b.axvline(0, color="#777777", lw=1)
    b.set_yticks(np.arange(len(names)), [n.capitalize() for n in names])
    b.invert_yaxis()
    b.set_xlabel("AP gain over source-only retrieval")
    b.set_title("Per-relation · descriptive 95% intervals", fontsize=10)
    for ax in [a, b]:
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="x", alpha=0.16)
    fig.tight_layout()
    fig.savefig(OUT / "state_transfer_effects.png", dpi=200)
    fig.savefig(OUT / "state_transfer_effects.pdf")
    plt.close(fig)
    # Illustrations: use the first saved instance for each endpoint, not favorable retrievals.
    example = [
        ("cleaning", "carpet"),
        ("cooking", "chicken"),
        ("ripening", "tomato"),
        ("thawing", "fish"),
        ("drying", "dog"),
    ]
    from PIL import Image

    fig, axes = plt.subplots(2, 5, figsize=(10, 4))
    for col, (name, noun) in enumerate(example):
        rel = next(x for x in manifest["config"]["relations"] if x["name"] == name)
        for row, state in enumerate([rel["source"], rel["target"]]):
            record = next(
                x
                for x in manifest["records"]
                if x["state"] == state and x["noun"] == noun and x["instance"] == 0
            )
            with Image.open(RUN / record["image"]) as image:
                axes[row, col].imshow(image.convert("RGB"))
            axes[row, col].axis("off")
            axes[row, col].set_title(f"{state} {noun}", fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / "state_examples.png", dpi=180)
    plt.close(fig)
    # Descriptive Pareto view: higher noun retention and AP are both desirable.
    fig, ax = plt.subplots(figsize=(7, 3.4))
    points = [
        ("no_shift", "Source only", (-68, -15)),
        ("text", "Universal text", (8, -4)),
        ("image_reference", "Image relation", (8, -16)),
        ("procrustes", "Procrustes", (-70, 12)),
        ("direct_target_text", "Direct target text", (8, 8)),
    ]
    for name, label, offset in points:
        value = conditions[name]
        ax.scatter(
            value["noun"] * 100,
            value["ap"],
            s=35,
            color="#246078" if name == "text" else "#777777",
        )
        ax.annotate(
            label,
            (value["noun"] * 100, value["ap"]),
            xytext=offset,
            textcoords="offset points",
            fontsize=9,
        )
    ax.set_xlabel("Noun top-1 retrieval (%)")
    ax.set_ylabel("Target-state-and-noun AP")
    ax.set_xlim(50, 94)
    ax.set_ylim(0.32, 0.51)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(alpha=0.14)
    fig.tight_layout()
    fig.savefig(OUT / "state_category_tradeoff.png", dpi=200)
    plt.close(fig)
    pct = lambda x: f"{100*x:.1f}%"
    ci = lambda x: f"[{x[0]:+.4f}, {x[1]:+.4f}]"
    overall = inf["text_minus_no_shift"]
    guard = inf["noun_preservation"]
    passed = r["passes_predeclared_criteria"]
    decision = (
        "The aggregate passes all four adjusted primary contrasts and the noun-preservation guardrail."
        if passed
        else "All four adjusted AP contrasts are positive, but the noun-preservation guardrail fails. The complete predeclared success criterion is therefore not met; retrieval gains do not establish a state edit that preserves object category."
    )
    positive = [
        name
        for name, x in rels.items()
        if x["contrasts"]["text_minus_no_shift"]["effect"] > 0
    ]
    support = ", ".join(
        f'{name} ({len(x["nouns"])} noun labels)' for name, x in rels.items()
    )
    foldscales = {
        name: [f["relations"][name]["scale"] for f in r["folds"]] for name in rels
    }
    bank_sizes = [
        v["candidates_per_query"] for f in r["folds"] for v in f["relations"].values()
    ]
    lines = [
        "# Cross-Modal State Retrieval: Gains and Category Drift",
        "",
        "Working research manuscript | October 15, 2026 | Not submitted or peer reviewed",
        "",
        "## Abstract",
        "",
        f'Cross-modal alignment does not by itself establish reusable semantic relations. We first audit supervised arbitrary endpoint-difference transfer and then evaluate a more useful task: state-conditioned retrieval across held-out noun labels in natural images. Using independently pretrained BLOOMZ1.7B and DINOv2 base representations, a supervised ridge map, and training-noun text contrasts, we test cleaning, cooking, ripening, thawing and drying. A frozen subset contains 992 images, 51 unique noun labels and 62 relation/noun cells. Five global noun-group folds yield 496 source-image queries. The text direction changes macro target-state-and-noun average precision from {conditions["no_shift"]["ap"]:.4f} to {conditions["text"]["ap"]:.4f}; its paired gain is {overall["effect"]:+.4f}, with a Bonferroni-adjusted 98.75% bootstrap interval {ci(overall["ci_adjusted"])}. {decision} The benchmark measures associations against released labels: state ambiguity, noun taxonomy overlap and source dependence limit interpretation. We provide a reproducible pipeline and distinguish this result from physical-state understanding or a general semantic algebra.',
        "",
        "## 1. Introduction",
        "",
        "A useful cross-modal relation should help answer a query such as “find this kind of food cooked rather than raw” or “find a clean version of this kind of surface.” This goes beyond matching a caption to an image or transferring differences between unrelated example endpoints. Our central question is whether a universal text-derived state change, estimated on fitting nouns, improves retrieval for noun labels excluded from fitting.",
        "",
        "The original project established measurable local agreement between language and vision representations, with preprocessing effects more consistent than model-size effects in a small roster. That observation motivated a supervised endpoint map. However, arbitrary difference retrieval is not an independent test of semantic analogy: for f(x)=xW+b, f(x_b)-f(x_a)=(x_b-x_a)W exactly. Useful difference retrieval may therefore follow from endpoint correspondence alone. The identity is elementary; our contribution is an auditable empirical distinction between those evaluation tasks.",
        "",
        "This manuscript adds five natural-state relation families and a target-state-plus-noun retrieval endpoint. It reports both primary paired comparisons and the scope of the annotations. The focus is useful retrieval under state changes, with separate checks on noun retention.",
        "",
        "## 2. Related work",
        "",
        "Huh et al. [1] motivates cross-modal geometric agreement. Fournier et al. [2] and Linzen [3] explain why analogy scores can combine different sources of regularity. Visalogy [4] learns representations for visual analogy questions, while StyleCLIP [5] uses cross-modally pretrained CLIP guidance for image manipulation. These works establish substantial precedent; we do not claim that vector arithmetic or input-independent edit directions are new.",
        "",
        "MIT-States [6] provides the natural state/noun image associations used here. BLOOMZ [7] and DINOv2 [8] supply frozen independently pretrained encoders; the downstream ridge bridge is explicitly supervised. Recent object-state studies [9,10] motivate evaluation beyond object recognition. We investigate a specific held-out-noun transfer protocol, not a state-of-the-art claim against those benchmarks. A broader novelty assessment and direct comparisons remain necessary.",
        "",
        "## 3. Data and experimental design",
        "",
        f"We selected five physical-state relations before inspecting transfer scores: {support}. Every supported state/noun composition had at least10 images in the original catalog; after deterministic archive duplicate screening we sampled8 per composition with fixed seeds. The final set contains124 compositions and992 images. All selected SHA256 hashes are distinct. The original dataset derives labels from image-search queries and human cleaning; its README explicitly notes residual ambiguity and mislabeled images. We performed visual QA but did not reannotate the frozen sample.",
        "",
        f'![Natural-state examples]({OUT/"state_examples.png"})',
        "",
        "Figure 1. First sampled source/target examples from five specified noun labels. Images are independent photographs; rows do not depict paired physical instances or controlled interventions. Labels are from the original MIT-States release [6].",
        "",
        "A seeded, coverage-balanced assignment partitions noun labels into five groups. In each fold one group supplies test queries, the next supplies validation, and the remaining three fit the representation transforms, map and universal relations. Each eligible noun is tested once per relation. All appearances of an exact test noun across relations are absent from fitting. Related labels such as fish/salmon/seafood or fruit/apple remain possible across groups, so the design establishes lexical noun-label separation, not taxonomic novelty.",
        "",
        f"For each relation/fold, the gallery contains source-state and target-state pictures from its test nouns. For source query index i we remove source index i and target index (i+1) mod8 from every noun. Thus the query is excluded and each noun contributes7 candidates in each state. Bank sizes range from {min(bank_sizes)} to {max(bank_sizes)} images; the same banks are used for every control. With few test nouns, noun recognition can be easy, so state-sensitive metrics and per-relation reporting are required.",
        "",
        "## 4. Representation mapping and relation transfer",
        "",
        "We fix BLOOMZ1.7B hidden-state index19 (including the embedding output as index0) and DINOv2 base block10 (zero-based), using layers chosen earlier on the unrelated WIT study. Four matched prompt templates describe each state/noun pair. Float32 MPS extraction uses pinned cached model revisions; CPU float64 analysis fits95th-percentile clipping, L2 normalization, centered PCA64 and projected endpoint normalization on map-fitting rows only.",
        "",
        "The first4 sampled images per fitting composition fit a centered ridge endpoint map with alpha=1. The remaining4 supply an image-derived relation reference. For a relation r, average normalized text differences across templates within a fitting noun, then average fitting nouns equally. The transferred direction is d_r=mean(x_target-x_source)W. A source image is translated as normalize(y_source+s*d_r). Select s from {0,.25,.5,1,2,4} using validation macro noun AP, with ascending-scale tie break. No test-noun text contrast enters this universal direction.",
        "",
        "Controls are source-only retrieval, reversed correct direction,100 fixed equal-norm isotropic directions, the other four state relations evaluated separately after norm matching, an image-derived relation, and orthogonal Procrustes transfer. Direct target-text endpoint retrieval and a source-anchored noun-specific text contrast deliberately use test-noun wording; they are stronger-information recognition references. Every map or scale fit is confined to fitting/validation groups.",
        "",
        "## 5. Outcomes and uncertainty",
        "",
        "Average precision treats an image as relevant only when both its released state and noun match the request. We macro-average source queries within noun, nouns within relation, then the five fixed relations. This prevents frequent categories or relation families from dominating. Additional outcomes are target-state top1, noun top1, joint top1 and within-noun state AUC/AP. No exact “matching target photograph” is assumed.",
        "",
        "Four primary paired contrasts compare the text direction with source-only, random, reversed and wrong-relation shifts. A20,000-draw paired bootstrap resamples unique noun labels jointly across all their relation appearances and source queries within relation/noun cells. Two-sided98.75% percentile intervals implement Bonferroni adjustment over these four aggregate contrasts; ordinary95% and category-only intervals are sensitivity summaries. Noun-top1 preservation uses a one-sided95% lower bound above -.05. Gallery banks, fitted maps and validation-selected scales stay fixed; their uncertainty is omitted. The relation roster is fixed and is not resampled as a population of semantic transformations.",
        "",
        "## 6. Results",
        "",
        "| Primary paired AP contrast | Gain | Adjusted 98.75% interval |",
        "|---|---:|---|",
    ]
    for label, key in zip(labels, keys):
        lines.append(
            f'| Text minus {label.lower()} | {inf[key]["effect"]:+.4f} | {ci(inf[key]["ci_adjusted"])} |'
        )
    lines += [
        "",
        decision,
        "",
        f'Noun-top1 changes by {guard["effect"]:+.4f}; its one-sided95% lower bound is {guard["one_sided_lower95"]:+.4f}, compared with the -.05 guardrail. These are aggregate criteria. A positive result for the fixed suite would not establish improvement for every individual relation.',
        "",
        f'![Paired retrieval effects]({OUT/"state_transfer_effects.png"})',
        "",
        "Figure 2. Left: four primary paired macro AP contrasts with multiplicity-adjusted intervals. Right: relation-specific gains over source-only retrieval, with descriptive95% intervals. Per-relation intervals are not a second family of significance claims.",
        "",
        "| Relation | Nouns | Source-only AP | Text AP | AP gain |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, v in rels.items():
        lines.append(
            f'| {name.capitalize()} | {len(v["nouns"])} | {v["metrics"]["no_shift"]["ap"]:.4f} | {v["metrics"]["text"]["ap"]:.4f} | {v["contrasts"]["text_minus_no_shift"]["effect"]:+.4f} |'
        )
    lines += [
        "",
        f"The observed gain over source-only is positive in {len(positive)} of the five relation means. This is a descriptive count, not an independent-replication test.",
        "",
        "| Condition | Macro AP | Target-state top1 | Noun top1 | Joint top1 |",
        "|---|---:|---:|---:|---:|",
    ]
    names = {
        "no_shift": "Source only",
        "text": "Universal text relation",
        "reverse": "Reverse relation",
        "mean_random": "Mean random shift",
        "mean_wrong": "Mean wrong relation",
        "image_reference": "Image relation reference",
        "procrustes": "Procrustes text relation",
        "direct_target_text": "Direct target text",
        "noun_specific_text": "Noun-specific text offset",
    }
    for condition, label in names.items():
        a = conditions[condition]
        lines.append(
            f'| {label} | {a["ap"]:.4f} | {pct(a["state"])} | {pct(a["noun"])} | {pct(a["joint"])} |'
        )
    lines += [
        "",
        "### Validation choices and robustness",
        "",
        f"Frozen validation-selected text scales across folds: {foldscales}. A zero scale is retained when selected and is not replaced by a favorable test setting. The saved results contain train/validation/test noun lists, all validation curves, per-noun effects, category-only intervals and leave-one-noun-out estimates.",
        "",
        "### Connection to the endpoint pilot",
        "",
        "The earlier WIT audit obtains19.5-25.8% arbitrary difference top1 among256 candidates across three overlapping splits, with single-endpoint ablations at3.9-8.6%. That correspondence result alone cannot establish semantic transfer. The present task instead transfers one named relation to excluded noun labels without using their text endpoints to form the universal direction. The two experiments measure different capabilities and should not be pooled statistically.",
        "",
        "## 7. Limitations and interpretation",
        "",
        "Released state labels are the central validity limitation. Visual QA found some “thawed” examples depicting cooked dishes, some unripe labels on ripe-looking or unrelated fruit, and states not visually decidable from product or generic object photographs. These observations do not estimate a label-error rate. Our numerical endpoint therefore measures retrieval against released labels, not verified physical states. We retain every frozen sample and report this limitation rather than repairing labels after seeing outcomes.",
        "",
        "Exact noun labels are separated, but taxonomic, photographer, web-source and near-duplicate dependence can remain. The sampled galleries are small and use eight images per composition; source URLs and photographer IDs are not available in the archive. The bootstrap conditions on overlapping fitted folds and banks and uses approximate coverage. Five selected relation families and one encoder pair cannot establish a general semantic algebra or cross-family scaling. Hyperparameters and research direction were selected through prior exploratory work; the saved local design lock is not external preregistration.",
        "",
        "Earlier exploratory alignment and color-sketch work informed the research protocol. Neither supplies independent replication evidence for the natural-state result.",
        "",
        "## 8. Next experiments and paper scope",
        "",
        "The immediate next milestone is a verified-label replication, with source-video or photographer separation and broader galleries. The ChangeIt-derived object-state golden subset [9] provides an alternative with existing human-verified state annotations; STATUS Bench [10] provides paired state descriptions but requires access to its underlying Ego4D images. Neither is claimed as evaluated here. We should also compare a second language/vision model family, test source/taxonomy-blocked splits, and estimate whether improvements persist with hundreds of same-noun and wrong-noun distractors.",
        "",
        "If only the image-derived direction works, investigate whether the endpoint map discards relation information. If direct target text works but the universal direction does not, recognition is present without demonstrated reusable transfer. If effects vary by relation, investigate category-conditioned or nonlinear transformations on validation data and evaluate those on a fresh untouched split. Those alternatives are future experiments, not retrospective replacements for the current primary criterion.",
        "",
        "The current deliverable is a results-bearing working manuscript. Submission readiness depends on label validation, broader baselines and a fuller novelty review. A separate3D viewer can later display the saved high-dimensional source, target and relation vectors; projection retention and full-space metrics must remain visible.",
        "",
        "## Reproducibility",
        "",
        "From sz684/, use run.py state-data, run.py state-extract for language and vision, run.py state-evaluate, and run.py state-report. The exact configuration is experiments/state_transfer/configs/state_transfer.json. The analysis lock and source snapshot precede natural-image feature extraction. Source image hashes, archive checksums, noun groups, sampled image indices, model revisions, library versions, controls, bootstrap draws and per-query scores are saved under experiments/state_transfer/results/. That directory is Git-ignored and must accompany any shared research archive. Tests are collected from experiment and shared test folders; archived results are excluded.",
        "",
        "## References",
        "",
        "[1] Huh et al. (2024). Position: The Platonic Representation Hypothesis. ICML. https://proceedings.mlr.press/v235/huh24a.html",
        "",
        "[2] Fournier, Dupoux and Dunbar (2020). Analogies minus analogy test: measuring regularities in word embeddings. CoNLL. https://aclanthology.org/2020.conll-1.29/",
        "",
        "[3] Linzen (2016). Issues in evaluating semantic spaces using word analogies. https://arxiv.org/abs/1606.07736",
        "",
        "[4] Sadeghi, Zitnick and Farhadi (2015). Visalogy: Answering Visual Analogy Questions. NeurIPS. https://papers.neurips.cc/paper_files/paper/2015/hash/45f31d16b1058d586fc3be7207b58053-Abstract.html",
        "",
        "[5] Patashnik et al. (2021). StyleCLIP: Text-Driven Manipulation of StyleGAN Imagery. ICCV. https://arxiv.org/abs/2103.17249",
        "",
        "[6] Isola, Lim and Adelson (2015). Discovering States and Transformations in Image Collections. CVPR. https://web.mit.edu/phillipi/Public/states_and_transformations/index.html",
        "",
        "[7] Muennighoff et al. (2023). Crosslingual Generalization through Multitask Finetuning. ACL. https://arxiv.org/abs/2211.01786",
        "",
        "[8] Oquab et al. DINOv2: Learning Robust Visual Features without Supervision. https://arxiv.org/abs/2304.07193",
        "",
        "[9] Newman et al. (2024). Do Pre-trained Vision-Language Models Encode Object States? https://arxiv.org/abs/2409.10488",
        "",
        "[10] Ukai, Kurita and Inoue (2025). STATUS Bench: A Rigorous Benchmark for Evaluating Object State Understanding in Vision-Language Models. ACM Multimedia. https://arxiv.org/abs/2510.22571",
        "",
    ]
    follow_path = RUN / "secondary_identity/results.json"
    if follow_path.exists():
        follow = json.loads(follow_path.read_text())
        section = [
            "### Exploratory preservation experiments",
            "",
            "After inspecting the primary result, we tested two remedies using the same test set. These are explicitly post-primary exploratory analyses. The first constrains scale selection to a validation noun-top1 loss no worse than five percentage points. The second estimates a training-noun centroid subspace retaining80% of between-centroid variance, removes0%,50% or100% of the relation component in that subspace, norm-matches the direction, and selects strength/scale under the same validation guardrail. Training centroids are an identity proxy, not a proven disentangled factor.",
            "",
            "| Exploratory method | Macro AP | Noun top1 | AP gain versus source only (95%) |",
            "|---|---:|---:|---|",
        ]
        for name, label in [
            ("constrained_scale", "Constrained scale"),
            ("identity_projection", "Identity-subspace projection"),
        ]:
            value = follow["conditions"][name]
            estimate = follow["exploratory_intervals"][name + "_ap_minus_no_shift"]
            section.append(
                f'| {label} | {value["ap"]:.4f} | {pct(value["noun"])} | {estimate["effect"]:+.4f} {ci(estimate["ci95"])} |'
            )
        section += [
            "",
            "Neither remedy demonstrates the predeclared noun-preservation tolerance on the test queries. Both trade away some of the original AP gain; a validation-time guardrail did not ensure a test-time guarantee. These ordinary95% intervals are descriptive, unadjusted follow-up summaries. Primary results are unchanged.",
            "",
            f'![State/category tradeoff]({OUT/"state_category_tradeoff.png"})',
            "",
            "Figure 3. Descriptive state/noun AP versus noun top1 for selected primary methods. Higher is better on both axes. More aggressive state targeting can sacrifice noun retention; no confidence intervals are attached to these method coordinates.",
            "",
        ]
        position = lines.index("## 7. Limitations and interpretation")
        lines[position:position] = section
    lines.insert(lines.index("## 7. Limitations and interpretation"), "")
    old = next(
        line
        for line in lines
        if line.startswith("Frozen validation-selected text scales")
    )
    position = lines.index(old)
    table = [
        "Text translation scales selected on validation only:",
        "",
        "| Relation | Fold 0 | Fold 1 | Fold 2 | Fold 3 | Fold 4 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    table += [
        "| " + name.capitalize() + " | " + " | ".join(map(str, values)) + " |"
        for name, values in foldscales.items()
    ]
    table += [
        "",
        "A zero scale is retained when selected and is not replaced using test performance. Saved results include every validation curve, train/validation/test noun list, category-only interval and leave-one-noun-out estimate.",
    ]
    lines[position : position + 1] = table
    manuscript = (
        "\n".join(lines)
        .replace("BLOOMZ1.7B", "BLOOMZ 1.7B")
        .replace("top1", "top-1")
        .replace("direction,100", "direction, 100")
        .replace("separate3D", "separate 3D")
        .replace("strength/scale", "strength and scale")
    )
    manuscript = re.sub(
        r"\b(at least|contains|sampled|with|and|among|using|first|remaining|per|of|A|a|in|to|by|over|retaining|removes)(?=\d)",
        r"\1 ",
        manuscript,
    )
    manuscript = re.sub(
        r"(sided|descriptive|ordinary|index|block|fits|contributes|mod)(?=\d)",
        r"\1 ",
        manuscript,
    )
    # Editorial manuscripts are maintained separately; rebuilding figures must not overwrite them.
    (RUN / "GENERATED_MANUSCRIPT.md").write_text(manuscript)

    # Compact companion report is useful before opening the full manuscript.
    (RUN / "REPORT.md").write_text(
        "\n".join(
            [
                "# Five-relation state retrieval",
                "",
                decision,
                "",
                f'Macro AP: {conditions["no_shift"]["ap"]:.4f} → {conditions["text"]["ap"]:.4f}; adjusted gain interval {ci(overall["ci_adjusted"])}.',
                "",
                "This is retrieval against released MIT-States labels, not independently verified physical-state understanding. See STATE_TRANSFER_PAPER.md and STATE_DATA_AUDIT.md.",
                "",
            ]
        )
    )
    print(
        "Wrote figures, companion report and results/state_transfer/GENERATED_MANUSCRIPT.md; editorial manuscripts preserved."
    )


if __name__ == "__main__":
    build()
