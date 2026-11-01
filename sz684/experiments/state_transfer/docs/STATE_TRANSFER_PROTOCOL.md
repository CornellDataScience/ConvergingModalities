# Natural-image state transfer: primary protocol

October 15,2026. Replaces the retired red→blue main study at the user's request. Written after inspecting dataset label coverage and before downloading/extracting the selected natural images or inspecting any state-transfer scores. Local prospective specification, not external preregistration.

## Research question and practical endpoint

Can a language-derived relation learned on some noun categories improve retrieval of the requested state of a held-out noun from an image query? Evaluate **cleaning(dirty→clean), cooking(raw→cooked), ripening(unripe→ripe), thawing(frozen→thawed), and drying(wet→dry)**. This is useful state-conditioned image retrieval, not image synthesis, causal prediction of a physical process, or recognition of a recoloring.

MIT-States source: Isola,Lim,Adelson(2015), https://web.mit.edu/phillipi/Public/states_and_transformations/index.html . Original public ZIP downloaded selectively with HTTP byte ranges; validate every member with ZIP CRC and SHA256. Preserve archive metadata, central directory identity, exact file names, selection indices and hashes. These are original internet-image associations with weak state/noun labels, not independently verified counterfactuals. Images in two states need not depict the same physical object. Do not claim instance preservation or causal effects. Do not imply original image copyright ownership or redistribute the corpus as newly licensed data.

## Coverage and sampling

Include all nouns with at least10 available images in both states for a named relation. Catalog inspection before filtering finds9,10,11,17,15 supported nouns respectively;51 unique nouns and124 distinct state/noun compositions. “Open→closed” has only3 supported nouns and is excluded from this main roster on coverage grounds, not performance.

Remove exact ZIP duplicate contents globally (matching CRC32 and uncompressed length; verify selected bytes with SHA256). Retain one deterministic canonical archive member, and list removals. Recheck support after duplicate removal; require at least8 distinct candidates per composition. Randomly sample8 images per composition with saved category-specific seeds; use the same sampled composition wherever it appears. Record a public gallery contact sheet for QA, without score-based exclusions. Label ambiguity and near duplicates remain limitations.

Assign nouns globally to five groups using a seeded random search, accepting the first assignment giving each relation at least one eligible noun in every group. Each of five folds holds one group out for test and the next for validation; remaining three groups fit transformations and maps. Each eligible noun is tested once per relation. An exact noun label is absent from fitting across all relations in its test fold. This is not taxonomic exclusion: labels such as fish and salmon or fruit and apple are related. Explicitly restrict the claim to noun-label-held-out transfer.

## Models and training

Pinned current BLOOMZ1.7B and DINOv2 base weights; fixed layers19(language hidden-state index including embeddings) and10(vision zero-based block), chosen in the earlier unrelated WIT study. Float32 MPS extraction; CPU float64 analysis. Embed four fixed prompt templates per state/noun composition. No main test text contrast enters a universal direction; direct target-text baselines deliberately have stronger information.

Within each fold, first4 sampled images per training composition fit a centered ridge endpoint map(alpha1), and the remaining4 supply an image-direction diagnostic. Fit original mean-per-example95th-percentile clipping across all layers, L2 normalization, centered PCA64 and projected endpoint normalization only on map-fitting rows. PCA64 is a fixed resource/rank choice, not validated on state-transfer outcomes. No layer search. Compare an orthogonal Procrustes map as a fixed secondary baseline.

For each relation, average text contrasts within training nouns over four templates, then average equally over those nouns. Transfer this one direction through the fitted ridge map. Choose scale from{0,.25,.5,1,2,4} by validation macro noun average precision, ties favoring smaller scale. Freeze map, preprocessing, direction and scale before scoring test nouns. Apply normalize(image_source+s*d).

## Candidate banks and outcomes

For a relation/fold, gallery includes the sampled source and target images of all test nouns eligible for that relation. For source query index i, exclude source image i in every noun and target image(i+1)%8 in every noun. This excludes the query and gives7 source and7 target candidates per noun with identical counts across conditions. Targets are independently photographed instances; there is no “correct matching endpoint” by image index.

A relevant retrieval is an image with BOTH the requested target state and the query's noun. Primary outcome: query average precision for this conjunction, then macro average over queries within noun, nouns within relation, and the fixed five-relation roster. Always report the per-relation and per-noun effects; a positive aggregate does not show every relation succeeds. Also report target-state top1, noun top1, joint state/noun top1 and within-noun state AP. No-shift is the source-only retrieval baseline.

Controls: no shift; reverse of correct direction at the selected scale;100 fixed-seed isotropic equal-norm directions at that same scale; mean of the other four relation directions, each norm matched to the correct direction and evaluated separately before averaging metrics; image-derived training relation(own validation scale); Procrustes text relation(own scale); affine direct target-text endpoint(averaged four target prompts for the test noun); and source-anchored target-noun text offset as a stronger-information reference. Wrong-relation controls test specificity beyond generic target/geometry drift. Do not average vectors before scoring the wrong/random controls.

## Inference and decision

Four planned primary paired AP-gain contrasts: correct text direction minus no shift, mean random shift, reverse shift, and mean wrong-relation shift. Bonferroni percentile intervals at98.75% for these four contrasts, plus ordinary95% intervals. Primary macro weights keep each relation equal regardless of dataset frequency.

Conditional paired bootstrap: resample unique noun labels jointly across all their relation appearances, then source queries within each relation/noun cluster. Use the same draws for all contrasts;20,000 saved resamples. Keep gallery banks, fitted folds and selected scales fixed. This omits map-training/bank-selection uncertainty and assumes exchangeability of the fixed selected noun labels. Preserve source-label overlap across relations in the outer resampling. Also report category-only and leave-one-noun-out sensitivity. The relation roster itself is fixed; do not infer over an unlimited population of semantic relations. Per-relation intervals are descriptive, without new significance declarations.

Preservation guardrail: one-sided95% lower bound for noun-top1(correct direction)−noun-top1(no shift) must exceed−.05. Claim useful aggregate transfer only if all four adjusted primary intervals exclude zero positively and the preservation guardrail passes. If only some relations improve, report mixed/conditional results. Failure to pass is not evidence of equivalence. Direct target text is a recognition reference, not a fair weaker-information comparator.

## Boundaries and follow-up

The current model pair, small sampled gallery, weak labels, potential web-source dependence, related noun taxonomy and adaptive choice of research direction restrict conclusions. Source URLs/photographer IDs are not available in this archive; source independence cannot be guaranteed. This experiment adds useful natural-image relations but does not make the paper submission-ready automatically. Follow with independently verified labels, source/taxonomy-blocked replication, a second model family and a paired natural-state dataset. Preserve retired sketch artifacts and clearly distinguish them from this main experiment.
