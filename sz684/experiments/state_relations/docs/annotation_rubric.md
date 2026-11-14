# Image review

Review the contact sheets by image ID, before looking at model outputs. Check whether the named object is the main subject and whether the released state is visibly supported. Use the full-resolution ZIP member for close calls. Mark each reviewed image `accept`, `reject`, or `uncertain`, with a reason for rejection or uncertainty. Exclude ambiguous cases such as cooked food labeled thawed or a wet/dry state inferred only from the setting. Source and target images may depict different instances of the same noun.

Record reviews in JSONL with `image_id`, `reviewer`, `decision` and `reason`. Have a second reviewer independently review ambiguous images and a seeded sample of the accepted images from each transition. Resolve disagreements before accepting images for the gallery or as queries. Keep the review records with the final manifest.

Check selected images against one another and the historical sample for near duplicates, including crops and resized copies. Assign any verified duplicate cluster to one split or exclude its copies. The cached catalog provides no reliable original page identifiers, so record visible shared-source evidence and any remaining uncertainty.
