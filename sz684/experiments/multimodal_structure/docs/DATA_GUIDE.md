# Data guide

CSV columns:

| Column | Meaning |
| --- | --- |
| sample_id | Unique record ID across all modalities |
| concept_id | Shared concept/event identity across modalities |
| family_id | Object/event family held out as a whole |
| source_id | Original source shared across crops, clips and modalities |
| modality | text, vision, or audio |
| split | train, validation, or test |
| text | Nonempty input for text records |
| path | Local media path relative to CSV, or absolute |
| relation | Optional relation name |
| label | Optional endpoint: 0=source, 1=target |

Images and sounds are different rows. A paired recording gives both rows the
same source_id. Multiple crops/frames do not become independent sources.
Family/source groups cannot cross splits. The helper `assign_splits` links
families sharing a source, then performs seeded 60/20/20 assignment. It needs at
least five independent groups, but does not guarantee relation coverage.

`prepare` rejects missing input, duplicate IDs and exact content crossing splits;
it records per-input SHA256s. Perceptual duplicates, uploader/session grouping
and taxonomy separation still need curation. Never infer those from exact hashes.

VGGSounder annotations are pinned to commit
`5a91bf6851d8cbf830d85b00ba0953f1befaffa4`; media to
`dc7815aa65132c48f070686e0004180f03b5abcf`. Only single-AV-label clips without
voice-over, background music or static images are sampled. Additional A-only or
V-only labels may remain. Text examples are label-derived descriptions, not
independent human paraphrases. Video-level AV labels do not guarantee that the
midpoint still shows the event. Review frames before stronger conclusions.

Family grouping currently uses exact labels, not verified superclasses. Splits
are for development within a released test corpus; they are not a new untouched
confirmatory test set. The download log records incomplete acquisition explicitly.

New directional axes need separate human validation in each channel. Do not
invent raw/cooked or wet/dry audio endpoints. Source and target images are not
physical before/after pairs unless explicitly established by acquisition.

For local clip import, CSV columns are
`clip_id,path,concept_id,family_id,split`. IDs must use letters, numbers,
underscores or hyphens. The importer extracts a midpoint image and centered audio
window up to ten seconds. Consolidate source IDs when clips share an original
recording before preparation.
