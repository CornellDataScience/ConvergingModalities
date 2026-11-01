# Cleanup record

Scope: `sz684/` only. Cleanup and documentation organization were authorized; Track 1 and Track 2 implementation and research reruns were explicitly excluded.

## Removed disposable files

Removed 141 disposable files (6,382,590 bytes, approximately 6.1 MiB). No research artifacts or installed packages were deleted.

| Removed path | Files | Bytes | Reason |
| --- | ---: | ---: | --- |
| `.DS_Store` | 1 | 6,148 | macOS Finder metadata |
| `.cache/fontconfig/` | 4 | 4,909,048 | Regenerable font/plot cache |
| `.cache/matplotlib/` | 1 | 594,994 | Regenerable font/plot cache |
| `.pytest_cache/` | 5 | 9,064 | Regenerable Python bytecode or pytest cache |
| `__pycache__/` | 19 | 269,901 | Regenerable Python bytecode or pytest cache |
| `experiments/__pycache__/` | 1 | 161 | Regenerable Python bytecode or pytest cache |
| `experiments/baseline/__pycache__/` | 1 | 170 | Regenerable Python bytecode or pytest cache |
| `experiments/baseline/data/__pycache__/` | 2 | 20,979 | Regenerable Python bytecode or pytest cache |
| `experiments/baseline/evaluation/__pycache__/` | 3 | 16,458 | Regenerable Python bytecode or pytest cache |
| `experiments/baseline/tests/__pycache__/` | 4 | 23,682 | Regenerable Python bytecode or pytest cache |
| `experiments/color_transfer/__pycache__/` | 1 | 176 | Regenerable Python bytecode or pytest cache |
| `experiments/color_transfer/data/__pycache__/` | 2 | 14,123 | Regenerable Python bytecode or pytest cache |
| `experiments/color_transfer/evaluation/__pycache__/` | 2 | 23,027 | Regenerable Python bytecode or pytest cache |
| `experiments/color_transfer/tests/__pycache__/` | 2 | 17,038 | Regenerable Python bytecode or pytest cache |
| `experiments/difference_transfer/__pycache__/` | 1 | 181 | Regenerable Python bytecode or pytest cache |
| `experiments/difference_transfer/evaluation/__pycache__/` | 2 | 10,719 | Regenerable Python bytecode or pytest cache |
| `experiments/difference_transfer/tests/__pycache__/` | 2 | 5,489 | Regenerable Python bytecode or pytest cache |
| `experiments/endpoint_audit/__pycache__/` | 1 | 176 | Regenerable Python bytecode or pytest cache |
| `experiments/endpoint_audit/evaluation/__pycache__/` | 2 | 10,991 | Regenerable Python bytecode or pytest cache |
| `experiments/endpoint_audit/tests/__pycache__/` | 2 | 4,163 | Regenerable Python bytecode or pytest cache |
| `experiments/heldout_layers/__pycache__/` | 1 | 176 | Regenerable Python bytecode or pytest cache |
| `experiments/heldout_layers/evaluation/__pycache__/` | 2 | 11,824 | Regenerable Python bytecode or pytest cache |
| `experiments/heldout_layers/tests/__pycache__/` | 2 | 4,981 | Regenerable Python bytecode or pytest cache |
| `experiments/multimodal_structure/__pycache__/` | 3 | 11,179 | Regenerable Python bytecode or pytest cache |
| `experiments/multimodal_structure/data/__pycache__/` | 9 | 54,821 | Regenerable Python bytecode or pytest cache |
| `experiments/multimodal_structure/evaluation/__pycache__/` | 11 | 44,049 | Regenerable Python bytecode or pytest cache |
| `experiments/multimodal_structure/extraction/__pycache__/` | 8 | 24,539 | Regenerable Python bytecode or pytest cache |
| `experiments/multimodal_structure/reporting/__pycache__/` | 3 | 17,322 | Regenerable Python bytecode or pytest cache |
| `experiments/multimodal_structure/tests/__pycache__/` | 5 | 40,442 | Regenerable Python bytecode or pytest cache |
| `experiments/multimodal_structure/training/__pycache__/` | 5 | 17,031 | Regenerable Python bytecode or pytest cache |
| `experiments/paired_uncertainty/__pycache__/` | 1 | 180 | Regenerable Python bytecode or pytest cache |
| `experiments/paired_uncertainty/evaluation/__pycache__/` | 2 | 13,218 | Regenerable Python bytecode or pytest cache |
| `experiments/paired_uncertainty/tests/__pycache__/` | 2 | 3,639 | Regenerable Python bytecode or pytest cache |
| `experiments/robustness/__pycache__/` | 1 | 172 | Regenerable Python bytecode or pytest cache |
| `experiments/robustness/evaluation/__pycache__/` | 2 | 17,057 | Regenerable Python bytecode or pytest cache |
| `experiments/robustness/tests/__pycache__/` | 2 | 10,925 | Regenerable Python bytecode or pytest cache |
| `experiments/state_transfer/__pycache__/` | 1 | 176 | Regenerable Python bytecode or pytest cache |
| `experiments/state_transfer/data/__pycache__/` | 4 | 31,762 | Regenerable Python bytecode or pytest cache |
| `experiments/state_transfer/evaluation/__pycache__/` | 3 | 40,924 | Regenerable Python bytecode or pytest cache |
| `experiments/state_transfer/reporting/__pycache__/` | 2 | 44,724 | Regenerable Python bytecode or pytest cache |
| `experiments/state_transfer/tests/__pycache__/` | 3 | 13,878 | Regenerable Python bytecode or pytest cache |
| `experiments/state_transfer/training/__pycache__/` | 3 | 11,331 | Regenerable Python bytecode or pytest cache |
| `shared/__pycache__/` | 7 | 18,874 | Regenerable Python bytecode or pytest cache |
| `shared/tests/__pycache__/` | 1 | 12,678 | Regenerable Python bytecode or pytest cache |
| `tmp/` | 0 | 0 | Empty scratch directory |

## Relocated and organized documentation

- Moved the superseded `docs/planning/MULTIMODAL_EXPANSION_PLAN.md` to [docs/history/MULTIMODAL_EXPANSION_PLAN.md](history/MULTIMODAL_EXPANSION_PLAN.md). Its original text is retained beneath a historical-status notice; old repository findings are not current instructions.
- Updated the root experiment index and research status to include the existing multimodal study and distinguish completed, exploratory, retired and planned work.
- Added the [newcomer map and diagrams](NEWCOMER_GUIDE.md), [file inventory](FILE_INVENTORY.md), and [planning-only team/file/rerun checklist](planning/STATE_RELATIONS_PLAN.md).
- Updated the state-transfer and multimodal README status/navigation. Existing experiment paths, generated reports and public runner commands stay stable.

## Preserved dependencies and evidence

- All executable source, configuration, requirements and tests remain unchanged.
- All experiment results and rendered outputs remain in place, including partial Quick, Draw! features and frozen source snapshots.
- Kept the baseline manifest: state data preparation reads its pinned model revisions. Historical analyses also import the baseline feature loader.
- Kept `shared/paths.py`, `shared/cache.py` and `shared/extractor_migrations.json`: these preserve historical path resolution and exact extractor identity checks.
- Kept Hugging Face weights/dataset caches, MIT-States and Quick, Draw! downloads, the OpenWebText archive and `.venv/`. Removing them is not necessary for navigation cleanup and would reduce reproduction readiness.
- Kept historical experiment code and overlapping manuscripts: they document distinct runs, interpretations or editorial versions. Their age alone does not establish that they are irrelevant.

## Verification

Verification passed: 282 local documentation links resolve; diagram/code fences are balanced; the inventory individually lists 192 maintained files and accounts for all 43,767 local files through non-overlapping groups. All 152 tracked non-Markdown files are byte-for-byte unchanged. All 15,536 research artifacts retain their paths, sizes and modification times. The relocated historical proposal retains its original text. `git diff --check` passes.

Research stages, model extraction, fitting, scoring, report generation and tests were not run for this documentation/cache-only change. No Track 1/2 implementation package was created. Mermaid diagrams are provided as Markdown source; their visual rendering was not independently tested.
