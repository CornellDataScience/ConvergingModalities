"""Resolve canonical project paths and filenames recorded before reorganization.

Saved manifests and source snapshots retain their original relative filenames.
Only working code is redirected; this never edits historical artifacts.
"""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LEGACY_PATHS = {
    "experiment.py": "experiments/baseline/data/experiment.py",
    "alignment.py": "experiments/baseline/evaluation/alignment.py",
    "language_eval.py": "experiments/baseline/evaluation/language_eval.py",
    "verify_extraction.py": "experiments/baseline/evaluation/verify_extraction.py",
    "report.py": "experiments/baseline/reporting/report.py",
    "PROTOCOL.md": "experiments/baseline/docs/PROTOCOL.md",
    "LEGACY_BASELINE.md": "experiments/baseline/docs/LEGACY_BASELINE.md",
    "configs/mac.json": "experiments/baseline/configs/mac.json",
    "tests/test_alignment.py": "experiments/baseline/tests/test_alignment.py",
    "tests/test_cache.py": "experiments/baseline/tests/test_cache.py",
    "tests/test_language_eval.py": "experiments/baseline/tests/test_language_eval.py",
    "heldout_layers.py": "experiments/heldout_layers/evaluation/heldout_layers.py",
    "RESEARCH_PLAN.md": "experiments/heldout_layers/docs/RESEARCH_PLAN.md",
    "tests/test_heldout_layers.py": "experiments/heldout_layers/tests/test_heldout_layers.py",
    "robustness.py": "experiments/robustness/evaluation/robustness.py",
    "ROBUSTNESS_PROTOCOL.md": "experiments/robustness/docs/ROBUSTNESS_PROTOCOL.md",
    "tests/test_robustness.py": "experiments/robustness/tests/test_robustness.py",
    "paired_uncertainty.py": "experiments/paired_uncertainty/evaluation/paired_uncertainty.py",
    "PAIRED_PROTOCOL.md": "experiments/paired_uncertainty/docs/PAIRED_PROTOCOL.md",
    "tests/test_paired_uncertainty.py": "experiments/paired_uncertainty/tests/test_paired_uncertainty.py",
    "difference_transfer.py": "experiments/difference_transfer/evaluation/difference_transfer.py",
    "TRANSFER_PROTOCOL.md": "experiments/difference_transfer/docs/TRANSFER_PROTOCOL.md",
    "tests/test_difference_transfer.py": "experiments/difference_transfer/tests/test_difference_transfer.py",
    "endpoint_audit.py": "experiments/endpoint_audit/evaluation/endpoint_audit.py",
    "vector_explorer.html": "experiments/endpoint_audit/reporting/vector_explorer.html",
    "ENDPOINT_AUDIT_PROTOCOL.md": "experiments/endpoint_audit/docs/ENDPOINT_AUDIT_PROTOCOL.md",
    "tests/test_endpoint_audit.py": "experiments/endpoint_audit/tests/test_endpoint_audit.py",
    "quickdraw_data.py": "experiments/color_transfer/data/quickdraw_data.py",
    "semantic_extract.py": "experiments/color_transfer/data/semantic_extract.py",
    "semantic_analysis.py": "experiments/color_transfer/evaluation/semantic_analysis.py",
    "pilot_planning.py": "experiments/color_transfer/evaluation/pilot_planning.py",
    "QUICKDRAW_PROTOCOL.md": "experiments/color_transfer/docs/QUICKDRAW_PROTOCOL.md",
    "SEMANTIC_PROTOCOL.md": "experiments/color_transfer/docs/SEMANTIC_PROTOCOL.md",
    "CONCEPT_TRANSFER_NEXT.md": "experiments/color_transfer/docs/CONCEPT_TRANSFER_NEXT.md",
    "PAPER_DRAFT.md": "experiments/color_transfer/docs/PAPER_DRAFT.md",
    "configs/semantic_quickdraw.json": "experiments/color_transfer/configs/semantic_quickdraw.json",
    "tests/test_semantic.py": "experiments/color_transfer/tests/test_semantic.py",
    "state_data.py": "experiments/state_transfer/data/state_data.py",
    "mit_states_data.py": "experiments/state_transfer/data/mit_states_data.py",
    "state_extract.py": "experiments/state_transfer/data/state_extract.py",
    "state_analysis.py": "experiments/state_transfer/evaluation/state_analysis.py",
    "state_identity_followup.py": "experiments/state_transfer/evaluation/state_identity_followup.py",
    "state_report.py": "experiments/state_transfer/reporting/state_report.py",
    "paper_pdf.py": "experiments/state_transfer/reporting/paper_pdf.py",
    "STATE_TRANSFER_PROTOCOL.md": "experiments/state_transfer/docs/STATE_TRANSFER_PROTOCOL.md",
    "STATE_FOLLOWUP_PROTOCOL.md": "experiments/state_transfer/docs/STATE_FOLLOWUP_PROTOCOL.md",
    "STATE_DATA_AUDIT.md": "experiments/state_transfer/docs/STATE_DATA_AUDIT.md",
    "STATE_TRANSFER_PAPER.md": "experiments/state_transfer/docs/STATE_TRANSFER_PAPER.md",
    "STATE_TRANSFER_PAPER_V2.md": "experiments/state_transfer/docs/STATE_TRANSFER_PAPER_V2.md",
    "configs/state_transfer.json": "experiments/state_transfer/configs/state_transfer.json",
    "tests/test_state_transfer.py": "experiments/state_transfer/tests/test_state_transfer.py",
    "tests/test_state_identity.py": "experiments/state_transfer/tests/test_state_identity.py",
    "RESEARCH_REPORT.md": "docs/history/RESEARCH_REPORT.md",
    "RESEARCH_REVIEW.md": "docs/history/RESEARCH_REVIEW.md",
    "results/heldout_layers": "experiments/heldout_layers/results",
    "results/robustness": "experiments/robustness/results",
    "results/paired_uncertainty": "experiments/paired_uncertainty/results",
    "results/difference_transfer": "experiments/difference_transfer/results",
    "results/endpoint_audit": "experiments/endpoint_audit/results",
    "results/state_transfer": "experiments/state_transfer/results",
    "results/baseline": "experiments/baseline/results/baseline",
    "results/pilot": "experiments/baseline/results/pilot",
    "results/semantic_main": "experiments/color_transfer/results/main",
    "results/semantic_pilot": "experiments/color_transfer/results/pilot",
    "output": "experiments/state_transfer/output",
}


def project_path(value):
    path = Path(value)
    if path.is_absolute():
        return path
    key = path.as_posix()
    for old in sorted(LEGACY_PATHS, key=len, reverse=True):
        if key == old or key.startswith(old + "/"):
            return PROJECT_ROOT / (LEGACY_PATHS[old] + key[len(old) :])
    return PROJECT_ROOT / path
