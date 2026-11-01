"""Run an experiment stage from any working directory; use --list for commands."""

import argparse
import runpy
import sys
import os
from shared.paths import PROJECT_ROOT

COMMANDS = {
    "baseline": "experiments.baseline.data.experiment",
    "baseline-language": "experiments.baseline.evaluation.language_eval",
    "baseline-verify": "experiments.baseline.evaluation.verify_extraction",
    "baseline-report": "experiments.baseline.reporting.report",
    "heldout-layers": "experiments.heldout_layers.evaluation.heldout_layers",
    "robustness": "experiments.robustness.evaluation.robustness",
    "paired-uncertainty": "experiments.paired_uncertainty.evaluation.paired_uncertainty",
    "difference-vectors": "experiments.difference_transfer.evaluation.difference_transfer",
    "endpoint-audit": "experiments.endpoint_audit.evaluation.endpoint_audit",
    "state-data": "experiments.state_transfer.data.state_data",
    "state-extract": "experiments.state_transfer.data.state_extract",
    "state-evaluate": "experiments.state_transfer.evaluation.state_analysis",
    "state-followup": "experiments.state_transfer.evaluation.state_identity_followup",
    "state-report": "experiments.state_transfer.reporting.state_report",
    "state-pdf": "experiments.state_transfer.reporting.paper_pdf",
    "color-data": "experiments.color_transfer.data.quickdraw_data",
    "color-extract": "experiments.color_transfer.data.semantic_extract",
    "color-evaluate": "experiments.color_transfer.evaluation.semantic_analysis",
    "color-planning": "experiments.color_transfer.evaluation.pilot_planning",
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list", action="store_true")
    parser.add_argument("command", nargs="?", choices=COMMANDS)
    # Forward stage flags (including --help) without consuming them here.
    if len(sys.argv) > 1 and sys.argv[1] in COMMANDS:
        args = argparse.Namespace(command=sys.argv[1], list=False)
        remaining = sys.argv[2:]
    else:
        args, remaining = parser.parse_known_args()
    if args.list or args.command is None:
        for command, module in COMMANDS.items():
            print(f"{command:20} {module}")
        return
    os.chdir(PROJECT_ROOT)
    sys.argv = [args.command, *remaining]
    runpy.run_module(COMMANDS[args.command], run_name="__main__")


if __name__ == "__main__":
    main()
