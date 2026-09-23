#!/usr/bin/env python3
"""Audit one formal training checkpoint before a dependent stage starts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from pi05_fabric.training.audit import audit_training_checkpoint


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--expected-stage", required=True)
    parser.add_argument("--expected-step", type=int, required=True)
    parser.add_argument("--expected-parent-model-sha256")
    parser.add_argument("--skip-model-verify", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    manifest = audit_training_checkpoint(
        args.checkpoint,
        expected_stage=args.expected_stage,
        expected_step=args.expected_step,
        verify_model=not args.skip_model_verify,
        expected_parent_model_sha256=args.expected_parent_model_sha256,
    )
    print(
        json.dumps(
            {
                "status": "PASS",
                "checkpoint": str(args.checkpoint.resolve()),
                "schema_version": manifest["schema_version"],
                "stage": manifest["stage"],
                "step": manifest["step"],
                "model_sha256": manifest["model_sha256"],
                "parent_model_sha256": manifest["parent_model_sha256"],
            },
            indent=2,
            sort_keys=True,
        )
    )

if __name__ == "__main__":
    main()
