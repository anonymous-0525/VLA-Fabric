#!/usr/bin/env python3
"""Audit and merge the four RoboTwin2 validation shards."""

from __future__ import annotations

import argparse
import json

from pi05_fabric.evaluation.robotwin2_protocol import (
    load_condition_csv,
    merge_rollout_shards,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--conditions-file", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--purpose", required=True)
    parser.add_argument("sources", nargs="+")
    args = parser.parse_args()

    conditions = load_condition_csv(
        args.conditions_file, expected_ids=tuple(range(200))
    )
    summary = merge_rollout_shards(
        args.sources,
        output_dir=args.output_dir,
        conditions=conditions,
        purpose=args.purpose,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
