#!/usr/bin/env python3
"""Validate a canonical multi-arm HDF5 episode and print JSON."""

import argparse
import json
import sys
from pathlib import Path

from multi_arm_episode import validate_episode


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("episode", type=Path)
    args = parser.parse_args(argv)
    try:
        summary = validate_episode(args.episode)
    except Exception as error:
        print(f"INVALID: {error}", file=sys.stderr)
        return 1
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
