"""Audit that every expanded optimizer group changes during the GPU gate."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
from pathlib import Path

import numpy as np

from pi05_fabric.agents.pi05_strong import FROZEN_GROUP
from pi05_fabric.agents.pi05_strong import native_parameter_group
from pi05_fabric.training.checkpoint import restore_training_state


def _leaves(tree, path=()):
    if isinstance(tree, dict):
        for key in sorted(tree):
            yield from _leaves(tree[key], (*path, key))
        return
    yield path, tree


def group_digests(params):
    groups = {}
    for path, value in _leaves(params):
        group = native_parameter_group(
            path,
            train_action_ffw=True,
            train_action_attention=True,
            train_paligemma_kv=True,
            train_paligemma_qo=True,
            separate_expanded_groups=True,
        )
        if group == FROZEN_GROUP:
            raise ValueError(f"expanded checkpoint contains frozen path: {'/'.join(path)}")
        record = groups.setdefault(group, {
            "digest": hashlib.sha256(),
            "tensor_count": 0,
            "parameter_count": 0,
        })
        array = np.asarray(value)
        record["digest"].update("/".join(path).encode("utf-8"))
        record["digest"].update(str(array.dtype).encode("ascii"))
        record["digest"].update(np.asarray(array.shape, dtype=np.int64).tobytes())
        record["digest"].update(array.tobytes(order="C"))
        record["tensor_count"] += 1
        record["parameter_count"] += int(array.size)
    return {
        group: {
            "sha256": record["digest"].hexdigest(),
            "tensor_count": record["tensor_count"],
            "parameter_count": record["parameter_count"],
        }
        for group, record in groups.items()
    }


def checkpoint_group_digests(path: Path):
    snapshot = restore_training_state(path)
    try:
        return group_digests(snapshot.params)
    finally:
        del snapshot
        gc.collect()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--before", type=Path, required=True)
    parser.add_argument("--after", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    before = checkpoint_group_digests(args.before)
    after = checkpoint_group_digests(args.after)
    if set(before) != set(after):
        raise ValueError("expanded checkpoint optimizer groups changed")
    unchanged = [group for group in before if before[group]["sha256"] == after[group]["sha256"]]
    if unchanged:
        raise ValueError(f"expanded optimizer groups did not update: {unchanged}")
    record = {
        "status": "PASS",
        "before": str(args.before.resolve()),
        "after": str(args.after.resolve()),
        "groups": {
            group: {
                "parameter_count": before[group]["parameter_count"],
                "tensor_count": before[group]["tensor_count"],
                "before_sha256": before[group]["sha256"],
                "after_sha256": after[group]["sha256"],
                "changed": True,
            }
            for group in sorted(before)
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(args.output)
    print(json.dumps(record, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
