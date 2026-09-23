#!/usr/bin/env python3
"""Reject files that do not belong in the public code release."""

from __future__ import annotations

import re
import sys
from pathlib import Path


MAX_FILE_BYTES = 10 * 1024 * 1024
TEXT_SUFFIXES = {
    ".cfg",
    ".ini",
    ".json",
    ".md",
    ".py",
    ".sh",
    ".toml",
    ".txt",
    ".yaml",
    ".yml",
}
GENERATED_SUFFIXES = {".ckpt", ".orig", ".pt", ".pth", ".safetensors"}
FORBIDDEN_DIRS = {
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    "__pycache__",
    "artifacts",
    "checkpoints",
    "evaluations",
    "logs",
}
FORBIDDEN_MODULES = {
    "eagle/src/commvla/communication/nspr.py",
    "eagle/src/commvla/models/multi_arm.py",
    "pi05/src/pi05_fabric/data/robotwin2_dataset.py",
    "pi05/src/pi05_fabric/evaluation/robotwin2_adapter.py",
    "pi05/src/pi05_fabric/evaluation/robotwin2_ipc.py",
    "pi05/src/pi05_fabric/evaluation/robotwin2_policy.py",
    "pi05/src/pi05_fabric/evaluation/robotwin2_protocol.py",
}
PRIVATE_PATH_TOKENS = (
    "/data" + "/private",
    "/home" + "/",
    "/scratch" + "/",
    "/gpfs" + "/",
    "/lustre" + "/",
)
CREDENTIAL_PATTERNS = (
    re.compile(r"github_pat_[A-Za-z0-9_]{20,}"),
    re.compile(r"ghp_[A-Za-z0-9]{20,}"),
    re.compile(r"hf_[A-Za-z0-9]{20,}"),
    re.compile(r"AKIA[0-9A-Z]{16}"),
    re.compile(r"BEGIN (?:RSA|OPENSSH|EC) PRIVATE KEY"),
)


def _relative(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def audit_tree(root: Path) -> list[str]:
    root = root.resolve()
    violations: list[str] = []

    for path in sorted(root.rglob("*")):
        relative = _relative(path, root)
        parts = set(path.relative_to(root).parts)
        if ".git" in parts:
            continue
        bad_dirs = parts & FORBIDDEN_DIRS
        if bad_dirs:
            violations.append(
                f"forbidden generated directory ({sorted(bad_dirs)[0]}): {relative}"
            )
            continue
        if path.is_dir():
            continue
        if relative in FORBIDDEN_MODULES:
            violations.append(f"forbidden out-of-scope module: {relative}")
        if path.suffix in GENERATED_SUFFIXES:
            violations.append(f"generated artifact: {relative}")
        if path.stat().st_size > MAX_FILE_BYTES:
            violations.append(f"file exceeds 10 MiB: {relative}")
        if path.suffix not in TEXT_SUFFIXES and path.name != "LICENSE":
            continue
        try:
            content = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            violations.append(f"non-UTF-8 source/configuration file: {relative}")
            continue
        if any(token in content for token in PRIVATE_PATH_TOKENS):
            violations.append(f"private path in {relative}")
        if relative == "LICENSE":
            copyright_lines = [
                line for line in content.splitlines() if line.startswith("Copyright (c)")
            ]
            if copyright_lines != ["Copyright (c) 2026 VLA-Fabric Authors"]:
                violations.append("unexpected license attribution")
        if any(pattern.search(content) for pattern in CREDENTIAL_PATTERNS):
            violations.append(f"credential-like content in {relative}")

    return violations


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    violations = audit_tree(root)
    if violations:
        for violation in violations:
            print(violation)
        return 1
    print("release audit: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
