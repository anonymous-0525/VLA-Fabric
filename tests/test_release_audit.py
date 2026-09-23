from pathlib import Path

from scripts.audit_release import audit_tree


def test_audit_reports_private_paths_and_generated_files(tmp_path: Path) -> None:
    (tmp_path / "config.yaml").write_text(
        "dataset: " + "/data" + "/private/example/dataset\n", encoding="utf-8"
    )
    (tmp_path / "checkpoint.pt").write_bytes(b"weights")

    violations = audit_tree(tmp_path)

    assert any("private path" in violation for violation in violations)
    assert any("generated artifact" in violation for violation in violations)


def test_audit_accepts_portable_source_tree(tmp_path: Path) -> None:
    (tmp_path / "config.yaml").write_text(
        "dataset: external/aloha_rlds\n", encoding="utf-8"
    )
    (tmp_path / "module.py").write_text("VALUE = 1\n", encoding="utf-8")

    assert audit_tree(tmp_path) == []


def test_audit_rejects_any_absolute_home_directory(tmp_path: Path) -> None:
    (tmp_path / "config.yaml").write_text(
        "weights: " + "/home" + "/example/model\n", encoding="utf-8"
    )

    assert any("private path" in item for item in audit_tree(tmp_path))


def test_audit_rejects_machine_specific_scratch_path(tmp_path: Path) -> None:
    (tmp_path / "config.yaml").write_text(
        "output: " + "/scratch" + "/example/run\n", encoding="utf-8"
    )

    assert any("private path" in item for item in audit_tree(tmp_path))


def test_audit_rejects_nonanonymous_license_attribution(tmp_path: Path) -> None:
    (tmp_path / "LICENSE").write_text(
        "Copyright (c) 2026 Unapproved Attribution\n", encoding="utf-8"
    )

    assert "unexpected license attribution" in audit_tree(tmp_path)
