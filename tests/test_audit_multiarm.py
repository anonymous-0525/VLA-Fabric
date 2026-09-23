from pathlib import Path

from scripts.audit_release import audit_tree


def test_audit_rejects_internal_robotwin2_modules(tmp_path: Path) -> None:
    module = (
        tmp_path
        / "pi05"
        / "src"
        / "pi05_fabric"
        / "evaluation"
        / "robotwin2_policy.py"
    )
    module.parent.mkdir(parents=True)
    module.write_text("VALUE = 1\n", encoding="utf-8")

    assert any("forbidden out-of-scope module" in item for item in audit_tree(tmp_path))

