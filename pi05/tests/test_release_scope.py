from pathlib import Path


ROOT = Path(__file__).parents[1]


def test_release_keeps_final_v2_interaction() -> None:
    package = ROOT / "src" / "pi05_fabric"
    assert (package / "agents" / "dual_pi05.py").is_file()
    assert (package / "communication" / "paired_gemma.py").is_file()
    assert (package / "communication" / "residual_action.py").is_file()


def test_release_includes_robotwin2_without_backup_files() -> None:
    package = ROOT / "src" / "pi05_fabric"
    assert (package / "data" / "robotwin2_dataset.py").exists()
    assert (package / "evaluation" / "robotwin2_adapter.py").exists()
    assert not list(ROOT.rglob("*.orig"))
