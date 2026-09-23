from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "src" / "pi05_fabric"


def test_multiarm_release_modules_exist() -> None:
    required = (
        PACKAGE / "agents" / "multiagent_pi05.py",
        PACKAGE / "communication" / "multiagent_collectives.py",
        PACKAGE / "communication" / "multiagent_gemma.py",
        PACKAGE / "training" / "agent_parallel.py",
        PACKAGE / "training" / "multiagent_checkpoint.py",
        PACKAGE / "data" / "stack_cube_multiagent.py",
        PACKAGE / "data" / "four_arm_tasks.py",
        PACKAGE / "evaluation" / "stack_cube_protocol.py",
        PACKAGE / "evaluation" / "four_arm.py",
    )
    assert all(path.is_file() for path in required)


def test_private_workspace_artifacts_remain_absent() -> None:
    assert not list(ROOT.rglob("*.orig"))
    assert not (PACKAGE / "data" / "robotwin2_dataset.py").exists()
    assert not (PACKAGE / "evaluation" / "robotwin2_policy.py").exists()

