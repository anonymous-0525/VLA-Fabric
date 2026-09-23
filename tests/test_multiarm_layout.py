from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_multiarm_documentation_and_configs_are_present() -> None:
    required = (
        ROOT / "docs" / "multiarm-hpc.md",
        ROOT / "pi05" / "scripts" / "multiarm" / "env.example",
        ROOT / "pi05" / "scripts" / "multiarm" / "launch_agent_ranks.sh",
        ROOT / "pi05" / "configs" / "multiarm" / "three_arm_stack_cube.yaml",
        ROOT / "pi05" / "configs" / "multiarm" / "four_arm_frame_insertion.yaml",
        ROOT / "pi05" / "configs" / "multiarm" / "four_arm_arch_assembly.yaml",
    )
    assert all(path.is_file() for path in required)

