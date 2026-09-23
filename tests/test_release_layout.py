from pathlib import Path


ROOT = Path(__file__).parents[1]


def test_release_has_two_scoped_packages() -> None:
    assert (ROOT / "eagle" / "src" / "commvla").is_dir()
    assert (ROOT / "pi05" / "src" / "pi05_fabric").is_dir()


def test_unfinished_and_infocom_modules_are_absent() -> None:
    forbidden = (
        ROOT / "eagle" / "src" / "commvla" / "models" / "multi_arm.py",
        ROOT / "eagle" / "src" / "commvla" / "communication" / "nspr.py",
        ROOT / "pi05" / "src" / "pi05_fabric" / "data" / "robotwin2_dataset.py",
    )
    assert all(not path.exists() for path in forbidden)


def test_public_documentation_is_present() -> None:
    required = (
        ROOT / "README.md",
        ROOT / "docs" / "installation.md",
        ROOT / "docs" / "datasets.md",
        ROOT / "docs" / "reproducibility.md",
    )
    assert all(path.is_file() for path in required)
