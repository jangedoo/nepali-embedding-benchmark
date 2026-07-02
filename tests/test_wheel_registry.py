from pathlib import Path


def test_wheel_configuration_bundles_registries() -> None:
    config = (Path(__file__).parents[1] / "pyproject.toml").read_text(encoding="utf-8")
    assert '"registries" = "neb/registries"' in config
