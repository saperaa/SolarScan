"""Tests for YAML pipeline configuration loading."""

import pytest

from src.config import load_pipeline_config


def test_load_pipeline_config_requires_yaml_mapping(tmp_path):
    config_path = tmp_path / "pipeline.yaml"
    config_path.write_text("name: pipeline_a\nseed: 2\n", encoding="utf-8")

    assert load_pipeline_config(config_path) == {"name": "pipeline_a", "seed": 2}


@pytest.mark.parametrize("contents", ["- a\n- b\n", "null\n", "name: ["])
def test_load_pipeline_config_rejects_invalid_documents(tmp_path, contents):
    config_path = tmp_path / "pipeline.yaml"
    config_path.write_text(contents, encoding="utf-8")

    with pytest.raises(ValueError):
        load_pipeline_config(config_path)


def test_load_pipeline_config_requires_existing_file(tmp_path):
    with pytest.raises(FileNotFoundError, match="config"):
        load_pipeline_config(tmp_path / "missing.yaml")
