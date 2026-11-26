"""Pipeline configuration loading."""

from pathlib import Path
from typing import Any

import yaml


def load_pipeline_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML pipeline configuration and require a top-level mapping."""
    config_path = Path(path)
    if not config_path.is_file():
        raise FileNotFoundError(f"pipeline config does not exist: {config_path}")
    try:
        with config_path.open(encoding="utf-8") as config_file:
            config = yaml.safe_load(config_file)
    except yaml.YAMLError as exc:
        raise ValueError(f"invalid YAML pipeline config: {config_path}") from exc
    if not isinstance(config, dict):
        raise ValueError("pipeline config must contain a YAML mapping")
    return config
