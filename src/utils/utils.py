import logging
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict

import yaml

# Constants should be uppercase
PROJECT_ROOT = Path(__file__).parents[2]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


@lru_cache(maxsize=None)
def load_config(config_file: str) -> Dict[str, Any]:
    """Load configuration from YAML file.

    Args:
        config_file: Name of the YAML configuration file

    Returns:
        Dictionary containing configuration

    Raises:
        FileNotFoundError: If configuration file doesn't exist
        yaml.YAMLError: If YAML parsing fails
    """
    config_path = PROJECT_ROOT / "config" / config_file
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except FileNotFoundError:
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    except yaml.YAMLError as e:
        raise yaml.YAMLError(f"YAML parsing error: {e}")
