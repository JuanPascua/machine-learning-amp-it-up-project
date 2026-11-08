"""Config loading, seeding, and path helpers."""
import os
import random
from pathlib import Path

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]

# Loads the configuration yaml file into a python variable
def load_config(path):
    with open(path, "r") as file:
        config = yaml.safe_load(file)
    config["_config_path"] = str(path)
    return config

#Sets the seed of all three random generators to the same seed so that the seed used for generations and generations don't change depending on the person and device.
def set_seed(seed):
    """Seed every random source used in this project (no deep-learning libraries yet)."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

#Resolves the conflic if where the enviornment variable says the data is located is different from the root of the repo(where you are storing amp-it-up)
def resolve_data_dir(config):
    """DATA_DIR env var wins over the config file. Relative paths are relative to the repo root."""
    data_path = os.environ.get("DATA_DIR") or config["data"].get("data_dir")
    if not data_path:
        raise SystemExit(
            "No data directory set. Export DATA_DIR=/path/to/tonetwist_afx "
            "(see README, 'Data access')."
        )
    path = Path(data_path)
    return path if path.is_absolute() else REPO_ROOT / path

#Creates an absolute for relatives path using the repositories root as the root to create an absolute path from for the relative path.
def repo_path(relative_path):
    path = Path(relative_path)
    return path if path.is_absolute() else REPO_ROOT / path
