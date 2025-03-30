import argparse
import os
import yaml
from typing import List

def load_config(config_file: str, params: List[str] = []):
    assert os.path.exists(config_file), f"Invalid config file: {config_file}"
    with open(config_file) as reader:
        config = yaml.safe_load(reader)
    # Parse overriden params.
    for param in params:
        fqn_key, value = param.split("=")
        entry_to_change = config
        keys = fqn_key.split(".")
        for k in keys[:-1]:
            entry_to_change = entry_to_change[k]
        entry_to_change[keys[-1]] = value
    return config