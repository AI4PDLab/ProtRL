import yaml
import os

def save_config(config, path):
    """Saves a dictionary as a YAML file."""
    with open(path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
