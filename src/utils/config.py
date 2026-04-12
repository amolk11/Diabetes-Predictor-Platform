"""Configuration management"""
import yaml
from pathlib import Path

def load_config(config_path="config/config.yaml"):
    """Load YAML configuration"""
    config_file = Path(config_path)
    
    if not config_file.exists():
        raise FileNotFoundError(f"Config not found: {config_file}")
    
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    
    return config
