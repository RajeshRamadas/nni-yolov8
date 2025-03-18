#!/usr/bin/env python3
"""
Updates training configuration YAML file with parameters from YOLOv8 JSON configuration.
"""

import json
import yaml
import os
import logging
import argparse
from pathlib import Path
from typing import Dict, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

def load_config(file_path: str) -> Dict[str, Any]:
    """Load configuration from JSON or YAML file."""
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {file_path}")
    
    if file_path.suffix.lower() == '.json':
        with open(file_path, 'r') as f:
            return json.load(f)
    elif file_path.suffix.lower() in ['.yaml', '.yml']:
        with open(file_path, 'r') as f:
            return yaml.safe_load(f)
    else:
        raise ValueError(f"Unsupported file format: {file_path.suffix}")

def save_yaml_config(config: Dict[str, Any], file_path: str) -> None:
    """Save configuration to YAML file."""
    file_path = Path(file_path)
    
    # Create parent directories if they don't exist
    file_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(file_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    
    logger.info(f"Updated configuration saved to: {file_path}")

def update_training_config(yolo_config: Dict[str, Any], train_config: Dict[str, Any]) -> Dict[str, Any]:
    """Update training configuration with values from YOLOv8 configuration."""
    # Mapping of YOLO config keys to training config keys (if different)
    param_mapping = {
        "epochs": "epochs",
        "batch_size": "batch_size",
        "lr0": "learning_rate",
        "optimizer": "optimizer",
        "resolution": "img_size"
    }
    
    updated_config = train_config.copy()
    
    for yolo_key, train_key in param_mapping.items():
        if yolo_key in yolo_config:
            updated_config[train_key] = yolo_config[yolo_key]
            logger.debug(f"Updated {train_key}: {updated_config[train_key]}")
        else:
            logger.warning(f"Key '{yolo_key}' not found in YOLOv8 configuration")
    
    return updated_config

def main():
    parser = argparse.ArgumentParser(description="Update training configuration with YOLOv8 parameters")
    parser.add_argument("--yolo-config", default="../configs/best_yolov8_config.json", 
                        help="Path to YOLOv8 JSON configuration file")
    parser.add_argument("--train-config", default="../configs/train_config.yaml", 
                        help="Path to training YAML configuration file")
    args = parser.parse_args()
    
    # Get absolute paths based on script location
    base_dir = Path(__file__).parent.absolute()
    yolo_config_path = Path(args.yolo_config)
    train_config_path = Path(args.train_config)
    
    # Convert to absolute paths if relative paths were provided
    if not yolo_config_path.is_absolute():
        yolo_config_path = base_dir / yolo_config_path
    
    if not train_config_path.is_absolute():
        train_config_path = base_dir / train_config_path
    
    try:
        # Load configurations
        yolo_config = load_config(yolo_config_path)
        train_config = load_config(train_config_path)
        
        # Update training configuration
        updated_config = update_training_config(yolo_config, train_config)
        
        # Save updated configuration
        save_yaml_config(updated_config, train_config_path)
        
    except Exception as e:
        logger.error(f"Error updating configuration: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())