import os
import torch
import yaml
from pathlib import Path
from ultralytics import YOLO
import logging
import argparse
import sys
from typing import Dict, Any, Optional, Union

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("model_retraining.log")
    ]
)
logger = logging.getLogger(__name__)

# Get the base directory (parent of src)
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Set paths for config, dataset, and models based on correct directory structure
configs_dir = os.path.join(base_dir, "configs")
datasets_dir = os.path.join(base_dir, "datasets")
default_config_path = os.path.join(configs_dir, "train_config.yaml")


def load_config(config_path: str = default_config_path) -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Args:
        config_path (str): Path to the configuration file.
        
    Returns:
        dict: Configuration parameters.
        
    Raises:
        FileNotFoundError: If the configuration file is not found.
    """
    config_file = Path(config_path)
    
    if not config_file.exists():
        logger.error(f"Configuration file not found at {config_path}")
        raise FileNotFoundError(f"Configuration file not found at {config_path}")
    
    try:
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
        
        logger.info(f"Loaded configuration from {config_path}")
        return config
    except yaml.YAMLError as e:
        logger.error(f"Error parsing YAML configuration: {str(e)}")
        raise ValueError(f"Invalid YAML format in configuration file: {str(e)}")
    except Exception as e:
        logger.error(f"Unexpected error loading configuration: {str(e)}")
        raise


def find_dataset_yaml(dataset_dir: str = datasets_dir) -> str:
    """
    Search for a 'data.yaml' file within the dataset directory.
    
    Args:
        dataset_dir (str): The directory where the dataset is expected to be located.
    
    Returns:
        str: The path to the 'data.yaml' file if found.
    
    Raises:
        FileNotFoundError: If no 'data.yaml' file is found in the dataset directory.
    """
    dataset_path = Path(dataset_dir)

    if not dataset_path.exists():
        logger.error(f"Dataset directory '{dataset_dir}' not found.")
        raise FileNotFoundError(f"Dataset directory '{dataset_dir}' not found.")

    yaml_files = list(dataset_path.rglob("data.yaml"))
    if not yaml_files:
        logger.error(f"No 'data.yaml' file found in '{dataset_dir}'.")
        raise FileNotFoundError(f"No 'data.yaml' file found in '{dataset_dir}'.")

    data_yaml_path = str(yaml_files[0])  # Use the first found data.yaml file
    absolute_data_yaml_path = os.path.abspath(data_yaml_path)
    logger.info(f"Found dataset YAML: {absolute_data_yaml_path}")

    # Validate dataset structure using proper YAML parsing
    try:
        with open(data_yaml_path, "r") as f:
            yaml_data = yaml.safe_load(f)
        
        # Check if the required keys exist
        if not isinstance(yaml_data, dict):
            raise ValueError(f"Invalid data.yaml format: expected a dictionary")
            
        if "train" not in yaml_data or "val" not in yaml_data:
            logger.error("Dataset YAML missing required 'train' or 'val' keys")
            raise ValueError("Dataset YAML missing required 'train' or 'val' keys")
        
        # Get train and val paths
        train_path = yaml_data["train"]
        val_path = yaml_data["val"]
        
        # Make paths absolute if they are relative
        yaml_dir = os.path.dirname(data_yaml_path)
        
        # Check if paths are already absolute
        train_full_path = train_path if os.path.isabs(train_path) else os.path.join(yaml_dir, train_path)
        val_full_path = val_path if os.path.isabs(val_path) else os.path.join(yaml_dir, val_path)
        
        # Verify paths exist
        if not os.path.exists(train_full_path):
            logger.error(f"Training dataset path not found: {train_full_path}")
            raise FileNotFoundError(f"Training dataset path not found: {train_full_path}")
            
        if not os.path.exists(val_full_path):
            logger.error(f"Validation dataset path not found: {val_full_path}")
            raise FileNotFoundError(f"Validation dataset path not found: {val_full_path}")
            
        logger.info(f"Validated dataset paths - Train: {train_full_path}, Val: {val_full_path}")
        
    except yaml.YAMLError as e:
        logger.error(f"Error parsing dataset YAML: {str(e)}")
        raise ValueError(f"Invalid YAML format in dataset file: {str(e)}")
    except Exception as e:
        logger.error(f"Error validating dataset structure: {str(e)}")
        raise

    return absolute_data_yaml_path


def select_device(device_preference: str) -> str:
    """
    Select the appropriate device for training based on preference and availability.
    
    Args:
        device_preference (str): The preferred device ('cuda', 'cpu', or a specific GPU index)
        
    Returns:
        str: The selected device string for YOLO training
    """
    if device_preference.lower() == 'cpu':
        logger.info("Using CPU for training as specified")
        return 'cpu'
    
    # Check if CUDA is available when GPU is requested
    if device_preference.lower() in ['cuda', '0'] or device_preference.isdigit():
        if not torch.cuda.is_available():
            logger.warning("CUDA requested but not available, falling back to CPU")
            return 'cpu'
        
        # If specific GPU index is requested, check if it exists
        if device_preference.isdigit() and int(device_preference) >= torch.cuda.device_count():
            logger.warning(f"GPU {device_preference} requested but only {torch.cuda.device_count()} GPUs available")
            logger.warning(f"Using GPU 0 instead")
            return '0'
            
        device = device_preference if device_preference.lower() != 'cuda' else '0'
        gpu_name = torch.cuda.get_device_name(int(device))
        logger.info(f"Using GPU {device} ({gpu_name}) for training")
        return device
    
    # Default to CPU for unrecognized options
    logger.warning(f"Unrecognized device option '{device_preference}', falling back to CPU")
    return 'cpu'
    
    
def retrain_yolo_model(
    config: Dict[str, Any],
    data_yaml_path: str
) -> str:
    """
    Retrains a YOLO model using the specified parameters from the config.
    
    Args:
        config (dict): Configuration parameters.
        data_yaml_path (str): Path to the dataset YAML file.
        
    Returns:
        str: Path to the saved retrained model.
    
    Raises:
        FileNotFoundError: If the base model is not found.
        RuntimeError: If training fails.
    """
    # Extract parameters from config with proper type checking
    saved_models_dir = os.path.join(base_dir, config.get("saved_models_dir", "saved_models"))
    base_model_name = config.get("base_model_name", "best_yolov8_model.pt")
    output_model_name = config.get("output_model_name", "retrained_yolov8_model.pt")
    
    # Ensure numeric parameters have appropriate types
    epochs = int(config.get("epochs", 20))
    batch_size = int(config.get("batch_size", 16))
    img_size = int(config.get("img_size", 640))
    learning_rate = float(config.get("learning_rate", 0.01))
    
    optimizer = config.get("optimizer", "Adam")
    device_preference = config.get("device", "cpu")
    
    # Select appropriate device
    device = select_device(device_preference)
    
    # Create saved_models directory if it doesn't exist
    os.makedirs(saved_models_dir, exist_ok=True)
    logger.info(f"Ensuring models directory exists: {saved_models_dir}")
    
    # Set paths
    base_model_path = os.path.join(saved_models_dir, base_model_name)
    retrained_model_path = os.path.join(saved_models_dir, output_model_name)
    
    # Check if the base model exists
    if not os.path.exists(base_model_path):
        logger.error(f"Base model not found at {base_model_path}")
        raise FileNotFoundError(f"Base model not found at {base_model_path}")
    
    logger.info(f"Loading base model from: {base_model_path}")
    
    try:
        # Load the model for retraining
        model = YOLO(base_model_path)
        
        # Log training parameters
        logger.info(f"Starting retraining with: epochs={epochs}, batch={batch_size}, img_size={img_size}, lr={learning_rate}")
        
        # Set up augmentation parameters from config if available
        augmentation_args = {}
        aug_params = ["mosaic", "fliplr", "scale", "hsv_h", "hsv_s", "hsv_v"]
        for aug_param in aug_params:
            if aug_param in config:
                augmentation_args[aug_param] = config[aug_param]
                
        # Log augmentation settings if any
        if augmentation_args:
            logger.info(f"Using augmentation settings: {augmentation_args}")
        
        # Setup training parameters
        train_args = {
            "data": data_yaml_path,
            "epochs": epochs,
            "batch": batch_size,
            "imgsz": img_size,
            "lr0": learning_rate,
            "optimizer": optimizer,
            "device": device,
            "pretrained": True,  # Fine-tune using base weights
            "cache": config.get("cache", False),  # Cache images for faster training
            "resume": config.get("resume", False)  # Resume training if interrupted
        }
        
        # Add optional parameters if in config
        optional_params = ["workers", "patience", "val", "iou", "cos_lr", "project", "name"]
        for param in optional_params:
            if param in config:
                train_args[param] = config[param]
        
        # Merge augmentation parameters
        train_args.update(augmentation_args)
            
        # Start retraining
        logger.info("Beginning model training")
        results = model.train(**train_args)
        
        # Check training results
        if results is None:
            raise RuntimeError("Training failed with no results returned")
        
        # Save retrained model
        model.save(retrained_model_path)
        logger.info(f"Retrained model saved at: {retrained_model_path}")
        
        # Log additional metrics if available
        if hasattr(results, "results_dict"):
            metrics = results.results_dict
            logger.info(f"Final training metrics: {metrics}")
        
        return retrained_model_path
    
    except Exception as e:
        logger.error(f"Error during model retraining: {str(e)}", exc_info=True)
        raise


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Retrain YOLO model with custom dataset")
    parser.add_argument("--config", type=str, default=default_config_path,
                        help=f"Path to config file (default: {default_config_path})")
    parser.add_argument("--dataset", type=str, default=datasets_dir,
                        help=f"Path to dataset directory (default: {datasets_dir})")
    parser.add_argument("--device", type=str,
                        help="Override device setting from config (e.g., 'cpu', '0', '1')")
    parser.add_argument("--epochs", type=int, 
                        help="Override epochs setting from config")
    parser.add_argument("--batch", type=int, 
                        help="Override batch size setting from config")
    return parser.parse_args()


if __name__ == "__main__":
    try:
        # Parse command line arguments
        args = parse_arguments()
        
        # Load configuration
        config = load_config(args.config)
        
        # Override config with command line arguments if provided
        if args.device:
            config["device"] = args.device
            logger.info(f"Overriding device from command line: {args.device}")
        if args.epochs:
            config["epochs"] = args.epochs
            logger.info(f"Overriding epochs from command line: {args.epochs}")
        if args.batch:
            config["batch_size"] = args.batch
            logger.info(f"Overriding batch size from command line: {args.batch}")
        
        # Use the dataset path from command line or default
        data_path = find_dataset_yaml(args.dataset)
        logger.info(f"Using dataset: {data_path}")
        
        # Record environment info
        logger.info(f"Python version: {sys.version}")
        logger.info(f"PyTorch version: {torch.__version__}")
        if torch.cuda.is_available():
            logger.info(f"CUDA available: Yes, version {torch.version.cuda}")
            logger.info(f"GPU(s): {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                logger.info(f"  {i}: {torch.cuda.get_device_name(i)}")
        else:
            logger.info("CUDA available: No")
        
        # Retrain the model using config parameters
        retrained_model = retrain_yolo_model(
            config=config,
            data_yaml_path=data_path
        )
        print(f"✅ Retrained YOLO model saved at {retrained_model}")
        
    except KeyboardInterrupt:
        logger.warning("Training interrupted by user")
        print("❌ Training interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Retraining failed: {str(e)}", exc_info=True)
        print(f"❌ Retraining failed: {str(e)}")
        sys.exit(1)