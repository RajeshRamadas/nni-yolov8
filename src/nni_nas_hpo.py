import torch
import torch.nn.utils.prune as prune
import nni
from ultralytics import YOLO
import os
from pathlib import Path
import sys
import traceback
import time
import json
import logging
import yaml
import shutil
import random

# Set up logging (avoid using emoji characters for Windows compatibility)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Get the base directory (parent of src)
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Set paths for config, dataset, and models based on correct directory structure
configs_dir = os.path.join(base_dir, "configs")
datasets_dir = os.path.join(base_dir, "datasets")
saved_models_dir = os.path.join(base_dir, "saved_models")

def apply_pruning(model, amount=0.3):
    """Apply pruning to model"""
    try:
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Conv2d):
                prune.l1_unstructured(module, name="weight", amount=amount)
        logger.info(f"Successfully applied pruning with amount {amount}")
        return model
    except Exception as e:
        logger.error(f"Error during pruning: {e}")
        return model  # Return original model if pruning fails

def apply_quantization(model):
    """Apply quantization to model"""
    try:
        quantized_model = torch.quantization.quantize_dynamic(
            model, {torch.nn.Linear, torch.nn.Conv2d}, dtype=torch.qint8
        )
        logger.info("Successfully applied quantization")
        return quantized_model
    except Exception as e:
        logger.error(f"Error during quantization: {e}")
        return model  # Return original model if quantization fails

def modify_kernel_size(model, kernel_size):
    """Modify kernel size of Conv2D layers dynamically"""
    try:
        for name, module in model.model.named_modules():
            if isinstance(module, torch.nn.Conv2d):
                module.kernel_size = (kernel_size, kernel_size)
        logger.info(f"Applied kernel size {kernel_size} to model.")
    except Exception as e:
        logger.error(f"Error modifying kernel size: {e}")

def save_best_config(params, map50, output_folder=configs_dir):
    """Save the best YOLOv8 NAS + hyperparameters configuration in a folder."""
    os.makedirs(output_folder, exist_ok=True)
    output_file = os.path.join(output_folder, "best_yolov8_config.json")
    best_config = params.copy()
    best_config["mAP@50"] = map50  # Save model performance
    with open(output_file, "w") as f:
        json.dump(best_config, f, indent=4)
    logger.info(f"Best YOLOv8 config saved to {output_file}")


def find_dataset_yaml(dataset_dir=datasets_dir):
    """
    Dynamically find `data.yaml` inside the dataset directory and resolve paths.

    Args:
        dataset_dir (str): Base directory where dataset folders are stored.

    Returns:
        str: The absolute path of the fixed `data.yaml`.

    Raises:
        FileNotFoundError: If no `data.yaml` file is found.
    """
    dataset_path = Path(dataset_dir)

    if not dataset_path.exists():
        logger.error(f"Dataset directory '{dataset_dir}' not found.")
        raise FileNotFoundError(f"Dataset directory '{dataset_dir}' not found.")

    # Find any data.yaml file inside datasets/
    yaml_files = list(dataset_path.rglob("data.yaml"))
    if not yaml_files:
        logger.error(f"No 'data.yaml' file found in '{dataset_dir}'.")
        raise FileNotFoundError(f"No 'data.yaml' file found in '{dataset_dir}'.")

    data_yaml_path = yaml_files[0].resolve()  # Use the first found data.yaml
    logger.info(f"Using dataset YAML: {data_yaml_path}")

    # Load and fix paths in YAML
    with open(data_yaml_path, "r") as f:
        yaml_data = yaml.safe_load(f)

    base_path = data_yaml_path.parent  # Get the folder containing data.yaml

    # Convert all paths to absolute paths
    yaml_data["train"] = str((base_path / yaml_data["train"]).resolve())
    yaml_data["val"] = str((base_path / yaml_data["val"]).resolve())
    yaml_data["test"] = str((base_path / yaml_data["test"]).resolve())

    # Save updated YAML
    with open(data_yaml_path, "w") as f:
        yaml.safe_dump(yaml_data, f)

    return str(data_yaml_path)

def train_with_nni():
    """Train YOLOv8 with NNI parameters"""
    try:
        logger.info(f"Python version: {sys.version}")
        logger.info(f"PyTorch version: {torch.__version__}")
        logger.info(f"CUDA available: {torch.cuda.is_available()}")
        logger.info(f"Current working directory: {os.getcwd()}")
        logger.info(f"Base directory: {base_dir}")
        
        # Create saved_models directory if it doesn't exist
        os.makedirs(saved_models_dir, exist_ok=True)
        logger.info(f"Ensuring models directory exists: {saved_models_dir}")
        
        # Use the dynamically found dataset YAML file
        data_path = find_dataset_yaml()
        logger.info(f"Using dataset: {data_path}")
        
        search_space_path = os.path.join(base_dir, "configs", "search_space.json")
        
        # MODIFIED SECTION: Parameter handling
        try:
            if os.path.exists(search_space_path):
                with open(search_space_path, "r") as f:
                    search_space = json.load(f)
                    logger.info(f"Loaded search space: {search_space}")
                    
                    # Check if NNI is properly initialized
                    try:
                        params = nni.get_next_parameter()
                        # If params is empty (empty dict), generate parameters from search space
                        if not params:
                            logger.warning("NNI returned empty parameters, generating from search space manually")
                            
                            # Generate parameters from search space
                            params = {
                                "batch_size": random.choice(search_space["batch_size"]["_value"]),
                                "epochs": random.choice(search_space["epochs"]["_value"]),
                                "resolution": random.choice(search_space["resolution"]["_value"]),
                                "lr0": random.uniform(search_space["lr0"]["_value"][0], search_space["lr0"]["_value"][1]),
                                "optimizer": random.choice(search_space["optimizer"]["_value"]),
                                "pruning_amount": random.uniform(search_space["pruning_amount"]["_value"][0], 
                                                              search_space["pruning_amount"]["_value"][1]),
                                "apply_quantization": random.choice(search_space["apply_quantization"]["_value"]),
                                "kernel_size": random.choice(search_space["kernel_size"]["_value"]),
                                "nas_method": random.choice(search_space["nas_method"]["_value"]),
                                "depth_multiple": random.choice(search_space["depth_multiple"]["_value"]),
                                "width_multiple": random.choice(search_space["width_multiple"]["_value"])
                            }
                            logger.info(f"Generated parameters from search space: {params}")
                    except Exception as nni_e:
                        logger.error(f"Error with NNI initialization: {nni_e}")
                        # Generate parameters from search space if NNI fails completely
                        
                        params = {
                            "batch_size": random.choice(search_space["batch_size"]["_value"]),
                            "epochs": random.choice(search_space["epochs"]["_value"]),
                            "resolution": random.choice(search_space["resolution"]["_value"]),
                            "lr0": random.uniform(search_space["lr0"]["_value"][0], search_space["lr0"]["_value"][1]),
                            "optimizer": random.choice(search_space["optimizer"]["_value"]),
                            "pruning_amount": random.uniform(search_space["pruning_amount"]["_value"][0], 
                                                          search_space["pruning_amount"]["_value"][1]),
                            "apply_quantization": random.choice(search_space["apply_quantization"]["_value"]),
                            "kernel_size": random.choice(search_space["kernel_size"]["_value"]),
                            "nas_method": random.choice(search_space["nas_method"]["_value"]),
                            "depth_multiple": random.choice(search_space["depth_multiple"]["_value"]),
                            "width_multiple": random.choice(search_space["width_multiple"]["_value"])
                        }
                        logger.info(f"Generated parameters from search space (NNI failed): {params}")
            else:
                logger.warning(f"Search space file not found at {search_space_path}")
                raise FileNotFoundError(f"Search space file not found")
                
        except Exception as e:
            logger.error(f"Error getting parameters: {e}")
            params = {
                "batch_size": 8,
                "epochs": 30,
                "resolution": 320,
                "lr0": 0.01,
                "optimizer": "Adam",
                "pruning_amount": 0.3,
                "apply_quantization": False,
                "kernel_size": 3,
                "nas_method": "TPE",
                "depth_multiple": 0.33,
                "width_multiple": 0.5
            }
            logger.info(f"Using default parameters: {params}")
        # END OF MODIFIED SECTION
        
        nas_method = params.get("nas_method", "TPE")
        logger.info(f"Using NAS method: {nas_method}")
        
        try:
            trial_id = nni.get_trial_id()
            logger.info(f"Current trial ID: {trial_id}")
        except Exception as e:
            logger.error(f"Error getting trial ID: {e}")
            trial_id = f"trial_{int(time.time())}"
        
        try:
            logger.info("Loading YOLOv8 model...")
            # Check for model in different potential locations
            model_path = "yolov8n.pt"
            
            # Check if model exists in base directory
            if not os.path.exists(model_path):
                # Check if model exists in saved_models directory
                saved_model_path = os.path.join(saved_models_dir, "yolov8n.pt")
                if os.path.exists(saved_model_path):
                    model_path = saved_model_path
            
            model = YOLO(model_path)
            logger.info(f"Model loaded successfully from {model_path}")
            
            initial_model_path = os.path.join(saved_models_dir, f"{trial_id}_initial.pt")
            model.save(initial_model_path)
            logger.info(f"Initial model saved as {initial_model_path}")
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            traceback.print_exc()
            raise
        
        img_size = params.get('resolution', 640)
        batch_size = params.get('batch_size', 8)
        epochs = params.get('epochs', 3)
        kernel_size = params.get('kernel_size', 3)
        
        # Determine device - prefer GPU if available
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {device}")
        
        logger.info(f"Training configuration:")
        logger.info(f"  - Image size: {img_size}")
        logger.info(f"  - Batch size: {batch_size}")
        logger.info(f"  - Epochs: {epochs}")
        logger.info(f"  - Kernel Size: {kernel_size}")
        logger.info(f"  - NAS Method: {nas_method}")
        logger.info(f"  - Device: {device}")
        logger.info(f"  - Data path: {data_path}")
        
        modify_kernel_size(model, kernel_size)
        
        # Define early stopping threshold based on dataset complexity
        # You may need to adjust this based on your specific dataset
        early_stopping_threshold = 0.3  # Example: Stop if mAP50 is below 0.3 after several epochs
        
        # Track metrics
        all_metrics = []
        best_map50 = 0.0
        best_epoch = 0
        
        # NEW SECTION: Train epoch by epoch with early stopping
        logger.info(f"Starting epoch-by-epoch training with early stopping")
        
        for epoch in range(1, epochs + 1):
            logger.info(f"Starting epoch {epoch}/{epochs}")
            
            # Train for a single epoch
            results = model.train(
                data=data_path,
                epochs=1,  # Train for just one epoch at a time
                batch=batch_size,
                imgsz=img_size,
                lr0=params.get("lr0", 0.01),
                optimizer=params.get("optimizer", "Adam"),
                device=device,
                resume=True if epoch > 1 else False,  # Resume after first epoch
                verbose=True
            )
            
            # Validate after each epoch
            logger.info(f"Validating after epoch {epoch}...")
            val_results = model.val(data=data_path)
            
            # Extract map50 value
            map50 = float(val_results.box.map50)
            logger.info(f"Epoch {epoch} mAP@50: {map50}")
            
            # Store metric
            all_metrics.append(map50)
            
            # Update best metrics
            if map50 > best_map50:
                best_map50 = map50
                best_epoch = epoch
                # Save best model
                best_model_path = os.path.join(saved_models_dir, f"{trial_id}_best_epoch_{epoch}.pt")
                model.save(best_model_path)
                logger.info(f"New best model saved at epoch {epoch} with mAP50: {map50}")
            
            # Early stopping logic
            if epoch % 5 == 0 and map50 < early_stopping_threshold:
                logger.info(f"Early stopping at epoch {epoch}, mAP50 ({map50}) below threshold ({early_stopping_threshold})")
                break
                
            # No improvement for 10 epochs
            if epoch > 10 and best_epoch <= epoch - 10:
                logger.info(f"Early stopping at epoch {epoch}, no improvement for 10 epochs. Best was epoch {best_epoch}")
                break
            
            # Report intermediate result (for NNI)
            try:
                logger.info(f"Reporting intermediate result to NNI: {map50}")
                nni.report_intermediate_result(map50)
            except Exception as e:
                logger.error(f"Error reporting result to NNI: {e}")
                traceback.print_exc()
        
        logger.info("Training completed")
        
        # Use the best metric as the final result
        if all_metrics:
            final_map50 = max(all_metrics)
            logger.info(f"Best mAP@50 across all epochs: {final_map50} (at epoch {best_epoch})")
        else:
            # Perform final validation if no metrics were collected
            logger.info("Performing final validation...")
            try:
                final_val_results = model.val(data=data_path)
                final_map50 = float(final_val_results.box.map50)
                logger.info(f"Final mAP@50: {final_map50}")
            except Exception as e:
                logger.error(f"Error in final validation: {e}")
                final_map50 = 0.0
        
        # Apply pruning and quantization if specified
        if params.get("pruning_amount", 0) > 0:
            pruning_amount = params.get("pruning_amount")
            logger.info(f"Applying pruning with amount {pruning_amount}...")
            model = apply_pruning(model, amount=pruning_amount)
            
        if params.get("apply_quantization", False):
            logger.info("Applying quantization...")
            model = apply_quantization(model)
            
        # Save final model and configuration
        try:
            trained_model_path = os.path.join(saved_models_dir, f"{trial_id}_trained_final.pt")
            model.save(trained_model_path)
            logger.info(f"Final trained model saved as {trained_model_path}")
            
            # Copy the best model to a fixed name for easy reference
            best_model_path = os.path.join(saved_models_dir, "best_yolov8_model.pt")
            best_epoch_model = os.path.join(saved_models_dir, f"{trial_id}_best_epoch_{best_epoch}.pt")
            
            if os.path.exists(best_epoch_model):
                shutil.copy2(best_epoch_model, best_model_path)
                logger.info(f"Best model (epoch {best_epoch}) copied to {best_model_path}")
            else:
                shutil.copy2(trained_model_path, best_model_path)
                logger.info(f"Final model copied to {best_model_path}")
            
            save_best_config(params, final_map50)
        except Exception as e:
            logger.error(f"Error saving model: {e}")
            traceback.print_exc()
        
        # Report final result
        try:
            logger.info(f"Reporting final result to NNI: {final_map50}")
            nni.report_final_result(final_map50)
            logger.info("Successfully reported final result")
        except Exception as e:
            logger.error(f"Error reporting final result: {e}")
            traceback.print_exc()
    
    except Exception as e:
        logger.error(f"Critical error: {e}")
        traceback.print_exc()
        try:
            nni.report_final_result(0.0)
            logger.info("Reported error status to NNI")
        except Exception as nested_e:
            logger.error(f"Failed to report error status to NNI: {nested_e}")

if __name__ == "__main__":
    logger.info("Starting YOLOv8 NNI experiment")
    train_with_nni()
    logger.info("Experiment completed")