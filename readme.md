# YOLOv8 Neural Architecture Search & Training Pipeline

This repository contains a complete ML pipeline for optimizing and training YOLOv8 object detection models using Neural Architecture Search (NAS) and hyperparameter optimization.

## Project Structure

```
project_root/
├── configs/
│   ├── best_yolov8_config.json    # Stores best model configuration found by NAS
│   ├── search_space.json          # Defines parameter search space for NAS
│   └── train_config.yaml          # Configuration for model training/retraining
├── datasets/
│   └── [downloaded datasets]      # Where datasets are downloaded and stored
├── saved_models/
│   └── [saved model files]        # Where trained models are saved
└── src/
    ├── fetch_dataset.py           # Script to download dataset from Roboflow
    ├── nni_nas_hpo.py             # Neural Architecture Search and Hyperparameter Optimization
    ├── responsive_kd.py           # Knowledge Distillation script (placeholder)
    ├── retrain_model.py           # Script to retrain models on specific datasets
    └── update_yolo_training_config.py  # Updates training config with best parameters
```

## Features

- **Neural Architecture Search**: Automatically discovers optimal YOLOv8 model architectures
- **Hyperparameter Optimization**: Finds the best training parameters for your specific dataset
- **Automated Dataset Fetching**: Downloads and prepares datasets from Roboflow
- **Model Compression**: Applies pruning and quantization techniques to optimize model size
- **CI/CD Integration**: Jenkins pipeline for automated training and deployment
- **Configurable Training**: Easily customize training parameters via YAML configuration

## Getting Started

### Prerequisites

- Python 3.8+
- PyTorch
- CUDA-compatible GPU (recommended)
- Roboflow API key

### Environment Setup

1. Create a `.env` file in the project root with your Roboflow API key:
   ```
   ROBOFLOW_API_KEY=your_api_key_here
   ```

2. Install required packages:
   ```
   pip install roboflow ultralytics torch nni python-dotenv pyyaml
   ```

### Workflow

1. **Download Dataset**:
   ```
   python src/fetch_dataset.py
   ```

2. **Run Neural Architecture Search**:
   ```
   nnictl create --config configs/nni_config.yml
   ```

3. **Update Training Configuration**:
   ```
   python src/update_yolo_training_config.py
   ```

4. **Retrain Model**:
   ```
   python src/retrain_model.py
   ```

## Neural Architecture Search

The NAS process optimizes:
- Network depth and width
- Resolution
- Kernel sizes
- Optimization parameters
- Training hyperparameters

The search space is defined in `configs/search_space.json` and can be customized for your specific requirements.

## Jenkins Integration

The project includes a Jenkins pipeline configuration (`jenkins-pipeline.groovy`) for automating the training process. To use it:

1. Configure Jenkins with Roboflow API credentials
2. Create a new pipeline job using the provided Jenkinsfile
3. Run the pipeline to execute the full training workflow

Example pipeline usage:
```groovy
pipeline {
    agent any
    
    stages {
        stage('Setup') {
            steps {
                sh 'pip install roboflow'
            }
        }
        
        stage('Fetch Dataset') {
            steps {
                withCredentials([string(credentialsId: 'roboflow-api-key', variable: 'ROBOFLOW_API_KEY')]) {
                    sh 'python src/fetch_dataset.py'
                }
            }
        }
        
        // Additional stages...
    }
}
```

## Dataset Fetching

The `fetch_dataset.py` script supports:
- Direct download from Roboflow using API key
- Automatic conversion to YOLOv8 format
- Flexible configuration via command-line arguments or environment variables

Usage:
```
python src/fetch_dataset.py [dataset_link] [api_key]
```

## Retraining Models

The `retrain_model.py` script provides:
- Fine-tuning of optimized models on specific datasets
- Customizable training parameters via configuration
- Detailed logging and performance metrics

Configuration options in `train_config.yaml` include:
- Batch size
- Learning rate
- Image resolution
- Optimizer selection
- Data augmentation parameters

## Contributing

Contributions to improve the pipeline are welcome. Please feel free to submit pull requests or open issues for bugs and feature requests.

## License

[MIT License](LICENSE)
