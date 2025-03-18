# fetch_dataset.py
import os
from dotenv import load_dotenv
from roboflow import Roboflow

# Load environment variables from .env file
load_dotenv()

# Get API key from environment variable
api_key = os.getenv("ROBOFLOW_API_KEY")

if not api_key:
    raise ValueError("No API key found. Please set the ROBOFLOW_API_KEY environment variable.")

# Initialize Roboflow with the API key
rf = Roboflow(api_key=api_key)

# Access the COCO dataset
project = rf.workspace("microsoft").project("coco")
version = project.version(13)

# Download the dataset in YOLOv8 format
dataset = version.download("yolov8")

print(f"Dataset downloaded to: {dataset.location}")