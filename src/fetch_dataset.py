# fetch_dataset.py
import os
import sys
from dotenv import load_dotenv
from roboflow import Roboflow

# Try to get API key from command-line argument first
if len(sys.argv) > 1:
    api_key = sys.argv[1]
else:
    # Fall back to environment variable if no command-line argument
    load_dotenv()
    api_key = os.getenv("ROBOFLOW_API_KEY")

if not api_key:
    raise ValueError("No API key found. Please provide it as a command-line argument or set the ROBOFLOW_API_KEY environment variable.")

# Initialize Roboflow with the API key
rf = Roboflow(api_key=api_key)

# Access the Chess Pieces dataset
project = rf.workspace("roboflow-100").project("chess-pieces-mjzgj")
version = project.version(2)

# Download the dataset in YOLOv8 format
dataset = version.download("yolov8")

print(f"Dataset downloaded to: {dataset.location}")