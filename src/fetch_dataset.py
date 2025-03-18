# fetch_dataset.py
import os
import sys
from roboflow import Roboflow


# Initialize Roboflow with the API key
rf = Roboflow(api_key="tAEs7pzo53VAUo8ui0h2")

# Access the Chess Pieces dataset
project = rf.workspace("roboflow-100").project("chess-pieces-mjzgj")
version = project.version(2)

# Download the dataset in YOLOv8 format
dataset = version.download("yolov8")

print(f"Dataset downloaded to: {dataset.location}")