from ultralytics import YOLO
import os
import cv2
import numpy as np
import gc
import torch
import yaml



# Set the path to your data
data_path = r"your_path_to\data.yaml"

# Load paths from YAML
with open(data_path, 'r') as f:
    yaml_data = yaml.safe_load(f)
    
# All paths will be based on YAML configuration

# Train YOLOv12s from scratch
print("Training YOLOv12s from scratch...")
model_12s_scratch = YOLO('yolov12s.yaml')  # Initialize YOLOv12s with random weights
model_12s_scratch.train(
    data=data_path,
    epochs=100,  # More epochs when training from scratch
    imgsz=640,
    batch=16,
    device='cuda',
    name='yolov12s_scratch',
    project='runs/scratch'
)

# Fine-tune YOLOv12s model
print("Fine-tuning YOLOv12s model...")
model_12s_finetune = YOLO('yolov12s.pt')  # Load the pretrained YOLOv12s model
model_12s_finetune.train(
    data=data_path,
    epochs=50,
    imgsz=640,
    batch=16,
    device='cuda',
    freeze=3,  # Freeze first 3 backbone layers to preserve COCO features and reduce overfitting
    name='yolov12s_finetune',
    project='runs/finetune'
)

# Fine-tune YOLOv12n model
print("Fine-tuning YOLOv12n model...")
model_12n_finetune = YOLO('yolov12n.pt')  # Load the pretrained YOLOv12n model
model_12n_finetune.train(
    data=data_path,
    epochs=50,
    imgsz=640,
    batch=16,
    device='cuda',
    freeze=3,  # Freeze first 3 backbone layers to preserve COCO features and reduce overfitting
    name='yolov12n_finetune',
    project='runs/finetune'
)
