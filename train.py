from pathlib import Path

from ultralytics import YOLO

model = YOLO("yolov8n.pt")

# 이미지 640 * 640임
# [Train - Ultralytics YOLO Docs](https://docs.ultralytics.com/modes/train/#multi-gpu-training)
#                 data (str): Path to dataset configuration file.
#                 epochs (int): Number of training epochs.
#                 batch_size (int): Batch size for training.
#                 imgsz (int): Input image size.
#                 device (str): Device to run training on (e.g., 'cuda', 'cpu').
#                 workers (int): Number of worker threads for data loading.
#                 optimizer (str): Optimizer to use for training.
#                 lr0 (float): Initial learning rate.
#                 patience (int): Epochs to wait for no observable improvement for early stopping of training.
train_results = model.train(
    # data="disaster_detection/data.yaml",  # path to dataset YAML
    data="/media/goldenmine/Data/Project/datasets/disaster_detection/data.yaml",  # path to dataset YAML
    epochs=100,  # number of training epochs
    imgsz=640,  # training image size
    batch=16,
    lr0=1e-5,
    # lrf=1e-3,
    device="cuda",  # device to run on, i.e. device=0 or device=0,1,2,3 or device=cpu
)


# Evaluate model performance on the validation set
metrics = model.val()

# Perform object detection on an image
test_image_path = 'dataset/test/MYH20230211008300032.jpg'
results = model(test_image_path)
results[0].save(f'results/{Path(test_image_path).name}')

# Export the model to ONNX format
path = model.export(format="onnx")  # return path to exported model

# [Benchmark Datasets for Machine Learning for Natural Disasters](https://roc-hci.github.io/NADBenchmarks/)
# xBD Dataset:
# Contains 850,736 building polygons across 22,068 images
# Covers 45,361.79 km²
# Designed for change detection and building damage assessment
# Suitable for multiclass (ordinal) classification
# FloodNet Dataset:
# Contains about 11,000 question-image pairs and 3,200 images
# Focused on post-flood scene understanding
# Useful for image classification, semantic segmentation, and visual question answering
# MEDIC Dataset:
# Contains 71,198 images
# Designed for disaster type detection, informativeness classification, humanitarian task categorization, and damage severity assessment
# Suitable for multitask learning
# CrisisMMD Dataset:
# Contains 18,082 images and 16,058 tweets
# Useful for informativeness classification, humanitarian task categorization, and damage severity assessment
# Suitable for binary classification, multiclass classification, and multiclass (ordinal) classification
# RescueNet Dataset:
# High-resolution UAV imagery
# Provides pixel-level annotations for 10 classes across 6 categories
# Includes buildings (with damage levels), roads, vehicles, trees, and water
# Designed for semantic segmentation
# AIDER (Aerial Image Database for Emergency Response):
# Image classification dataset
# Covers four disaster events: Fire/Smoke, Flood, Collapsed Building/Rubble, and Traffic Accidents
# FEMA and NOAA Dataset:
# Contains vector (FEMA) and image (NOAA) data
# Focused on hurricane-damaged building detection
# Suitable for image segmentation and multiclass (ordinal) classification
# Incidents Dataset:
# Contains 1,144,148 images (446,684 positive samples)
# Designed for general disaster detection
# Suitable for multiclass classification