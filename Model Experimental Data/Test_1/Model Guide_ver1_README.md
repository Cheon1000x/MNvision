# YOLOv8 Training Guide

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python: 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Ultralytics: YOLOv8](https://img.shields.io/badge/Ultralytics-YOLOv8-green.svg)](https://github.com/ultralytics/ultralytics)

## 📝 Overview

This repository contains a Python script for training custom YOLOv8 object detection models. The code supports JSON label conversion, dataset preparation, model training, and evaluation with comprehensive metrics.

## ✨ Features

- Convert JSON annotations to YOLO format
- Automatic dataset splitting (train/validation)
- YOLOv8m model training
- Continue training from existing checkpoints
- Comprehensive model evaluation
- Automatic detection of CUDA for GPU acceleration

## 🚀 Getting Started

### Prerequisites

```bash
pip install ultralytics torch opencv-python pillow matplotlib tqdm
```

### Basic Usage

1. Prepare your dataset:
   - Images (JPG, PNG)
   - JSON annotation files
   - `classes.txt` file with class names (one per line)

2. Run the script:
   ```bash
   python model_ver_1.py
   ```

3. Follow the prompts to choose between:
   - Training a new model
   - Continuing training from an existing model

## 📋 Code Structure

| Component | Description |
|-----------|-------------|
| Environment Setup | Directory creation and OpenMP configuration |
| Data Conversion | JSON to YOLO format annotation conversion |
| Dataset Preparation | Random split into training (80%) and validation (20%) sets |
| YAML Configuration | Automatic generation of YOLOv8 configuration file |
| Model Training | New model training or continued training options |
| Model Evaluation | Comprehensive metrics calculation and visualization |
| Result Visualization | Training curves and metrics plotting |

## 🧩 Main Functions

### Data Processing

- `convert_json_to_yolo_format()`: Converts JSON annotations to YOLO format
- `process_data()`: Processes input data and saves in YOLO format
- `split_dataset()`: Splits the dataset into training and validation sets

### Model Operations

- `train_yolov8()`: Trains a new YOLOv8 model
- `continue_training()`: Continues training from an existing model checkpoint
- `evaluate_model()`: Evaluates model performance
- `plot_results()`: Visualizes training results

## 📊 Evaluation

The model is evaluated with:
- mAP50
- mAP50-95
- Precision
- Recall
- F1-Score

## 📈 Performance Tips

If model performance is insufficient:

1. Consider transitioning to segmentation:
   - Start with object detection, evaluate performance
   - Switch to segmentation if needed

2. Try lighter segmentation models:
   - YOLOv8n-seg instead of YOLOv8m-seg
   - Reduce input image size (1280 → 640)

3. Apply model optimization:
   - Quantization
   - Pruning
   - Knowledge distillation

4. Hybrid approach:
   - Fast object detection, then selective segmentation

## ⚙️ Configuration Options

- Default image size: 640 (can be changed to 1280 for higher accuracy)
- Default epochs: 50 (set to 1 in example for quick testing)
- Default batch size: 16
- Training/validation split: 80%/20%

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.