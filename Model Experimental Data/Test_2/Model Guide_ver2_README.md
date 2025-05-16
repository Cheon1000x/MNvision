# YOLOv8 Training Guide

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python: 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Ultralytics: YOLOv8](https://img.shields.io/badge/Ultralytics-YOLOv8-green.svg)](https://github.com/ultralytics/ultralytics)

## 📝 Overview

This repository contains Python scripts for training custom YOLOv8 object detection models. The code supports JSON label conversion, dataset preparation, model training, and evaluation with comprehensive metrics.

## ✨ Features

- Convert JSON annotations to YOLO format
- Dataset splitting and preparation
- Custom model training with YOLOv8
- Advanced data augmentation (v2)
- Model evaluation and visualization
- Transfer learning support

## 🚀 Getting Started

### Prerequisites

```bash
pip install ultralytics torch albumentations opencv-python pillow matplotlib tqdm
```

### Basic Usage

1. Prepare your dataset:
   - Images (JPG, PNG)
   - JSON annotation files
   - `classes.txt` file with class names (one per line)

2. Run the script:
   ```bash
   python model_ver_1.py  # Basic version
   # OR
   python model_ver_2_with_data_augmentation.py  # With data augmentation
   ```

3. Follow the prompts to choose between:
   - Training a new model
   - Continuing training from an existing model

## 📋 Version Comparison

| Feature | Version 1 | Version 2 |
|---------|-----------|-----------|
| Basic YOLOv8 Training | ✅ | ✅ |
| JSON to YOLO Conversion | ✅ | ✅ |
| Dataset Splitting | ✅ | ✅ |
| External Data Augmentation | ❌ | ✅ |
| Albumentations Integration | ❌ | ✅ |
| Enhanced Learning Schedule | ❌ | ✅ |
| Augmentation Visualization | ❌ | ✅ |

## 💡 Data Augmentation (Version 2)

Version 2 implements extensive data augmentation using Albumentations:

- Horizontal flips (50% probability)
- 90° rotations (50% probability)
- Brightness/contrast adjustments (30% probability)
- Gamma adjustments (30% probability)
- Gaussian blur (10% probability)
- CLAHE (30% probability)
- Gaussian noise (20% probability)
- Random shadows (10% probability)
- Tone curve adjustments (20% probability)

Additionally, YOLOv8 built-in augmentations are enabled:
- Mosaic (100% probability)
- Mixup (30% probability)
- Copy-paste (30% probability)

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

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.