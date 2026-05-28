"""
Configuration file for Lung Cancer Detection CNN
Contains all constants and paths used in training and prediction
"""

import os

# Image Configuration
IMG_WIDTH = 224
IMG_HEIGHT = 224
IMG_CHANNELS = 3

# Training Configuration
EPOCHS = 10
BATCH_SIZE = 32
VALIDATION_SPLIT = 0.2
LEARNING_RATE = 0.001

# Dataset Paths
DATASET_DIR = "dataset"
CANCER_PATH = os.path.join(DATASET_DIR, "cancer")
NORMAL_PATH = os.path.join(DATASET_DIR, "normal")

# Output Paths
OUTPUT_DIR = "outputs"
MODEL_PATH = os.path.join("models", "lung_cancer_model.h5")
METRICS_PATH = os.path.join(OUTPUT_DIR, "training_metrics.png")
