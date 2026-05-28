"""
Utility functions for Lung Cancer Detection CNN
Includes dataset loading and data analysis functions
"""

import os
import cv2
import numpy as np
from sklearn.preprocessing import normalize
import config


def load_dataset():
    """
    Load images from cancer and normal directories
    Returns: (images, labels) where labels are 1 for cancer, 0 for normal
    """
    images = []
    labels = []
    
    # Load cancer images (label = 1)
    if os.path.exists(config.CANCER_PATH):
        for filename in os.listdir(config.CANCER_PATH):
            if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                img_path = os.path.join(config.CANCER_PATH, filename)
                img = cv2.imread(img_path)
                if img is not None:
                    img = cv2.resize(img, (config.IMG_WIDTH, config.IMG_HEIGHT))
                    images.append(img)
                    labels.append(1)
    
    # Load normal images (label = 0)
    if os.path.exists(config.NORMAL_PATH):
        for filename in os.listdir(config.NORMAL_PATH):
            if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                img_path = os.path.join(config.NORMAL_PATH, filename)
                img = cv2.imread(img_path)
                if img is not None:
                    img = cv2.resize(img, (config.IMG_WIDTH, config.IMG_HEIGHT))
                    images.append(img)
                    labels.append(0)
    
    # Convert to numpy arrays
    images = np.array(images, dtype=np.float32)
    labels = np.array(labels)
    
    # Normalize images
    images = images / 255.0
    
    print(f"✓ Loaded {len(images)} images:")
    print(f"  - Cancer images: {np.sum(labels)}")
    print(f"  - Normal images: {len(labels) - np.sum(labels)}")
    
    return images, labels


def get_sample_stats(images):
    """
    Calculate statistics about the dataset samples
    Returns: dictionary with statistics
    """
    stats = {
        'total_images': len(images),
        'mean_value': np.mean(images),
        'std_value': np.std(images),
        'min_value': np.min(images),
        'max_value': np.max(images),
        'image_shape': images[0].shape if len(images) > 0 else None
    }
    
    print("\n✓ Dataset Statistics:")
    print(f"  - Total images: {stats['total_images']}")
    print(f"  - Mean pixel value: {stats['mean_value']:.4f}")
    print(f"  - Std deviation: {stats['std_value']:.4f}")
    print(f"  - Min value: {stats['min_value']:.4f}")
    print(f"  - Max value: {stats['max_value']:.4f}")
    print(f"  - Image shape: {stats['image_shape']}")
    
    return stats
