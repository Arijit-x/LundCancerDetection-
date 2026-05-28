"""
Training script for Lung Cancer Detection CNN
Implements the CNN pipeline: Input → Conv Layers → Pooling + Dropout → Dense → Output
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Sequential
import config
import utils


def build_cnn_model(img_width=config.IMG_WIDTH, img_height=config.IMG_HEIGHT, img_channels=config.IMG_CHANNELS):
    """
    Build a CNN model for binary classification (Cancer vs Normal).
    
    Architecture:
    - Input layer
    - Conv2D (32 filters) → ReLU → MaxPool → Dropout
    - Conv2D (64 filters) → ReLU → MaxPool → Dropout
    - Conv2D (128 filters) → ReLU → MaxPool → Dropout
    - Flatten → Dense (128) → ReLU → Dropout
    - Dense (1) → Sigmoid (binary classification)
    """
    model = Sequential([
        # Input layer
        layers.Input(shape=(img_height, img_width, img_channels)),
        
        # First Convolutional Block
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Second Convolutional Block
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Third Convolutional Block
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Flatten and Dense layers
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        
        # Output layer (binary classification)
        layers.Dense(1, activation='sigmoid')
    ])
    
    return model


def compile_model(model, learning_rate=config.LEARNING_RATE):
    """Compile the model with binary crossentropy loss."""
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss='binary_crossentropy',
        metrics=['accuracy', keras.metrics.Precision(), keras.metrics.Recall()]
    )
    return model


def plot_training_metrics(history, save_path=config.METRICS_PATH):
    """
    Plot training and validation accuracy across epochs.
    
    Args:
        history: Training history object from model.fit()
        save_path: Path to save the figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Accuracy plot
    axes[0].plot(history.history['accuracy'], label='Train Accuracy', marker='o')
    axes[0].plot(history.history['val_accuracy'], label='Val Accuracy', marker='s')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy')
    axes[0].set_title('Model Accuracy Across Epochs')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Loss plot
    axes[1].plot(history.history['loss'], label='Train Loss', marker='o')
    axes[1].plot(history.history['val_loss'], label='Val Loss', marker='s')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].set_title('Model Loss Across Epochs')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Metrics saved to {save_path}")
    plt.close()


def train_model(images, labels, epochs=config.EPOCHS, batch_size=config.BATCH_SIZE):
    """
    Train the CNN model.
    
    Args:
        images: Array of preprocessed images
        labels: Array of binary labels (0 or 1)
        epochs: Number of training epochs
        batch_size: Batch size
    
    Returns:
        Trained model and training history
    """
    # Split data into train and validation sets
    X_train, X_val, y_train, y_val = train_test_split(
        images, labels, 
        test_size=config.VALIDATION_SPLIT, 
        random_state=42,
        stratify=labels
    )
    
    print(f"Training set size: {len(X_train)}")
    print(f"Validation set size: {len(X_val)}")
    
    # Build and compile model
    model = build_cnn_model()
    model = compile_model(model)
    
    print("\nModel Summary:")
    model.summary()
    
    # Train model
    print("\nTraining model...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        verbose=1
    )
    
    # Plot and save metrics
    plot_training_metrics(history)
    
    # Evaluate on validation set
    print("\nEvaluating on validation set...")
    val_loss, val_accuracy, val_precision, val_recall = model.evaluate(X_val, y_val, verbose=0)
    print(f"Validation Accuracy: {val_accuracy:.4f}")
    print(f"Validation Precision: {val_precision:.4f}")
    print(f"Validation Recall: {val_recall:.4f}")
    
    return model, history


def main():
    """Main training pipeline."""
    print("=" * 60)
    print("LUNG CANCER DETECTION CNN - TRAINING PIPELINE")
    print("=" * 60)
    
    # Check if dataset exists
    if not os.path.exists(config.CANCER_PATH) or not os.path.exists(config.NORMAL_PATH):
        print("\n⚠️  Dataset directories not found!")
        print(f"Please ensure the following directories exist with images:")
        print(f"  - {config.CANCER_PATH}/")
        print(f"  - {config.NORMAL_PATH}/")
        print("\nDataset sources:")
        print("  - Kaggle IQ-OTH/NCCD Lung Cancer Dataset")
        print("  - LUNA16 Dataset")
        return
    
    # Load dataset
    print("\nLoading dataset...")
    images, labels = utils.load_dataset()
    
    # Print dataset statistics
    stats = utils.get_sample_stats(images)
    print(f"\nDataset Statistics:")
    print(f"  Total images: {stats['total_images']}")
    print(f"  Image shape: {stats['image_shape']}")
    print(f"  Value range: [{stats['min_value']:.4f}, {stats['max_value']:.4f}]")
    print(f"  Cancer images: {np.sum(labels == 1)}")
    print(f"  Normal images: {np.sum(labels == 0)}")
    
    # Train model
    model, history = train_model(images, labels)
    
    # Save model
    os.makedirs(os.path.dirname(config.MODEL_PATH), exist_ok=True)
    model.save(config.MODEL_PATH)
    print(f"\n✓ Model saved to {config.MODEL_PATH}")
    
    print("\n" + "=" * 60)
    print("Training complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
