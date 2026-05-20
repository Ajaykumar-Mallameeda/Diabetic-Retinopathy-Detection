"""Visualization utilities for retinal images and model results"""

import numpy as np
import matplotlib.pyplot as plt
import cv2
import seaborn as sns
from typing import List, Tuple, Optional
from sklearn.metrics import confusion_matrix


def show_images(images: List[np.ndarray], titles: List[str] = None,
               cols: int = 5, figsize: Tuple[float, float] = (20, 20),
               cmap: str = 'gray') -> None:
    """
    Display a list of images in a grid.

    Args:
        images: List of images to display
        titles: Optional list of titles for each image
        cols: Number of columns in the grid
        figsize: Figure size
        cmap: Colormap for grayscale images
    """
    n_images = len(images)
    if titles is None:
        titles = [f'Image {i+1}' for i in range(n_images)]

    # Calculate rows needed
    rows = (n_images + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    axes = axes.flatten() if rows > 1 else [axes] if cols == 1 else axes

    for i, (img, title) in enumerate(zip(images, titles)):
        if i < len(axes):
            if len(img.shape) == 2:
                axes[i].imshow(img, cmap=cmap)
            else:
                axes[i].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            axes[i].set_title(title)
            axes[i].axis('off')

    # Hide unused subplots
    for i in range(n_images, len(axes)):
        axes[i].axis('off')

    plt.tight_layout()
    plt.show()


def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray,
                         class_names: List[str] = None,
                         save_path: str = None) -> None:
    """
    Plot confusion matrix.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: Names of classes
        save_path: Optional path to save the plot
    """
    if class_names is None:
        class_names = [f'Class {i}' for i in range(len(np.unique(y_true)))]

    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    plt.show()


def plot_training_curves(history: dict, save_path: str = None) -> None:
    """
    Plot training loss and accuracy curves.

    Args:
        history: Training history dictionary
        save_path: Optional path to save the plot
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # Loss curves
    if 'loss' in history:
        ax1.plot(history['loss'], label='Training Loss')
    if 'val_loss' in history:
        ax1.plot(history['val_loss'], label='Validation Loss')
    ax1.set_title('Model Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)

    # Accuracy curves
    if 'accuracy' in history:
        ax2.plot(history['accuracy'], label='Training Accuracy')
    if 'val_accuracy' in history:
        ax2.plot(history['val_accuracy'], label='Validation Accuracy')
    ax2.set_title('Model Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    plt.show()


def visualize_grad_cam(image: np.ndarray, heatmap: np.ndarray,
                       alpha: float = 0.4, save_path: str = None) -> None:
    """
    Visualize Grad-CAM heatmap overlaid on original image.

    Args:
        image: Original image
        heatmap: Grad-CAM heatmap
        alpha: Transparency for overlay
        save_path: Optional path to save the visualization
    """
    # Convert image to RGB if needed
    if len(image.shape) == 3 and image.shape[2] == 3:
        img_display = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    else:
        img_display = image

    # Create heatmap colormap
    heatmap_colored = plt.cm.jet(heatmap)[:, :, :3]
    heatmap_colored = (heatmap_colored * 255).astype(np.uint8)

    # Resize heatmap to match image size
    if heatmap.shape != img_display.shape[:2]:
        heatmap_colored = cv2.resize(
            heatmap_colored,
            (img_display.shape[1], img_display.shape[0])
        )

    # Overlay heatmap on image
    overlay = cv2.addWeighted(
        img_display.astype(np.uint8),
        1 - alpha,
        heatmap_colored,
        alpha,
        0
    )

    # Display
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Original image
    axes[0].imshow(img_display)
    axes[0].set_title('Original Image')
    axes[0].axis('off')

    # Heatmap
    axes[1].imshow(heatmap, cmap='jet')
    axes[1].set_title('Grad-CAM Heatmap')
    axes[1].axis('off')

    # Overlay
    axes[2].imshow(overlay)
    axes[2].set_title('Overlay')
    axes[2].axis('off')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    plt.show()


def plot_preprocessing_pipeline(original: np.ndarray,
                               processed_images: dict,
                               save_path: str = None) -> None:
    """
    Visualize preprocessing pipeline steps.

    Args:
        original: Original image
        processed_images: Dictionary of processed images
        save_path: Optional path to save the visualization
    """
    # Count images
    n_images = 1 + len(processed_images)
    cols = min(n_images, 4)
    rows = (n_images + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 4*rows))
    axes = axes.flatten() if rows > 1 else [axes] if cols == 1 else axes

    # Original image
    idx = 0
    if len(original.shape) == 3:
        axes[idx].imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
    else:
        axes[idx].imshow(original, cmap='gray')
    axes[idx].set_title('Original')
    axes[idx].axis('off')

    # Processed images
    for title, img in processed_images.items():
        idx += 1
        if idx < len(axes):
            if len(img.shape) == 3:
                axes[idx].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            else:
                axes[idx].imshow(img, cmap='gray')
            axes[idx].set_title(title)
            axes[idx].axis('off')

    # Hide unused subplots
    for i in range(idx + 1, len(axes)):
        axes[i].axis('off')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    plt.show()


def plot_class_distribution(labels: np.ndarray, class_names: List[str] = None,
                           save_path: str = None) -> None:
    """
    Plot distribution of classes in dataset.

    Args:
        labels: Array of labels
        class_names: Optional names for classes
        save_path: Optional path to save the plot
    """
    if class_names is None:
        class_names = [f'Class {i}' for i in range(len(np.unique(labels)))]

    # Count samples per class
    unique, counts = np.unique(labels, return_counts=True)

    plt.figure(figsize=(10, 5))
    bars = plt.bar(unique, counts)

    # Customize plot
    plt.title('Class Distribution')
    plt.xlabel('Class')
    plt.ylabel('Number of Samples')
    plt.xticks(unique, [class_names[i] for i in unique])

    # Add count labels on bars
    for bar, count in zip(bars, counts):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                str(count), ha='center', va='bottom')

    plt.grid(True, axis='y', alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    plt.show()