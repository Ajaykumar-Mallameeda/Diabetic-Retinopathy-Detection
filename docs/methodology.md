# Methodology

## Overview

This document outlines the methodology used in the diabetic retinopathy detection project, including data preprocessing, feature extraction, model training, and evaluation approaches.

## Datasets

### DIARETDB1
- **Source**: Kaarina Kalesnykiene et al., DIARETDB1: Diabetic Retinopathy Database and Evaluation Protocol
- **Size**: 89 fundus images
- **Resolution**: 1152 × 1500 pixels
- **Labels**: Binary (0: No DR, 1: DR)
- **Characteristics**: Contains various lesion types typical of diabetic retinopathy

### IDRiD
- **Source**: Indian Diabetic Retinopathy Image Dataset
- **Size**: 540 images (440 train, 103 test)
- **Resolution**: Original 4288 × 2848, resized to 536 × 356
- **Labels**: 5-class grading (0: No DR, 1: Mild, 2: Moderate, 3: Severe, 4: Proliferative)
- **Characteristics**: Graded by ophthalmology experts

## Classical ML Pipeline

### 1. Preprocessing Steps

#### Grayscale Conversion
- RGB to grayscale conversion using OpenCV
- Reduces computational complexity while preserving vessel information

#### Adaptive Histogram Equalization
- OpenCV's `equalizeHist` for contrast enhancement
- Improves visibility of retinal structures
- Essential for consistent feature extraction

#### Discrete Wavelet Transform (DWT)
- Haar wavelet decomposition with `pywt.dwt2`
- Captures multi-scale features
- Reconstruction preserves spatial information

#### Gaussian Matched Filtering
- Custom kernel for vessel enhancement
- Multiple orientations (0°, 45°, 90°, 135°)
- Maximum response across orientations

#### Gabor Filtering
- Bank of 16 Gabor filters
- Parameters: σ=6, λ=12, γ=0.37
- Captures texture information at multiple scales

#### K-means Segmentation
- 2-cluster segmentation with K-means++
- Reduces color complexity
- Highlights important regions

### 2. Feature Engineering

- Final feature vector size: 1,728,000 (flattened 1152×1500)
- Features represent processed pixel intensities
- No explicit feature selection (learned by classifier)

### 3. Model Selection

#### SVM (Support Vector Machine)
- Kernel: Radial Basis Function (RBF)
- C parameter: 1.0
- Gamma: 'auto' (1/n_features)
- Advantages: Handles high-dimensional data well

#### KNN (K-Nearest Neighbors)
- K=3 neighbors
- Euclidean distance metric
- Uniform weight for all neighbors
- Advantages: Simple, non-parametric

## Deep Learning Pipeline

### 1. Data Preparation

#### Resizing
- Original IDRiD images resized from 4288×2848 to 536×356
- Maintains aspect ratio while reducing computational load

#### Normalization
- Pixel values scaled to [0, 1] range
- Standard preprocessing for neural networks

#### Data Augmentation
- Rotation: ±20°
- Width/Height shift: ±10%
- Shear: ±10%
- Zoom: ±10%
- Horizontal flipping
- Purpose: Increase training data diversity

### 2. Model Architectures

#### Custom CNN
```
Input (356×536×3)
↓ Conv2D(32, 4×4, stride=4)
↓ Conv2D(64, 3×3, stride=2)
↓ Conv2D(128, 3×3, stride=2)
↓ Flatten
↓ Dense(512)
↓ Dense(512)
↓ Dense(512)
↓ Dense(5, softmax)
```

#### MobileNetV2 Transfer Learning
- Pre-trained on ImageNet
- Fine-tuned from layer 100 onwards
- Custom classifier head:
  - Dense(256) → Dense(256) → Dense(256) → Dense(5)

### 3. Training Strategy

#### Optimization
- Optimizer: Adam
- Learning rate: 0.0001
- Loss: Sparse categorical cross-entropy
- Metrics: Accuracy

#### Regularization
- L2 regularization (λ=0.0001) on all layers
- Dropout not used (relying on L2 and data augmentation)

#### Training Schedule
- Batch size: 32
- Epochs: 20 (for demonstration)
- Early stopping not implemented (for reproducibility)

## Evaluation Metrics

### Classification Metrics
- **Accuracy**: Overall correct predictions
- **Precision**: TP / (TP + FP)
- **Recall**: TP / (TP + FN)
- **F1-Score**: Harmonic mean of precision and recall

### Confusion Matrix Analysis
- Per-class performance
- Misclassification patterns
- Class imbalance effects

### Cross-Validation
- 5-fold cross-validation for classical models
- Ensures robustness of results

## Model Interpretation

### Classical ML Interpretation
- Support vectors indicate critical examples
- Feature weights (for linear kernels)
- Decision boundary analysis

### Deep Learning Interpretation
- Grad-CAM visualization
- Feature map analysis
- Confidence score distribution
- Attention pattern analysis

## Implementation Details

### Code Organization
- Modular design with clear separation
- Configuration management
- Type hints for clarity
- Error handling

### Reproducibility
- Fixed random seeds
- Deterministic algorithms where possible
- Saved model weights
- Documented preprocessing pipeline

## Limitations

### Data Limitations
- Small dataset sizes
- Class imbalance in IDRiD
- Different resolutions between datasets

### Methodological Limitations
- No extensive hyperparameter tuning
- Limited cross-dataset validation
- No external validation set

### Computational Constraints
- Training on limited hardware
- Batch size restrictions
- Memory limitations for large images