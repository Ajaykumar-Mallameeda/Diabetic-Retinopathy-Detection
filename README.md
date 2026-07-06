# Diabetic Retinopathy Detection

![Status](https://img.shields.io/badge/Status-Active-059669?style=flat)
![License](https://img.shields.io/badge/License-MIT-2563EB?style=flat)
![Last Updated](https://img.shields.io/badge/Last%20Updated-2026--07-6B7280?style=flat)

![Python](https://img.shields.io/badge/Python-0D9488?style=flat&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-0D9488?style=flat&logo=pytorch&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=flat&logo=tensorflow&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-0D9488?style=flat&logo=opencv&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-0D9488?style=flat&logo=scikitlearn&logoColor=white)

A machine learning project exploring automated detection of diabetic retinopathy from retinal fundus images using classical ML and deep learning approaches.

---

## Overview

Diabetic retinopathy is a leading cause of blindness worldwide. This project explores automated detection using computer vision and machine learning techniques, implementing two distinct approaches:

- **Classical ML Pipeline** — Traditional image processing with SVM/KNN classifiers
- **Deep Learning Pipeline** — CNN architectures with transfer learning (MobileNetV2)

**Current status:** Both pipelines are implemented and evaluated on the DIARETDB1 dataset. The deep learning pipeline is actively being developed with additional architectures and datasets.

> **Note:** This project is for educational and research purposes. Not intended for clinical use.

---

## Methods

### Classical ML Approach

1. **Preprocessing Pipeline:** Grayscale conversion, adaptive histogram equalization, discrete wavelet transform (Haar), Gaussian matched filtering, Gabor filter banks, K-means segmentation
2. **Models:** SVM with RBF kernel, K-Nearest Neighbors

### Deep Learning Approach

1. **Architecture:** Custom CNN with 3 convolutional layers, MobileNetV2 with transfer learning, data augmentation
2. **Training:** Adam optimizer with fixed learning rate, sparse categorical cross-entropy loss

---

## Results

*Preliminary results from single-dataset experiments. Not independently validated.*

| Model | Dataset | Accuracy | Status |
|-------|---------|----------|--------|
| SVM | DIARETDB1 (89 images) | ~96% | Complete |
| KNN | DIARETDB1 (89 images) | ~94% | Complete |
| MobileNetV2 | DIARETDB1 + IDRiD | TBD | In progress |

> **Results context:** These results are from single-dataset experiments on small datasets. They demonstrate the pipeline's functionality but should not be interpreted as clinically validated performance.

---

## Technology Stack

| Technology | Purpose |
|------------|---------|
| Python 3.8+ | Core language |
| scikit-learn | Classical ML models |
| TensorFlow 2.x / PyTorch | Deep learning frameworks |
| OpenCV / scikit-image | Computer vision preprocessing |
| PyWavelets | Wavelet transforms |
| Pillow | Image handling |
| Matplotlib / Seaborn | Visualization |

---

## Getting Started

### Prerequisites

- Python 3.8 or later
- Jupyter Notebook or Jupyter Lab

### Installation

```bash
# Clone the repository
git clone https://github.com/Ajaykumar-Mallameeda/Diabetic-Retinopathy-Detection.git
cd Diabetic-Retinopathy-Detection

# Install dependencies
pip install -r requirements.txt
```

### Setup Configuration

Update the dataset paths in `src/utils/config.py`:

```python
from src.utils.config import Config

Config.update_config({
    'DATASET_PATHS': {
        'diaretdb1': 'path/to/diaretdb1/images',
        'idrid_train': 'path/to/idrid/train',
        'idrid_test': 'path/to/idrid/test',
        'idrid_labels': 'path/to/idrid/labels.csv'
    }
})
```

### Run Notebooks

```bash
jupyter notebook
```

Start with `notebooks/01_exploratory_analysis.ipynb` and follow the notebooks in order.

---

## Project Structure

```
Diabetic-Retinopathy-Detection/
├── src/                         # Source code modules
│   ├── preprocessing/           # Image preprocessing utilities
│   ├── models/                  # ML/DL model definitions
│   ├── data/                    # Data loading utilities
│   └── utils/                   # Helper functions
├── notebooks/                   # Jupyter notebooks for experiments
├── assets/                      # Generated assets and diagrams
├── docs/                        # Additional documentation
├── models/                      # Trained model files
├── data/                        # Dataset files
├── requirements.txt
├── LICENSE
└── README.md
```

---

## Model Analysis

- **Classical ML** — Support vector analysis for feature understanding
- **Deep Learning** — Grad-CAM implementation for attention visualization
- **Error Patterns** — Manual analysis of misclassifications
- **Confidence** — Basic prediction confidence scoring

---

## Dataset Information

### DIARETDB1
- 89 fundus images
- Binary classification (DR / No DR)
- High resolution (1152 x 1500)

### IDRiD
- 540+ fundus images
- 5-level severity grading
- Lower resolution (356 x 536)

---

## Lessons Learned

- **Small datasets overfit quickly** — The 96% SVM accuracy on 89 images is almost certainly optimistic. Cross-dataset validation is essential before drawing conclusions.
- **Classical features + simple models can be competitive** — Hand-crafted features (wavelets, Gabor filters) with SVM achieved surprisingly strong results, even compared to deep learning approaches on small datasets.
- **Dual pipeline architecture aided debugging** — Implementing both classical and deep learning pipelines helped isolate whether errors came from feature extraction or model architecture.

---

## License & Author

**License:** This project is licensed under the MIT License — see the [LICENSE](./LICENSE) file for details.

**Author:** [Ajaykumar Mallameeda](https://github.com/Ajaykumar-Mallameeda) · Indian Institute of Technology Palakkad

---

*Built at IIT Palakkad as part of a continuous learning journey in AI and Backend Engineering.*
