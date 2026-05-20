# Diabetic Retinopathy Detection

A machine learning project exploring automated detection of diabetic retinopathy from retinal fundus images. Demonstrates classical ML and deep learning approaches for medical image classification.

## 🎯 Project Overview

Diabetic retinopathy is a leading cause of blindness worldwide. This project explores automated detection using computer vision and machine learning techniques, implementing two distinct approaches:

- **Classical ML Pipeline**: Traditional image processing with SVM/KNN classifiers
- **Deep Learning Pipeline**: CNN architectures with transfer learning

## 📁 Project Structure

```
Diabetic-Retinopathy-Detection/
├── README.md
├── requirements.txt
├── LICENSE
├── .gitignore
│
├── notebooks/                    # Jupyter notebooks for experiments
│   ├── 01_exploratory_analysis.ipynb
│   ├── 02_classical_preprocessing.ipynb
│   ├── 03_classical_training.ipynb
│   ├── 04_deep_learning_pipeline.ipynb
│   └── 05_model_interpretation.ipynb
│
├── src/                         # Source code modules
│   ├── preprocessing/          # Image preprocessing utilities
│   ├── models/                 # ML/DL model definitions
│   ├── data/                   # Data loading utilities
│   └── utils/                  # Helper functions
│
├── assets/                      # Generated assets
│   ├── architecture/           # Architecture diagrams
│   └── screenshots/           # Result visualizations
│
├── docs/                       # Additional documentation
├── models/                     # Trained model files
└── data/                       # Dataset files
```

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd Diabetic-Retinopathy-Detection

# Install dependencies
pip install -r requirements.txt
```

### Setup Configuration

Update the dataset paths in `src/utils/config.py` or create a local config:

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

Start with `01_exploratory_analysis.ipynb` to understand the datasets and follow the notebooks in order.

## 🧠 Methods

### Classical ML Approach

1. **Preprocessing Pipeline**:
   - Grayscale conversion
   - Adaptive histogram equalization
   - Discrete Wavelet Transform (Haar)
   - Gaussian matched filtering
   - Gabor filter banks
   - K-means segmentation

2. **Models**:
   - SVM with RBF kernel
   - K-Nearest Neighbors

### Deep Learning Approach

1. **Architecture**:
   - Custom CNN with 3 convolutional layers
   - MobileNetV2 with transfer learning
   - Data augmentation for robustness

2. **Training**:
   - Adam optimizer with fixed learning rate
   - Sparse categorical cross-entropy loss
   - Basic training loop implementation

## 📊 Results

The original notebooks report:
- SVM (Classical): ~96% accuracy on DIARETDB1 dataset
- KNN (Classical): ~94% accuracy on DIARETDB1 dataset
- MobileNetV2: Training in progress

*Note: Results from single dataset experiments. Not independently validated.*

## 🔬 Model Analysis

- **Classical ML**: Support vector analysis for feature understanding
- **Deep Learning**: Grad-CAM implementation for attention visualization
- **Error Patterns**: Manual analysis of misclassifications
- **Confidence**: Basic prediction confidence scoring

## 📈 Visualizations

Example visualizations included:

- Preprocessing pipeline demonstration
- Dataset samples showing severity levels
- Example training curves
- Example confusion matrices

See `assets/screenshots/` for demonstration images.

## 🛠️ Technologies Used

- **Python 3.8+**
- **Machine Learning**: scikit-learn, TensorFlow 2.x
- **Computer Vision**: OpenCV, scikit-image
- **Image Processing**: PyWavelets, Pillow
- **Visualization**: Matplotlib, Seaborn

## 📚 Dataset Information

### DIARETDB1
- 89 fundus images
- Binary classification (DR/No DR)
- High resolution (1152x1500)

### IDRiD
- 540+ fundus images
- 5-level severity grading
- Lower resolution (356x536)

## 🔧 Development Notes

### Code Organization
- Modular design with clear separation of concerns
- Reusable preprocessing pipeline
- Configurable parameters through centralized config
- Type hints for better code clarity

### Best Practices
- Data validation and path checking
- Clear error messages for missing datasets
- Basic model weight persistence
- Fixed random seeds where applicable

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- DIARETDB1 dataset providers
- IDRiD dataset organizers
- Open source computer vision community

## 📈 Future Improvements

- [ ] Ensemble methods for better performance
- [ ] Attention mechanisms for interpretability
- [ ] Multi-scale feature fusion
- [ ] Cross-dataset validation
- [ ] Real-time inference optimization

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

## 📧 Contact

Ajay Kumar Mallameeda  
Indian Institute of Technology Palakkad

---

**Note**: This project is for educational and research purposes. Not intended for clinical use.