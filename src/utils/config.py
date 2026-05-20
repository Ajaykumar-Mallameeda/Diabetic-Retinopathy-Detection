"""Configuration management for the project"""

import os
from typing import Dict, Any


class Config:
    """Configuration class for managing project settings"""

    # Dataset paths (should be overridden by user)
    DATASET_PATHS = {
        'diaretdb1': 'data/diaretdb1',
        'idrid_train': 'data/idrid/train',
        'idrid_test': 'data/idrid/test',
        'idrid_labels': 'data/idrid/labels.csv'
    }

    # Image dimensions
    IMAGE_DIMENSIONS = {
        'diaretdb1': (1152, 1500),
        'idrid': (356, 536),  # Original: (2848, 4288) divided by 8
    }

    # Preprocessing parameters
    PREPROCESSING = {
        'wavelet_type': 'haar',
        'gabor_filters': 16,
        'gabor_sigma': 6,
        'gabor_lambda': 12,
        'gabor_gamma': 0.37,
        'kmeans_clusters': 2,
        'gaussian_sigma': 5,
        'gaussian_length': 20,
    }

    # Model parameters
    MODEL_PARAMS = {
        'svm_kernel': 'rbf',
        'svm_c': 1.0,
        'knn_neighbors': 3,
        'cnn_hidden_dim': 512,
        'cnn_num_classes': 5,
        'cnn_learning_rate': 0.0001,
        'cnn_epochs': 100,
        'cnn_batch_size': 32,
    }

    # Training parameters
    TRAINING = {
        'test_size': 0.2,
        'random_state': 42,
        'stratify': True,
    }

    @classmethod
    def update_config(cls, config_dict: Dict[str, Any]) -> None:
        """Update configuration with user-provided values"""
        for section, values in config_dict.items():
            if hasattr(cls, section.upper()):
                section_config = getattr(cls, section.upper())
                section_config.update(values)

    @classmethod
    def get_dataset_path(cls, dataset_name: str) -> str:
        """Get dataset path by name"""
        return cls.DATASET_PATHS.get(dataset_name, '')

    @classmethod
    def get_image_size(cls, dataset_name: str) -> tuple:
        """Get image dimensions for dataset"""
        return cls.IMAGE_DIMENSIONS.get(dataset_name, (256, 256))

    @classmethod
    def save_config(cls, filepath: str) -> None:
        """Save configuration to file"""
        import json
        config = {
            'DATASET_PATHS': cls.DATASET_PATHS,
            'IMAGE_DIMENSIONS': cls.IMAGE_DIMENSIONS,
            'PREPROCESSING': cls.PREPROCESSING,
            'MODEL_PARAMS': cls.MODEL_PARAMS,
            'TRAINING': cls.TRAINING,
        }
        with open(filepath, 'w') as f:
            json.dump(config, f, indent=2)

    @classmethod
    def load_config(cls, filepath: str) -> None:
        """Load configuration from file"""
        import json
        with open(filepath, 'r') as f:
            config = json.load(f)
            cls.update_config(config)