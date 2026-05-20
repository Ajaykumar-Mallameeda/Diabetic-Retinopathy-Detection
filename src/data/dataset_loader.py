"""Dataset loading utilities for different retinal image datasets"""

import os
import glob
import numpy as np
import pandas as pd
from PIL import Image
from typing import Tuple, List, Optional, Union
import cv2
from ..utils.config import Config


class DatasetLoader:
    """Base class for loading retinal image datasets"""

    def __init__(self, dataset_path: str):
        """
        Initialize dataset loader.

        Args:
            dataset_path: Path to the dataset directory
        """
        self.dataset_path = dataset_path
        self.image_paths = []
        self.labels = []

    def validate_path(self) -> bool:
        """Validate that the dataset path exists"""
        return os.path.exists(self.dataset_path)

    def load_image(self, path: str, color_mode: str = 'BGR') -> np.ndarray:
        """
        Load a single image.

        Args:
            path: Path to the image file
            color_mode: 'BGR', 'RGB', or 'GRAY'

        Returns:
            Loaded image as numpy array
        """
        if color_mode == 'GRAY':
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        else:
            img = cv2.imread(path)
            if color_mode == 'RGB':
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        return img


class DIARETDB1Loader(DatasetLoader):
    """Loader for DIARETDB1 dataset"""

    def __init__(self, dataset_path: str = None):
        """
        Initialize DIARETDB1 loader.

        Args:
            dataset_path: Path to DIARETDB1 dataset
                         (default: from config)
        """
        if dataset_path is None:
            dataset_path = Config.get_dataset_path('diaretdb1')
        super().__init__(dataset_path)
        self.image_size = Config.get_image_size('diaretdb1')

    def load_dataset(self, num_images: int = 90) -> Tuple[List[np.ndarray], List[int]]:
        """
        Load DIARETDB1 dataset.

        Args:
            num_images: Number of images to load (default: 90)

        Returns:
            Tuple of (images, labels)
        """
        if not self.validate_path():
            raise ValueError(f"Dataset path not found: {self.dataset_path}")

        images = []
        labels = np.ones(num_images)

        # Load images with hardcoded naming pattern from notebook
        for i in range(1, num_images + 1):
            # Construct filename with zero padding
            if i < 10:
                filename = f"image00{i}.png"
            else:
                filename = f"image0{i}.png"

            img_path = os.path.join(self.dataset_path, filename)

            if os.path.exists(img_path):
                img = self.load_image(img_path, color_mode='BGR')
                images.append(img)
            else:
                print(f"Warning: Image not found: {img_path}")

        # Mark specific images as non-affected (from notebook)
        non_affected_indices = [1, 5, 7, 17, 6]
        for idx in non_affected_indices:
            if idx < len(labels):
                labels[idx] = 0

        return images, labels.tolist()


class IDRiDLoader(DatasetLoader):
    """Loader for IDRiD dataset with folder structure"""

    def __init__(self, train_path: str = None, test_path: str = None,
                 labels_path: str = None):
        """
        Initialize IDRiD loader.

        Args:
            train_path: Path to training images
            test_path: Path to test images
            labels_path: Path to labels CSV file
        """
        if train_path is None:
            train_path = Config.get_dataset_path('idrid_train')
        if test_path is None:
            test_path = Config.get_dataset_path('idrid_test')
        if labels_path is None:
            labels_path = Config.get_dataset_path('idrid_labels')

        self.train_path = train_path
        self.test_path = test_path
        self.labels_path = labels_path
        self.image_size = Config.get_image_size('idrid')

    def load_from_folders(self, directory: str) -> Tuple[List[np.ndarray], List[int]]:
        """
        Load images from class-organized folder structure.

        Args:
            directory: Path to directory with subfolders for each class

        Returns:
            Tuple of (images, labels)
        """
        images = []
        labels = []

        if not os.path.exists(directory):
            raise ValueError(f"Directory not found: {directory}")

        # Get all class folders (should be numbered 0-4)
        class_folders = sorted(glob.glob(os.path.join(directory, '*')))
        class_folders = [f for f in class_folders if os.path.isdir(f)]

        for class_folder in class_folders:
            class_name = os.path.basename(class_folder)
            try:
                class_label = int(class_name)
            except ValueError:
                continue

            # Load all images in this class folder
            image_files = glob.glob(os.path.join(class_folder, '*.jpg'))
            image_files.extend(glob.glob(os.path.join(class_folder, '*.png')))

            for img_path in image_files:
                img = self.load_image(img_path, color_mode='RGB')
                if img is not None:
                    # Resize to target size
                    img = cv2.resize(img, self.image_size[::-1])  # OpenCV uses (width, height)
                    images.append(img)
                    labels.append(class_label)

        return images, labels

    def load_train_data(self) -> Tuple[List[np.ndarray], List[int]]:
        """Load training data"""
        return self.load_from_folders(self.train_path)

    def load_test_data(self) -> Tuple[List[np.ndarray], List[int]]:
        """Load test data"""
        return self.load_from_folders(self.test_path)

    def load_labels_from_csv(self) -> pd.DataFrame:
        """
        Load labels from CSV file.

        Returns:
            DataFrame with labels
        """
        if os.path.exists(self.labels_path):
            return pd.read_csv(self.labels_path)
        else:
            print(f"Warning: Labels file not found: {self.labels_path}")
            return pd.DataFrame()


class DataGenerator:
    """Data generator for training deep learning models"""

    def __init__(self, images: List[np.ndarray], labels: List[int],
                 batch_size: int = 32, shuffle: bool = True):
        """
        Initialize data generator.

        Args:
            images: List of images
            labels: List of labels
            batch_size: Batch size
            shuffle: Whether to shuffle data
        """
        self.images = np.array(images)
        self.labels = np.array(labels)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_samples = len(images)
        self.indexes = np.arange(self.num_samples)

    def __len__(self):
        """Return number of batches"""
        return int(np.ceil(self.num_samples / self.batch_size))

    def __iter__(self):
        """Iterate over batches"""
        self.on_epoch_end()
        for i in range(len(self)):
            yield self.__getitem__(i)

    def on_epoch_end(self):
        """Shuffle indexes after each epoch"""
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __getitem__(self, index):
        """Get a batch of data"""
        # Get batch indexes
        start_idx = index * self.batch_size
        end_idx = min((index + 1) * self.batch_size, self.num_samples)
        batch_indexes = self.indexes[start_idx:end_idx]

        # Get batch data
        batch_images = self.images[batch_indexes]
        batch_labels = self.labels[batch_indexes]

        # Normalize images
        batch_images = batch_images.astype(np.float32) / 255.0

        return batch_images, batch_labels