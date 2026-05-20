"""Complete preprocessing pipeline for retinal images"""

import numpy as np
from typing import List, Tuple
import cv2

from ..preprocessing.classical_filters import (
    gaussian_matched_filter_kernel,
    create_matched_filter_bank,
    apply_filters,
    create_gabor_filter_bank,
    adaptive_histogram_equalization,
    kmeans_segmentation
)
from ..preprocessing.wavelet_transforms import apply_wavelet_transform_batch
from ..utils.config import Config


class PreprocessingPipeline:
    """Complete preprocessing pipeline for retinal images"""

    def __init__(self, config: dict = None):
        """
        Initialize preprocessing pipeline.

        Args:
            config: Custom configuration dictionary
        """
        if config:
            Config.update_config({'PREPROCESSING': config})

        # Initialize filter banks
        self.gaussian_kernel = gaussian_matched_filter_kernel(
            Config.PREPROCESSING['gaussian_length'],
            Config.PREPROCESSING['gaussian_sigma']
        )
        self.gaussian_bank = create_matched_filter_bank(
            self.gaussian_kernel,
            n=4
        )
        self.gabor_bank = create_gabor_filter_bank()

    def preprocess_classical_ml(self, images: List[np.ndarray]) -> List[np.ndarray]:
        """
        Apply full preprocessing pipeline for classical ML approach.

        Args:
            images: List of input images (BGR format)

        Returns:
            List of preprocessed feature vectors
        """
        processed = []

        for img in images:
            # Step 1: Convert to grayscale
            if len(img.shape) == 3:
                img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            else:
                img_gray = img

            # Step 2: Adaptive histogram equalization
            img_equalized = adaptive_histogram_equalization(img_gray)

            # Step 3: Wavelet transform
            img_shape = img_equalized.shape
            coeffs = (img_equalized, (np.zeros_like(img_equalized),
                                    np.zeros_like(img_equalized),
                                    np.zeros_like(img_equalized)))
            from ..preprocessing.wavelet_transforms import inverse_discrete_wavelet_transform_2d
            img_wavelet = inverse_discrete_wavelet_transform_2d(
                coeffs,
                Config.PREPROCESSING['wavelet_type']
            )

            # Step 4: Gaussian matched filtering
            img_gaussian = apply_filters(img_wavelet, self.gaussian_bank)

            # Step 5: Gabor filtering
            img_gabor = apply_filters(img_wavelet, self.gabor_bank)

            # Step 6: K-means segmentation
            # Convert to 3-channel for kmeans
            img_3channel = cv2.cvtColor(img_gabor, cv2.COLOR_GRAY2BGR)
            img_segmented = kmeans_segmentation(
                img_3channel,
                k=Config.PREPROCESSING['kmeans_clusters']
            )

            # Convert back to grayscale and flatten
            img_final = cv2.cvtColor(img_segmented, cv2.COLOR_BGR2GRAY)
            processed.append(img_final.flatten())

        return processed

    def preprocess_deep_learning(self, images: List[np.ndarray],
                               target_size: Tuple[int, int] = None) -> np.ndarray:
        """
        Apply preprocessing for deep learning approach.

        Args:
            images: List of input images
            target_size: Target resize size (height, width)

        Returns:
            Preprocessed image array
        """
        if target_size is None:
            target_size = Config.get_image_size('idrid')

        processed = []

        for img in images:
            # Resize if needed
            if img.shape[:2] != target_size:
                img = cv2.resize(img, target_size[::-1])  # OpenCV uses (width, height)

            # Normalize to [0, 1]
            img_normalized = img.astype(np.float32) / 255.0

            processed.append(img_normalized)

        return np.array(processed)

    def visualize_preprocessing_steps(self, img: np.ndarray,
                                    save_path: str = None) -> dict:
        """
        Generate intermediate preprocessing results for visualization.

        Args:
            img: Input image
            save_path: Optional path to save visualization

        Returns:
            Dictionary containing intermediate results
        """
        results = {}

        # Original
        results['original'] = img.copy()

        # Grayscale
        if len(img.shape) == 3:
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            img_gray = img.copy()
        results['grayscale'] = img_gray

        # Histogram equalization
        results['equalized'] = adaptive_histogram_equalization(img_gray)

        # Wavelet reconstruction
        coeffs = (results['equalized'], (np.zeros_like(results['equalized']),
                                       np.zeros_like(results['equalized']),
                                       np.zeros_like(results['equalized'])))
        from ..preprocessing.wavelet_transforms import inverse_discrete_wavelet_transform_2d
        results['wavelet'] = inverse_discrete_wavelet_transform_2d(
            coeffs,
            Config.PREPROCESSING['wavelet_type']
        )

        # Gaussian filtering
        results['gaussian'] = apply_filters(results['wavelet'], self.gaussian_bank)

        # Gabor filtering
        results['gabor'] = apply_filters(results['wavelet'], self.gabor_bank)

        # K-means segmentation
        img_3channel = cv2.cvtColor(results['gabor'], cv2.COLOR_GRAY2BGR)
        results['segmented'] = kmeans_segmentation(
            img_3channel,
            k=Config.PREPROCESSING['kmeans_clusters']
        )

        # Save visualization if requested
        if save_path:
            import matplotlib.pyplot as plt
            fig, axes = plt.subplots(2, 4, figsize=(16, 8))
            axes = axes.flatten()

            titles = ['Original', 'Grayscale', 'Equalized', 'Wavelet',
                     'Gaussian', 'Gabor', 'Segmented', 'Final']

            for ax, title, key in zip(axes, titles, list(results.keys())):
                if key == 'original' and len(results[key].shape) == 3:
                    ax.imshow(cv2.cvtColor(results[key], cv2.COLOR_BGR2RGB))
                else:
                    ax.imshow(results[key], cmap='gray')
                ax.set_title(title)
                ax.axis('off')

            plt.tight_layout()
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()

        return results