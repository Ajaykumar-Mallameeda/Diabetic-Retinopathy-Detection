"""Wavelet transform methods for retinal image feature extraction"""

import numpy as np
import pywt
from typing import Tuple, Optional


def discrete_wavelet_transform_2d(img: np.ndarray, wavelet: str = 'haar') -> Tuple:
    """
    Apply 2D Discrete Wavelet Transform to an image.

    Args:
        img: 2D input image
        wavelet: Wavelet type (default: 'haar')

    Returns:
        Wavelet coefficients tuple (cA, (cH, cV, cD))
    """
    return pywt.dwt2(img, wavelet)


def inverse_discrete_wavelet_transform_2d(coeffs: Tuple, wavelet: str = 'haar') -> np.ndarray:
    """
    Apply 2D Inverse Discrete Wavelet Transform.

    Args:
        coeffs: Wavelet coefficients from DWT
        wavelet: Wavelet type (default: 'haar')

    Returns:
        Reconstructed image
    """
    return pywt.idwt2(coeffs, wavelet)


def wavelet_denoise(img: np.ndarray, wavelet: str = 'haar',
                    mode: str = 'soft', sigma: Optional[float] = None) -> np.ndarray:
    """
    Denoise image using wavelet thresholding.

    Args:
        img: 2D input image
        wavelet: Wavelet type
        mode: Thresholding mode ('soft' or 'hard')
        sigma: Noise standard deviation (if None, estimated from data)

    Returns:
        Denoised image
    """
    # Decompose
    coeffs = pywt.wavedec2(img, wavelet, level=3)

    # Estimate noise if not provided
    if sigma is None:
        # Use median absolute deviation (MAD) estimator
        sigma_est = np.median(np.abs(coeffs[-1] - np.median(coeffs[-1]))) / 0.6745
    else:
        sigma_est = sigma

    # Calculate threshold
    threshold = sigma_est * np.sqrt(2 * np.log(img.size))

    # Threshold detail coefficients
    coeffs_thresh = list(coeffs)
    coeffs_thresh[1:] = [pywt.threshold(detail, threshold, mode)
                        for detail in coeffs_thresh[1:]]

    # Reconstruct
    return pywt.waverec2(coeffs_thresh, wavelet)


def wavelet_feature_extraction(img: np.ndarray, wavelet: str = 'haar',
                              level: int = 3) -> dict:
    """
    Extract wavelet-based features from an image.

    Args:
        img: 2D input image
        wavelet: Wavelet type
        level: Decomposition level

    Returns:
        Dictionary of wavelet features
    """
    # Multi-level decomposition
    coeffs = pywt.wavedec2(img, wavelet, level=level)

    features = {}

    # Energy features for each sub-band
    for i, coeff in enumerate(coeffs):
        if i == 0:  # Approximation coefficients
            features[f'energy_level_{i}'] = np.sum(coeff ** 2)
            features[f'mean_level_{i}'] = np.mean(coeff)
            features[f'std_level_{i}'] = np.std(coeff)
        else:  # Detail coefficients (horizontal, vertical, diagonal)
            for j, (name, c) in enumerate(zip(['H', 'V', 'D'], coeff)):
                features[f'energy_level_{i}_{name}'] = np.sum(c ** 2)
                features[f'mean_level_{i}_{name}'] = np.mean(c)
                features[f'std_level_{i}_{name}'] = np.std(c)

    return features


def apply_wavelet_transform_batch(images: list, wavelet: str = 'haar') -> list:
    """
    Apply wavelet transform to a batch of images.

    Args:
        images: List of 2D images
        wavelet: Wavelet type

    Returns:
        List of transformed images
    """
    transformed = []
    for img in images:
        if len(img.shape) == 1:
            # Reshape flat image back to 2D if needed
            # Assuming original shape was (1152, 1500) based on notebook
            img = img.reshape((1152, 1500))

        coeffs = pywt.dwt2(img, wavelet)
        reconstructed = pywt.idwt2(coeffs, wavelet)
        transformed.append(np.array(reconstructed).flatten())

    return transformed