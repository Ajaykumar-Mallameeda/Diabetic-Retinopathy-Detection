"""Classical filtering methods for retinal image preprocessing"""

import cv2
import numpy as np
from typing import List, Tuple


def gaussian_matched_filter_kernel(L: int, sigma: float, t: int = 3) -> np.ndarray:
    """
    Generate Gaussian matched filter kernel for blood vessel detection.

    Args:
        L: Length of the kernel
        sigma: Standard deviation of Gaussian
        t: Scale factor for kernel width

    Returns:
        2D Gaussian matched filter kernel
    """
    return _filter_kernel_mf_fdog(L, sigma, t, True)


def _filter_kernel_mf_fdog(L: int, sigma: float, t: int = 3, mf: bool = True) -> np.ndarray:
    """
    Internal function to generate filter kernels.

    Args:
        L: Length of the kernel
        sigma: Standard deviation
        t: Scale factor
        mf: Whether to generate matched filter or derivative of Gaussian

    Returns:
        2D filter kernel
    """
    dim_y = int(L)
    dim_x = 2 * int(t * sigma)
    arr = np.zeros((dim_y, dim_x), 'f')

    ctr_x = dim_x / 2
    ctr_y = int(dim_y / 2.)

    # Set elements to their x coordinate
    it = np.nditer(arr, flags=['multi_index'])
    while not it.finished:
        arr[it.multi_index] = it.multi_index[1] - ctr_x
        it.iternext()

    two_sigma_sq = 2 * sigma * sigma
    sqrt_w_pi_sigma = 1. / (np.sqrt(2 * np.pi) * sigma)

    if not mf:
        sqrt_w_pi_sigma = sqrt_w_pi_sigma / sigma ** 2

    def k_fun(x):
        return sqrt_w_pi_sigma * np.exp(-x * x / two_sigma_sq)

    def k_fun_derivative(x):
        return -x * sqrt_w_pi_sigma * np.exp(-x * x / two_sigma_sq)

    if mf:
        kernel = k_fun(arr)
        kernel = kernel - kernel.mean()
    else:
        kernel = k_fun_derivative(arr)

    return cv2.flip(kernel, -1)


def create_matched_filter_bank(K: np.ndarray, n: int = 12) -> List[np.ndarray]:
    """
    Create a bank of rotated matched filters.

    Args:
        K: Base kernel
        n: Number of rotations

    Returns:
        List of rotated kernels
    """
    rotate = 180 / n
    center = (K.shape[1] / 2, K.shape[0] / 2)
    cur_rot = 0
    kernels = [K]

    for i in range(1, n):
        cur_rot += rotate
        r_mat = cv2.getRotationMatrix2D(center, cur_rot, 1)
        k = cv2.warpAffine(K, r_mat, (K.shape[1], K.shape[0]))
        kernels.append(k)

    return kernels


def apply_filters(im: np.ndarray, kernels: List[np.ndarray]) -> np.ndarray:
    """
    Apply filter bank to image and return maximum response.

    Args:
        im: Input image
        kernels: List of filter kernels

    Returns:
        Maximum response image
    """
    images = np.array([cv2.filter2D(im, -1, k) for k in kernels])
    return np.max(images, 0)


def create_gabor_filter_bank() -> List[np.ndarray]:
    """
    Create a bank of Gabor filters for texture analysis.

    Returns:
        List of Gabor filter kernels
    """
    filters = []
    ksize = 31
    for theta in np.arange(0, np.pi, np.pi / 16):
        kern = cv2.getGaborKernel((ksize, ksize), 6, theta, 12, 0.37, 0, ktype=cv2.CV_32F)
        kern /= 1.5 * kern.sum()
        filters.append(kern)
    return filters


def adaptive_histogram_equalization(img: np.ndarray) -> np.ndarray:
    """
    Apply adaptive histogram equalization to enhance image contrast.

    Args:
        img: Input grayscale image

    Returns:
        Enhanced image
    """
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.equalizeHist(img)


def kmeans_segmentation(img: np.ndarray, k: int = 2) -> np.ndarray:
    """
    Apply K-means clustering for image segmentation.

    Args:
        img: Input image
        k: Number of clusters

    Returns:
        Segmented image
    """
    # Reshape image to be a list of pixels
    Z = img.reshape((-1, 3))

    # Convert to np.float32
    Z = np.float32(Z)

    # Define criteria and apply kmeans
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    _, label, center = cv2.kmeans(Z, k, None, criteria, 10, cv2.KMEANS_PP_CENTERS)

    # Convert back to uint8 and reshape to original image shape
    center = np.uint8(center)
    res = center[label.flatten()]
    return res.reshape((img.shape))