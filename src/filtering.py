"""
CS4059: Fundamentals of Computer Vision - Fall 2026
Assignment 1: Industrial Defect Inspection & Surface Crack Detection

Task 2: Filtering, Contrast Sharpening & Subsampling from Mathematical First Principles.
Strict Prohibition: Zero use of built-in processing functions (cv2.filter2D, cv2.GaussianBlur, etc.).
"""

from typing import Tuple, Union
import numpy as np
from convolution import conv2d


def generate_gaussian_kernel_2d(ksize: int, sigma: float) -> np.ndarray:
    """
    Generate an isotropic 2D Gaussian filter kernel from mathematical first principles.

    G_sigma(x, y) = (1 / (2 * pi * sigma^2)) * exp(-(x^2 + y^2) / (2 * sigma^2))

    Parameters:
    -----------
    ksize : int
        Odd integer dimension of the square kernel (e.g., 3, 5, 7).
    sigma : float
        Standard deviation of the Gaussian distribution (sigma > 0).

    Returns:
    --------
    kernel : np.ndarray
        2D normalized Gaussian kernel of shape (ksize, ksize) summing to 1.0.
    """
    if ksize % 2 == 0 or ksize < 1:
        raise ValueError(f"ksize must be an odd positive integer, got {ksize}")
    if sigma <= 0:
        raise ValueError(f"sigma must be strictly positive, got {sigma}")

    radius = (ksize - 1) // 2
    # Coordinate grid centered at (0, 0)
    x = np.arange(-radius, radius + 1, dtype=np.float64)
    y = np.arange(-radius, radius + 1, dtype=np.float64)
    xx, yy = np.meshgrid(x, y)

    # Isotropic 2D Gaussian density function
    normalizer = 1.0 / (2.0 * np.pi * (sigma**2))
    kernel = normalizer * np.exp(-(xx**2 + yy**2) / (2.0 * (sigma**2)))

    # Discrete spatial normalization to guarantee energy conservation (sum = 1)
    kernel_sum = np.sum(kernel)
    if kernel_sum > 0:
        kernel = kernel / kernel_sum

    return kernel


def unsharp_mask(
    image: np.ndarray,
    sigma: float = 1.5,
    alpha: float = 1.2,
    ksize: int = None,
) -> np.ndarray:
    """
    Apply unsharp masking (high-pass detail accentuation) to an image.

    F_sharp = F + alpha * (F - F * G_sigma)

    Parameters:
    -----------
    image : np.ndarray
        Input 2D grayscale image array of shape (H, W).
    sigma : float, default=1.5
        Gaussian smoothing scale parameter.
    alpha : float, default=1.2
        High-pass amplification gain factor.
    ksize : int, optional
        Gaussian kernel size. If None, chosen adaptively as 2 * ceil(3 * sigma) + 1.

    Returns:
    --------
    f_sharp : np.ndarray
        Sharpened image strictly clipped to [0, 255] as float64.
    """
    if image.ndim != 2:
        raise ValueError(f"Expected 2D grayscale image, got shape {image.shape}")

    # Determine odd kernel size covering at least 3 standard deviations
    if ksize is None:
        radius = int(np.ceil(3.0 * sigma))
        ksize = 2 * radius + 1
        if ksize % 2 == 0:
            ksize += 1

    gaussian_kernel = generate_gaussian_kernel_2d(ksize, sigma)

    # Low-pass smoothed baseline
    f_blur = conv2d(image, gaussian_kernel, stride=1, padding="same")

    # High-pass detail mask = Original - Low-Pass
    detail_mask = image - f_blur

    # Detail amplification
    f_sharp = image + alpha * detail_mask

    # Strictly clip output values to [0, 255] to prevent overflow/underflow artifacts
    f_sharp_clipped = np.clip(f_sharp, 0.0, 255.0)

    return f_sharp_clipped


def pool2d(
    feature_map: np.ndarray,
    pool_size: Union[int, Tuple[int, int]] = (2, 2),
    stride: int = 2,
    mode: str = "max",
) -> np.ndarray:
    """
    Perform 2D spatial pooling (feature subsampling) supporting 'max' and 'average' modes.

    Parameters:
    -----------
    feature_map : np.ndarray
        2D array (H, W) or 3D array (H, W, C).
    pool_size : int or tuple of (int, int), default=(2, 2)
        Spatial extent of the pooling window (p_h, p_w).
    stride : int, default=2
        Step size with which the pooling window slides.
    mode : str, default='max'
        Pooling mode: 'max' (maximum activation) or 'average' (mean activation).

    Returns:
    --------
    output : np.ndarray
        Subsampled pooled feature map satisfying discrete spatial dimension formula:
        H_out = floor((H_in - p_h) / stride) + 1
        W_out = floor((W_in - p_w) / stride) + 1
    """
    if isinstance(pool_size, int):
        p_h, p_w = pool_size, pool_size
    elif isinstance(pool_size, (tuple, list)) and len(pool_size) == 2:
        p_h, p_w = pool_size[0], pool_size[1]
    else:
        raise ValueError(f"Invalid pool_size: {pool_size}")

    if stride < 1:
        raise ValueError(f"Stride must be a positive integer >= 1, got {stride}")

    mode_lower = mode.lower()
    if mode_lower not in ["max", "average", "avg"]:
        raise ValueError(f"Mode must be 'max' or 'average', got '{mode}'")

    is_2d = feature_map.ndim == 2
    if is_2d:
        H, W = feature_map.shape
        data = np.expand_dims(feature_map, axis=2)
        C = 1
    elif feature_map.ndim == 3:
        H, W, C = feature_map.shape
        data = feature_map
    else:
        raise ValueError(f"Expected 2D or 3D array, got ndim={feature_map.ndim}")

    out_h = (H - p_h) // stride + 1
    out_w = (W - p_w) // stride + 1

    if out_h <= 0 or out_w <= 0:
        raise ValueError(
            f"Pooling window ({p_h}, {p_w}) larger than feature map ({H}, {W})"
        )

    output = np.zeros((out_h, out_w, C), dtype=np.float64)

    for i in range(out_h):
        for j in range(out_w):
            r_start = i * stride
            c_start = j * stride
            window = data[r_start : r_start + p_h, c_start : c_start + p_w, :]

            if mode_lower == "max":
                output[i, j, :] = np.max(window, axis=(0, 1))
            else:  # average
                output[i, j, :] = np.mean(window, axis=(0, 1))

    return output[:, :, 0] if is_2d else output
