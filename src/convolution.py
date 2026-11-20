"""
CS4059: Fundamentals of Computer Vision - Fall 2026
Assignment 1: Industrial Defect Inspection & Surface Crack Detection

Task 1: Generalized 2D and 3D Convolution Engines from Mathematical First Principles.
Strict Prohibition: Zero use of cv2.filter2D, cv2.GaussianBlur, cv2.Sobel, cv2.Canny, or scipy filtering functions.
"""

import numpy as np


def conv2d(
    image: np.ndarray,
    kernel: np.ndarray,
    stride: int = 1,
    padding: str = "same",
    pad_value: float = 0.0,
) -> np.ndarray:
    """
    Pure 2D spatial convolution (cross-correlation) from mathematical first principles.

    Assumes odd-sized kernels for 'same' padding logic.

    Parameters:
    -----------
    image : np.ndarray
        2D single-channel input image of shape (H, W).
    kernel : np.ndarray
        2D filter kernel of shape (k_h, k_w) with odd spatial dimensions.
    stride : int, default=1
        Step size with which the kernel slides across the image.
    padding : str, default='same'
        Padding mode: 'same' (output size matches ceil(input / stride)) or 'valid' (no padding).
    pad_value : float, default=0.0
        Constant value used for boundary padding.

    Returns:
    --------
    output : np.ndarray
        2D convolved activation map satisfying the discrete spatial dimension formula.
    """
    if image.ndim != 2:
        raise ValueError(f"Expected 2D image array of shape (H, W), got shape {image.shape}")
    if kernel.ndim != 2:
        raise ValueError(f"Expected 2D kernel array of shape (k_h, k_w), got shape {kernel.shape}")
    if stride < 1:
        raise ValueError(f"Stride must be a positive integer >= 1, got {stride}")

    k_h, k_w = kernel.shape

    # Assumes odd-sized kernels for 'same' padding logic
    if padding.lower() == "same":
        pad_h = (k_h - 1) // 2
        pad_w = (k_w - 1) // 2
        padded_img = np.pad(
            image,
            ((pad_h, pad_h), (pad_w, pad_w)),
            mode="constant",
            constant_values=pad_value,
        )
    elif padding.lower() == "valid":
        pad_h = 0
        pad_w = 0
        padded_img = image
    else:
        raise ValueError(f"Padding mode must be 'same' or 'valid', got '{padding}'")

    # Output dimensions strictly satisfy the discrete spatial formula:
    # W_out = floor((W_in - F + 2P) / S) + 1, H_out = floor((H_in - F + 2P) / S) + 1
    out_h = (padded_img.shape[0] - k_h) // stride + 1
    out_w = (padded_img.shape[1] - k_w) // stride + 1

    output = np.zeros((out_h, out_w), dtype=np.float64)

    # Sliding window dot product
    for i in range(out_h):
        for j in range(out_w):
            r_start = i * stride
            c_start = j * stride
            region = padded_img[r_start : r_start + k_h, c_start : c_start + k_w]
            output[i, j] = np.sum(region * kernel)

    return output


def conv3d(
    image: np.ndarray,
    kernel_stack: np.ndarray,
    stride: int = 1,
    padding: str = "same",
    pad_value: float = 0.0,
) -> np.ndarray:
    """
    3D Multi-Channel Spatial Convolution Layer from mathematical first principles.

    Parameters:
    -----------
    image : np.ndarray
        3D multi-channel input image of shape (H, W, C).
    kernel_stack : np.ndarray
        Bank of K filters of shape (K, k_h, k_w, C), or a single 3D filter of shape (k_h, k_w, C).
    stride : int, default=1
        Step size with which the filters slide across spatial dimensions (H, W).
    padding : str, default='same'
        Padding mode: 'same' or 'valid'.
    pad_value : float, default=0.0
        Constant value used for boundary padding.

    Returns:
    --------
    output : np.ndarray
        3D activation volume of shape (H_out, W_out, K) as float64.
    """
    if image.ndim != 3:
        raise ValueError(f"Expected 3D input image of shape (H, W, C), got shape {image.shape}")

    # Standardize kernel_stack to shape (K, k_h, k_w, C)
    if kernel_stack.ndim == 3:
        kernel_stack = np.expand_dims(kernel_stack, axis=0)
    elif kernel_stack.ndim != 4:
        raise ValueError(
            f"Expected kernel_stack of shape (K, k_h, k_w, C) or (k_h, k_w, C), got {kernel_stack.shape}"
        )

    K, k_h, k_w, k_c = kernel_stack.shape
    H, W, C = image.shape

    if C != k_c:
        raise ValueError(
            f"Channel mismatch: image has {C} channels but kernel stack has depth {k_c}"
        )
    if stride < 1:
        raise ValueError(f"Stride must be a positive integer >= 1, got {stride}")

    # Spatial padding across height and width only
    if padding.lower() == "same":
        pad_h = (k_h - 1) // 2
        pad_w = (k_w - 1) // 2
        padded_img = np.pad(
            image,
            ((pad_h, pad_h), (pad_w, pad_w), (0, 0)),
            mode="constant",
            constant_values=pad_value,
        )
    elif padding.lower() == "valid":
        padded_img = image
    else:
        raise ValueError(f"Padding mode must be 'same' or 'valid', got '{padding}'")

    out_h = (padded_img.shape[0] - k_h) // stride + 1
    out_w = (padded_img.shape[1] - k_w) // stride + 1

    output = np.zeros((out_h, out_w, K), dtype=np.float64)

    # Convolve each filter k across the spatial grid
    for k in range(K):
        current_kernel = kernel_stack[k]  # shape: (k_h, k_w, C)
        for i in range(out_h):
            for j in range(out_w):
                r_start = i * stride
                c_start = j * stride
                region = padded_img[r_start : r_start + k_h, c_start : c_start + k_w, :]
                output[i, j, k] = np.sum(region * current_kernel)

    return output
