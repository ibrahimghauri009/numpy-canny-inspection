"""
CS4059: Fundamentals of Computer Vision - Fall 2026
Assignment 1: Industrial Defect Inspection & Surface Crack Detection

Task 3: First-Principles Canny Edge Detection Pipeline.
Strict Prohibition: Zero use of cv2.Sobel, cv2.Canny, or scipy filtering functions.
"""

from collections import deque
from typing import Dict, Tuple
import numpy as np

from convolution import conv2d
from filtering import generate_gaussian_kernel_2d


def sobel_gradients(
    image: np.ndarray,
    sigma: float = 1.0,
    ksize: int = None,
    pre_smooth: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Stage 1: Gaussian pre-smoothing and Sobel gradient computation from mathematical first principles.

    Normalized Sobel kernels:
        S_x = (1/8) * [[-1, 0, 1],
                       [-2, 0, 2],
                       [-1, 0, 1]]

        S_y = (1/8) * [[ 1,  2,  1],
                       [ 0,  0,  0],
                       [-1, -2, -1]]

    Magnitude: M = sqrt(S_x^2 + S_y^2)
    Orientation: theta = atan2(S_y, S_x) mapped to [0, 180) degrees.

    Parameters:
    -----------
    image : np.ndarray
        2D single-channel input image array of shape (H, W).
    sigma : float, default=1.0
        Standard deviation for Gaussian pre-smoothing.
    ksize : int, optional
        Gaussian kernel dimension. If None, chosen adaptively as 2 * ceil(3 * sigma) + 1.
    pre_smooth : bool, default=True
        Whether to apply Gaussian pre-smoothing before gradient calculation.

    Returns:
    --------
    smoothed : np.ndarray
        Gaussian smoothed image (or original image if pre_smooth=False).
    s_x : np.ndarray
        Horizontal gradient response map.
    s_y : np.ndarray
        Vertical gradient response map.
    magnitude : np.ndarray
        Gradient magnitude map M.
    orientation : np.ndarray
        Gradient orientation map in degrees within [0, 180).
    """
    if image.ndim != 2:
        raise ValueError(f"Expected 2D single-channel image, got shape {image.shape}")

    # Step 1: Pre-smoothing
    if pre_smooth:
        if ksize is None:
            radius = int(np.ceil(3.0 * sigma))
            ksize = 2 * radius + 1
            if ksize % 2 == 0:
                ksize += 1
        gauss_k = generate_gaussian_kernel_2d(ksize, sigma)
        smoothed = conv2d(image, gauss_k, stride=1, padding="same")
    else:
        smoothed = image.copy()

    # Step 2: Normalized Sobel Operators
    s_x_kernel = (
        np.array(
            [
                [-1.0, 0.0, 1.0],
                [-2.0, 0.0, 2.0],
                [-1.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        / 8.0
    )

    s_y_kernel = (
        np.array(
            [
                [1.0, 2.0, 1.0],
                [0.0, 0.0, 0.0],
                [-1.0, -2.0, -1.0],
            ],
            dtype=np.float64,
        )
        / 8.0
    )

    s_x = conv2d(smoothed, s_x_kernel, stride=1, padding="same")
    s_y = conv2d(smoothed, s_y_kernel, stride=1, padding="same")

    # Step 3: Gradient Magnitude and Direction
    magnitude = np.sqrt(s_x**2 + s_y**2)
    theta_rad = np.arctan2(s_y, s_x)

    # Convert orientation to degrees in [0, 180)
    theta_deg = np.rad2deg(theta_rad)
    theta_deg = (theta_deg + 180.0) % 180.0

    return smoothed, s_x, s_y, magnitude, theta_deg


def non_max_suppression(magnitude: np.ndarray, orientation: np.ndarray) -> np.ndarray:
    """
    Stage 2: 4-Sector Non-Maximum Suppression (NMS) from mathematical first principles.

    Converts continuous gradient orientations into 4 discrete angular sectors:
    - Sector 0 (0 deg +- 22.5 deg or 180 deg +- 22.5 deg): Horizontal neighbors (i, j - 1) and (i, j + 1)
    - Sector 1 (45 deg +- 22.5 deg): Diagonal neighbors (i - 1, j + 1) and (i + 1, j - 1)
    - Sector 2 (90 deg +- 22.5 deg): Vertical neighbors (i - 1, j) and (i + 1, j)
    - Sector 3 (135 deg +- 22.5 deg): Anti-diagonal neighbors (i - 1, j - 1) and (i + 1, j + 1)

    Suppression Rule:
    Pixel is set to 0 if its magnitude is NOT strictly greater than both directional neighbors.
    Image borders are set to 0.

    Parameters:
    -----------
    magnitude : np.ndarray
        Gradient magnitude map of shape (H, W).
    orientation : np.ndarray
        Gradient orientation map in degrees [0, 180) of shape (H, W).

    Returns:
    --------
    nms : np.ndarray
        Thinned 1-pixel wide edge ridges of shape (H, W) as float64.
    """
    H, W = magnitude.shape
    nms = np.zeros((H, W), dtype=np.float64)

    # Quantize angles into 4 sectors for interior pixels
    # Sectors:
    # 0: [0, 22.5) or [157.5, 180)
    # 1: [22.5, 67.5)
    # 2: [67.5, 112.5)
    # 3: [112.5, 157.5)

    # Vectorized neighbor comparison for interior pixels [1:H-1, 1:W-1]
    sub_mag = magnitude[1 : H - 1, 1 : W - 1]
    sub_theta = orientation[1 : H - 1, 1 : W - 1]

    # Sector masks
    sec0 = (sub_theta < 22.5) | (sub_theta >= 157.5)
    sec1 = (sub_theta >= 22.5) & (sub_theta < 67.5)
    sec2 = (sub_theta >= 67.5) & (sub_theta < 112.5)
    sec3 = (sub_theta >= 112.5) & (sub_theta < 157.5)

    q = np.zeros_like(sub_mag)
    r = np.zeros_like(sub_mag)

    # Sector 0: Horizontal (i, j-1) and (i, j+1)
    q[sec0] = magnitude[1 : H - 1, 0 : W - 2][sec0]
    r[sec0] = magnitude[1 : H - 1, 2:W][sec0]

    # Sector 1: Diagonal (i-1, j+1) and (i+1, j-1)
    q[sec1] = magnitude[0 : H - 2, 2:W][sec1]
    r[sec1] = magnitude[2:H, 0 : W - 2][sec1]

    # Sector 2: Vertical (i-1, j) and (i+1, j)
    q[sec2] = magnitude[0 : H - 2, 1 : W - 1][sec2]
    r[sec2] = magnitude[2:H, 1 : W - 1][sec2]

    # Sector 3: Anti-diagonal (i-1, j-1) and (i+1, j+1)
    q[sec3] = magnitude[0 : H - 2, 0 : W - 2][sec3]
    r[sec3] = magnitude[2:H, 2:W][sec3]

    # Strict inequality suppression
    keep = (sub_mag > q) & (sub_mag > r)
    nms[1 : H - 1, 1 : W - 1][keep] = sub_mag[keep]

    # Borders remain 0 as initialized
    return nms


def hysteresis_tracking(
    img_nms: np.ndarray,
    low_thresh: float,
    high_thresh: float,
) -> np.ndarray:
    """
    Stage 3: Double thresholding and 8-connectivity edge linking via Breadth-First Search (BFS).

    Parameters:
    -----------
    img_nms : np.ndarray
        Non-maximum suppressed gradient magnitude map of shape (H, W).
    low_thresh : float
        Lower threshold T_low for weak edge candidates.
    high_thresh : float
        Higher threshold T_high for strong edge candidates.

    Returns:
    --------
    output : np.ndarray
        Binary edge map of shape (H, W) with edge pixels as 255 and non-edges as 0 (uint8).
    """
    H, W = img_nms.shape
    strong_i, strong_j = np.where(img_nms >= high_thresh)

    output = np.zeros((H, W), dtype=np.uint8)
    output[strong_i, strong_j] = 255

    # Breadth-first edge tracking using an efficient deque (O(1) pop operations)
    queue = deque(zip(strong_i, strong_j))

    while queue:
        ci, cj = queue.popleft()
        for ni in range(max(0, ci - 1), min(H, ci + 2)):
            for nj in range(max(0, cj - 1), min(W, cj + 2)):
                if (img_nms[ni, nj] >= low_thresh) and (output[ni, nj] == 0):
                    output[ni, nj] = 255
                    queue.append((ni, nj))

    return output


def canny_edge_detector(
    image: np.ndarray,
    sigma: float = 1.4,
    low_thresh: float = 20.0,
    high_thresh: float = 50.0,
    ksize: int = None,
) -> Dict[str, np.ndarray]:
    """
    Complete 4-Stage First-Principles Canny Edge Detection Pipeline.

    Parameters:
    -----------
    image : np.ndarray
        Input 2D single-channel grayscale image array.
    sigma : float, default=1.4
        Standard deviation for Gaussian pre-smoothing.
    low_thresh : float, default=20.0
        Lower threshold for hysteresis edge linking.
    high_thresh : float, default=50.0
        Upper threshold for strong edge seed identification.
    ksize : int, optional
        Odd kernel size for Gaussian filter.

    Returns:
    --------
    results : dict
        Dictionary containing intermediate and final activation maps:
        - 'smoothed': Gaussian filtered image
        - 'grad_x': Horizontal Sobel gradient
        - 'grad_y': Vertical Sobel gradient
        - 'magnitude': Gradient magnitude map M
        - 'orientation': Gradient orientation in degrees [0, 180)
        - 'nms': Non-maximum suppressed thin edge map
        - 'edges': Final binary edge map after 8-connected hysteresis tracking (uint8)
    """
    # Stage 1: Smoothing and Sobel Gradients
    smoothed, s_x, s_y, magnitude, orientation = sobel_gradients(
        image=image, sigma=sigma, ksize=ksize, pre_smooth=True
    )

    # Stage 2: 4-Sector Non-Maximum Suppression
    nms = non_max_suppression(magnitude, orientation)

    # Stage 3: Double Thresholding and 8-Connected Hysteresis
    edges = hysteresis_tracking(nms, low_thresh=low_thresh, high_thresh=high_thresh)

    return {
        "smoothed": smoothed,
        "grad_x": s_x,
        "grad_y": s_y,
        "magnitude": magnitude,
        "orientation": orientation,
        "nms": nms,
        "edges": edges,
    }
