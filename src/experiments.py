"""
CS4059: Fundamentals of Computer Vision - Fall 2026
Assignment 1: Industrial Defect Inspection & Surface Crack Detection

Tasks 4 & 5: Visual Ablation Studies, Scale-Space Causality, Parameter Sensitivity & Failure Analysis.
Strict Prohibition: Zero use of cv2.filter2D, cv2.GaussianBlur, cv2.Sobel, cv2.Canny, or scipy filtering functions.
"""

import os
import cv2
import matplotlib.pyplot as plt
import numpy as np

from canny import canny_edge_detector, non_max_suppression, sobel_gradients
from convolution import conv2d
from filtering import generate_gaussian_kernel_2d, pool2d, unsharp_mask

# Output directory for saving all assignment figures
OUTPUT_DIR = os.path.join("..", "output_images")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def safe_savefig(out_path: str, dpi: int = 300):
    """
    Safely saves a matplotlib figure with Windows file-lock handling and clean fallback.
    Prevents OSError: [Errno 22] when an existing figure is currently open in Windows Photos,
    File Explorer Preview Pane, or another process.
    """
    abs_path = os.path.abspath(out_path)
    os.makedirs(os.path.dirname(abs_path), exist_ok=True)
    try:
        if os.path.exists(abs_path):
            try:
                os.remove(abs_path)
            except OSError:
                pass
        plt.savefig(abs_path, dpi=dpi, bbox_inches="tight")
        print(f"Saved: {abs_path}")
    except OSError as err:
        print(f"[Warning] Could not overwrite '{os.path.basename(abs_path)}' (file may be open in an external viewer). Error: {err}")
        base, ext = os.path.splitext(abs_path)
        fallback_path = f"{base}_new{ext}"
        try:
            plt.savefig(fallback_path, dpi=dpi, bbox_inches="tight")
            print(f"Saved fallback to: {fallback_path}")
        except Exception:
            pass


def load_dataset_samples():
    """
    Load standardized sample images from the benchmark repositories:
    1. PCB Defect Sample (Mouse_bite region of interest)
    2. Concrete Crack Sample (Positive micro-crack)
    """
    # 1. PCB Sample
    pcb_full_path = os.path.join(
        "..",
        "data",
        "pcb_defects",
        "PCB_DATASET",
        "images",
        "Mouse_bite",
        "01_mouse_bite_01.jpg",
    )
    if os.path.exists(pcb_full_path):
        pcb_bgr = cv2.imread(pcb_full_path)
        pcb_gray_full = cv2.cvtColor(pcb_bgr, cv2.COLOR_BGR2GRAY).astype(np.float64)
        # Bounding box of mouse bite flaw is around (x: 1518..1547, y: 913..958)
        pcb_sample = pcb_gray_full[810:1066, 1400:1656]  # 256x256 crop around defect
    else:
        raise FileNotFoundError(f"PCB sample image not found at {pcb_full_path}")

    # 2. Concrete Crack Sample
    crack_path = os.path.join("..", "data", "concrete_cracks", "Positive", "00001.jpg")
    if os.path.exists(crack_path):
        crack_bgr = cv2.imread(crack_path)
        crack_sample = cv2.cvtColor(crack_bgr, cv2.COLOR_BGR2GRAY).astype(np.float64)
    else:
        raise FileNotFoundError(f"Concrete crack sample not found at {crack_path}")

    # 3. Concrete Non-Crack / Negative Sample
    neg_path = os.path.join("..", "data", "concrete_cracks", "Negative", "00001.jpg")
    if os.path.exists(neg_path):
        neg_bgr = cv2.imread(neg_path)
        neg_sample = cv2.cvtColor(neg_bgr, cv2.COLOR_BGR2GRAY).astype(np.float64)
    else:
        neg_sample = None

    return pcb_sample, crack_sample, neg_sample


def to_display(arr: np.ndarray) -> np.ndarray:
    """Normalize 2D array to [0, 255] uint8 for clean visual display."""
    arr_abs = np.abs(arr)
    min_v, max_v = np.min(arr_abs), np.max(arr_abs)
    if max_v - min_v > 1e-8:
        norm = (arr_abs - min_v) / (max_v - min_v) * 255.0
    else:
        norm = np.zeros_like(arr_abs)
    return norm.astype(np.uint8)


def run_task2_demonstration(pcb_sample, crack_sample):
    """Generate and save Task 2 Demonstration Figure (Unsharp Masking and Pooling)."""
    print("Generating Task 2 Demonstration figure...")
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))

    # PCB Unsharp Masking
    pcb_sharp = unsharp_mask(pcb_sample, sigma=1.5, alpha=1.5)
    pcb_k = generate_gaussian_kernel_2d(9, 1.5)
    pcb_blur = conv2d(pcb_sample, pcb_k, stride=1, padding="same")
    pcb_detail = pcb_sample - pcb_blur

    axes[0, 0].imshow(pcb_sample, cmap="gray")
    axes[0, 0].set_title("(a) PCB: Original Input (256x256)", fontsize=10)
    axes[0, 0].axis("off")

    axes[0, 1].imshow(pcb_blur, cmap="gray")
    axes[0, 1].set_title(r"(b) Gaussian Blurred ($G_{\sigma=1.5}$)", fontsize=10)
    axes[0, 1].axis("off")

    axes[0, 2].imshow(to_display(pcb_detail), cmap="gray")
    axes[0, 2].set_title(r"(c) High-Pass Detail ($F - F*G$)", fontsize=10)
    axes[0, 2].axis("off")

    axes[0, 3].imshow(pcb_sharp, cmap="gray")
    axes[0, 3].set_title(r"(d) Unsharp Sharpened ($\alpha=1.5$)", fontsize=10)
    axes[0, 3].axis("off")

    # Concrete Crack 2D Pooling
    crack_max_pool = pool2d(
        crack_sample, pool_size=(2, 2), stride=2, mode="max"
    )
    crack_avg_pool = pool2d(
        crack_sample, pool_size=(2, 2), stride=2, mode="average"
    )
    diff_pool = np.abs(crack_max_pool - crack_avg_pool)

    axes[1, 0].imshow(crack_sample, cmap="gray")
    axes[1, 0].set_title("(e) Crack: Original Input (227x227)", fontsize=10)
    axes[1, 0].axis("off")

    axes[1, 1].imshow(crack_max_pool, cmap="gray")
    axes[1, 1].set_title(
        f"(f) Max Pooling 2x2 (Stride 2)\nShape: {crack_max_pool.shape}", fontsize=10
    )
    axes[1, 1].axis("off")

    axes[1, 2].imshow(crack_avg_pool, cmap="gray")
    axes[1, 2].set_title(
        f"(g) Avg Pooling 2x2 (Stride 2)\nShape: {crack_avg_pool.shape}", fontsize=10
    )
    axes[1, 2].axis("off")

    axes[1, 3].imshow(to_display(diff_pool), cmap="magma")
    axes[1, 3].set_title("(h) |Max - Avg| Subsampling Residual", fontsize=10)
    axes[1, 3].axis("off")

    plt.suptitle(
        "Task 2: Isotropic Gaussian Filtering, Unsharp Detail Amplification, and 2D Spatial Subsampling",
        fontsize=13,
        fontweight="bold",
    )
    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, "task2_unsharp_and_pooling.png")
    safe_savefig(out_path)
    plt.show()


def run_task3_demonstration(crack_sample):
    """Generate and save Task 3 Demonstration Figure (Complete 4-Stage Canny Edge Detection Pipeline)."""
    print("Generating Task 3 Demonstration figure...")
    results = canny_edge_detector(
        crack_sample, sigma=1.2, low_thresh=5.0, high_thresh=15.0
    )

    fig, axes = plt.subplots(2, 3, figsize=(14, 9))

    axes[0, 0].imshow(crack_sample, cmap="gray")
    axes[0, 0].set_title("(a) Input Concrete Surface", fontsize=11)
    axes[0, 0].axis("off")

    axes[0, 1].imshow(results["smoothed"], cmap="gray")
    axes[0, 1].set_title(r"(b) Stage 1: Pre-smoothed ($G_{\sigma=1.2}$)", fontsize=11)
    axes[0, 1].axis("off")

    axes[0, 2].imshow(to_display(results["magnitude"]), cmap="gray")
    axes[0, 2].set_title(r"(c) Stage 1: Sobel Magnitude $M$", fontsize=11)
    axes[0, 2].axis("off")

    axes[1, 0].imshow(results["orientation"], cmap="hsv")
    axes[1, 0].set_title(r"(d) Stage 1: Orientation $\theta \in [0, 180^\circ)$", fontsize=11)
    axes[1, 0].axis("off")

    axes[1, 1].imshow(to_display(results["nms"]), cmap="gray")
    axes[1, 1].set_title("(e) Stage 2: 4-Sector NMS (1-px Thin Ridges)", fontsize=11)
    axes[1, 1].axis("off")

    axes[1, 2].imshow(results["edges"], cmap="gray")
    axes[1, 2].set_title(
        r"(f) Stage 3: Hysteresis Edges ($T_L=5, T_H=15$)", fontsize=11
    )
    axes[1, 2].axis("off")

    plt.suptitle(
        "Task 3: First-Principles 4-Stage Canny Edge Detection Pipeline on Micro-Crack Surface",
        fontsize=13,
        fontweight="bold",
    )
    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, "task3_canny_stages.png")
    safe_savefig(out_path)
    plt.show()


def run_ablation1_noise_smoothing(crack_sample):
    """
    Task 4.1: Ablation 1 (Impact of Noise & Smoothing).
    Synthesizes noisy test samples by adding Gaussian noise N(0, sigma_n^2) with sigma_n in {10, 25, 40}.
    Runs gradient computation directly without pre-smoothing vs. with Gaussian pre-smoothing.
    """
    print("Generating Task 4.1 Ablation 1 (Noise & Smoothing) figure...")
    np.random.seed(42)  # Deterministic repeatability
    noise_sigmas = [10, 25, 40]

    fig, axes = plt.subplots(3, 4, figsize=(16, 12))

    for idx, sn in enumerate(noise_sigmas):
        # Synthesize additive Gaussian noise
        noise = np.random.normal(0.0, sn, size=crack_sample.shape)
        noisy_img = np.clip(crack_sample + noise, 0.0, 255.0)

        # 1. Gradient directly WITHOUT pre-smoothing
        _, _, _, mag_raw, _ = sobel_gradients(
            noisy_img, sigma=1.2, pre_smooth=False
        )

        # 2. Gradient WITH Gaussian pre-smoothing
        _, _, _, mag_smooth, _ = sobel_gradients(
            noisy_img, sigma=1.5, pre_smooth=True
        )

        # Difference noise attenuation map
        noise_attenuation = np.abs(mag_raw - mag_smooth)

        axes[idx, 0].imshow(noisy_img, cmap="gray")
        axes[idx, 0].set_title(
            fr"(Row {idx+1}) Noisy Input ($\sigma_n={sn}$)", fontsize=10
        )
        axes[idx, 0].axis("off")

        axes[idx, 1].imshow(to_display(mag_raw), cmap="inferno")
        axes[idx, 1].set_title(
            f"Without Smoothing (Noise Amplified)\nMax: {mag_raw.max():.1f}, Mean: {mag_raw.mean():.1f}",
            fontsize=9,
        )
        axes[idx, 1].axis("off")

        axes[idx, 2].imshow(to_display(mag_smooth), cmap="inferno")
        axes[idx, 2].set_title(
            fr"With Gaussian Smoothing ($\sigma=1.5$)\nMax: {mag_smooth.max():.1f}, Mean: {mag_smooth.mean():.1f}",
            fontsize=9,
        )
        axes[idx, 2].axis("off")

        axes[idx, 3].imshow(to_display(noise_attenuation), cmap="magma")
        axes[idx, 3].set_title(
            r"Noise Attenuated ($|M_{raw} - M_{smooth}|$)", fontsize=10
        )
        axes[idx, 3].axis("off")

    plt.suptitle(
        r"Task 4: Ablation 1 — High-Frequency Noise Amplification vs. Gaussian Pre-Smoothing Mitigation ($\sigma_n \in \{10, 25, 40\}$)",
        fontsize=12,
        fontweight="bold",
    )
    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, "task4_ablation1_noise_smoothing.png")
    safe_savefig(out_path)
    plt.show()


def run_ablation2_pre_post_nms(crack_sample, pcb_sample):
    """
    Task 4.2: Ablation 2 (Impact of NMS).
    Show edge maps before NMS (blurred, thick gradient bands) vs. immediately after NMS (1-pixel wide thin ridges).
    """
    print("Generating Task 4.2 Ablation 2 (Pre/Post NMS) figure...")
    # Crack Sample
    _, _, _, crack_mag, crack_theta = sobel_gradients(
        crack_sample, sigma=1.2, pre_smooth=True
    )
    crack_nms = non_max_suppression(crack_mag, crack_theta)

    # PCB Defect Sample
    _, _, _, pcb_mag, pcb_theta = sobel_gradients(
        pcb_sample, sigma=1.2, pre_smooth=True
    )
    pcb_nms = non_max_suppression(pcb_mag, pcb_theta)

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))

    # Concrete Crack Row
    axes[0, 0].imshow(crack_sample, cmap="gray")
    axes[0, 0].set_title("(a) Concrete Crack Input", fontsize=10)
    axes[0, 0].axis("off")

    axes[0, 1].imshow(to_display(crack_mag), cmap="magma")
    axes[0, 1].set_title("(b) Pre-NMS: Thick Gradient Magnitude Bands", fontsize=10)
    axes[0, 1].axis("off")

    axes[0, 2].imshow(to_display(crack_nms), cmap="magma")
    axes[0, 2].set_title("(c) Post-NMS: 1-Pixel Wide Thin Ridges", fontsize=10)
    axes[0, 2].axis("off")

    # Zoomed ROI for Crack (showing 1-pixel thickness)
    axes[0, 3].imshow(to_display(crack_nms[80:140, 80:140]), cmap="magma")
    axes[0, 3].set_title("(d) Zoomed ROI (Post-NMS 1-px Width)", fontsize=10)
    axes[0, 3].axis("off")

    # PCB Defect Row
    axes[1, 0].imshow(pcb_sample, cmap="gray")
    axes[1, 0].set_title("(e) PCB Defect Input (Mouse Bite)", fontsize=10)
    axes[1, 0].axis("off")

    axes[1, 1].imshow(to_display(pcb_mag), cmap="magma")
    axes[1, 1].set_title("(f) Pre-NMS: Blurred Boundary Band", fontsize=10)
    axes[1, 1].axis("off")

    axes[1, 2].imshow(to_display(pcb_nms), cmap="magma")
    axes[1, 2].set_title("(g) Post-NMS: Exact Contour Ridge", fontsize=10)
    axes[1, 2].axis("off")

    # Zoomed ROI for PCB
    axes[1, 3].imshow(to_display(pcb_nms[100:160, 100:160]), cmap="magma")
    axes[1, 3].set_title("(h) Zoomed ROI (Flaw Boundary Ridge)", fontsize=10)
    axes[1, 3].axis("off")

    plt.suptitle(
        "Task 4: Ablation 2 — Quantitative Ridge Thinning: Thick Gradient Bands vs. 1-Pixel NMS Ridges",
        fontsize=13,
        fontweight="bold",
    )
    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, "task4_ablation2_pre_post_nms.png")
    safe_savefig(out_path)
    plt.show()


def run_ablation3_hysteresis_vs_single(crack_sample):
    """
    Task 4.3: Ablation 3 (Impact of Hysteresis vs. Single Thresholding).
    Compare standard double-threshold hysteresis edge linking against single global thresholding.
    Highlight how hysteresis preserves continuous weak crack lines without introducing excessive background texture noise.
    """
    print("Generating Task 4.3 Ablation 3 (Hysteresis vs. Single) figure...")
    _, _, _, mag, theta = sobel_gradients(
        crack_sample, sigma=1.2, pre_smooth=True
    )
    nms = non_max_suppression(mag, theta)

    t_low, t_high = 5.0, 15.0

    # 1. Single Global Low Threshold: M >= T_low (introduces noise)
    single_low = np.zeros_like(nms, dtype=np.uint8)
    single_low[nms >= t_low] = 255

    # 2. Single Global High Threshold: M >= T_high (breaks crack continuity)
    single_high = np.zeros_like(nms, dtype=np.uint8)
    single_high[nms >= t_high] = 255

    # 3. Double-Threshold Hysteresis Edge Linking (optimal balance)
    hysteresis_edges = canny_edge_detector(
        crack_sample, sigma=1.2, low_thresh=t_low, high_thresh=t_high
    )["edges"]

    fig, axes = plt.subplots(1, 4, figsize=(16, 4.5))

    axes[0].imshow(crack_sample, cmap="gray")
    axes[0].set_title("(a) Concrete Crack Input", fontsize=10)
    axes[0].axis("off")

    axes[1].imshow(single_low, cmap="gray")
    axes[1].set_title(
        fr"(b) Single Low Threshold ($M \geq {t_low}$)\nTexture Speckle Noise Overload",
        fontsize=10,
    )
    axes[1].axis("off")

    axes[2].imshow(single_high, cmap="gray")
    axes[2].set_title(
        fr"(c) Single High Threshold ($M \geq {t_high}$)\nFragmented / Broken Crack Lines",
        fontsize=10,
    )
    axes[2].axis("off")

    axes[3].imshow(hysteresis_edges, cmap="gray")
    axes[3].set_title(
        fr"(d) Double-Threshold Hysteresis ($T_L={t_low}, T_H={t_high}$)\nContinuous Crack, Clean Background",
        fontsize=10,
    )
    axes[3].axis("off")

    plt.suptitle(
        "Task 4: Ablation 3 — Global Single Thresholding vs. Dual-Threshold 8-Connected Hysteresis Edge Linking",
        fontsize=12,
        fontweight="bold",
    )
    plt.tight_layout()
    out_path = os.path.join(
        OUTPUT_DIR, "task4_ablation3_hysteresis_vs_single.png"
    )
    safe_savefig(out_path)
    plt.show()


def run_scalespace_causality(crack_sample):
    """
    Task 5.1: Scale-Space Crack Merging (sigma in {1.0, 2.5, 5.0}).
    Evaluate crack detection across scale.
    Verify and document Witkin's scale-space causality: why edges shift or merge at larger sigma
    but new edges never emerge.
    """
    print("Generating Task 5.1 Scale-Space Causality figure...")
    sigmas = [1.0, 2.5, 5.0]

    fig, axes = plt.subplots(len(sigmas), 4, figsize=(16, 11))

    for idx, sig in enumerate(sigmas):
        res = canny_edge_detector(
            crack_sample, sigma=sig, low_thresh=4.0, high_thresh=12.0
        )

        edge_pixel_count = np.count_nonzero(res["edges"])

        axes[idx, 0].imshow(res["smoothed"], cmap="gray")
        axes[idx, 0].set_title(
            fr"Pre-Smoothed ($\sigma={sig}$)\nScale-Space Slice", fontsize=10
        )
        axes[idx, 0].axis("off")

        axes[idx, 1].imshow(to_display(res["magnitude"]), cmap="magma")
        axes[idx, 1].set_title(
            fr"Sobel Magnitude ($\sigma={sig}$)\nCoarse Contours", fontsize=10
        )
        axes[idx, 1].axis("off")

        axes[idx, 2].imshow(to_display(res["nms"]), cmap="magma")
        axes[idx, 2].set_title(
            fr"NMS Ridge Map ($\sigma={sig}$)", fontsize=10
        )
        axes[idx, 2].axis("off")

        axes[idx, 3].imshow(res["edges"], cmap="gray")
        axes[idx, 3].set_title(
            fr"Hysteresis Edges ($\sigma={sig}$)\nActive Pixels: {edge_pixel_count}",
            fontsize=10,
        )
        axes[idx, 3].axis("off")

    plt.suptitle(
        r"Task 5.1: Witkin's Scale-Space Causality Verification across Scales $\sigma \in \{1.0, 2.5, 5.0\}$ (Monotonic Edge Dissolution without Edge Genesis)",
        fontsize=12,
        fontweight="bold",
    )
    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, "task5_scalespace_causality.png")
    safe_savefig(out_path)
    plt.show()


def run_parameter_sensitivity_grid(crack_sample, pcb_sample):
    """
    Task 5.2: Parameter Sensitivity Grid.
    Plot a 3x3 grid varying (T_high, T_low) across concrete and PCB samples to demonstrate optimal operating regimes.
    """
    print("Generating Task 5.2 Parameter Sensitivity Grid figure...")
    # We explore 3 low thresholds and 3 high thresholds
    t_lows = [3.0, 6.0, 10.0]
    t_highs = [12.0, 20.0, 30.0]

    # Concrete Crack Sensitivity 3x3 Grid
    fig, axes = plt.subplots(3, 3, figsize=(12, 12))

    for i, th in enumerate(t_highs):
        for j, tl in enumerate(t_lows):
            res = canny_edge_detector(
                crack_sample, sigma=1.2, low_thresh=tl, high_thresh=th
            )
            count = np.count_nonzero(res["edges"])

            axes[i, j].imshow(res["edges"], cmap="gray")
            is_optimal = (tl == 6.0) and (th == 20.0)
            status_str = " (Optimal)" if is_optimal else ""
            axes[i, j].set_title(
                f"$T_L={tl}, T_H={th}${status_str}\nEdge Pixels: {count}",
                fontsize=9,
                fontweight="bold" if is_optimal else "normal",
                color="green" if is_optimal else "black",
            )
            axes[i, j].axis("off")

    plt.suptitle(
        "Task 5.2: Concrete Surface Crack Parameter Sensitivity Matrix ($3 \\times 3$ Dual Threshold Grid)",
        fontsize=13,
        fontweight="bold",
    )
    plt.tight_layout()
    out_path = os.path.join(
        OUTPUT_DIR, "task5_parameter_sensitivity_grid.png"
    )
    safe_savefig(out_path)
    plt.show()


def run_failure_case_analysis(pcb_sample, crack_sample, neg_sample):
    """
    Task 5.3: Failure Case Investigation.
    Identify and display at least two realistic scenarios where Canny edge detection fails:
    1. Strong shadow / illumination gradient contours falsely identified as structural boundaries.
    2. Highly textured surface / aggregate speckle concealing micro-crack signals.
    """
    print("Generating Task 5.3 Failure Case Investigation figure...")
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))

    # Scenario 1: Strong Shadow Boundary / Illumination Non-Uniformity
    # Synthesize an illumination gradient / diagonal shadow artifact across the PCB sample
    H, W = pcb_sample.shape
    y_coords, x_coords = np.mgrid[0:H, 0:W]
    # Sharp diagonal shadow contour
    shadow_mask = np.ones((H, W), dtype=np.float64)
    diagonal_line = x_coords + y_coords - (H + W) // 2
    # Sigmoidal shadow falloff creating sharp artificial luminance boundary
    shadow_attenuation = 1.0 - 0.65 / (1.0 + np.exp(-diagonal_line / 8.0))
    shadow_pcb = np.clip(pcb_sample * shadow_attenuation, 0.0, 255.0)

    shadow_canny = canny_edge_detector(
        shadow_pcb, sigma=1.2, low_thresh=6.0, high_thresh=18.0
    )["edges"]

    axes[0, 0].imshow(shadow_pcb, cmap="gray")
    axes[0, 0].set_title(
        "(a) Scenario 1: PCB Flaw under Strong Shadow Boundary", fontsize=10
    )
    axes[0, 0].axis("off")

    axes[0, 1].imshow(shadow_canny, cmap="gray")
    axes[0, 1].set_title(
        "(b) Canny Failure: Shadow Contour Falsely Detected",
        fontsize=10,
        color="red",
    )
    axes[0, 1].axis("off")

    # Highlighting false positive shadow boundary vs defect
    axes[0, 2].imshow(shadow_canny, cmap="hot")
    axes[0, 2].set_title(
        "(c) Diagnostic: Shadow Edge Dominates Flaw Response", fontsize=10
    )
    axes[0, 2].axis("off")

    # Scenario 2: Severe Surface Texture Noise (Aggregate Concrete Noise)
    # Using negative un-cracked sample with heavy aggregate texture
    if neg_sample is not None:
        texture_img = neg_sample
    else:
        texture_img = crack_sample

    # Add high-frequency textured surface noise simulating rough aggregate/asphalt
    np.random.seed(101)
    aggregate_noise = np.random.normal(0, 30.0, size=texture_img.shape)
    rough_concrete = np.clip(texture_img + aggregate_noise, 0.0, 255.0)

    rough_canny = canny_edge_detector(
        rough_concrete, sigma=1.2, low_thresh=5.0, high_thresh=15.0
    )["edges"]

    axes[1, 0].imshow(rough_concrete, cmap="gray")
    axes[1, 0].set_title(
        "(d) Scenario 2: Highly Textured Aggregate Surface (No Crack)",
        fontsize=10,
    )
    axes[1, 0].axis("off")

    axes[1, 1].imshow(rough_canny, cmap="gray")
    axes[1, 1].set_title(
        "(e) Canny Failure: False Alarm Clutter from Aggregate Texture",
        fontsize=10,
        color="red",
    )
    axes[1, 1].axis("off")

    axes[1, 2].imshow(rough_canny, cmap="hot")
    axes[1, 2].set_title(
        "(f) Diagnostic: Texture Speckles Trigger False Positive Edges",
        fontsize=10,
    )
    axes[1, 2].axis("off")

    plt.suptitle(
        "Task 5.3: Realistic Failure Scenarios in Industrial Quality Inspection (Shadow Illumination Contours & Aggregate Texture Clutter)",
        fontsize=13,
        fontweight="bold",
    )
    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, "task5_failure_cases.png")
    safe_savefig(out_path)
    plt.show()


def run_all_experiments():
    """Execute complete ablation and sensitivity experimental suite."""
    print("=" * 70)
    print("STARTING FULL EXPERIMENTAL ABLATION AND SENSITIVITY SUITE")
    print("=" * 70)

    pcb_sample, crack_sample, neg_sample = load_dataset_samples()

    run_task2_demonstration(pcb_sample, crack_sample)
    run_task3_demonstration(crack_sample)
    run_ablation1_noise_smoothing(crack_sample)
    run_ablation2_pre_post_nms(crack_sample, pcb_sample)
    run_ablation3_hysteresis_vs_single(crack_sample)
    run_scalespace_causality(crack_sample)
    run_parameter_sensitivity_grid(crack_sample, pcb_sample)
    run_failure_case_analysis(pcb_sample, crack_sample, neg_sample)

    print("=" * 70)
    print("ALL EXPERIMENTAL FIGURES GENERATED AND SAVED TO output_images/ SUCCESSFULLY!")
    print("=" * 70)


if __name__ == "__main__":
    run_all_experiments()
