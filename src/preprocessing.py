"""
Image preprocessing for DigitVision inference.

This module bridges the gap between what the user provides and what
the model expects. That gap is both real and non-trivial.

What the user gives us (canvas):
    - RGBA image (4 channels)
    - Black stroke on a white background
    - Resolution of 280×280 pixels (our canvas size)

What the model expects (MNIST format):
    - Grayscale (1 channel)
    - White digit on a black background
    - 28×28 pixels
    - Normalized float32 values in [0.0, 1.0]

The inversion step is the most critical and the most commonly missed.
Skipping it means the model sees a mostly white tensor where it expects
a mostly black one — prediction quality drops dramatically.

Interview Note:
    "The preprocessing pipeline is often more important than the model
    architecture for real-world inference. A model that achieves 99.3%
    on MNIST can fail visibly if the canvas image isn't correctly
    transformed to match the training distribution."
"""

from __future__ import annotations

import cv2
import numpy as np
from PIL import Image

from config.config import IMAGE_SIZE
from src.logger import get_logger

logger = get_logger(__name__)


def canvas_image_to_model_input(canvas_image_data: np.ndarray) -> np.ndarray:
    """
    Convert a drawable-canvas RGBA image into a model-ready input tensor.

    This is the primary function called during live prediction on the
    Draw page. It applies the complete transformation pipeline.

    Pipeline:
        1. Validate input shape (must be RGBA, i.e., 4 channels)
        2. Convert RGB channels to grayscale — this correctly captures
           the white stroke on the black background that st_canvas produces
           when background_color="#000000" is set.
        3. Resize to 28x28 using area interpolation (best for downscaling)
        4. Normalize to [0.0, 1.0]
        5. Reshape to (1, 28, 28, 1) — batch size 1, single channel

    Why NOT the alpha channel:
        When st_canvas is configured with background_color="#000000", the
        background pixels are rendered as opaque black (R=0, G=0, B=0,
        A=255). The stroke pixels are opaque white (R=255, G=255, B=255,
        A=255). Both background AND stroke have alpha=255, so extracting
        the alpha channel returns a flat all-255 array. After inversion
        (255 - 255 = 0), the model receives an all-zero tensor and always
        predicts the same digit regardless of what the user draws.

    Why RGB grayscale works:
        Background: RGB=(0,0,0)       -> grayscale=0   -> normalized=0.0
        Stroke:     RGB=(255,255,255) -> grayscale=255 -> normalized=1.0
        This is exactly the MNIST distribution: white digit on black
        background. No inversion is required.

    Args:
        canvas_image_data: RGBA numpy array from streamlit-drawable-canvas,
            shape (H, W, 4), dtype uint8.

    Returns:
        Model-ready float32 array of shape (1, 28, 28, 1).

    Raises:
        ValueError: If the input does not have exactly 4 channels.
    """
    if canvas_image_data.ndim != 3 or canvas_image_data.shape[-1] != 4:
        raise ValueError(
            f"Expected an RGBA image with shape (H, W, 4), "
            f"but received shape {canvas_image_data.shape}. "
            "Ensure the canvas component is configured to output RGBA."
        )

    # Convert RGB channels to grayscale.
    # The canvas uses background_color="#000000" (opaque black) and
    # stroke_color="#FFFFFF" (opaque white), so the grayscale signal
    # is: 0 where no stroke was drawn, 255 where the user drew.
    # This is already in MNIST format (white digit on black) — no inversion.
    gray = cv2.cvtColor(canvas_image_data[:, :, :3], cv2.COLOR_RGB2GRAY)

    resized = _resize_to_mnist(gray)
    normalized = resized.astype(np.float32) / 255.0

    return normalized.reshape(1, *IMAGE_SIZE, 1)


def uploaded_image_to_model_input(pil_image: Image.Image) -> np.ndarray:
    """
    Convert a user-uploaded PIL Image into a model-ready input tensor.

    Handles images in any PIL-supported mode (RGB, RGBA, L, P, etc.)
    and any resolution. The function auto-detects whether inversion is
    needed based on the brightness of the image corners.

    The inversion heuristic: sample 5×5 pixel patches from each of the
    four corners and compute their average brightness. If the corners are
    light (mean > 127), the image has a light background (digit is dark
    on white — typical of scanned documents). We invert it to match
    MNIST's white-on-black format.

    Why corners instead of the global mean?
        The global mean is unreliable when the digit is thick or large,
        because the digit pixels drag the overall mean below the 127
        threshold even on a white background. Corner pixels are almost
        always background — they are far from the centre where digits
        appear — making them a reliable proxy for background colour.

    Args:
        pil_image: A PIL Image in any mode.

    Returns:
        Model-ready float32 array of shape (1, 28, 28, 1).
    """
    grayscale = pil_image.convert("L")
    image_array = np.array(grayscale, dtype=np.uint8)

    if _has_light_background(image_array):
        logger.debug("Uploaded image has light background — inverting colours.")
        image_array = _invert_for_mnist(image_array)

    resized = _resize_to_mnist(image_array)
    normalized = resized.astype(np.float32) / 255.0

    return normalized.reshape(1, *IMAGE_SIZE, 1)


# ─── Private pipeline steps ────────────────────────────────────────────────────


def _has_light_background(grayscale: np.ndarray) -> bool:
    """
    Determine whether an image has a light background by sampling corner pixels.

    Samples four 5×5 patches from the image corners and returns True if their
    mean brightness exceeds 127. Corner regions are almost exclusively
    background pixels — digits rarely extend into the extreme corners.

    This is more reliable than the global mean when the digit is thick or
    large, because global averaging is pulled towards the digit intensity.

    Args:
        grayscale: 2-D uint8 numpy array (H, W).

    Returns:
        True if the background appears light (inversion is needed).
    """
    patch = 5  # 5×5 pixels per corner
    h, w = grayscale.shape
    corners = np.array([
        grayscale[:patch, :patch],          # top-left
        grayscale[:patch, w - patch:],       # top-right
        grayscale[h - patch:, :patch],       # bottom-left
        grayscale[h - patch:, w - patch:],   # bottom-right
    ])
    return float(corners.mean()) > 127



def _extract_alpha_channel(rgba_image: np.ndarray) -> np.ndarray:
    """
    Isolate the alpha (transparency) channel from an RGBA image.

    The alpha channel represents exactly where the user drew — fully
    transparent (0) where they didn't draw, fully opaque (255) where
    they did. Converting RGB to grayscale would include background
    colour information and produce a noisier signal.
    """
    return rgba_image[:, :, 3]


def _invert_for_mnist(grayscale: np.ndarray) -> np.ndarray:
    """
    Invert pixel intensities so the digit is white on a black background.

    MNIST images have white digits (255) on a black background (0).
    Canvas images have dark strokes (255) on a white background (0)
    when using the alpha channel representation.

    This single step is the most common source of failure in digit
    recognition demos. Without it, the model sees background as signal.
    """
    return 255 - grayscale


def _resize_to_mnist(image: np.ndarray) -> np.ndarray:
    """
    Resize the image to the 28×28 dimensions all models expect.

    Uses INTER_AREA interpolation — the correct choice for downscaling.
    Unlike nearest-neighbour (blocky) or bilinear (blurry), INTER_AREA
    averages pixels in each destination region, preserving stroke detail.
    """
    height, width = IMAGE_SIZE
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
