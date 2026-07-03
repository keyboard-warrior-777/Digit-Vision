"""
Tests for src/preprocessing.py — image transformation pipeline.

Why these tests matter:
    The preprocessing pipeline is the most critical and most commonly broken
    component in digit recognition demos. A model achieving 99.3% on MNIST
    will predict incorrectly on every input if the inversion step is skipped,
    or if the image is not resized before inference.

    These tests lock in the correct behaviour so refactoring the pipeline
    never silently breaks inference.

Tests in this file:
    - Valid RGBA canvas → correct shape, dtype, value range
    - Blank canvas → produces all-zero tensor (not an error)
    - Invalid canvas shapes → raises informative ValueError
    - Uploaded PIL image (light background) → inverted correctly
    - Uploaded PIL image (dark background) → not inverted
    - Normalization → all values in [0.0, 1.0]
    - Output shape → always (1, 28, 28, 1)
    - Wrong number of channels → raises ValueError with helpful message
"""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from src.preprocessing import canvas_image_to_model_input, uploaded_image_to_model_input

# ─── Canvas preprocessing ─────────────────────────────────────────────────────


class TestCanvasImageToModelInput:
    """Tests for canvas_image_to_model_input()."""

    def test_valid_canvas_returns_correct_shape(
        self, valid_canvas_rgba: np.ndarray
    ) -> None:
        """
        What: A valid RGBA canvas returns a tensor of shape (1, 28, 28, 1).
        Why:  model.predict() requires exactly this shape. Any other shape
              causes a silent shape mismatch or a cryptic TensorFlow error.
        Prevents: Shape errors reaching the model, causing silent wrong predictions.
        """
        result = canvas_image_to_model_input(valid_canvas_rgba)
        assert result.shape == (
            1,
            28,
            28,
            1,
        ), f"Expected (1, 28, 28, 1), got {result.shape}"

    def test_valid_canvas_returns_float32(self, valid_canvas_rgba: np.ndarray) -> None:
        """
        What: Output dtype is float32.
        Why:  TensorFlow models are compiled for float32. uint8 or float64
              inputs cause a dtype mismatch that silently produces wrong results
              on some backends.
        Prevents: Subtle dtype-related inference errors.
        """
        result = canvas_image_to_model_input(valid_canvas_rgba)
        assert result.dtype == np.float32

    def test_valid_canvas_values_in_unit_range(
        self, valid_canvas_rgba: np.ndarray
    ) -> None:
        """
        What: All output values are in [0.0, 1.0].
        Why:  The model was trained on normalized data. Values outside [0, 1]
              push activations out of the expected range, degrading accuracy.
        Prevents: Forgetting the /255.0 normalization step.
        """
        result = canvas_image_to_model_input(valid_canvas_rgba)
        assert result.min() >= 0.0, f"Min value {result.min()} < 0"
        assert result.max() <= 1.0, f"Max value {result.max()} > 1"

    def test_blank_canvas_produces_zero_tensor(
        self, blank_canvas_rgba: np.ndarray
    ) -> None:
        """
        What: A blank canvas (all-black RGB, no stroke drawn) produces an
              all-zero output tensor (0.0 everywhere).
        Why:  With the RGB grayscale approach, background pixels are
              black (RGB=0) -> grayscale=0 -> normalized=0.0. A blank
              canvas with no stroke drawn should produce a zero tensor.
              The UI prevents blank-canvas predictions with a sum>0 check.
        Prevents: Misunderstanding the direction of the preprocessing output.
        """
        result = canvas_image_to_model_input(blank_canvas_rgba)
        assert result.shape == (1, 28, 28, 1)
        # Blank canvas: all RGB=0 -> grayscale=0 -> normalized=0.0
        assert (
            result.max() == 0.0
        ), f"Blank canvas should produce all-zero output (0.0), got max={result.max()}"

    def test_invalid_shape_rgb_raises_value_error(self) -> None:
        """
        What: Passing an RGB (3-channel) array raises a ValueError.
        Why:  The function expects RGBA from streamlit-drawable-canvas.
              An RGB image (e.g. from a different source) should fail fast
              with a message explaining what shape is expected.
        Prevents: Silent wrong behaviour when a 3-channel image is passed.
        """
        rgb_image = np.zeros((280, 280, 3), dtype=np.uint8)
        with pytest.raises(ValueError, match="RGBA"):
            canvas_image_to_model_input(rgb_image)

    def test_invalid_shape_grayscale_raises_value_error(self) -> None:
        """
        What: Passing a 2-D grayscale array raises a ValueError.
        Why:  A 2-D array has ndim=2, not 3. The error should be immediate
              and descriptive rather than propagating as an IndexError deeper
              in the pipeline.
        Prevents: Confusing IndexError instead of clear ValueError.
        """
        gray_image = np.zeros((280, 280), dtype=np.uint8)
        with pytest.raises(ValueError, match="RGBA"):
            canvas_image_to_model_input(gray_image)

    def test_error_message_includes_received_shape(self) -> None:
        """
        What: The ValueError message includes the bad input shape.
        Why:  When a shape error occurs during debugging, knowing "got (280, 280, 3)"
              immediately points to the root cause without adding print statements.
        Prevents: Vague error messages that slow down debugging.
        """
        bad_input = np.zeros((280, 280, 3), dtype=np.uint8)
        with pytest.raises(ValueError) as exc_info:
            canvas_image_to_model_input(bad_input)
        assert "280" in str(
            exc_info.value
        ), "Error message should contain the bad shape"

    def test_different_canvas_sizes_produce_28x28_output(self) -> None:
        """
        What: Canvas images at any resolution produce a 28×28 output.
        Why:  If the canvas size is changed in config (e.g. 560×560 for a
              Retina display), the model input must still be 28×28. This test
              ensures the resize step is size-agnostic.
        Prevents: Hardcoded resize dimensions breaking on config changes.
        """
        large_canvas = np.zeros((560, 560, 4), dtype=np.uint8)
        large_canvas[200:350, 270:290, 3] = 255  # stroke
        result = canvas_image_to_model_input(large_canvas)
        assert result.shape == (1, 28, 28, 1)

    def test_stroke_appears_bright_in_output(self) -> None:
        """
        What: A white RGB stroke on a black background appears bright (near 1.0)
              in the output tensor, and the background appears dark (near 0.0).
        Why:  The canvas pipeline converts RGB to grayscale without inversion.
              Stroke pixels (R=255, G=255, B=255) -> grayscale=255 -> 1.0.
              Background pixels (R=0, G=0, B=0) -> grayscale=0 -> 0.0.
              This matches MNIST format: white digit on black background.
        Prevents: The most common demo bug: wrong signal polarity causing all
                  predictions to be the same regardless of what is drawn.
        """
        # Canvas with white RGB stroke only in the bottom-right quadrant.
        # Top-left is all-black RGB (no stroke) = background.
        canvas = np.zeros((280, 280, 4), dtype=np.uint8)
        canvas[:, :, 3] = 255  # all pixels opaque (opaque black BG)
        canvas[140:, 140:, 0] = 255  # white stroke in bottom-right (R)
        canvas[140:, 140:, 1] = 255  # white stroke in bottom-right (G)
        canvas[140:, 140:, 2] = 255  # white stroke in bottom-right (B)
        result = canvas_image_to_model_input(canvas)
        # Top-left corresponds to background (black RGB -> grayscale=0 -> 0.0).
        top_left_mean = result[0, :14, :14, 0].mean()
        assert top_left_mean < 0.1, (
            f"Background region should be near 0.0 (black), "
            f"got mean={top_left_mean:.3f}"
        )
        # Bottom-right corresponds to stroke (white RGB -> grayscale=255 -> 1.0).
        bottom_right_mean = result[0, 14:, 14:, 0].mean()
        assert bottom_right_mean > 0.9, (
            f"Stroke region should be near 1.0 (white), "
            f"got mean={bottom_right_mean:.3f}"
        )


# ─── Uploaded image preprocessing ─────────────────────────────────────────────


class TestUploadedImageToModelInput:
    """Tests for uploaded_image_to_model_input()."""

    def test_light_background_image_is_inverted(
        self, white_pil_image: Image.Image
    ) -> None:
        """
        What: An all-white image (mean > 127) is inverted.
        Why:  Scanned documents and photos of handwriting have dark digits
              on white paper — the opposite of MNIST format. Auto-inversion
              makes the function work correctly for both cases.
        Prevents: Wrong predictions for every uploaded scan or photo.
        """
        result = uploaded_image_to_model_input(white_pil_image)
        # After inverting a white image, all pixels should be 0.0 (black)
        assert (
            result.max() < 0.01
        ), f"Inverted white image should be near-zero, max={result.max():.3f}"

    def test_dark_background_image_is_not_inverted(
        self, dark_pil_image: Image.Image
    ) -> None:
        """
        What: A dark image (mean < 127) is NOT inverted.
        Why:  An image already in MNIST format (white digit on black) should
              not be double-inverted — that would turn it back into black-on-white
              and the model would see noise.
        Prevents: Double-inversion bug destroying correctly-formatted uploads.
        """
        result = uploaded_image_to_model_input(dark_pil_image)
        assert result.shape == (1, 28, 28, 1)
        # The image had a bright region — it should still be visible after processing
        assert result.max() > 0.5, "Dark image should not be inverted"

    def test_uploaded_image_output_shape(self, white_pil_image: Image.Image) -> None:
        """
        What: Output shape is always (1, 28, 28, 1).
        Why:  Uploaded images can be any resolution. The function must always
              produce exactly the shape the model expects.
        Prevents: Shape mismatches when users upload large images.
        """
        result = uploaded_image_to_model_input(white_pil_image)
        assert result.shape == (1, 28, 28, 1)

    def test_uploaded_image_output_dtype(self, white_pil_image: Image.Image) -> None:
        """
        What: Output dtype is float32.
        Why:  Consistent with canvas pipeline — both must produce float32.
        Prevents: dtype mismatch between upload and canvas paths in batch prediction.
        """
        result = uploaded_image_to_model_input(white_pil_image)
        assert result.dtype == np.float32

    def test_uploaded_image_handles_rgb_mode(self) -> None:
        """
        What: RGB PIL images are converted to grayscale correctly.
        Why:  Users upload PNG (RGBA), JPEG (RGB), BMP, etc. The function
              must handle any PIL mode without raising an exception.
        Prevents: Mode errors crashing the upload pipeline.
        """
        rgb_image = Image.fromarray(np.zeros((28, 28, 3), dtype=np.uint8), mode="RGB")
        result = uploaded_image_to_model_input(rgb_image)
        assert result.shape == (1, 28, 28, 1)

    def test_uploaded_image_handles_rgba_mode(self) -> None:
        """
        What: RGBA PIL images are handled correctly (converted to grayscale via L).
        Why:  PNG files commonly have an alpha channel. PIL's .convert('L')
              collapses the alpha compositely — the test ensures this doesn't crash.
        Prevents: AttributeError or mode errors on PNG uploads.
        """
        rgba = np.zeros((64, 64, 4), dtype=np.uint8)
        rgba[:, :, 3] = 128  # semi-transparent
        pil_rgba = Image.fromarray(rgba, mode="RGBA")
        result = uploaded_image_to_model_input(pil_rgba)
        assert result.shape == (1, 28, 28, 1)

    def test_uploaded_image_normalisation(self) -> None:
        """
        What: Output values are in [0.0, 1.0].
        Why:  Identical to the canvas normalisation requirement.
        Prevents: Unnormalized inputs being passed to the model.
        """
        mid_gray = Image.fromarray(np.full((28, 28), 128, dtype=np.uint8), mode="L")
        result = uploaded_image_to_model_input(mid_gray)
        assert result.min() >= 0.0
        assert result.max() <= 1.0

    def test_uploaded_image_handles_large_resolution(self) -> None:
        """
        What: High-resolution uploads (e.g. 1024×1024) are resized to 28×28.
        Why:  Users may upload high-DPI scans. The pipeline must resize without
              crashing or producing wrong output shapes.
        Prevents: Memory errors or shape errors on large image uploads.
        """
        large_img = Image.fromarray(np.zeros((1024, 1024), dtype=np.uint8), mode="L")
        result = uploaded_image_to_model_input(large_img)
        assert result.shape == (1, 28, 28, 1)
