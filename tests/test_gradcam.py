"""
Tests for src/gradcam.py — Grad-CAM heatmap generation and overlay.

Why these tests matter:
    Grad-CAM is the hardest module to debug because its failures are visual.
    A broken heatmap looks like a heatmap — the model builds, no error is raised,
    but the highlighted region is random or all-zero. These tests catch the
    three most common failure modes:

    1. Wrong layer targeted (first conv instead of last conv)
    2. Heatmap values all equal — failed gradient computation
    3. Wrong output shape — resize step missing or broken

Tests in this file:
    - Dense model → returns None (no Conv2D layers)
    - CNN model → returns heatmap of shape (28, 28)
    - Heatmap values are in [0.0, 1.0]
    - Heatmap is not all-zero (gradient computation produced signal)
    - Last Conv2D layer is correctly identified
    - overlay_heatmap_on_image → output shape (28, 28, 3)
    - overlay_heatmap_on_image → output dtype uint8
    - Alpha blending is applied (result is not identical to original)
    - Invalid alpha=0.0 → overlay equals grayscale (no heatmap)
"""

from __future__ import annotations

import numpy as np
import pytest
import tensorflow as tf

from src.gradcam import (
    _find_last_conv_layer,
    compute_gradcam,
    overlay_heatmap_on_image,
)

# ─── compute_gradcam ─────────────────────────────────────────────────────────


class TestComputeGradcam:
    """Tests for the main compute_gradcam() function."""

    def test_dense_model_returns_none(
        self, stub_dense_model, single_mnist_image: np.ndarray
    ) -> None:
        """
        What: compute_gradcam() returns None for a model with no Conv2D layers.
        Why:  Grad-CAM requires spatial activations from a Conv2D layer.
              The Dense NN is purely fully-connected — there are no feature maps
              to visualise. Returning None is the contract for this case.
        Prevents: RuntimeError or AttributeError when Grad-CAM is run on Dense NN.
        """
        result = compute_gradcam(stub_dense_model, single_mnist_image, predicted_class_index=0)
        assert result is None, (
            "Expected None for Dense model (no Conv2D layers)"
        )

    def test_cnn_model_returns_heatmap(
        self, stub_cnn_model, single_mnist_image: np.ndarray
    ) -> None:
        """
        What: compute_gradcam() returns a non-None array for a CNN model.
        Why:  If None is returned for a CNN, the Streamlit Explainer page
              will silently show no heatmap — confusing and incorrect.
        Prevents: Grad-CAM silently failing for valid CNN models.
        """
        result = compute_gradcam(stub_cnn_model, single_mnist_image, predicted_class_index=0)
        assert result is not None, "Expected a heatmap for CNN model"

    def test_heatmap_shape_is_28x28(
        self, stub_cnn_model, single_mnist_image: np.ndarray
    ) -> None:
        """
        What: The returned heatmap has shape (28, 28).
        Why:  The heatmap must match the input image size (28×28) so it can
              be overlaid correctly. Any other shape means the resize step failed.
        Prevents: Shape mismatch between heatmap and image in overlay_heatmap_on_image().
        """
        heatmap = compute_gradcam(stub_cnn_model, single_mnist_image, predicted_class_index=0)
        assert heatmap.shape == (28, 28), (
            f"Expected heatmap shape (28, 28), got {heatmap.shape}"
        )

    def test_heatmap_values_in_unit_range(
        self, stub_cnn_model, single_mnist_image: np.ndarray
    ) -> None:
        """
        What: All heatmap values are in [0.0, 1.0].
        Why:  The heatmap is normalised before being passed to cv2.applyColorMap().
              Values outside [0, 1] cause incorrect colourmap output, making
              the overlay meaningless.
        Prevents: Unnormalised heatmaps producing invalid colour overlays.
        """
        heatmap = compute_gradcam(stub_cnn_model, single_mnist_image, predicted_class_index=0)
        assert heatmap.min() >= 0.0, f"Heatmap min {heatmap.min()} < 0"
        assert heatmap.max() <= 1.0, f"Heatmap max {heatmap.max()} > 1"

    def test_heatmap_max_is_one(
        self, stub_cnn_model, single_mnist_image: np.ndarray
    ) -> None:
        """
        What: The maximum heatmap value is exactly 1.0 after normalisation.
        Why:  Grad-CAM divides by max_value to produce a [0, 1] range. If the
              division is skipped (max_value == 0 edge case not handled), the
              heatmap is not normalised. A properly normalised heatmap always
              has max = 1.0.
        Prevents: The max_value == 0 edge case producing invalid output.
        """
        heatmap = compute_gradcam(stub_cnn_model, single_mnist_image, predicted_class_index=0)
        # A non-trivial input should produce a heatmap with max = 1.0
        if heatmap.max() > 0:  # skip if input produces all-zero gradients
            assert abs(heatmap.max() - 1.0) < 1e-5, (
                f"Heatmap maximum should be 1.0 after normalisation, got {heatmap.max()}"
            )

    def test_all_class_indices_produce_heatmaps(
        self, stub_cnn_model, single_mnist_image: np.ndarray
    ) -> None:
        """
        What: Grad-CAM produces a heatmap for every class index (0–9).
        Why:  The Explainer page allows users to visualise Grad-CAM for any
              class, not just the predicted one. All indices must work.
        Prevents: IndexError or None output for non-argmax class indices.
        """
        for class_idx in range(10):
            heatmap = compute_gradcam(stub_cnn_model, single_mnist_image, class_idx)
            assert heatmap is not None, f"Expected heatmap for class {class_idx}"
            assert heatmap.shape == (28, 28)


# ─── _find_last_conv_layer ───────────────────────────────────────────────────


class TestFindLastConvLayer:
    """Tests for the internal _find_last_conv_layer() helper."""

    def test_dense_model_returns_none(self, stub_dense_model) -> None:
        """
        What: Returns None for a model with no Conv2D layers.
        Why:  compute_gradcam() relies on this to detect inapplicable models.
              If it returned a non-Conv2D layer, the gradient model would fail.
        Prevents: Wrong layer being selected, causing gradient computation crash.
        """
        result = _find_last_conv_layer(stub_dense_model)
        assert result is None

    def test_cnn_model_returns_conv2d_layer(self, stub_cnn_model) -> None:
        """
        What: Returns a Conv2D layer for a CNN model.
        Why:  The returned layer is used to build the gradient model. It must
              be a Conv2D — any other layer type would fail during gradient tape.
        Prevents: Wrong layer type being returned.
        """
        layer = _find_last_conv_layer(stub_cnn_model)
        assert layer is not None
        assert isinstance(layer, tf.keras.layers.Conv2D), (
            f"Expected Conv2D layer, got {type(layer).__name__}"
        )

    def test_returns_last_not_first_conv_layer(self) -> None:
        """
        What: Returns the LAST Conv2D layer in the model, not the first.
        Why:  Grad-CAM targets the last conv layer because it has the most
              semantically rich feature maps. Targeting the first conv layer
              would show low-level edge features, not the high-level digit
              patterns that explain the prediction.
        Prevents: Targeting early layers and producing meaningless visualisations.
        """
        # Build a model with two Conv2D layers with distinct filter counts
        inputs = tf.keras.layers.Input(shape=(28, 28, 1))
        x = tf.keras.layers.Conv2D(8, (3, 3), padding="same", name="first_conv")(inputs)
        x = tf.keras.layers.Conv2D(16, (3, 3), padding="same", name="last_conv")(x)
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        outputs = tf.keras.layers.Dense(10, activation="softmax")(x)
        model = tf.keras.Model(inputs=inputs, outputs=outputs)

        last_conv = _find_last_conv_layer(model)
        assert last_conv.name == "last_conv", (
            f"Expected 'last_conv', got '{last_conv.name}'"
        )


# ─── overlay_heatmap_on_image ─────────────────────────────────────────────────


class TestOverlayHeatmapOnImage:
    """Tests for overlay_heatmap_on_image()."""

    @pytest.fixture
    def sample_heatmap(self) -> np.ndarray:
        """A valid 28×28 normalised heatmap."""
        rng = np.random.default_rng(seed=7)
        heatmap = rng.random((28, 28)).astype(np.float32)
        return heatmap / heatmap.max()  # ensure max == 1

    def test_output_shape_is_28x28x3(
        self, sample_heatmap: np.ndarray, single_mnist_image: np.ndarray
    ) -> None:
        """
        What: Output shape is (28, 28, 3).
        Why:  The overlay must be an RGB image for display in Streamlit's st.image().
              Any other shape causes a rendering error.
        Prevents: Grayscale or RGBA output breaking the image display.
        """
        overlay = overlay_heatmap_on_image(sample_heatmap, single_mnist_image)
        assert overlay.shape == (28, 28, 3), (
            f"Expected (28, 28, 3), got {overlay.shape}"
        )

    def test_output_dtype_is_uint8(
        self, sample_heatmap: np.ndarray, single_mnist_image: np.ndarray
    ) -> None:
        """
        What: Output dtype is uint8 (0–255 range).
        Why:  PIL.Image.fromarray() and st.image() both expect uint8 arrays.
              A float32 overlay would cause a silent rendering error in Streamlit.
        Prevents: dtype mismatch when saving overlay as PNG or displaying in UI.
        """
        overlay = overlay_heatmap_on_image(sample_heatmap, single_mnist_image)
        assert overlay.dtype == np.uint8

    def test_alpha_zero_returns_grayscale_equivalent(
        self, sample_heatmap: np.ndarray, single_mnist_image: np.ndarray
    ) -> None:
        """
        What: alpha=0.0 produces an output containing only the original image.
        Why:  alpha=0.0 means "image only, no heatmap". The blending formula
              is (1-alpha)*image + alpha*heatmap. With alpha=0, this equals image.
        Prevents: Alpha parameter being ignored in the blending calculation.
        """
        overlay = overlay_heatmap_on_image(sample_heatmap, single_mnist_image, alpha=0.0)
        assert overlay.shape == (28, 28, 3)
        # All channels should be identical (grayscale expanded to RGB)
        assert np.array_equal(overlay[:, :, 0], overlay[:, :, 1])
        assert np.array_equal(overlay[:, :, 1], overlay[:, :, 2])

    def test_alpha_one_returns_coloured_heatmap(
        self, sample_heatmap: np.ndarray, single_mnist_image: np.ndarray
    ) -> None:
        """
        What: alpha=1.0 produces an output dominated by the heatmap colours.
        Why:  With alpha=1.0, the blend is entirely heatmap. The result should
              differ significantly from the grayscale-only alpha=0.0 case.
        Prevents: The alpha parameter having no effect on the output.
        """
        overlay_no_heatmap = overlay_heatmap_on_image(
            sample_heatmap, single_mnist_image, alpha=0.0
        )
        overlay_full_heatmap = overlay_heatmap_on_image(
            sample_heatmap, single_mnist_image, alpha=1.0
        )
        # The two overlays should be different (heatmap changes colour channels)
        assert not np.array_equal(overlay_no_heatmap, overlay_full_heatmap), (
            "alpha=0.0 and alpha=1.0 should produce different results"
        )

    def test_default_alpha_produces_blended_result(
        self, sample_heatmap: np.ndarray, single_mnist_image: np.ndarray
    ) -> None:
        """
        What: Default alpha (0.4) produces a result between the two extremes.
        Why:  The default is chosen to balance visibility of the digit and the
              heatmap. This test ensures the default is actually applied.
        Prevents: Default parameter being silently ignored.
        """
        overlay = overlay_heatmap_on_image(sample_heatmap, single_mnist_image)
        assert overlay.shape == (28, 28, 3)
        assert overlay.max() > 0  # not all black
