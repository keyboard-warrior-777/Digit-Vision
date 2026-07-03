"""
Tests for src/models/ — architecture construction and output shapes.

Why these tests matter:
    Model architecture bugs are silent. build_model() succeeds even if
    you mistype a layer count, use the wrong activation, or accidentally
    build a model that outputs shape (N, 1) instead of (N, 10). These
    tests catch structural regressions immediately.

Tests in this file:
    - All three architectures build without error
    - Output shape is always (batch, 10) — one probability per digit class
    - Parameter counts are in expected ranges (sanity check)
    - Model is compiled and can call predict()
    - build_model() raises ValueError for unknown model names
    - list_available_models() returns all registered names
    - Dense NN has no Conv2D layers (important for Grad-CAM logic)
    - LeNet-5 and Custom CNN have Conv2D layers (required for Grad-CAM)
"""

from __future__ import annotations

import numpy as np
import pytest
import tensorflow as tf

from src.models import build_model, list_available_models

# ─── Registry ────────────────────────────────────────────────────────────────


class TestModelRegistry:
    """Tests for the model registry and build_model() factory."""

    def test_list_available_models_returns_all_three(self) -> None:
        """
        What: list_available_models() returns exactly the three registered names.
        Why:  Training, evaluation, and CLI scripts all iterate this list.
              A missing or extra entry would cause silent skips or crashes.
        Prevents: A model being dropped from the registry without anyone noticing.
        """
        models = list_available_models()
        assert set(models) == {
            "dense_nn",
            "lenet5",
            "custom_cnn",
        }, f"Expected three specific models, got: {models}"

    def test_list_available_models_preserves_order(self) -> None:
        """
        What: Models are returned in registration order (dense_nn, lenet5, custom_cnn).
        Why:  The training summary log and Streamlit comparison table depend on
              this order representing increasing architectural complexity.
        Prevents: Misleading comparison tables that show models out of order.
        """
        models = list_available_models()
        assert models[0] == "dense_nn"
        assert models[-1] == "custom_cnn"

    def test_build_unknown_model_raises_value_error(self) -> None:
        """
        What: build_model("nonexistent") raises ValueError with a helpful message.
        Why:  Without this, a typo in --model argument would cause a KeyError
              deep inside training code, far from where the typo occurred.
        Prevents: Cryptic KeyErrors instead of a clear, actionable error message.
        """
        with pytest.raises(ValueError, match="nonexistent"):
            build_model("nonexistent")

    def test_error_message_lists_valid_options(self) -> None:
        """
        What: The ValueError for unknown models lists the valid options.
        Why:  "Valid model names are: ['dense_nn', 'lenet5', 'custom_cnn']" is
              immediately actionable. An error without this forces the user to
              read source code.
        Prevents: Time wasted looking up valid model names.
        """
        with pytest.raises(ValueError) as exc_info:
            build_model("typo_model")
        error_message = str(exc_info.value)
        assert "dense_nn" in error_message
        assert "lenet5" in error_message
        assert "custom_cnn" in error_message


# ─── Dense NN ─────────────────────────────────────────────────────────────────


class TestDenseNN:
    """Tests for the Dense Neural Network architecture."""

    @pytest.fixture(autouse=True)
    def build(self) -> None:
        """Build the dense model once per test class."""
        self.model = build_model("dense_nn")

    def test_builds_successfully(self) -> None:
        """
        What: build_model('dense_nn') returns a tf.keras.Model.
        Why:  Architectural changes (wrong layer arg, missing import) cause
              build failures. This test acts as a smoke test.
        Prevents: Broken imports or constructor errors going undetected.
        """
        assert isinstance(self.model, tf.keras.Model)

    def test_output_shape_is_10_classes(self) -> None:
        """
        What: A forward pass on a single image produces output shape (1, 10).
        Why:  The model must output one logit per digit class. An incorrect
              final layer (e.g. Dense(1)) would give wrong output shape.
        Prevents: Final layer misconfigurations producing wrong probability vectors.
        """
        dummy_input = np.zeros((1, 28, 28, 1), dtype=np.float32)
        output = self.model.predict(dummy_input, verbose=0)
        assert output.shape == (1, 10), f"Expected (1, 10), got {output.shape}"

    def test_output_probabilities_sum_to_one(self) -> None:
        """
        What: Softmax output sums to 1.0 for each sample.
        Why:  The softmax activation is the last layer. If it were accidentally
              removed or replaced with linear, the outputs would not sum to 1
              and cannot be interpreted as probabilities.
        Prevents: Missing softmax activation producing invalid confidence scores.
        """
        dummy_input = np.random.rand(4, 28, 28, 1).astype(np.float32)
        output = self.model.predict(dummy_input, verbose=0)
        row_sums = output.sum(axis=1)
        np.testing.assert_allclose(row_sums, np.ones(4), atol=1e-5)

    def test_parameter_count_in_expected_range(self) -> None:
        """
        What: Dense NN has between 400K and 700K parameters.
        Why:  The Dense NN is intentionally large (to show its parameter
              inefficiency vs CNNs). If a layer is accidentally removed,
              the count drops out of range. If an extra layer is added, it rises.
        Prevents: Silent architecture changes breaking the educational comparison.
        """
        n_params = self.model.count_params()
        assert (
            400_000 <= n_params <= 700_000
        ), f"Dense NN parameter count {n_params:,} is out of expected range [400K, 700K]"

    def test_has_no_conv2d_layers(self) -> None:
        """
        What: Dense NN contains no Conv2D layers.
        Why:  The Grad-CAM code returns None for models without Conv2D layers.
              If someone accidentally adds a Conv2D to the Dense NN, Grad-CAM
              would activate for it — breaking the "Dense NN = no spatial features"
              educational narrative.
        Prevents: Unexpected Grad-CAM output for the Dense NN.
        """
        conv_layers = [
            layer
            for layer in self.model.layers
            if isinstance(layer, tf.keras.layers.Conv2D)
        ]
        assert (
            len(conv_layers) == 0
        ), f"Dense NN should have no Conv2D layers, found: {[layer.name for layer in conv_layers]}"

    def test_is_compiled(self) -> None:
        """
        What: The model is compiled and has an optimizer and loss configured.
        Why:  build_model() must return a model ready for .fit(). An uncompiled
              model raises a RuntimeError when .fit() is called, with no clear
              explanation.
        Prevents: Uncompiled models reaching the training loop.
        """
        assert self.model.optimizer is not None
        assert self.model.loss is not None


# ─── LeNet-5 ──────────────────────────────────────────────────────────────────


class TestLeNet5:
    """Tests for the LeNet-5 architecture."""

    @pytest.fixture(autouse=True)
    def build(self) -> None:
        self.model = build_model("lenet5")

    def test_builds_successfully(self) -> None:
        """What: build_model('lenet5') succeeds. Prevents: broken imports."""
        assert isinstance(self.model, tf.keras.Model)

    def test_output_shape_is_10_classes(self) -> None:
        """
        What: Output shape is (1, 10) for a single image.
        Why:  Same as Dense NN — must output exactly 10 class probabilities.
        """
        dummy_input = np.zeros((1, 28, 28, 1), dtype=np.float32)
        output = self.model.predict(dummy_input, verbose=0)
        assert output.shape == (1, 10)

    def test_has_conv2d_layers(self) -> None:
        """
        What: LeNet-5 contains at least one Conv2D layer.
        Why:  Conv2D layers are required for Grad-CAM. LeNet-5's educational
              value is that it introduced convolution for image recognition.
              If Conv2D layers are replaced with Dense layers, the architecture
              is no longer LeNet-5.
        Prevents: Accidentally flattening LeNet-5 into a Dense network.
        """
        conv_layers = [
            layer
            for layer in self.model.layers
            if isinstance(layer, tf.keras.layers.Conv2D)
        ]
        assert (
            len(conv_layers) >= 2
        ), f"LeNet-5 should have at least 2 Conv2D layers, found {len(conv_layers)}"

    def test_parameter_count_in_expected_range(self) -> None:
        """
        What: LeNet-5 has between 40K and 90K parameters.
        Why:  LeNet-5 is the compact historical architecture. If it exceeds
              90K parameters, it has likely been incorrectly configured.
        Prevents: LeNet-5 being made larger than the Custom CNN (breaks the
              parameter-efficiency comparison).
        """
        n_params = self.model.count_params()
        assert (
            40_000 <= n_params <= 90_000
        ), f"LeNet-5 parameter count {n_params:,} out of expected range [40K, 90K]"

    def test_output_probabilities_sum_to_one(self) -> None:
        """What: Softmax outputs sum to 1. Prevents: missing softmax activation."""
        dummy = np.random.rand(2, 28, 28, 1).astype(np.float32)
        output = self.model.predict(dummy, verbose=0)
        np.testing.assert_allclose(output.sum(axis=1), np.ones(2), atol=1e-5)


# ─── Custom CNN ───────────────────────────────────────────────────────────────


class TestCustomCNN:
    """Tests for the Custom CNN architecture."""

    @pytest.fixture(autouse=True)
    def build(self) -> None:
        self.model = build_model("custom_cnn")

    def test_builds_successfully(self) -> None:
        """What: build_model('custom_cnn') succeeds. Prevents: broken imports."""
        assert isinstance(self.model, tf.keras.Model)

    def test_output_shape_is_10_classes(self) -> None:
        """What: Output shape is (batch, 10). Prevents: wrong final layer size."""
        dummy_input = np.zeros((1, 28, 28, 1), dtype=np.float32)
        output = self.model.predict(dummy_input, verbose=0)
        assert output.shape == (1, 10)

    def test_has_conv2d_layers(self) -> None:
        """
        What: Custom CNN contains at least four Conv2D layers (two per block).
        Why:  The architecture document specifies 2×2 conv blocks. If a block
              is missing, the model is shallower and less accurate.
        Prevents: Accidentally building a one-block CNN instead of two-block.
        """
        conv_layers = [
            layer
            for layer in self.model.layers
            if isinstance(layer, tf.keras.layers.Conv2D)
        ]
        assert (
            len(conv_layers) >= 4
        ), f"Custom CNN should have at least 4 Conv2D layers, found {len(conv_layers)}"

    def test_has_batch_normalization_layers(self) -> None:
        """
        What: Custom CNN contains BatchNormalization layers.
        Why:  BatchNorm is a key architectural feature enabling higher learning
              rates and stable training. Its absence would be a regression.
        Prevents: Accidentally removing BatchNorm during refactoring.
        """
        bn_layers = [
            layer
            for layer in self.model.layers
            if isinstance(layer, tf.keras.layers.BatchNormalization)
        ]
        assert (
            len(bn_layers) >= 3
        ), f"Expected ≥3 BatchNorm layers, found {len(bn_layers)}"

    def test_has_global_average_pooling(self) -> None:
        """
        What: Custom CNN uses GlobalAveragePooling2D instead of Flatten.
        Why:  GlobalAveragePooling is the key efficiency feature of the Custom CNN
              — it dramatically reduces parameter count vs Flatten+Dense. Its
              absence would mean the model is not the architecture documented.
        Prevents: GlobalAvgPool being replaced with Flatten during editing.
        """
        gap_layers = [
            layer
            for layer in self.model.layers
            if isinstance(layer, tf.keras.layers.GlobalAveragePooling2D)
        ]
        assert (
            len(gap_layers) == 1
        ), f"Expected exactly 1 GlobalAveragePooling2D, found {len(gap_layers)}"

    def test_parameter_count_smaller_than_dense_nn(self) -> None:
        """
        What: Custom CNN has fewer parameters than the Dense NN.
        Why:  The core educational claim is "CNN outperforms Dense NN with fewer
              parameters." If this relationship breaks, the comparison narrative
              is false — and an interviewer who checks will notice.
        Prevents: Architecture changes that invalidate the project's key insight.
        """
        dense_params = build_model("dense_nn").count_params()
        cnn_params = self.model.count_params()
        assert (
            cnn_params < dense_params
        ), f"Custom CNN ({cnn_params:,}) should have fewer params than Dense NN ({dense_params:,})"

    def test_parameter_count_in_expected_range(self) -> None:
        """
        What: Custom CNN has between 50K and 120K parameters.
        Why:  GlobalAveragePooling makes the model compact. A count above 120K
              suggests Flatten was used instead of GAP.
        """
        n_params = self.model.count_params()
        assert (
            50_000 <= n_params <= 120_000
        ), f"Custom CNN parameter count {n_params:,} out of expected range [50K, 120K]"

    def test_batch_prediction_output_shape(self) -> None:
        """
        What: A batch of 8 images produces shape (8, 10).
        Why:  Batch prediction is used in the playground comparison. The shape
              must scale correctly with batch size.
        Prevents: Batch dimension handling bugs that only appear with N>1 inputs.
        """
        batch = np.random.rand(8, 28, 28, 1).astype(np.float32)
        output = self.model.predict(batch, verbose=0)
        assert output.shape == (8, 10)
