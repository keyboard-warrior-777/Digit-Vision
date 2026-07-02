"""
Shared pytest fixtures for DigitVision.

All fixtures are designed to be fast, offline, and deterministic:

  - No real MNIST download is triggered (synthetic arrays are used).
  - No trained model is required on disk (stub Sequential models are built).
  - No file I/O side-effects persist between tests (tmp_path is used).

Why fixtures instead of test helpers?
    pytest fixtures enforce dependency injection, making each test's
    requirements explicit. A test that needs a canvas image declares it
    in its signature — the reader immediately knows what it depends on.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pytest

# ── Suppress TensorFlow C++ startup logs during testing ───────────────────────
# Level 3 = only errors. Removes the "This TensorFlow binary is optimized..."
# banner that floods test output and makes failures hard to spot.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

# ── Ensure project root and streamlit_app are on the import path ──────────────
# Running pytest from the project root is the expected workflow, but explicitly
# adding these paths makes the suite robust to invocation from any directory.
_ROOT = Path(__file__).parent.parent
_STREAMLIT_APP = _ROOT / "streamlit_app"
for _p in (_ROOT, _STREAMLIT_APP):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))


# ─── Synthetic dataset ────────────────────────────────────────────────────────


@pytest.fixture
def mnist_batch() -> dict[str, np.ndarray]:
    """
    Return a small synthetic MNIST-like batch (32 samples).

    Why it matters:
        Tests that verify normalisation, shape expectations, or label
        encoding should not depend on a real MNIST download. This fixture
        gives them fast, deterministic data.

    Prevents:
        Tests failing in CI due to network unavailability or slow downloads.
    """
    rng = np.random.default_rng(seed=0)
    images = rng.integers(0, 256, size=(32, 28, 28), dtype=np.uint8)
    labels = rng.integers(0, 10, size=(32,), dtype=np.int32)
    return {"images": images, "labels": labels}


@pytest.fixture
def normalized_batch(mnist_batch: dict) -> dict[str, np.ndarray]:
    """
    Return the synthetic batch pre-processed to model-ready format.

    Shape: (32, 28, 28, 1), dtype float32, values in [0.0, 1.0].

    Prevents:
        Confusing dtype or shape errors in prediction/evaluation tests.
    """
    images = mnist_batch["images"].astype(np.float32) / 255.0
    images = images.reshape(-1, 28, 28, 1)
    labels = mnist_batch["labels"]
    return {"X": images, "y": labels}


@pytest.fixture
def single_mnist_image() -> np.ndarray:
    """
    Return a single preprocessed image, shape (1, 28, 28, 1), float32.

    This is the exact format expected by model.predict() and all
    DigitVision prediction functions.
    """
    rng = np.random.default_rng(seed=42)
    image = rng.random((1, 28, 28, 1)).astype(np.float32)
    return image


# ─── Canvas inputs ────────────────────────────────────────────────────────────


@pytest.fixture
def valid_canvas_rgba() -> np.ndarray:
    """
    Return a valid 280x280 RGBA canvas image with a white stroke.

    Mimics the exact output of streamlit-drawable-canvas: RGBA uint8,
    black background (transparent), white foreground stroke.

    Prevents:
        Bugs in the RGBA -> grayscale -> inversion pipeline going undetected.
    """
    canvas = np.zeros((280, 280, 4), dtype=np.uint8)
    # Draw a simple vertical stroke in the centre
    canvas[80:200, 135:145, 3] = 255  # alpha = fully opaque stroke
    return canvas


@pytest.fixture
def blank_canvas_rgba() -> np.ndarray:
    """Return an all-zero 280x280 RGBA canvas (nothing drawn)."""
    return np.zeros((280, 280, 4), dtype=np.uint8)


@pytest.fixture
def white_pil_image():
    """
    Return a white PIL Image (28x28, mode 'L') simulating a scanned digit.

    A white image has mean > 127, so the preprocessing pipeline should
    invert it before passing to the model.

    Prevents:
        The inversion heuristic being skipped for light-background images.
    """
    from PIL import Image
    return Image.fromarray(np.full((28, 28), 255, dtype=np.uint8), mode="L")


@pytest.fixture
def dark_pil_image():
    """
    Return a dark PIL Image (28x28, mode 'L') already in MNIST format.

    Mean < 127, so no inversion should be applied.
    """
    from PIL import Image
    arr = np.zeros((28, 28), dtype=np.uint8)
    arr[10:20, 10:20] = 200  # small bright region (the digit)
    return Image.fromarray(arr, mode="L")


# ─── Stub Keras models ────────────────────────────────────────────────────────


@pytest.fixture
def stub_dense_model():
    """
    Return a minimal compiled Keras model with no Conv2D layers.

    Used to test:
        - Grad-CAM correctly returns None for non-convolutional models.
        - Prediction output shape is (N, 10) regardless of architecture.
        - predict_batch works with dense-only architectures.

    Prevents:
        Grad-CAM code crashing instead of gracefully returning None.
    """
    import tensorflow as tf
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(28, 28, 1)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(10, activation="softmax"),
    ], name="stub_dense")
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    return model


@pytest.fixture
def stub_cnn_model():
    """
    Return a minimal compiled Keras model with one Conv2D layer.

    Used to test:
        - Grad-CAM correctly identifies the Conv2D layer.
        - Grad-CAM produces a heatmap of shape (28, 28).
        - The gradient model builds without error.

    Prevents:
        Grad-CAM failures caused by incorrect layer detection logic.
    """
    import tensorflow as tf
    inputs = tf.keras.layers.Input(shape=(28, 28, 1))
    x = tf.keras.layers.Conv2D(8, (3, 3), padding="same", activation="relu")(inputs)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    outputs = tf.keras.layers.Dense(10, activation="softmax")(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="stub_cnn")
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    return model


# ─── File system helpers ──────────────────────────────────────────────────────


@pytest.fixture
def saved_stub_model(stub_cnn_model, tmp_path: Path) -> Path:
    """
    Save the stub CNN model to a temporary .keras file and return its path.

    Used by prediction tests that exercise the file-loading path without
    requiring a fully trained model.

    Prevents:
        predict.py tests requiring real trained weights (~50 MB per model).
    """
    model_path = tmp_path / "stub_cnn.keras"
    stub_cnn_model.save(model_path)
    return model_path


@pytest.fixture
def saved_stub_dense_model(stub_dense_model, tmp_path: Path) -> Path:
    """Save the stub dense model to a temporary .keras file and return its path."""
    model_path = tmp_path / "stub_dense.keras"
    stub_dense_model.save(model_path)
    return model_path


@pytest.fixture
def sample_history_json(tmp_path: Path) -> Path:
    """
    Write a synthetic training history JSON file and return the path.

    Mimics the format written by train._save_training_history().
    Used by evaluate.load_training_history() tests.
    """
    import json
    history = {
        "accuracy":     [0.90, 0.93, 0.96, 0.97, 0.98],
        "val_accuracy": [0.88, 0.91, 0.94, 0.96, 0.97],
        "loss":         [0.33, 0.24, 0.16, 0.12, 0.09],
        "val_loss":     [0.38, 0.28, 0.20, 0.15, 0.11],
    }
    path = tmp_path / "test_history.json"
    path.write_text(json.dumps(history, indent=2), encoding="utf-8")
    return path


@pytest.fixture
def sample_metadata_json(tmp_path: Path) -> Path:
    """
    Write a synthetic model metadata JSON file and return the path.

    Mimics the format produced by model_card.generate_model_card().
    """
    import json
    metadata = {
        "model_name": "custom_cnn",
        "display_name": "Custom CNN",
        "dataset": "MNIST",
        "date_trained": "2026-07-01T12:00:00+00:00",
        "total_parameters": 75000,
        "trainable_parameters": 74500,
        "training_time_seconds": 420.0,
        "epochs_trained": 18,
        "optimizer": "Adam",
        "learning_rate": 0.001,
        "batch_size": 128,
        "final_train_accuracy": 0.9934,
        "final_val_accuracy": 0.9921,
        "test_accuracy": 0.9930,
        "test_loss": 0.0242,
        "macro_f1": 0.9929,
        "weighted_f1": 0.9930,
        "per_class_f1": {str(i): 0.99 + i * 0.001 for i in range(10)},
        "tensorflow_version": "2.17.0",
        "python_version": "3.11.0",
        "platform": "Windows",
        "git_commit": "abc1234",
    }
    path = tmp_path / "test_metadata.json"
    path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return path
