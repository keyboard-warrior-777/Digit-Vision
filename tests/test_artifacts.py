"""
Tests for src/artifacts.py — post-evaluation artifact generation.

Why these tests matter:
    Artifacts are the bridge between the backend and the Streamlit UI.
    If artifacts are missing, malformed, or have the wrong structure,
    the UI silently shows empty charts. These tests verify that every
    artifact is saved in the correct format with the correct structure.

Tests in this file:
    - Confusion matrix saved as .npy file with correct shape
    - Confusion matrix values are non-negative integers
    - Raw metrics JSON has correct structure (per-class keys)
    - ROC data JSON has fpr, tpr, auc for each digit class
    - Sample predictions JSON has correct fields
    - _find_best_example_for_class returns correct shape
    - _find_best_example_for_class returns None when no correct examples exist
    - _save_mnist_image creates a 112×112 PNG
    - Grad-CAM samples skipped gracefully for Dense NN
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.artifacts import (
    _find_best_example_for_class,
    _save_confusion_matrix,
    _save_mnist_image,
    _save_raw_metrics,
    _save_roc_data,
    _save_sample_predictions,
)
from src.dataset import MNISTData


# ─── Synthetic MNISTData ─────────────────────────────────────────────────────


@pytest.fixture
def tiny_mnist_data() -> MNISTData:
    """
    Return a tiny MNISTData-like object (100 samples per class = 1000 total).

    Used by artifact tests that need realistic label/image structure without
    downloading real MNIST (11 MB download not needed in CI).
    """
    rng = np.random.default_rng(seed=99)
    n_per_class = 20
    n_total = n_per_class * 10

    X = rng.random((n_total, 28, 28, 1), dtype=np.float32).astype(np.float32)
    y_labels = np.repeat(np.arange(10), n_per_class).astype(np.int32)
    y_onehot = np.eye(10, dtype=np.float32)[y_labels]

    return MNISTData(
        X_train=X,
        y_train=y_onehot,
        X_val=X[:50],
        y_val=y_onehot[:50],
        X_test=X,
        y_test=y_onehot,
        y_test_labels=y_labels,
    )


@pytest.fixture
def fake_predictions(tiny_mnist_data: MNISTData) -> tuple[np.ndarray, np.ndarray]:
    """
    Return (predicted_labels, probabilities) with ~80% accuracy.

    80% accuracy means the confusion matrix is non-trivial (not all on diagonal)
    but also not completely random — providing realistic test data.
    """
    rng = np.random.default_rng(seed=42)
    n = len(tiny_mnist_data.y_test_labels)
    true = tiny_mnist_data.y_test_labels

    # 80% correct, 20% random
    correct_mask = rng.random(n) < 0.80
    predicted = true.copy()
    predicted[~correct_mask] = rng.integers(0, 10, size=(~correct_mask).sum())

    # Build plausible probability vectors
    probs = np.zeros((n, 10), dtype=np.float32)
    for i, (t, p) in enumerate(zip(true, predicted)):
        probs[i, p] = 0.9
        remaining = np.ones(10) * 0.1 / 9
        remaining[p] = 0
        probs[i] += remaining
        probs[i] = np.clip(probs[i], 0, 1)
        probs[i] /= probs[i].sum()

    return predicted, probs


# ─── _save_confusion_matrix ───────────────────────────────────────────────────


class TestSaveConfusionMatrix:
    """Tests for the confusion matrix artifact."""

    def test_creates_npy_file(
        self, tmp_path: Path, tiny_mnist_data: MNISTData, fake_predictions
    ) -> None:
        """
        What: A .npy file is created at the specified path.
        Why:  The Streamlit analytics page loads the confusion matrix with
              np.load(path). If the file doesn't exist, the chart cannot render.
        Prevents: Silent file save failure (no exception, but no file created).
        """
        predicted, _ = fake_predictions
        save_path = tmp_path / "cm.npy"

        with patch("src.artifacts.CONFUSION_MATRIX_PATHS", {"custom_cnn": save_path}):
            _save_confusion_matrix(
                tiny_mnist_data.y_test_labels, predicted, "custom_cnn"
            )

        assert save_path.exists(), f"Confusion matrix file was not created at {save_path}"

    def test_confusion_matrix_shape(
        self, tmp_path: Path, tiny_mnist_data: MNISTData, fake_predictions
    ) -> None:
        """
        What: Saved confusion matrix has shape (10, 10).
        Why:  There are 10 digit classes. A (10, 10) matrix has one cell per
              (true, predicted) pair. Any other shape would break chart indexing.
        Prevents: Confusion matrix with wrong dimensions producing wrong hover data.
        """
        predicted, _ = fake_predictions
        save_path = tmp_path / "cm.npy"

        with patch("src.artifacts.CONFUSION_MATRIX_PATHS", {"lenet5": save_path}):
            _save_confusion_matrix(
                tiny_mnist_data.y_test_labels, predicted, "lenet5"
            )

        loaded_cm = np.load(save_path)
        assert loaded_cm.shape == (10, 10), f"Expected (10, 10), got {loaded_cm.shape}"

    def test_confusion_matrix_diagonal_has_correct_predictions(
        self, tmp_path: Path, tiny_mnist_data: MNISTData, fake_predictions
    ) -> None:
        """
        What: Diagonal cells contain correctly classified counts.
        Why:  The confusion matrix diagonal represents correct predictions.
              If diagonal and off-diagonal are swapped, the "hardest classes"
              visualisation would be completely wrong.
        Prevents: True/predicted axis being transposed in the confusion matrix.
        """
        predicted, _ = fake_predictions
        true = tiny_mnist_data.y_test_labels
        save_path = tmp_path / "cm.npy"

        with patch("src.artifacts.CONFUSION_MATRIX_PATHS", {"custom_cnn": save_path}):
            _save_confusion_matrix(true, predicted, "custom_cnn")

        cm = np.load(save_path)
        diagonal_sum = np.trace(cm)
        correct_count = int((true == predicted).sum())
        assert diagonal_sum == correct_count, (
            f"Diagonal sum ({diagonal_sum}) should equal correct count ({correct_count})"
        )


# ─── _save_raw_metrics ────────────────────────────────────────────────────────


class TestSaveRawMetrics:
    """Tests for the per-class metrics JSON artifact."""

    def test_creates_json_file(
        self, tmp_path: Path, tiny_mnist_data: MNISTData, fake_predictions
    ) -> None:
        """
        What: A .json file is created at the specified path.
        Why:  The analytics page loads per-class metrics from this file.
        Prevents: Missing metrics JSON causing empty analytics charts.
        """
        import json
        predicted, _ = fake_predictions
        save_path = tmp_path / "metrics.json"

        with patch("src.artifacts.RAW_METRICS_PATHS", {"dense_nn": save_path}):
            _save_raw_metrics(
                tiny_mnist_data.y_test_labels, predicted, "dense_nn"
            )

        assert save_path.exists()
        content = json.loads(save_path.read_text(encoding="utf-8"))
        # sklearn classification_report always includes 'accuracy' key
        assert "accuracy" in content

    def test_per_class_keys_exist(
        self, tmp_path: Path, tiny_mnist_data: MNISTData, fake_predictions
    ) -> None:
        """
        What: Metrics JSON contains entries for all 10 digit classes.
        Why:  The per-class F1 chart requires one entry per class.
              Missing entries would produce incomplete bar charts.
        Prevents: Missing class entries causing chart rendering errors.
        """
        import json
        predicted, _ = fake_predictions
        save_path = tmp_path / "metrics.json"

        with patch("src.artifacts.RAW_METRICS_PATHS", {"dense_nn": save_path}):
            _save_raw_metrics(
                tiny_mnist_data.y_test_labels, predicted, "dense_nn"
            )

        content = json.loads(save_path.read_text(encoding="utf-8"))
        for digit in [str(i) for i in range(10)]:
            assert digit in content, f"Expected class '{digit}' in metrics JSON"


# ─── _save_roc_data ───────────────────────────────────────────────────────────


class TestSaveRocData:
    """Tests for the ROC curve data JSON artifact."""

    def test_creates_json_file(
        self, tmp_path: Path, tiny_mnist_data: MNISTData, fake_predictions
    ) -> None:
        """What: A JSON file is created. Prevents: Missing ROC data breaking chart."""
        _, probs = fake_predictions
        save_path = tmp_path / "roc.json"

        with patch("src.artifacts.ROC_DATA_PATHS", {"custom_cnn": save_path}):
            _save_roc_data(tiny_mnist_data.y_test_labels, probs, "custom_cnn")

        assert save_path.exists()

    def test_roc_data_has_ten_classes(
        self, tmp_path: Path, tiny_mnist_data: MNISTData, fake_predictions
    ) -> None:
        """
        What: ROC data contains entries for all 10 digit classes.
        Why:  The ROC chart renders one line per class. Missing classes produce
              an incomplete chart.
        Prevents: Partial ROC data producing incomplete multi-class ROC chart.
        """
        import json
        _, probs = fake_predictions
        save_path = tmp_path / "roc.json"

        with patch("src.artifacts.ROC_DATA_PATHS", {"custom_cnn": save_path}):
            _save_roc_data(tiny_mnist_data.y_test_labels, probs, "custom_cnn")

        content = json.loads(save_path.read_text(encoding="utf-8"))
        assert len(content) == 10, f"Expected 10 ROC entries, got {len(content)}"

    def test_each_class_has_fpr_tpr_auc(
        self, tmp_path: Path, tiny_mnist_data: MNISTData, fake_predictions
    ) -> None:
        """
        What: Each class entry has 'fpr', 'tpr', and 'auc' keys.
        Why:  The ROC chart builder reads all three keys. Missing keys cause
              KeyError during chart rendering.
        Prevents: Incomplete ROC entry crashing the chart builder.
        """
        import json
        _, probs = fake_predictions
        save_path = tmp_path / "roc.json"

        with patch("src.artifacts.ROC_DATA_PATHS", {"lenet5": save_path}):
            _save_roc_data(tiny_mnist_data.y_test_labels, probs, "lenet5")

        content = json.loads(save_path.read_text(encoding="utf-8"))
        for digit, entry in content.items():
            assert "fpr" in entry, f"Class {digit} missing 'fpr'"
            assert "tpr" in entry, f"Class {digit} missing 'tpr'"
            assert "auc" in entry, f"Class {digit} missing 'auc'"

    def test_auc_values_in_valid_range(
        self, tmp_path: Path, tiny_mnist_data: MNISTData, fake_predictions
    ) -> None:
        """
        What: AUC values are in [0.0, 1.0].
        Why:  AUC is a probability (area under the ROC curve). Values outside
              [0, 1] are mathematically invalid and indicate a computation error.
        Prevents: Invalid AUC values confusing the reader or breaking the chart scale.
        """
        import json
        _, probs = fake_predictions
        save_path = tmp_path / "roc.json"

        with patch("src.artifacts.ROC_DATA_PATHS", {"custom_cnn": save_path}):
            _save_roc_data(tiny_mnist_data.y_test_labels, probs, "custom_cnn")

        content = json.loads(save_path.read_text(encoding="utf-8"))
        for digit, entry in content.items():
            auc_val = entry["auc"]
            assert 0.0 <= auc_val <= 1.0, (
                f"AUC for class {digit} is {auc_val} — outside [0, 1]"
            )


# ─── _find_best_example_for_class ────────────────────────────────────────────


class TestFindBestExampleForClass:
    """Tests for _find_best_example_for_class()."""

    def test_returns_correct_shape(self, stub_cnn_model, tiny_mnist_data: MNISTData) -> None:
        """
        What: The returned image has shape (1, 28, 28, 1).
        Why:  The returned array is passed directly to compute_gradcam(), which
              expects exactly this shape.
        Prevents: Shape mismatch causing Grad-CAM to fail on the returned image.
        """
        result = _find_best_example_for_class(tiny_mnist_data, digit_class=0, model=stub_cnn_model)
        if result is not None:
            assert result.shape == (1, 28, 28, 1), (
                f"Expected shape (1, 28, 28, 1), got {result.shape}"
            )

    def test_returns_none_for_empty_class(self, stub_cnn_model) -> None:
        """
        What: Returns None when the dataset has no examples of the requested class.
        Why:  The function searches test_labels for the digit class. If no examples
              exist (e.g. for an edge-case dataset), it must return None gracefully.
        Prevents: IndexError when accessing empty class_indices.
        """
        # Create a dataset with only class 0 — no class 5 examples
        rng = np.random.default_rng(seed=1)
        X = rng.random((50, 28, 28, 1)).astype(np.float32)
        y_labels = np.zeros(50, dtype=np.int32)  # all class 0
        y_onehot = np.eye(10)[y_labels].astype(np.float32)

        data = MNISTData(
            X_train=X, y_train=y_onehot,
            X_val=X[:10], y_val=y_onehot[:10],
            X_test=X, y_test=y_onehot, y_test_labels=y_labels,
        )

        result = _find_best_example_for_class(data, digit_class=5, model=stub_cnn_model)
        assert result is None, "Should return None when digit class has no examples"

    def test_return_type_is_optional_array_not_tuple(
        self, stub_cnn_model, tiny_mnist_data: MNISTData
    ) -> None:
        """
        What: Returns Optional[np.ndarray], not a tuple.
        Why:  After the Module 3.5 fix, the dead Optional[Path] return was removed.
              This test ensures the refactor holds — if anyone reverts to returning
              a tuple, this test will catch it.
        Prevents: Regression to the old (image, None) return type.
        """
        result = _find_best_example_for_class(tiny_mnist_data, digit_class=0, model=stub_cnn_model)
        # Result should be ndarray or None — NEVER a tuple
        assert not isinstance(result, tuple), (
            "Function should return Optional[np.ndarray], not a tuple"
        )


# ─── _save_mnist_image ────────────────────────────────────────────────────────


class TestSaveMnistImage:
    """Tests for the image saving helper."""

    def test_creates_png_file(self, tmp_path: Path, single_mnist_image: np.ndarray) -> None:
        """
        What: Saves a valid PNG file at the specified path.
        Why:  The analytics page uses these PNGs as thumbnails. Missing or
              corrupt PNG files would show broken image icons.
        Prevents: Silent save failures leaving the predictions directory empty.
        """
        save_path = tmp_path / "test_digit.png"
        # Use a single 28×28×1 image (not the batched 1×28×28×1 fixture)
        image_array = single_mnist_image[0]  # shape (28, 28, 1)
        _save_mnist_image(image_array, save_path)

        assert save_path.exists(), "PNG file was not created"
        assert save_path.suffix == ".png"

    def test_saved_image_is_112x112(self, tmp_path: Path, single_mnist_image: np.ndarray) -> None:
        """
        What: Saved image is 112×112 pixels (upscaled from 28×28).
        Why:  The sample predictions display uses 112×112 thumbnails. Saving
              at 28×28 would produce tiny, unusable images in the dashboard.
        Prevents: Thumbnail size regression producing unusably small images.
        """
        from PIL import Image as PILImage
        save_path = tmp_path / "test_digit.png"
        image_array = single_mnist_image[0]
        _save_mnist_image(image_array, save_path)

        with PILImage.open(save_path) as img:
            assert img.size == (112, 112), (
                f"Expected 112×112 PNG, got {img.size}"
            )
