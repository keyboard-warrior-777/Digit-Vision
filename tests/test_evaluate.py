"""
Tests for src/evaluate.py — metrics computation and artifact loading.

Why these tests matter:
    Evaluation bugs are the sneakiest class of bugs. The model trains and
    predicts correctly, but the metrics are computed on the wrong labels,
    or the confusion matrix is transposed, or the history file is read from
    the wrong path. These tests catch exactly that class of error.

Tests in this file:
    - load_training_history returns correct structure from JSON
    - load_training_history returns None for nonexistent file
    - _compute_per_class_f1 returns dict with 10 entries
    - _compute_per_class_f1 returns float values for all entries
    - Perfect predictions produce F1 = 1.0 for all classes
    - Random predictions produce F1 < 1.0
    - ModelEvaluation dataclass is frozen and typed correctly
    - evaluate_model raises FileNotFoundError for untrained model
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from config.config import CLASS_NAMES
from src.evaluate import (
    ModelEvaluation,
    _compute_per_class_f1,
    load_training_history,
)

# ─── load_training_history ────────────────────────────────────────────────────


class TestLoadTrainingHistory:
    """Tests for load_training_history()."""

    def test_returns_correct_keys(self, sample_history_json: Path) -> None:
        """
        What: The returned dict has the four expected metric keys.
        Why:  The Streamlit training curves page reads 'accuracy', 'val_accuracy',
              'loss', and 'val_loss'. Missing keys produce KeyError in the chart.
        Prevents: Charts crashing due to missing history keys.
        """
        with patch("src.evaluate.HISTORY_PATHS", {"custom_cnn": sample_history_json}):
            history = load_training_history("custom_cnn")

        assert history is not None
        for key in ("accuracy", "val_accuracy", "loss", "val_loss"):
            assert key in history, f"Expected key '{key}' in history"

    def test_returns_lists_of_floats(self, sample_history_json: Path) -> None:
        """
        What: Each value in the history dict is a list of floats.
        Why:  The chart builder iterates these lists and calls Plotly's
              go.Scatter(y=...). Non-float or non-list values would crash.
        Prevents: Type errors in chart rendering.
        """
        with patch("src.evaluate.HISTORY_PATHS", {"custom_cnn": sample_history_json}):
            history = load_training_history("custom_cnn")

        for key, values in history.items():
            assert isinstance(values, list), f"'{key}' should be a list"
            assert all(isinstance(v, float) for v in values), (
                f"All values in '{key}' should be float"
            )

    def test_returns_none_for_missing_model(self) -> None:
        """
        What: Returns None when no history file exists for the model.
        Why:  Models that haven't been trained yet have no history file.
              The Streamlit analytics page must handle this gracefully
              (show "no data" rather than crashing).
        Prevents: FileNotFoundError crashing the analytics page.
        """
        with patch("src.evaluate.HISTORY_PATHS", {"dense_nn": Path("/nonexistent/path.json")}):
            result = load_training_history("dense_nn")
        assert result is None

    def test_returns_none_for_unregistered_model(self) -> None:
        """
        What: Returns None for a model name not in HISTORY_PATHS.
        Why:  If the UI passes an invalid model name, it should receive None
              (safe default) rather than a KeyError.
        Prevents: KeyError when history is queried for unknown model names.
        """
        result = load_training_history("model_that_doesnt_exist")
        assert result is None

    def test_epoch_count_is_consistent(self, sample_history_json: Path) -> None:
        """
        What: All four metric lists have the same number of entries.
        Why:  Each list entry corresponds to one epoch. Mismatched lengths
              mean the file is corrupt — the chart would show misaligned curves.
        Prevents: Plotting epoch N accuracy against epoch M loss.
        """
        with patch("src.evaluate.HISTORY_PATHS", {"custom_cnn": sample_history_json}):
            history = load_training_history("custom_cnn")

        lengths = {key: len(vals) for key, vals in history.items()}
        unique_lengths = set(lengths.values())
        assert len(unique_lengths) == 1, (
            f"All history lists should have same length: {lengths}"
        )


# ─── _compute_per_class_f1 ────────────────────────────────────────────────────


class TestComputePerClassF1:
    """Tests for the per-class F1 computation helper."""

    def test_returns_ten_entries(self) -> None:
        """
        What: Returns a dict with exactly 10 entries (one per digit class).
        Why:  The F1 bar chart renders exactly 10 bars. Fewer entries would
              show a truncated chart; more would crash or produce extra bars.
        Prevents: Per-class F1 having wrong number of entries.
        """
        true = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9] * 100)
        pred = true.copy()  # perfect predictions
        result = _compute_per_class_f1(true, pred)
        assert len(result) == 10, f"Expected 10 entries, got {len(result)}"

    def test_uses_string_class_names(self) -> None:
        """
        What: Dict keys are CLASS_NAMES ('0'–'9'), not integers.
        Why:  The chart builder and metadata JSON both use string class names.
              Integer keys would cause KeyError in the chart and JSON.
        Prevents: Type mismatch breaking F1 chart and model metadata.
        """
        true = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9] * 10)
        pred = true.copy()
        result = _compute_per_class_f1(true, pred)
        for key in result:
            assert isinstance(key, str), f"Key {key!r} should be a string"
        assert set(result.keys()) == set(CLASS_NAMES)

    def test_perfect_predictions_give_f1_one(self) -> None:
        """
        What: Perfect predictions (true == predicted) give F1 = 1.0 for all classes.
        Why:  After a good training run, the model should produce near-perfect F1
              scores on MNIST. Confirming the computation is correct before training
              means we trust the numbers after training.
        Prevents: Wrong F1 calculation producing misleadingly low scores.
        """
        true = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9] * 100)
        pred = true.copy()
        result = _compute_per_class_f1(true, pred)
        for digit, score in result.items():
            assert abs(score - 1.0) < 1e-6, (
                f"Perfect predictions should give F1=1.0 for digit {digit}, got {score}"
            )

    def test_all_wrong_predictions_give_f1_zero(self) -> None:
        """
        What: Systematically wrong predictions produce F1 < 0.1 for all classes.
        Why:  If F1 was returning 1.0 for wrong predictions, it would be silently
              broken and all models would appear perfect.
        Prevents: F1 metric being computed on swapped (true, pred) arguments.
        """
        true = np.array([0] * 100 + [1] * 100)
        pred = np.array([1] * 100 + [0] * 100)  # always wrong
        result = _compute_per_class_f1(true, pred)
        for digit in ["0", "1"]:
            assert result[digit] < 0.1, (
                f"Completely wrong predictions should give near-zero F1 for digit {digit}, "
                f"got {result[digit]:.4f}"
            )

    def test_values_are_python_floats(self) -> None:
        """
        What: All F1 values are Python float, not numpy float32.
        Why:  json.dump() serializes Python float correctly but raises TypeError
              for numpy float32. If the values aren't converted, the metadata
              JSON cannot be saved.
        Prevents: JSON serialisation failure when saving model metadata.
        """
        true = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9] * 10)
        pred = true.copy()
        result = _compute_per_class_f1(true, pred)
        for digit, score in result.items():
            assert isinstance(score, float), (
                f"F1 score for '{digit}' should be Python float, got {type(score).__name__}"
            )


# ─── ModelEvaluation dataclass ────────────────────────────────────────────────


class TestModelEvaluationDataclass:
    """Tests for the ModelEvaluation return type."""

    def test_is_frozen(self) -> None:
        """
        What: ModelEvaluation is immutable after creation.
        Why:  Evaluation results are facts — they should not be mutable.
              Mutable results could be accidentally modified by UI code.
        Prevents: Evaluation results being corrupted by downstream code.
        """
        eval_result = ModelEvaluation(
            model_name="lenet5",
            test_accuracy=0.985,
            test_loss=0.052,
            per_class_f1={"0": 0.99},
            macro_f1=0.985,
            weighted_f1=0.985,
            classification_report_text="...",
            confusion_matrix_path=Path("cm.png"),
            f1_chart_path=Path("f1.png"),
        )
        with pytest.raises((AttributeError, TypeError)):
            eval_result.test_accuracy = 0.5  # type: ignore[misc]


# ─── evaluate_model error handling ───────────────────────────────────────────


class TestEvaluateModelErrorHandling:
    """Tests for evaluate_model() error cases."""

    def test_missing_model_file_raises_file_not_found(self, tmp_path: Path) -> None:
        """
        What: evaluate_model() raises FileNotFoundError when the model file
              does not exist.
        Why:  Running evaluation before training is a common mistake. The error
              message must tell the user to run training first — not a generic
              OSError or KeyError.
        Prevents: Confusing errors when evaluate is run before train.
        """
        from src.evaluate import evaluate_model

        with (
            patch("src.evaluate.MODEL_PATHS", {"dense_nn": tmp_path / "missing.keras"}),
            pytest.raises(FileNotFoundError, match="Train it first"),
        ):
            evaluate_model("dense_nn")
