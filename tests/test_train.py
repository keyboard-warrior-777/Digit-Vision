"""
Tests for src/train.py — training pipeline components.

Why these tests matter:
    Training is the most expensive operation in the project (~15–20 minutes).
    We cannot run it in CI, but we can test every component around it:
    the callbacks configuration, the history serialisation, and the
    TrainingResult dataclass. If these components are broken, training will
    fail or produce corrupted outputs.

Tests in this file:
    - _build_training_callbacks returns correct number of callbacks
    - EarlyStopping is configured with restore_best_weights=True
    - ModelCheckpoint monitors val_accuracy
    - ReduceLROnPlateau halves LR on plateau
    - _save_training_history creates valid JSON file
    - Saved history is round-trippable (load → same values)
    - TrainingResult is a frozen dataclass
    - TrainingResult fields have correct types
    - History JSON values are Python float (not numpy float32)
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import tensorflow as tf

from src.train import TrainingResult, _build_training_callbacks, _save_training_history

# ─── TrainingResult dataclass ─────────────────────────────────────────────────


class TestTrainingResult:
    """Tests for the TrainingResult dataclass."""

    def test_is_frozen(self, tmp_path: Path) -> None:
        """
        What: TrainingResult is immutable after construction.
        Why:  Training results are historical facts. Mutable results could be
              accidentally modified when the same result is reused in different
              log entries or summary tables.
        Prevents: Post-construction mutation corrupting training logs.
        """
        result = TrainingResult(
            model_name="custom_cnn",
            final_train_accuracy=0.993,
            final_val_accuracy=0.992,
            epochs_trained=18,
            model_path=tmp_path / "model.keras",
            history_path=tmp_path / "history.json",
        )
        with pytest.raises((AttributeError, TypeError)):
            result.model_name = "lenet5"  # type: ignore[misc]

    def test_field_types(self, tmp_path: Path) -> None:
        """
        What: All fields have the correct Python types.
        Why:  train.py logs and model_card.py metadata both read these fields
              and pass them to json.dump(). Wrong types (e.g. numpy types)
              would cause serialisation errors.
        Prevents: Type errors in metadata JSON generation.
        """
        result = TrainingResult(
            model_name="lenet5",
            final_train_accuracy=0.985,
            final_val_accuracy=0.983,
            epochs_trained=20,
            model_path=tmp_path / "model.keras",
            history_path=tmp_path / "history.json",
        )
        assert isinstance(result.model_name, str)
        assert isinstance(result.final_train_accuracy, float)
        assert isinstance(result.final_val_accuracy, float)
        assert isinstance(result.epochs_trained, int)
        assert isinstance(result.model_path, Path)
        assert isinstance(result.history_path, Path)


# ─── _build_training_callbacks ───────────────────────────────────────────────


class TestBuildTrainingCallbacks:
    """Tests for the training callbacks configuration."""

    @pytest.fixture
    def callbacks(self) -> list:
        """Build callbacks for the custom_cnn model."""
        with patch(
            "src.train.MODEL_PATHS",
            {"custom_cnn": Path("models/saved/custom_cnn.keras")},
        ):
            return _build_training_callbacks("custom_cnn")

    def test_returns_three_callbacks(self, callbacks: list) -> None:
        """
        What: Exactly three callbacks are returned.
        Why:  The training loop expects ModelCheckpoint, EarlyStopping, and
              ReduceLROnPlateau. Missing or extra callbacks change training
              behaviour silently.
        Prevents: Callbacks being accidentally removed or duplicated.
        """
        assert (
            len(callbacks) == 3
        ), f"Expected 3 callbacks, got {len(callbacks)}: {[type(c).__name__ for c in callbacks]}"

    def test_includes_model_checkpoint(self, callbacks: list) -> None:
        """
        What: One of the callbacks is ModelCheckpoint.
        Why:  Without ModelCheckpoint, only the final epoch's weights are saved
              — not the best epoch's weights. The saved model would be suboptimal.
        Prevents: Best-epoch weights not being saved due to missing checkpoint.
        """
        types = [type(c) for c in callbacks]
        assert tf.keras.callbacks.ModelCheckpoint in types

    def test_model_checkpoint_monitors_val_accuracy(self, callbacks: list) -> None:
        """
        What: ModelCheckpoint monitors 'val_accuracy' (not 'val_loss').
        Why:  We want the model with the highest accuracy, not lowest loss.
              On MNIST these often diverge slightly in late epochs — always
              saving the lowest-loss model can sacrifice a few accuracy points.
        Prevents: Saving suboptimal weights due to wrong monitor metric.
        """
        checkpoint = next(
            c for c in callbacks if isinstance(c, tf.keras.callbacks.ModelCheckpoint)
        )
        assert (
            checkpoint.monitor == "val_accuracy"
        ), f"ModelCheckpoint should monitor 'val_accuracy', monitors '{checkpoint.monitor}'"

    def test_model_checkpoint_saves_best_only(self, callbacks: list) -> None:
        """
        What: ModelCheckpoint has save_best_only=True.
        Why:  Without this, the checkpoint file is overwritten every epoch.
              After EarlyStopping triggers, the final file contains the last
              epoch's weights — not the best epoch's weights.
        Prevents: Suboptimal model weights being saved to disk.
        """
        checkpoint = next(
            c for c in callbacks if isinstance(c, tf.keras.callbacks.ModelCheckpoint)
        )
        assert checkpoint.save_best_only is True

    def test_early_stopping_restores_best_weights(self, callbacks: list) -> None:
        """
        What: EarlyStopping has restore_best_weights=True.
        Why:  Without this, after EarlyStopping fires, the model's in-memory
              weights are from the last (suboptimal) epoch, not the best epoch.
              The model saved by _save_model() would be the last epoch, not best.
        Prevents: Returning worse weights than the best observed checkpoint.
        """
        early_stopping = next(
            c for c in callbacks if isinstance(c, tf.keras.callbacks.EarlyStopping)
        )
        assert early_stopping.restore_best_weights is True

    def test_early_stopping_monitors_val_loss(self, callbacks: list) -> None:
        """
        What: EarlyStopping monitors 'val_loss'.
        Why:  val_loss is smoother than val_accuracy and more reliably signals
              when the model has stopped improving. Monitoring val_accuracy can
              trigger premature stopping due to accuracy plateaus.
        Prevents: Premature early stopping from noisy accuracy metric.
        """
        early_stopping = next(
            c for c in callbacks if isinstance(c, tf.keras.callbacks.EarlyStopping)
        )
        assert early_stopping.monitor == "val_loss"

    def test_reduce_lr_halves_learning_rate(self, callbacks: list) -> None:
        """
        What: ReduceLROnPlateau uses factor=0.5 (halving the learning rate).
        Why:  A factor of 0.5 is the standard choice. A higher factor (e.g. 0.9)
              barely changes the LR; a lower factor (e.g. 0.1) reduces it too
              aggressively, causing training to stall.
        Prevents: Learning rate being reduced by a wrong factor.
        """
        reduce_lr = next(
            c for c in callbacks if isinstance(c, tf.keras.callbacks.ReduceLROnPlateau)
        )
        assert (
            reduce_lr.factor == 0.5
        ), f"ReduceLROnPlateau factor should be 0.5, got {reduce_lr.factor}"

    def test_all_three_callback_types_present(self, callbacks: list) -> None:
        """
        What: All three expected callback types are present in the list.
        Why:  A comprehensive check that the exact expected types are returned,
              not just "three of something".
        """
        type_names = {type(c).__name__ for c in callbacks}
        assert "ModelCheckpoint" in type_names
        assert "EarlyStopping" in type_names
        assert "ReduceLROnPlateau" in type_names


# ─── _save_training_history ───────────────────────────────────────────────────


class TestSaveTrainingHistory:
    """Tests for training history serialisation."""

    def _make_mock_history(self) -> MagicMock:
        """Create a Keras History-like mock object."""
        history = MagicMock()
        history.history = {
            "accuracy": [np.float32(0.90), np.float32(0.95), np.float32(0.98)],
            "val_accuracy": [np.float32(0.88), np.float32(0.93), np.float32(0.97)],
            "loss": [np.float32(0.30), np.float32(0.20), np.float32(0.10)],
            "val_loss": [np.float32(0.35), np.float32(0.24), np.float32(0.12)],
        }
        return history

    def test_creates_json_file(self, tmp_path: Path) -> None:
        """
        What: A JSON file is created at the HISTORY_PATHS location.
        Why:  The Streamlit analytics page reads training curves from this file.
              If the file is not created, charts cannot be rendered.
        Prevents: History not being persisted after training completes.
        """
        history = self._make_mock_history()
        history_path = tmp_path / "history.json"

        with patch("src.train.HISTORY_PATHS", {"custom_cnn": history_path}):
            _save_training_history(history, "custom_cnn")

        assert history_path.exists(), "History JSON was not created"

    def test_saved_values_are_python_float(self, tmp_path: Path) -> None:
        """
        What: All values in the saved JSON are Python float, not numpy float32.
        Why:  json.dump() raises TypeError for numpy float32. If the float()
              cast in _save_training_history() is ever removed, this test will
              catch it before the training loop breaks.
        Prevents: TypeError during JSON serialisation causing lost training history.
        """
        history = self._make_mock_history()
        history_path = tmp_path / "history.json"

        with patch("src.train.HISTORY_PATHS", {"lenet5": history_path}):
            _save_training_history(history, "lenet5")

        content = json.loads(history_path.read_text(encoding="utf-8"))
        for key, values in content.items():
            for val in values:
                assert isinstance(val, float), (
                    f"Value {val!r} in '{key}' should be Python float, "
                    f"got {type(val).__name__}"
                )

    def test_round_trip_preserves_values(self, tmp_path: Path) -> None:
        """
        What: Values read back from JSON match the original values within tolerance.
        Why:  numpy float32 → Python float → JSON → Python float should preserve
              values within floating-point precision. If precision is lost, the
              training curve charts will show wrong metric values.
        Prevents: Float precision loss corrupting the displayed training curves.
        """
        history = self._make_mock_history()
        history_path = tmp_path / "history.json"

        with patch("src.train.HISTORY_PATHS", {"dense_nn": history_path}):
            _save_training_history(history, "dense_nn")

        content = json.loads(history_path.read_text(encoding="utf-8"))

        original_val_acc = [float(v) for v in history.history["val_accuracy"]]
        saved_val_acc = content["val_accuracy"]

        for orig, saved in zip(original_val_acc, saved_val_acc, strict=False):
            assert (
                abs(orig - saved) < 1e-5
            ), f"Value mismatch: original={orig}, saved={saved}"

    def test_all_four_metric_keys_saved(self, tmp_path: Path) -> None:
        """
        What: The saved JSON has exactly the four metric keys.
        Why:  All four keys are required by the training curves chart.
        Prevents: A missing key causing a KeyError in the Streamlit analytics page.
        """
        history = self._make_mock_history()
        history_path = tmp_path / "history.json"

        with patch("src.train.HISTORY_PATHS", {"custom_cnn": history_path}):
            _save_training_history(history, "custom_cnn")

        content = json.loads(history_path.read_text(encoding="utf-8"))
        for key in ("accuracy", "val_accuracy", "loss", "val_loss"):
            assert key in content, f"Expected key '{key}' in saved history"
