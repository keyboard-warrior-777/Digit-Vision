import json
import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import tensorflow as tf

from src.evaluate import ModelEvaluation
from src.model_card import (
    _format_per_class_f1_table,
    _get_architecture_description,
    _get_batch_size_from_config,
    _get_git_commit,
    _get_model_narrative,
    generate_model_card,
    load_metadata,
)
from src.train import TrainingResult


@pytest.fixture
def dummy_training_result() -> TrainingResult:
    return TrainingResult(
        model_name="test_model",
        final_train_accuracy=0.9,
        final_val_accuracy=0.92,
        epochs_trained=2,
        model_path=Path("dummy.keras"),
        history_path=Path("dummy.json"),
    )


@pytest.fixture
def dummy_evaluation_result() -> ModelEvaluation:
    return ModelEvaluation(
        model_name="test_model",
        test_loss=0.1,
        test_accuracy=0.95,
        macro_f1=0.94,
        weighted_f1=0.95,
        per_class_f1={"0": 0.99, "1": 0.98, "2": 0.90},
        classification_report_text="dummy report",
        confusion_matrix_path=Path("dummy_cm.png"),
        f1_chart_path=Path("dummy_f1.png"),
    )


@pytest.fixture
def dummy_model() -> tf.keras.Model:
    model = MagicMock(spec=tf.keras.Model)
    model.count_params.return_value = 1000
    model.trainable_weights = []
    model.optimizer = MagicMock()
    model.optimizer.learning_rate = 0.001
    return model


def test_generate_model_card(
    tmp_path: Path,
    dummy_training_result: TrainingResult,
    dummy_evaluation_result: ModelEvaluation,
    dummy_model: tf.keras.Model,
) -> None:
    meta_path = tmp_path / "metadata.json"
    summary_path = tmp_path / "summary.md"

    with (
        patch("src.model_card.METADATA_PATHS", {"test_model": meta_path}),
        patch("src.model_card.MODEL_SUMMARY_PATHS", {"test_model": summary_path}),
    ):
        generate_model_card(
            model_name="test_model",
            training_result=dummy_training_result,
            evaluation_result=dummy_evaluation_result,
            training_time_seconds=120.5,
            model=dummy_model,
        )

        assert meta_path.exists()
        assert summary_path.exists()

        # Verify JSON properties
        with meta_path.open("r", encoding="utf-8") as f:
            metadata = json.load(f)
            assert metadata["model_name"] == "test_model"
            assert metadata["test_accuracy"] == 0.95

        # Verify Markdown summary
        summary = summary_path.read_text(encoding="utf-8")
        assert "Model Card" in summary
        assert "test_model" in summary
        assert "120.5 minutes" not in summary  # 120.5 secs = 2.0 minutes
        assert "2.0 minutes" in summary


def test_load_metadata(tmp_path: Path) -> None:
    meta_path = tmp_path / "metadata.json"

    with patch("src.model_card.METADATA_PATHS", {"test_model": meta_path}):
        assert load_metadata("test_model") is None

        meta_path.write_text('{"key": "value"}', encoding="utf-8")
        assert load_metadata("test_model") == {"key": "value"}


def test_load_metadata_unregistered_model() -> None:
    with patch("src.model_card.METADATA_PATHS", {}):
        assert load_metadata("unknown_model") is None


def test_format_per_class_f1_table() -> None:
    scores = {"0": 0.995, "1": 0.975, "2": 0.90}
    table = _format_per_class_f1_table(scores)
    assert "Excellent" in table
    assert "Good" in table
    assert "Needs work" in table


def test_get_model_narrative() -> None:
    s, _, _ = _get_model_narrative("dense_nn")
    assert "Simple architecture" in s

    s, _, _ = _get_model_narrative("unknown")
    assert s == "No narrative available."


def test_get_architecture_description() -> None:
    desc = _get_architecture_description("lenet5")
    assert "Conv2D(6" in desc

    desc = _get_architecture_description("unknown")
    assert desc == "No architecture description available."


def test_get_batch_size_from_config() -> None:
    with patch("src.model_card.MODEL_TRAINING_CONFIG", {"test": {"batch_size": 64}}):
        assert _get_batch_size_from_config("test") == 64
        assert _get_batch_size_from_config("missing") == 128


def test_get_git_commit_success() -> None:
    with patch("src.model_card.subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0, stdout="abcdef12\n")
        assert _get_git_commit() == "abcdef12"


def test_get_git_commit_failure() -> None:
    with patch("src.model_card.subprocess.run") as mock_run:
        mock_run.side_effect = FileNotFoundError
        assert _get_git_commit() is None

    with patch("src.model_card.subprocess.run") as mock_run:
        mock_run.side_effect = subprocess.TimeoutExpired(cmd="git", timeout=3)
        assert _get_git_commit() is None

    with patch("src.model_card.subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=1)
        assert _get_git_commit() is None
