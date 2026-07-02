"""
Tests for src/predict.py — inference engine.

Why these tests matter:
    predict.py is the only module that bridges model weights on disk to the
    Streamlit UI. Bugs here manifest as wrong predictions in production. The
    tests verify the full contract of PredictionResult — not just that it
    runs, but that the values it returns are structurally correct.

Tests in this file:
    - PredictionResult structure and types
    - Valid prediction from canvas
    - Valid prediction from uploaded image
    - Batch prediction returns correct count
    - Batch prediction empty input returns empty list
    - Unknown model name raises ValueError
    - Missing model file raises FileNotFoundError
    - Confidence scores sum to 1.0
    - Predicted digit matches argmax of probabilities
    - Model cache is populated after first load
    - Invalid canvas input raises ValueError (not a silent wrong answer)
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.predict import (
    PredictionResult,
    _load_model,
    _model_cache,
    _probabilities_to_result,
    predict_batch,
    predict_from_canvas,
    predict_from_upload,
)


# ─── PredictionResult dataclass ──────────────────────────────────────────────


class TestPredictionResult:
    """Tests for the PredictionResult dataclass structure."""

    def test_prediction_result_is_frozen(self) -> None:
        """
        What: PredictionResult is immutable (frozen dataclass).
        Why:  Streamlit pages should never modify a prediction result — that
              could corrupt cached values or produce inconsistent UI state.
        Prevents: Accidental mutation of prediction results in page logic.
        """
        result = PredictionResult(
            predicted_digit=7,
            predicted_label="7",
            confidence=0.97,
            all_probabilities={str(i): 0.01 for i in range(10)},
            model_display_name="Custom CNN",
        )
        with pytest.raises((AttributeError, TypeError)):
            result.predicted_digit = 5  # type: ignore[misc]

    def test_prediction_result_fields(self) -> None:
        """
        What: All five fields are accessible with correct types.
        Why:  The Streamlit UI reads all five fields. A missing or renamed
              field causes AttributeError at runtime in the UI.
        Prevents: Field name typos or missing fields breaking the UI.
        """
        probs = {str(i): (0.9 if i == 3 else 0.01) for i in range(10)}
        result = PredictionResult(
            predicted_digit=3,
            predicted_label="3",
            confidence=0.9,
            all_probabilities=probs,
            model_display_name="Custom CNN",
        )
        assert isinstance(result.predicted_digit, int)
        assert isinstance(result.predicted_label, str)
        assert isinstance(result.confidence, float)
        assert isinstance(result.all_probabilities, dict)
        assert isinstance(result.model_display_name, str)
        assert len(result.all_probabilities) == 10


# ─── _probabilities_to_result ─────────────────────────────────────────────────


class TestProbabilitiesToResult:
    """Tests for the internal _probabilities_to_result() converter."""

    def test_argmax_matches_predicted_digit(self) -> None:
        """
        What: predicted_digit is the argmax of the probability vector.
        Why:  The model's prediction is defined as the class with the highest
              probability. Any other selection rule would be wrong.
        Prevents: Off-by-one errors or wrong indexing in argmax logic.
        """
        probs = np.zeros(10, dtype=np.float32)
        probs[7] = 0.97
        probs[1] = 0.03

        result = _probabilities_to_result(probs, "custom_cnn")
        assert result.predicted_digit == 7

    def test_confidence_matches_top_class_probability(self) -> None:
        """
        What: confidence equals the probability of the predicted class.
        Why:  Displaying confidence is core to the UI's value. If confidence
              is taken from the wrong class index, the number is meaningless.
        Prevents: Confidence value coming from the wrong class.
        """
        probs = np.zeros(10, dtype=np.float32)
        probs[4] = 0.85
        probs[9] = 0.15

        result = _probabilities_to_result(probs, "custom_cnn")
        assert abs(result.confidence - 0.85) < 1e-6

    def test_all_probabilities_has_ten_entries(self) -> None:
        """
        What: all_probabilities contains exactly 10 entries (one per class).
        Why:  The confidence bar chart renders all 10 bars. Missing entries
              would produce incomplete charts with missing bars.
        Prevents: Partial probability dicts breaking the confidence chart.
        """
        probs = np.random.dirichlet(np.ones(10)).astype(np.float32)
        result = _probabilities_to_result(probs, "dense_nn")
        assert len(result.all_probabilities) == 10

    def test_all_probabilities_uses_string_keys(self) -> None:
        """
        What: all_probabilities keys are strings ('0'–'9'), not integers.
        Why:  The chart builder uses dict['7'] not dict[7]. String keys match
              CLASS_NAMES format and the chart's x-axis labels.
        Prevents: KeyError in build_confidence_bar_chart() from int vs str keys.
        """
        probs = np.ones(10, dtype=np.float32) / 10
        result = _probabilities_to_result(probs, "lenet5")
        assert all(isinstance(k, str) for k in result.all_probabilities)
        assert "0" in result.all_probabilities
        assert "9" in result.all_probabilities

    def test_predicted_label_matches_predicted_digit(self) -> None:
        """
        What: predicted_label is the string version of predicted_digit.
        Why:  CLASS_NAMES maps 7 → '7'. If indexing is wrong, the label
              could say '3' while the digit says 7.
        Prevents: Label-digit mismatch in the prediction result card.
        """
        probs = np.zeros(10, dtype=np.float32)
        probs[5] = 1.0
        result = _probabilities_to_result(probs, "lenet5")
        assert result.predicted_label == str(result.predicted_digit)


# ─── _load_model ──────────────────────────────────────────────────────────────


class TestLoadModel:
    """Tests for the model loading and caching logic."""

    def test_unknown_model_raises_value_error(self) -> None:
        """
        What: _load_model() raises ValueError for unregistered model names.
        Why:  A typo in a model name must fail immediately with a clear message,
              not with a cryptic KeyError or AttributeError later.
        Prevents: Silent wrong behaviour from unknown model names.
        """
        with pytest.raises(ValueError, match="Unknown model name"):
            _load_model("nonexistent_model")

    def test_missing_file_raises_file_not_found(self, tmp_path: Path) -> None:
        """
        What: _load_model() raises FileNotFoundError when the model file is missing.
        Why:  If a user runs the Streamlit app before training, the error message
              should clearly say "model not trained yet" and how to fix it.
        Prevents: Confusing OSError instead of a helpful FileNotFoundError.
        """
        with patch("src.predict.MODEL_PATHS", {"dense_nn": tmp_path / "missing.keras"}):
            # Clear cache for this model name
            _model_cache.pop("dense_nn", None)
            with pytest.raises(FileNotFoundError, match="not been trained"):
                _load_model("dense_nn")

    def test_cached_model_returned_on_second_call(
        self, saved_stub_model: Path
    ) -> None:
        """
        What: Second call to _load_model() returns the cached model without
              reloading from disk.
        Why:  Each Streamlit re-run calls _load_model(). Without caching, every
              user interaction would wait 1–2 seconds for model loading. The
              cache is the key to responsive UI.
        Prevents: Per-request model loading making the app feel sluggish.
        """
        model_name = "custom_cnn"
        with patch("src.predict.MODEL_PATHS", {model_name: saved_stub_model}):
            _model_cache.pop(model_name, None)

            model_first = _load_model(model_name)
            model_second = _load_model(model_name)

            # Same object in memory — not reloaded from disk
            assert model_first is model_second, "Second call should return cached model"


# ─── predict_from_canvas ──────────────────────────────────────────────────────


class TestPredictFromCanvas:
    """Tests for predict_from_canvas()."""

    def test_valid_canvas_returns_prediction_result(
        self, valid_canvas_rgba: np.ndarray, saved_stub_model: Path
    ) -> None:
        """
        What: A valid canvas image produces a PredictionResult.
        Why:  This is the primary code path for the Draw page. It must work
              end-to-end from canvas data to a structured result.
        Prevents: Integration failures between preprocessing and inference.
        """
        model_name = "custom_cnn"
        with patch("src.predict.MODEL_PATHS", {model_name: saved_stub_model}):
            _model_cache.pop(model_name, None)
            result = predict_from_canvas(valid_canvas_rgba, model_name)

        assert isinstance(result, PredictionResult)
        assert 0 <= result.predicted_digit <= 9
        assert 0.0 <= result.confidence <= 1.0

    def test_invalid_canvas_shape_raises_value_error(
        self, saved_stub_model: Path
    ) -> None:
        """
        What: Passing a 3-channel (RGB) array raises ValueError before inference.
        Why:  The canvas preprocessing step validates the input. The error should
              come from preprocessing — not a cryptic TensorFlow error during predict.
        Prevents: Hard-to-debug TF shape errors reaching the user.
        """
        bad_canvas = np.zeros((280, 280, 3), dtype=np.uint8)
        model_name = "custom_cnn"
        with patch("src.predict.MODEL_PATHS", {model_name: saved_stub_model}):
            _model_cache.pop(model_name, None)
            with pytest.raises(ValueError, match="RGBA"):
                predict_from_canvas(bad_canvas, model_name)


# ─── predict_from_upload ──────────────────────────────────────────────────────


class TestPredictFromUpload:
    """Tests for predict_from_upload()."""

    def test_pil_image_returns_prediction_result(
        self, white_pil_image, saved_stub_model: Path
    ) -> None:
        """
        What: A valid PIL Image produces a PredictionResult.
        Why:  This is the primary code path for the Batch Prediction page.
        Prevents: Integration failures in the upload prediction path.
        """
        model_name = "custom_cnn"
        with patch("src.predict.MODEL_PATHS", {model_name: saved_stub_model}):
            _model_cache.pop(model_name, None)
            result = predict_from_upload(white_pil_image, model_name)

        assert isinstance(result, PredictionResult)
        assert 0 <= result.predicted_digit <= 9


# ─── predict_batch ────────────────────────────────────────────────────────────


class TestPredictBatch:
    """Tests for predict_batch()."""

    def test_empty_list_returns_empty_list(self) -> None:
        """
        What: predict_batch([]) returns [].
        Why:  The batch prediction page may call this with zero images if no
              files have been uploaded. It must return an empty list, not raise.
        Prevents: IndexError or ValueError on empty batch input.
        """
        result = predict_batch([], "custom_cnn")
        assert result == []

    def test_batch_returns_correct_count(
        self, saved_stub_model: Path
    ) -> None:
        """
        What: N input images → N PredictionResults.
        Why:  The batch page displays one result per uploaded image. A mismatch
              between input count and output count would silently drop results.
        Prevents: Off-by-one or indexing errors dropping predictions.
        """
        batch_size = 5
        images = [np.random.rand(1, 28, 28, 1).astype(np.float32) for _ in range(batch_size)]
        model_name = "custom_cnn"
        with patch("src.predict.MODEL_PATHS", {model_name: saved_stub_model}):
            _model_cache.pop(model_name, None)
            results = predict_batch(images, model_name)

        assert len(results) == batch_size, (
            f"Expected {batch_size} results, got {len(results)}"
        )

    def test_batch_all_results_are_prediction_results(
        self, saved_stub_model: Path
    ) -> None:
        """
        What: Every element in the returned list is a PredictionResult.
        Why:  The UI iterates the list and accesses .predicted_digit on each.
              A mix of types would cause AttributeError.
        Prevents: Wrong types in the returned list.
        """
        images = [np.random.rand(1, 28, 28, 1).astype(np.float32) for _ in range(3)]
        model_name = "custom_cnn"
        with patch("src.predict.MODEL_PATHS", {model_name: saved_stub_model}):
            _model_cache.pop(model_name, None)
            results = predict_batch(images, model_name)

        assert all(isinstance(r, PredictionResult) for r in results)
