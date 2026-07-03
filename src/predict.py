"""
Inference engine for DigitVision.

Provides clean, independently-testable functions for running predictions
against single images, canvas drawings, and image batches.

Design — separation from UI:
    The Streamlit app calls predict_from_canvas() and predict_from_upload().
    It never loads models or calls model.predict() directly. This means the
    prediction logic can be tested entirely without launching the UI.

Model caching:
    TensorFlow model loading takes 1–2 seconds. In Streamlit, every user
    interaction re-runs the script — without caching, every prediction
    would feel sluggish. Models are cached in a module-level dictionary
    after their first load and reused for all subsequent calls.

Return format:
    All prediction functions return a PredictionResult dataclass. The
    full probability distribution across all 10 classes is always
    included — required for the confidence bar chart in the UI.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import tensorflow as tf
from PIL import Image

from config.config import CLASS_NAMES, MODEL_DISPLAY_NAMES, MODEL_PATHS
from src.logger import get_logger
from src.preprocessing import canvas_image_to_model_input, uploaded_image_to_model_input

logger = get_logger(__name__)

# Module-level cache: maps model name → loaded Keras model.
# Populated lazily on first access. Lives for the lifetime of the process.
_model_cache: dict[str, tf.keras.Model] = {}


@dataclass(frozen=True)
class PredictionResult:
    """
    Immutable result of a single digit prediction.

    Contains everything the UI needs to render the prediction view —
    the top-1 class, its confidence score, the full probability vector,
    and the display name of the model that made the prediction.

    Attributes:
        predicted_digit:    Integer class index (0–9).
        predicted_label:    String label matching the digit ('0'–'9').
        confidence:         Probability assigned to the top-1 class, in [0, 1].
        all_probabilities:  Dict mapping digit label → probability for all 10 classes.
                            Used to render the full confidence bar chart.
        model_display_name: Human-readable model name for UI display.
    """

    predicted_digit: int
    predicted_label: str
    confidence: float
    all_probabilities: dict[str, float]
    model_display_name: str


def predict_from_canvas(
    canvas_image: np.ndarray,
    model_name: str,
) -> PredictionResult:
    """
    Run inference on a user-drawn canvas image.

    Applies the complete canvas preprocessing pipeline before prediction:
    RGBA extraction → inversion → resize → normalise → reshape.

    Args:
        canvas_image: RGBA numpy array from streamlit-drawable-canvas,
            shape (H, W, 4), dtype uint8.
        model_name: Registered model name.

    Returns:
        PredictionResult with the predicted digit and confidence scores.
    """
    model_input = canvas_image_to_model_input(canvas_image)
    return _run_inference(model_input, model_name)


def predict_from_upload(
    pil_image: Image.Image,
    model_name: str,
) -> PredictionResult:
    """
    Run inference on a user-uploaded image file.

    Accepts any PIL-supported format (PNG, JPEG, BMP, etc.) at any
    resolution and colour mode. Preprocessing normalises all inputs
    to MNIST format before inference.

    Args:
        pil_image: A PIL Image in any mode.
        model_name: Registered model name.

    Returns:
        PredictionResult with the predicted digit and confidence scores.
    """
    model_input = uploaded_image_to_model_input(pil_image)
    return _run_inference(model_input, model_name)


def predict_batch(
    preprocessed_images: list[np.ndarray],
    model_name: str,
) -> list[PredictionResult]:
    """
    Run inference on a list of preprocessed images in a single forward pass.

    Batching is more efficient than calling predict_from_* in a loop
    because TensorFlow executes all images in a single GPU/CPU kernel call.
    The speed advantage grows with batch size.

    Args:
        preprocessed_images: List of model-ready arrays, each shape
            (1, 28, 28, 1). Typically produced by uploaded_image_to_model_input().
        model_name: Registered model name.

    Returns:
        List of PredictionResults in the same order as the input images.
        Returns an empty list if the input is empty.
    """
    if not preprocessed_images:
        return []

    model = _load_model(model_name)

    # Concatenate along the batch axis: [(1,28,28,1), ...] → (N, 28, 28, 1)
    batch_input = np.concatenate(preprocessed_images, axis=0)
    batch_probabilities = model.predict(batch_input, verbose=0)

    return [
        _probabilities_to_result(probabilities, model_name)
        for probabilities in batch_probabilities
    ]


# ─── Private helpers ─────────────────────────────────────────────────────────


def _run_inference(model_input: np.ndarray, model_name: str) -> PredictionResult:
    """Load the model, run a forward pass, and return a structured result."""
    model = _load_model(model_name)
    probabilities = model.predict(model_input, verbose=0)[0]
    return _probabilities_to_result(probabilities, model_name)


def _probabilities_to_result(
    probabilities: np.ndarray,
    model_name: str,
) -> PredictionResult:
    """
    Convert a raw softmax probability vector into a PredictionResult.

    The argmax of the probability vector gives the predicted class.
    All ten probabilities are included so the UI can display a bar chart
    showing what the model considered for each digit (0–9).
    """
    predicted_digit = int(np.argmax(probabilities))
    confidence = float(probabilities[predicted_digit])

    all_probabilities = {
        label: float(prob)
        for label, prob in zip(CLASS_NAMES, probabilities, strict=False)
    }

    return PredictionResult(
        predicted_digit=predicted_digit,
        predicted_label=CLASS_NAMES[predicted_digit],
        confidence=confidence,
        all_probabilities=all_probabilities,
        model_display_name=MODEL_DISPLAY_NAMES.get(model_name, model_name),
    )


def _load_model(model_name: str) -> tf.keras.Model:
    """
    Load a model from disk, caching it in memory after the first load.

    On first call for a given model_name, the model is loaded from the
    .keras file and stored in _model_cache. All subsequent calls return
    the cached instance immediately — no disk access required.

    Args:
        model_name: Registered model name.

    Returns:
        A loaded tf.keras.Model.

    Raises:
        ValueError: If model_name is not in MODEL_PATHS (unrecognised).
        FileNotFoundError: If the model file does not exist (not trained yet).
    """
    if model_name in _model_cache:
        return _model_cache[model_name]

    model_path = MODEL_PATHS.get(model_name)
    if model_path is None:
        raise ValueError(
            f"Unknown model name '{model_name}'. "
            f"Valid options: {list(MODEL_PATHS.keys())}"
        )

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model '{model_name}' has not been trained yet.\n"
            f"Run: python -m src.train --model {model_name}"
        )

    logger.info("Loading '%s' from %s", model_name, model_path)
    model = tf.keras.models.load_model(model_path)
    _model_cache[model_name] = model
    logger.info("'%s' loaded and cached successfully.", model_name)

    return model
