"""
Post-evaluation artifact generation for DigitVision.

After a model is evaluated, this module generates the rich data files
that the Streamlit frontend needs for interactive visualisations:

1. Raw evaluation data (confusion matrix, ROC curves, per-class metrics)
   Saved as NumPy and JSON so Plotly charts can be built from real data
   rather than reading static PNG images.

2. Sample predictions
   ~10 correct + ~10 incorrect examples, each with image, labels,
   confidence score, and inference time. Displayed in the dashboard.

3. Grad-CAM samples
   One Grad-CAM overlay per digit class (0–9), saved as PNG files.
   Available in the Streamlit interface without retraining.

Design — Why a separate module?
    evaluate.py computes metrics and generates static plots.
    artifacts.py generates the richer, interactive-ready outputs.
    Keeping them separate lets you regenerate artifacts independently
    without re-running the full evaluation pipeline.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import tensorflow as tf
from PIL import Image
from sklearn.metrics import auc, classification_report, confusion_matrix, roc_curve
from sklearn.preprocessing import label_binarize

from config.config import (
    CLASS_NAMES,
    CONFUSION_MATRIX_PATHS,
    GRADCAM_SAMPLES_DIR,
    PREDICTION_SAMPLE_IMAGES_DIR,
    RAW_EVAL_DIR,
    RAW_METRICS_PATHS,
    ROC_DATA_PATHS,
    SAMPLE_PREDICTIONS_PATHS,
)
from src.dataset import MNISTData
from src.gradcam import compute_gradcam, overlay_heatmap_on_image
from src.logger import get_logger

logger = get_logger(__name__)

# Number of correct / incorrect samples to save per model
_CORRECT_SAMPLE_COUNT = 10
_INCORRECT_SAMPLE_COUNT = 10


def generate_all_artifacts(
    model_name: str,
    model: tf.keras.Model,
    data: MNISTData,
    y_predicted: np.ndarray,
    probabilities: np.ndarray,
) -> None:
    """
    Generate all post-evaluation artifacts for a trained model.

    Call this once after evaluate_model() has computed predictions.
    Each artifact type is generated independently — a failure in one
    does not prevent the others from being saved.

    Args:
        model_name: Registered model name.
        model: The loaded trained Keras model.
        data: Pre-loaded MNISTData with the test set.
        y_predicted: Integer predicted labels, shape (10000,).
        probabilities: Softmax probabilities, shape (10000, 10).
    """
    RAW_EVAL_DIR.mkdir(parents=True, exist_ok=True)
    PREDICTION_SAMPLE_IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    GRADCAM_SAMPLES_DIR.mkdir(parents=True, exist_ok=True)

    _save_confusion_matrix(data.y_test_labels, y_predicted, model_name)
    _save_raw_metrics(data.y_test_labels, y_predicted, model_name)
    _save_roc_data(data.y_test_labels, probabilities, model_name)
    _save_sample_predictions(model, data, y_predicted, probabilities, model_name)
    _save_gradcam_samples(model, data, model_name)


# ─── Raw evaluation data ────────────────────────────────────────────────────────


def _save_confusion_matrix(
    true_labels: np.ndarray,
    predicted_labels: np.ndarray,
    model_name: str,
) -> None:
    """
    Save the raw (integer count) confusion matrix as a NumPy .npy file.

    The Streamlit analytics page uses this to build an interactive
    Plotly heatmap with hover tooltips showing raw counts and percentages.
    """
    cm = confusion_matrix(true_labels, predicted_labels)

    save_path = CONFUSION_MATRIX_PATHS[model_name]
    np.save(save_path, cm)
    logger.info("Confusion matrix data saved → %s", save_path)


def _save_raw_metrics(
    true_labels: np.ndarray,
    predicted_labels: np.ndarray,
    model_name: str,
) -> None:
    """
    Save per-class precision, recall, F1, and support as a structured JSON.

    Using sklearn's classification_report with output_dict=True gives us
    a clean dictionary instead of a formatted string, making it trivial
    for the frontend to build tables and charts from real data.
    """
    report = classification_report(
        true_labels,
        predicted_labels,
        target_names=CLASS_NAMES,
        output_dict=True,
        zero_division=0,  # Return 0 for undefined metrics instead of warning
    )

    save_path = RAW_METRICS_PATHS[model_name]
    with save_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    logger.info("Raw metrics saved → %s", save_path)


def _save_roc_data(
    true_labels: np.ndarray,
    probabilities: np.ndarray,
    model_name: str,
) -> None:
    """
    Compute and save one-vs-rest ROC curve data for all 10 digit classes.

    Each class gets its own FPR and TPR arrays, plus the AUC score.
    The frontend reads this JSON to build an interactive multi-line ROC chart.
    """
    # Binarise labels for one-vs-rest ROC: shape (10000, 10)
    y_binary = label_binarize(true_labels, classes=list(range(10)))

    roc_data: dict[str, dict] = {}

    for class_idx, digit_label in enumerate(CLASS_NAMES):
        fpr, tpr, _ = roc_curve(y_binary[:, class_idx], probabilities[:, class_idx])
        roc_auc = auc(fpr, tpr)

        roc_data[digit_label] = {
            "fpr": fpr.tolist(),
            "tpr": tpr.tolist(),
            "auc": round(float(roc_auc), 6),
        }

    save_path = ROC_DATA_PATHS[model_name]
    with save_path.open("w", encoding="utf-8") as f:
        json.dump(roc_data, f, indent=2)

    logger.info("ROC data saved → %s", save_path)


# ─── Sample predictions ─────────────────────────────────────────────────────────


def _save_sample_predictions(
    model: tf.keras.Model,
    data: MNISTData,
    y_predicted: np.ndarray,
    probabilities: np.ndarray,
    model_name: str,
) -> None:
    """
    Save a curated set of correct and incorrect prediction examples.

    Selects high-confidence correct predictions and incorrect predictions
    to provide a balanced view of model behaviour. Each example includes
    the image path, labels, confidence score, and per-example inference time.

    Images are saved as 112×112 PNG files (upscaled from 28×28 with
    NEAREST interpolation to preserve the pixel structure of MNIST).
    """
    correct_mask = y_predicted == data.y_test_labels
    incorrect_mask = ~correct_mask

    correct_indices = np.where(correct_mask)[0]
    incorrect_indices = np.where(incorrect_mask)[0]

    # Sort correct by confidence (take the highest-confidence correct examples)
    correct_confidences = probabilities[correct_indices, y_predicted[correct_indices]]
    top_correct = correct_indices[np.argsort(correct_confidences)[-_CORRECT_SAMPLE_COUNT:]]

    # Take the first N incorrect examples (diverse failures are more informative)
    selected_incorrect = incorrect_indices[:_INCORRECT_SAMPLE_COUNT]

    all_indices = np.concatenate([top_correct, selected_incorrect])
    samples = []

    # Measure per-sample inference time with a single batched call.
    # Calling model.predict() 20 times in a loop is ~20x slower due to
    # per-call graph execution overhead in TensorFlow.
    image_batch = data.X_test[all_indices]  # shape (N, 28, 28, 1)
    t0 = time.perf_counter()
    _ = model.predict(image_batch, verbose=0)
    total_ms = (time.perf_counter() - t0) * 1000
    per_sample_ms = total_ms / len(all_indices) if all_indices.size > 0 else 0.0

    image_save_dir = PREDICTION_SAMPLE_IMAGES_DIR / model_name
    image_save_dir.mkdir(parents=True, exist_ok=True)

    for idx in all_indices:
        image_array = data.X_test[idx]
        true_label = int(data.y_test_labels[idx])
        predicted_label = int(y_predicted[idx])
        confidence = float(probabilities[idx, predicted_label])
        is_correct = true_label == predicted_label

        # Save the image
        image_filename = f"{'correct' if is_correct else 'incorrect'}_{idx}.png"
        image_path = image_save_dir / image_filename
        _save_mnist_image(image_array, image_path)

        samples.append(
            {
                "index": int(idx),
                "image_path": str(image_path),
                "true_label": true_label,
                "predicted_label": predicted_label,
                "confidence": round(confidence, 6),
                "is_correct": is_correct,
                "inference_ms": round(per_sample_ms, 3),
            }
        )

    save_path = SAMPLE_PREDICTIONS_PATHS[model_name]
    with save_path.open("w", encoding="utf-8") as f:
        json.dump(samples, f, indent=2)

    logger.info(
        "Sample predictions saved (%d correct, %d incorrect) → %s",
        len(top_correct),
        len(selected_incorrect),
        save_path,
    )


# ─── Grad-CAM samples ────────────────────────────────────────────────────────────


def _save_gradcam_samples(
    model: tf.keras.Model,
    data: MNISTData,
    model_name: str,
) -> None:
    """
    Save one Grad-CAM overlay image for each digit class (0–9).

    For each digit, finds a correctly classified example with high
    confidence and saves the Grad-CAM overlay. These images are displayed
    immediately in the Streamlit interface without requiring retraining.

    Skips gracefully if the model has no Conv2D layers (Dense NN).
    """
    # Check whether Grad-CAM is applicable before processing 10 digits
    test_image = data.X_test[0:1]
    test_heatmap = compute_gradcam(model, test_image, 0)

    if test_heatmap is None:
        logger.info(
            "Grad-CAM not applicable for '%s' (no Conv2D layers). "
            "Skipping Grad-CAM sample generation.",
            model_name,
        )
        return

    samples_dir = GRADCAM_SAMPLES_DIR / model_name
    samples_dir.mkdir(parents=True, exist_ok=True)

    for digit_class in range(10):
        image_array = _find_best_example_for_class(
            data, digit_class, model
        )
        if image_array is None:
            logger.warning(
                "Could not find a correct example for digit %d — skipping.",
                digit_class,
            )
            continue

        heatmap = compute_gradcam(model, image_array, digit_class)
        if heatmap is None:
            continue

        overlay = overlay_heatmap_on_image(heatmap, image_array)
        pil_overlay = Image.fromarray(overlay)
        pil_overlay = pil_overlay.resize((224, 224), Image.NEAREST)

        save_path = samples_dir / f"digit_{digit_class}_gradcam.png"
        pil_overlay.save(save_path)

    logger.info(
        "Grad-CAM samples saved (one per digit class) → %s",
        samples_dir,
    )


# ─── Private helpers ─────────────────────────────────────────────────────────────


def _find_best_example_for_class(
    data: MNISTData,
    digit_class: int,
    model: tf.keras.Model,
) -> np.ndarray | None:
    """
    Find the highest-confidence correctly-predicted example for a given digit class.

    Returns:
        Preprocessed image array of shape (1, 28, 28, 1), or None if no correct
        example exists for this class.
    """
    class_indices = np.where(data.y_test_labels == digit_class)[0]
    if len(class_indices) == 0:
        return None

    # Get predictions for all examples of this class
    class_images = data.X_test[class_indices]
    batch_probs = model.predict(class_images, verbose=0)
    batch_preds = np.argmax(batch_probs, axis=1)

    # Find correct predictions
    correct_mask = batch_preds == digit_class
    if not correct_mask.any():
        return None

    # Select the one with highest confidence among correct predictions
    correct_confidences = batch_probs[correct_mask, digit_class]
    best_within_correct = np.argmax(correct_confidences)
    best_image = class_images[correct_mask][best_within_correct]

    return best_image.reshape(1, 28, 28, 1)


def _save_mnist_image(image_array: np.ndarray, save_path: Path) -> None:
    """
    Save a normalised 28×28 numpy array as a 112×112 PNG.

    Upscaling with NEAREST interpolation preserves the pixel grid of MNIST —
    important for maintaining the authentic look in the dashboard.
    """
    img_uint8 = (image_array.squeeze() * 255).astype(np.uint8)
    pil_image = Image.fromarray(img_uint8, mode="L")
    pil_image = pil_image.resize((112, 112), Image.NEAREST)
    pil_image.save(save_path)
