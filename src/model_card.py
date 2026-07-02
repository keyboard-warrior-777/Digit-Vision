"""
Model card and training summary generation for DigitVision.

After a model is trained and evaluated, this module generates two files:

1. ``{model_name}_metadata.json``
   Machine-readable JSON with all training and evaluation metrics.
   Read by the Streamlit dashboard to populate performance cards without
   recomputing anything.

2. ``{model_name}_summary.md``
   Human-readable Markdown summary covering architecture, metrics,
   strengths, weaknesses, and recommended use case. Professional enough
   to discuss in interviews or include in project documentation.

Design — Single Responsibility:
    This module is deliberately separate from both ``train.py`` and
    ``evaluate.py``. It receives their results and combines them. Keeping
    it separate means training and evaluation stay focused on their own
    concerns, while card generation can evolve independently.
"""

from __future__ import annotations

import json
import platform
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import tensorflow as tf

from config.config import (
    METADATA_PATHS,
    MODEL_DISPLAY_NAMES,
    MODEL_SUMMARY_PATHS,
    MODEL_TRAINING_CONFIG,
)
from src.logger import get_logger

if TYPE_CHECKING:
    from src.evaluate import ModelEvaluation
    from src.train import TrainingResult

logger = get_logger(__name__)


def generate_model_card(
    model_name: str,
    training_result: "TrainingResult",
    evaluation_result: "ModelEvaluation",
    training_time_seconds: float,
    model: tf.keras.Model,
) -> None:
    """
    Generate and save both the metadata JSON and the Markdown summary.

    This is the single public entry point. Call it once after both
    training and evaluation are complete.

    Args:
        model_name: Registered model name.
        training_result: Result dataclass returned by ``train_model()``.
        evaluation_result: Result dataclass returned by ``evaluate_model()``.
        training_time_seconds: Wall-clock training duration in seconds.
        model: The trained Keras model (used for parameter count).
    """
    _save_metadata_json(
        model_name, training_result, evaluation_result, training_time_seconds, model
    )
    _save_summary_markdown(
        model_name, training_result, evaluation_result, training_time_seconds, model
    )


def load_metadata(model_name: str) -> Optional[dict]:
    """
    Load the saved metadata JSON for a model.

    Used by the Streamlit dashboard to display performance cards
    without recomputing metrics.

    Args:
        model_name: Registered model name.

    Returns:
        The parsed metadata dictionary, or None if the file does not exist.
    """
    path = METADATA_PATHS.get(model_name)
    if path is None or not path.exists():
        return None

    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


# ─── Private generators ────────────────────────────────────────────────────────


def _save_metadata_json(
    model_name: str,
    training_result: "TrainingResult",
    evaluation_result: "ModelEvaluation",
    training_time_seconds: float,
    model: tf.keras.Model,
) -> None:
    """Build and save the metadata.json file."""
    metadata = {
        # Identity
        "model_name": model_name,
        "display_name": MODEL_DISPLAY_NAMES.get(model_name, model_name),
        "dataset": "MNIST",
        "date_trained": datetime.now(tz=timezone.utc).isoformat(),
        # Architecture
        "total_parameters": model.count_params(),
        "trainable_parameters": sum(
            tf.keras.backend.count_params(w) for w in model.trainable_weights
        ),
        # Training configuration
        "training_time_seconds": round(training_time_seconds, 2),
        "epochs_trained": training_result.epochs_trained,
        "optimizer": type(model.optimizer).__name__,
        "learning_rate": float(model.optimizer.learning_rate),
        "batch_size": _get_batch_size_from_config(model_name),
        # Training metrics
        "final_train_accuracy": round(training_result.final_train_accuracy, 6),
        "final_val_accuracy": round(training_result.final_val_accuracy, 6),
        # Evaluation metrics (test set)
        "test_accuracy": round(evaluation_result.test_accuracy, 6),
        "test_loss": round(evaluation_result.test_loss, 6),
        "macro_f1": round(evaluation_result.macro_f1, 6),
        "weighted_f1": round(evaluation_result.weighted_f1, 6),
        "per_class_f1": {
            k: round(v, 6) for k, v in evaluation_result.per_class_f1.items()
        },
        # Environment
        "tensorflow_version": tf.__version__,
        "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        "platform": platform.system(),
        "git_commit": _get_git_commit(),
    }

    save_path = METADATA_PATHS[model_name]
    save_path.parent.mkdir(parents=True, exist_ok=True)

    with save_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    logger.info("Metadata saved → %s", save_path)


def _save_summary_markdown(
    model_name: str,
    training_result: "TrainingResult",
    evaluation_result: "ModelEvaluation",
    training_time_seconds: float,
    model: tf.keras.Model,
) -> None:
    """Build and save the human-readable model summary Markdown file."""
    display_name = MODEL_DISPLAY_NAMES.get(model_name, model_name)
    accuracy_pct = evaluation_result.test_accuracy * 100
    f1_pct = evaluation_result.macro_f1 * 100
    training_minutes = training_time_seconds / 60

    # Find the best and worst performing digit classes
    f1_scores = evaluation_result.per_class_f1
    best_class = max(f1_scores, key=f1_scores.get)
    worst_class = min(f1_scores, key=f1_scores.get)

    strengths, weaknesses, use_case = _get_model_narrative(model_name)

    markdown = f"""# Model Card — {display_name}

> Auto-generated after training and evaluation. Last updated: {datetime.now(tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")}

---

## Overview

| Field              | Value                            |
|--------------------|----------------------------------|
| **Model Name**     | `{model_name}`                   |
| **Dataset**        | MNIST (70,000 images, 10 classes)|
| **Parameters**     | {model.count_params():,}         |
| **Optimiser**      | Adam (lr = {float(model.optimizer.learning_rate):.0e}) |
| **Epochs Trained** | {training_result.epochs_trained} |
| **Training Time**  | {training_minutes:.1f} minutes   |

---

## Evaluation Metrics (Test Set — 10,000 samples)

| Metric              | Value        |
|---------------------|--------------|
| **Test Accuracy**   | {accuracy_pct:.2f}% |
| **Test Loss**       | {evaluation_result.test_loss:.4f} |
| **Macro F1**        | {f1_pct:.2f}% |
| **Weighted F1**     | {evaluation_result.weighted_f1 * 100:.2f}% |
| **Best Class**      | Digit {best_class} (F1 = {f1_scores[best_class]:.4f}) |
| **Hardest Class**   | Digit {worst_class} (F1 = {f1_scores[worst_class]:.4f}) |

---

## Per-Class F1 Scores

| Digit | F1 Score | Grade |
|-------|----------|-------|
{_format_per_class_f1_table(f1_scores)}

---

## Architecture

{_get_architecture_description(model_name)}

---

## Strengths

{strengths}

## Weaknesses

{weaknesses}

## Recommended Use Case

{use_case}

---

*Generated by DigitVision · TensorFlow {tf.__version__} · Python {sys.version_info.major}.{sys.version_info.minor}*
"""

    save_path = MODEL_SUMMARY_PATHS[model_name]
    save_path.parent.mkdir(parents=True, exist_ok=True)

    with save_path.open("w", encoding="utf-8") as f:
        f.write(markdown)

    logger.info("Model summary saved → %s", save_path)


# ─── Private helpers ────────────────────────────────────────────────────────────


def _format_per_class_f1_table(f1_scores: dict[str, float]) -> str:
    """Format per-class F1 scores as Markdown table rows."""
    rows = []
    for digit, score in sorted(f1_scores.items(), key=lambda x: int(x[0])):
        grade = "🟢 Excellent" if score >= 0.99 else "🟡 Good" if score >= 0.97 else "🔴 Needs work"
        rows.append(f"| {digit}     | {score:.4f}   | {grade} |")
    return "\n".join(rows)


def _get_model_narrative(
    model_name: str,
) -> tuple[str, str, str]:
    """Return (strengths, weaknesses, use_case) for each model."""
    narratives = {
        "dense_nn": (
            "- Simple architecture — easy to understand and explain\n"
            "- Fast training and inference\n"
            "- No hyperparameter tuning beyond layer sizes required",
            "- No spatial awareness — treats each pixel independently\n"
            "- More parameters than the CNN for worse performance\n"
            "- Sensitive to translation and rotation of the digit",
            "Best as a baseline comparison point. Shows why spatial feature "
            "extraction via convolution was such a significant advance.",
        ),
        "lenet5": (
            "- Historically significant — foundational CNN architecture\n"
            "- Demonstrates core CNN concepts (convolution, pooling)\n"
            "- Compact architecture with a clear two-stage design",
            "- AveragePooling loses some feature detail vs MaxPooling\n"
            "- No BatchNormalization — training is less stable\n"
            "- No Dropout — more susceptible to overfitting on small datasets",
            "Excellent for educational purposes. Demonstrates the original "
            "insights that enabled the deep learning revolution. Not recommended "
            "for production use.",
        ),
        "custom_cnn": (
            "- Best accuracy with fewest parameters among the three models\n"
            "- BatchNormalization enables stable training at higher learning rates\n"
            "- GlobalAveragePooling provides structural regularisation\n"
            "- MaxPooling retains strongest feature activations",
            "- More hyperparameters to tune than the simpler architectures\n"
            "- Grad-CAM explanations require the convolutional structure\n"
            "- Slightly longer training time due to deeper architecture",
            "Recommended for production deployment. Best accuracy/parameter "
            "trade-off of the three architectures. Suitable for real-time "
            "digit recognition in web or mobile applications.",
        ),
    }
    return narratives.get(
        model_name,
        ("No narrative available.", "No narrative available.", "General purpose."),
    )


def _get_architecture_description(model_name: str) -> str:
    """Return a Markdown architecture description for each model."""
    descriptions = {
        "dense_nn": (
            "```\n"
            "Input (28, 28, 1)\n"
            "  └─ Flatten → (784,)\n"
            "  └─ Dense(512) + ReLU + Dropout(0.3)\n"
            "  └─ Dense(256) + ReLU + Dropout(0.3)\n"
            "  └─ Dense(128) + ReLU + Dropout(0.2)\n"
            "  └─ Dense(10) + Softmax\n"
            "```"
        ),
        "lenet5": (
            "```\n"
            "Input (28, 28, 1)\n"
            "  └─ Conv2D(6, 5×5, same) + ReLU → AveragePool(2×2)\n"
            "  └─ Conv2D(16, 5×5, valid) + ReLU → AveragePool(2×2)\n"
            "  └─ Flatten → (400,)\n"
            "  └─ Dense(120) + ReLU\n"
            "  └─ Dense(84) + ReLU\n"
            "  └─ Dense(10) + Softmax\n"
            "```"
        ),
        "custom_cnn": (
            "```\n"
            "Input (28, 28, 1)\n"
            "  └─ [Conv2D(32) → BN → ReLU] × 2 → MaxPool → Dropout(0.25)\n"
            "  └─ [Conv2D(64) → BN → ReLU] × 2 → MaxPool → Dropout(0.25)\n"
            "  └─ GlobalAveragePooling2D → (64,)\n"
            "  └─ Dense(128) → BN → ReLU → Dropout(0.5)\n"
            "  └─ Dense(10) + Softmax\n"
            "```"
        ),
    }
    return descriptions.get(model_name, "No architecture description available.")


def _get_batch_size_from_config(model_name: str) -> int:
    """Retrieve the batch size for a given model from the central config."""
    return MODEL_TRAINING_CONFIG.get(model_name, {}).get("batch_size", 128)


def _get_git_commit() -> Optional[str]:
    """Return the current git commit hash, or None if git is not available."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=3,
        )
        return result.stdout.strip() if result.returncode == 0 else None
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None
