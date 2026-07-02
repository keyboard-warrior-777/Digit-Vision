"""
Model evaluation suite for DigitVision.

Computes a comprehensive set of metrics and generates visualisations
for each trained model. Goes significantly beyond overall accuracy.

Metrics computed:
    - Test accuracy and loss (overall)
    - Per-class precision, recall, F1-score (sklearn classification_report)
    - Macro-averaged and weighted-averaged F1 scores
    - Normalised confusion matrix

Plots generated and saved to performance_plots/:
    - Confusion matrix heatmap (per model)
    - Per-class F1 bar chart with colour coding (per model)
    - Combined training curves across all three models

Why go beyond accuracy?
    Accuracy averages over all classes. If the model confuses digit 4
    with digit 9 frequently, that weakness is hidden in an overall 99%
    figure but immediately visible in per-class F1 and the confusion matrix.

Interview Note:
    "I used per-class F1 because the confusion matrix immediately
    surfaces which digit pairs are hardest — 4↔9 and 3↔8 are the
    classic difficult pairs. You can see it visually and explain why:
    these digits share similar curves and stroke patterns."
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib

# Non-interactive backend: safe for servers, Docker, CI, and headless environments.
# Must be set before importing pyplot.
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix, f1_score

from config.config import (
    CLASS_NAMES,
    CONFUSION_MATRIX_FIGSIZE,
    CURVES_FIGSIZE,
    HISTORY_PATHS,
    MODEL_DISPLAY_NAMES,
    MODEL_PATHS,
    PERFORMANCE_PLOTS_DIR,
    PLOT_DPI,
    PLOT_STYLE,
)
from src.artifacts import generate_all_artifacts
from src.dataset import MNISTData, load_and_prepare_mnist
from src.logger import get_logger
from src.models import list_available_models

logger = get_logger(__name__)


@dataclass(frozen=True)
class ModelEvaluation:
    """
    Complete evaluation results for a single model.

    Stores both scalar metrics and paths to generated plot files.
    The Streamlit UI reads these paths to display visualisations
    without re-running evaluation.
    """

    model_name: str
    test_accuracy: float
    test_loss: float
    per_class_f1: dict[str, float]
    macro_f1: float
    weighted_f1: float
    classification_report_text: str
    confusion_matrix_path: Path
    f1_chart_path: Path


def evaluate_model(
    model_name: str, data: MNISTData | None = None
) -> ModelEvaluation:
    """
    Load a trained model, evaluate it on the test set, and save plots.

    If 'data' is not provided, the function loads MNIST internally.
    When evaluating all models, pass pre-loaded data to avoid redundant
    downloads (see evaluate_all_models).

    Args:
        model_name: Registered model name.
        data: Optional pre-loaded MNISTData. Loaded internally if None.

    Returns:
        ModelEvaluation with all metrics and plot file paths.

    Raises:
        FileNotFoundError: If the trained model file does not exist.
    """
    model_path = MODEL_PATHS[model_name]
    if not model_path.exists():
        raise FileNotFoundError(
            f"No trained model found at '{model_path}'.\n"
            f"Train it first: python -m src.train --model {model_name}"
        )

    logger.info("Evaluating: %s", model_name)

    if data is None:
        data = load_and_prepare_mnist()

    model = tf.keras.models.load_model(model_path)
    test_loss, test_accuracy = model.evaluate(data.X_test, data.y_test, verbose=0)

    logger.info(
        "%s — Test accuracy: %.4f | Test loss: %.4f",
        model_name,
        test_accuracy,
        test_loss,
    )

    probabilities = model.predict(data.X_test, verbose=0)
    predicted_labels = np.argmax(probabilities, axis=1)

    # Generate all interactive-ready artifacts (raw data, sample images, Grad-CAM)
    try:
        generate_all_artifacts(
            model_name=model_name,
            model=model,
            data=data,
            y_predicted=predicted_labels,
            probabilities=probabilities,
        )
    except (OSError, ValueError, RuntimeError) as artifact_error:
        # Artifact generation is non-critical — log and continue.
        # Catching specific types avoids masking programming errors during development.
        logger.warning(
            "Artifact generation partially failed for '%s': %s",
            model_name,
            artifact_error,
        )

    report_text = classification_report(
        data.y_test_labels, predicted_labels, target_names=CLASS_NAMES
    )
    per_class_f1 = _compute_per_class_f1(data.y_test_labels, predicted_labels)
    macro_f1 = float(f1_score(data.y_test_labels, predicted_labels, average="macro"))
    weighted_f1 = float(
        f1_score(data.y_test_labels, predicted_labels, average="weighted")
    )

    PERFORMANCE_PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    cm_path = _save_confusion_matrix_plot(
        data.y_test_labels, predicted_labels, model_name
    )
    f1_path = _save_f1_bar_chart(per_class_f1, model_name)

    return ModelEvaluation(
        model_name=model_name,
        test_accuracy=test_accuracy,
        test_loss=test_loss,
        per_class_f1=per_class_f1,
        macro_f1=macro_f1,
        weighted_f1=weighted_f1,
        classification_report_text=report_text,
        confusion_matrix_path=cm_path,
        f1_chart_path=f1_path,
    )


def evaluate_all_models() -> dict[str, ModelEvaluation]:
    """
    Evaluate all registered models and generate the combined training curves.

    MNIST is loaded once and shared across all evaluations.
    Models that have not been trained are skipped with a warning.

    Returns:
        Dictionary mapping model name → ModelEvaluation.
        Models that were not found are absent from the dictionary.
    """
    data = load_and_prepare_mnist()
    results: dict[str, ModelEvaluation] = {}

    for model_name in list_available_models():
        try:
            results[model_name] = evaluate_model(model_name, data=data)
        except FileNotFoundError as error:
            logger.warning("Skipping '%s': %s", model_name, error)

    if results:
        _save_training_curves_plot(list(results.keys()))

    return results


def load_training_history(model_name: str) -> dict[str, list[float]] | None:
    """
    Load the saved training history for a model from its JSON file.

    Used by the Streamlit comparison and evaluation pages to display
    learning curves without requiring the model to be retrained.

    Args:
        model_name: Registered model name.

    Returns:
        Dictionary with keys 'accuracy', 'val_accuracy', 'loss', 'val_loss'
        (each a list of float values per epoch), or None if the model
        has not been trained yet.
    """
    history_path = HISTORY_PATHS.get(model_name)

    if history_path is None or not history_path.exists():
        return None

    with history_path.open("r", encoding="utf-8") as f:
        return json.load(f)


# ─── Plot generators ────────────────────────────────────────────────────────────


def _save_confusion_matrix_plot(
    true_labels: np.ndarray,
    predicted_labels: np.ndarray,
    model_name: str,
) -> Path:
    """
    Generate and save a normalised confusion matrix heatmap.

    Normalisation (proportions rather than counts) makes the colour scale
    meaningful regardless of class size, and reveals percentage confusion
    between pairs — far more informative than raw counts.
    """
    cm = confusion_matrix(true_labels, predicted_labels, normalize="true")

    with plt.style.context(PLOT_STYLE):
        fig, ax = plt.subplots(figsize=CONFUSION_MATRIX_FIGSIZE)
        sns.heatmap(
            cm,
            annot=True,
            fmt=".2f",
            cmap="Blues",
            xticklabels=CLASS_NAMES,
            yticklabels=CLASS_NAMES,
            linewidths=0.5,
            ax=ax,
        )
        ax.set_xlabel("Predicted Label", fontsize=12, labelpad=10)
        ax.set_ylabel("True Label", fontsize=12, labelpad=10)
        ax.set_title(
            f"Normalised Confusion Matrix — {MODEL_DISPLAY_NAMES.get(model_name, model_name)}",
            fontsize=14,
            pad=15,
        )

        plt.tight_layout()
        save_path = PERFORMANCE_PLOTS_DIR / f"confusion_matrix_{model_name}.png"
        fig.savefig(save_path, dpi=PLOT_DPI, bbox_inches="tight")
        plt.close(fig)

    logger.info("Confusion matrix saved → %s", save_path)
    return save_path


def _save_f1_bar_chart(
    per_class_f1: dict[str, float],
    model_name: str,
) -> Path:
    """
    Generate and save a colour-coded horizontal bar chart of per-class F1 scores.

    Colour coding:
        Green  (≥ 0.990): excellent performance
        Amber  (≥ 0.970): acceptable, room for improvement
        Red    (< 0.970): needs attention

    This makes underperforming digit classes immediately obvious at a glance.
    """
    digits = list(per_class_f1.keys())
    scores = list(per_class_f1.values())
    bar_colours = [
        "#4ade80" if s >= 0.990 else "#fbbf24" if s >= 0.970 else "#f87171"
        for s in scores
    ]

    with plt.style.context(PLOT_STYLE):
        fig, ax = plt.subplots(figsize=(10, 6))

        bars = ax.barh(digits, scores, color=bar_colours, edgecolor="white", linewidth=0.5)
        ax.set_xlim(0.93, 1.005)
        ax.set_xlabel("F1 Score", fontsize=12)
        ax.set_ylabel("Digit Class", fontsize=12)
        ax.set_title(
            f"Per-Class F1 Score — {MODEL_DISPLAY_NAMES.get(model_name, model_name)}",
            fontsize=14,
            pad=15,
        )

        for bar, score in zip(bars, scores):
            ax.text(
                bar.get_width() + 0.001,
                bar.get_y() + bar.get_height() / 2,
                f"{score:.4f}",
                va="center",
                fontsize=10,
            )

        plt.tight_layout()
        save_path = PERFORMANCE_PLOTS_DIR / f"f1_per_class_{model_name}.png"
        fig.savefig(save_path, dpi=PLOT_DPI, bbox_inches="tight")
        plt.close(fig)

    logger.info("F1 chart saved → %s", save_path)
    return save_path


def _save_training_curves_plot(model_names: list[str]) -> None:
    """
    Generate and save a side-by-side validation accuracy and loss plot.

    Overlaying all three models on the same axes makes convergence speed
    and final performance directly comparable — a natural centrepiece for
    the Model Comparison page.
    """
    # Assign distinct colours so the three lines are immediately distinguishable
    line_colours = {
        "dense_nn":   "#60a5fa",  # blue
        "lenet5":     "#f59e0b",  # amber
        "custom_cnn": "#34d399",  # green
    }
    display_labels = {
        "dense_nn":   "Dense NN",
        "lenet5":     "LeNet-5",
        "custom_cnn": "Custom CNN",
    }

    with plt.style.context(PLOT_STYLE):
        fig, (ax_accuracy, ax_loss) = plt.subplots(1, 2, figsize=CURVES_FIGSIZE)

        for model_name in model_names:
            history = load_training_history(model_name)
            if history is None:
                logger.warning(
                    "No history found for '%s' — skipping from curves plot.", model_name
                )
                continue

            colour = line_colours.get(model_name, "#ffffff")
            label = display_labels.get(model_name, model_name)
            epochs = range(1, len(history["accuracy"]) + 1)

            ax_accuracy.plot(
                epochs, history["val_accuracy"], label=label, color=colour, linewidth=2
            )
            ax_loss.plot(
                epochs, history["val_loss"], label=label, color=colour, linewidth=2
            )

        ax_accuracy.set_title("Validation Accuracy", fontsize=14)
        ax_accuracy.set_xlabel("Epoch", fontsize=11)
        ax_accuracy.set_ylabel("Accuracy", fontsize=11)
        ax_accuracy.legend(fontsize=10)
        ax_accuracy.grid(alpha=0.2)

        ax_loss.set_title("Validation Loss", fontsize=14)
        ax_loss.set_xlabel("Epoch", fontsize=11)
        ax_loss.set_ylabel("Loss", fontsize=11)
        ax_loss.legend(fontsize=10)
        ax_loss.grid(alpha=0.2)

        fig.suptitle("All Models — Validation Metrics", fontsize=16, y=1.02)
        plt.tight_layout()

        save_path = PERFORMANCE_PLOTS_DIR / "training_curves.png"
        fig.savefig(save_path, dpi=PLOT_DPI, bbox_inches="tight")
        plt.close(fig)

    logger.info("Training curves saved → %s", save_path)


# ─── Private helpers ────────────────────────────────────────────────────────────


def _compute_per_class_f1(
    true_labels: np.ndarray,
    predicted_labels: np.ndarray,
) -> dict[str, float]:
    """Return a dict mapping digit label → F1 score for each of the 10 classes."""
    scores = f1_score(
        true_labels,
        predicted_labels,
        average=None,
        labels=list(range(10)),
    )
    return {digit: float(score) for digit, score in zip(CLASS_NAMES, scores)}



# ─── CLI entry point ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate DigitVision models and generate performance plots."
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--all", action="store_true", help="Evaluate all models.")
    group.add_argument(
        "--model",
        choices=list_available_models(),
        metavar="MODEL_NAME",
        help=f"Evaluate a single model. Choices: {list_available_models()}",
    )

    args = parser.parse_args()

    if args.all:
        evaluate_all_models()
    else:
        evaluate_model(args.model)

    sys.exit(0)
