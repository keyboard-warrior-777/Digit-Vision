"""
Training engine for DigitVision.

Trains one or all three registered models and saves:
    - The trained model weights in .keras format (native Keras format,
      portable across TensorFlow versions)
    - The training history as JSON (accuracy/loss per epoch, used by
      the Streamlit UI to display learning curves without retraining)

This module can be used in two ways:

1. As a command-line script:
       python -m src.train --all
       python -m src.train --model custom_cnn

2. Programmatically from tests or other modules:
       from src.train import train_model
       from src.dataset import load_and_prepare_mnist

       data = load_and_prepare_mnist()
       result = train_model("custom_cnn", data)
       print(result.final_val_accuracy)

Design — Data loading separation:
    train_model() accepts a pre-loaded MNISTData object rather than
    loading data internally. This means when training all three models,
    the dataset is downloaded and processed exactly once, not three times.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import tensorflow as tf

from config.config import (
    HISTORY_PATHS,
    MODEL_PATHS,
    MODEL_TRAINING_CONFIG,
    MODELS_DIR,
    RANDOM_SEED,
)
from src.dataset import MNISTData, create_augmented_generator, load_and_prepare_mnist
from src.logger import get_logger
from src.models import build_model, list_available_models

logger = get_logger(__name__)


@dataclass(frozen=True)
class TrainingResult:
    """
    Immutable summary of a completed training run.

    Returned by train_model() so callers can inspect results without
    parsing the Keras History object or reading files from disk.
    """

    model_name: str
    final_train_accuracy: float
    final_val_accuracy: float
    epochs_trained: int
    model_path: Path
    history_path: Path


def train_model(model_name: str, data: MNISTData) -> TrainingResult:
    """
    Train a single model on MNIST and save weights and history.

    Uses three training callbacks (see _build_training_callbacks) to
    prevent overfitting, avoid wasted compute, and save only the
    best checkpoint.

    Args:
        model_name: Registered model name ('dense_nn', 'lenet5', 'custom_cnn').
        data: Pre-loaded and preprocessed MNISTData. The caller is
            responsible for loading data — this avoids redundant downloads
            when training multiple models in sequence.

    Returns:
        TrainingResult with final metrics and paths to saved files.

    Raises:
        ValueError: If model_name is not a registered model.
    """
    config = MODEL_TRAINING_CONFIG[model_name]

    logger.info("=" * 60)
    logger.info("Training: %s", model_name.upper())
    logger.info(
        "Epochs: %d | Batch size: %d | Learning rate: %.0e",
        config["epochs"],
        config["batch_size"],
        config["learning_rate"],
    )
    logger.info("=" * 60)

    tf.random.set_seed(RANDOM_SEED)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    model = build_model(model_name)
    model.summary(print_fn=lambda line: logger.debug(line))

    train_generator = create_augmented_generator(
        data.X_train, data.y_train, batch_size=config["batch_size"]
    )
    steps_per_epoch = len(data.X_train) // config["batch_size"]

    history = model.fit(
        train_generator,
        steps_per_epoch=steps_per_epoch,
        epochs=config["epochs"],
        validation_data=(data.X_val, data.y_val),
        callbacks=_build_training_callbacks(model_name),
        verbose=1,
    )

    _save_model(model, model_name)
    _save_training_history(history, model_name)

    final_train_accuracy = float(history.history["accuracy"][-1])
    final_val_accuracy = float(history.history["val_accuracy"][-1])

    logger.info(
        "%s complete — Train acc: %.4f | Val acc: %.4f | Epochs: %d",
        model_name,
        final_train_accuracy,
        final_val_accuracy,
        len(history.history["accuracy"]),
    )

    return TrainingResult(
        model_name=model_name,
        final_train_accuracy=final_train_accuracy,
        final_val_accuracy=final_val_accuracy,
        epochs_trained=len(history.history["accuracy"]),
        model_path=MODEL_PATHS[model_name],
        history_path=HISTORY_PATHS[model_name],
    )


def train_all_models(data: MNISTData) -> dict[str, TrainingResult]:
    """
    Train all three registered models sequentially, sharing one dataset load.

    Trains in registry order: Dense NN → LeNet-5 → Custom CNN.
    This order mirrors increasing architectural complexity, making any
    training log naturally readable as a progression.

    Args:
        data: Pre-loaded MNISTData shared across all training runs.

    Returns:
        Dictionary mapping model name → TrainingResult for each model.
    """
    results: dict[str, TrainingResult] = {}

    for model_name in list_available_models():
        results[model_name] = train_model(model_name, data)

    _log_training_summary(results)
    return results


# ─── Private helpers ────────────────────────────────────────────────────────────


def _build_training_callbacks(model_name: str) -> list[tf.keras.callbacks.Callback]:
    """
    Build the standard set of training callbacks for a given model.

    Three callbacks are used:

    ModelCheckpoint:
        Saves the model weights only when val_accuracy improves.
        This means the final saved model is always the best checkpoint,
        not necessarily the last epoch — important when training curves
        show a late dip in validation accuracy.

    EarlyStopping:
        Halts training when val_loss stops improving for 5 consecutive
        epochs. Avoids wasted compute and prevents overfitting past the
        optimal point. restore_best_weights=True ensures we end up with
        the best weights, not the last epoch's weights.

    ReduceLROnPlateau:
        Halves the learning rate when val_loss plateaus for 3 epochs.
        This often unlocks a final accuracy boost by allowing finer
        gradient steps near a local optimum.

    Args:
        model_name: Used to determine the model checkpoint save path.

    Returns:
        List of configured Keras callback instances.
    """
    return [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(MODEL_PATHS[model_name]),
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            min_lr=1e-6,
            verbose=1,
        ),
    ]


def _save_model(model: tf.keras.Model, model_name: str) -> None:
    """Save model weights in the native Keras format (.keras)."""
    save_path = MODEL_PATHS[model_name]
    model.save(save_path)
    logger.info("Model saved → %s", save_path)


def _save_training_history(
    history: tf.keras.callbacks.History, model_name: str
) -> None:
    """
    Persist training history (accuracy/loss curves) to a JSON file.

    The Streamlit UI reads this file to render learning curves without
    requiring the model to be retrained. JSON is preferred over pickle
    because it is human-readable and framework-agnostic — the file
    can be opened in any text editor or consumed by any language.

    Numpy float32 values from the history dict must be cast to Python
    float before JSON serialisation, as json.dump does not handle
    numpy types natively.
    """
    serialisable = {
        metric: [float(value) for value in values]
        for metric, values in history.history.items()
    }

    history_path = HISTORY_PATHS[model_name]
    history_path.parent.mkdir(parents=True, exist_ok=True)

    with history_path.open("w", encoding="utf-8") as f:
        json.dump(serialisable, f, indent=2)

    logger.info("Training history saved → %s", history_path)


def _log_training_summary(results: dict[str, TrainingResult]) -> None:
    """Log a formatted summary table of all training results."""
    logger.info("=" * 60)
    logger.info("Training Complete — Results Summary")
    logger.info("%-20s %-15s %-10s", "Model", "Val Accuracy", "Epochs")
    logger.info("-" * 50)
    for name, result in results.items():
        logger.info(
            "%-20s %-15.4f %-10d",
            name,
            result.final_val_accuracy,
            result.epochs_trained,
        )
    logger.info("=" * 60)


# ─── CLI entry point ────────────────────────────────────────────────────────────


def _parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train DigitVision models on MNIST.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Train all three models:
      python -m src.train --all

  Train one model:
      python -m src.train --model custom_cnn
      python -m src.train --model dense_nn
      python -m src.train --model lenet5
        """,
    )

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--all",
        action="store_true",
        help="Train all registered models (dense_nn, lenet5, custom_cnn).",
    )
    group.add_argument(
        "--model",
        choices=list_available_models(),
        metavar="MODEL_NAME",
        help=f"Train a single model. Choices: {list_available_models()}",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_arguments()

    logger.info("Loading MNIST dataset...")
    mnist_data = load_and_prepare_mnist()

    if args.all:
        train_all_models(mnist_data)
    else:
        train_model(args.model, mnist_data)

    sys.exit(0)
