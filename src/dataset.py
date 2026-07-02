"""
MNIST dataset loading and preparation for DigitVision.

Responsibilities:
    - Download and cache MNIST via tf.keras.datasets
    - Normalize pixel values to [0.0, 1.0]
    - Reshape to (N, 28, 28, 1) — the channel dimension Conv2D requires
    - One-hot encode integer labels for categorical cross-entropy
    - Carve a reproducible validation split from the training set
    - Provide an augmented data generator for training

Design — Separation of Concerns:
    Data preparation is entirely decoupled from model code. You can
    swap datasets, change augmentation, or adjust the split strategy
    here without touching a single model file.

Interview Note:
    "I used a frozen dataclass to hold the dataset splits so access is
    named (data.X_train) rather than positional (data[0]). Frozen means
    no code can accidentally mutate the dataset after it's been prepared."
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import tensorflow as tf

from config.config import (
    AUGMENTATION_CONFIG,
    INPUT_SHAPE,
    NUM_CLASSES,
    RANDOM_SEED,
    VALIDATION_SPLIT,
)
from src.logger import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class MNISTData:
    """
    Immutable container for all MNIST dataset splits.

    All arrays are ready for direct use by Keras: normalized,
    reshaped, and encoded. No further transformation needed.

    Attributes:
        X_train:      Training images,   shape (N_train, 28, 28, 1), float32 in [0, 1].
        y_train:      Training labels,   shape (N_train, 10), one-hot encoded.
        X_val:        Validation images, shape (N_val,   28, 28, 1), float32 in [0, 1].
        y_val:        Validation labels, shape (N_val,   10), one-hot encoded.
        X_test:       Test images,       shape (10000,   28, 28, 1), float32 in [0, 1].
        y_test:       Test labels,       shape (10000,   10), one-hot encoded.
        y_test_labels: Raw integer labels, shape (10000,).
                       Kept alongside one-hot encoding because sklearn metrics
                       (confusion matrix, classification report) require integers.
    """

    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    y_test_labels: np.ndarray


def load_and_prepare_mnist() -> MNISTData:
    """
    Download MNIST, apply all preprocessing, and return the dataset splits.

    MNIST is fetched from tf.keras.datasets and cached locally after the
    first download (~11 MB). Subsequent calls use the cached version.

    Preprocessing pipeline:
        1. Normalize uint8 pixels [0, 255] → float32 [0.0, 1.0]
           Neural networks train faster on normalised input — gradients
           stay in a consistent scale throughout the network.

        2. Reshape (N, 28, 28) → (N, 28, 28, 1)
           TensorFlow's Conv2D expects a channel dimension. MNIST is
           grayscale, so we add a single channel.

        3. One-hot encode integer labels → class probability vectors
           Required by categorical_crossentropy loss. The label '7'
           becomes [0, 0, 0, 0, 0, 0, 0, 1, 0, 0].

        4. Carve a validation split from the training set
           10% of training data is held out for validation. The split
           is seeded for reproducibility.

    Returns:
        MNISTData with all splits ready for training and evaluation.
    """
    logger.info("Loading MNIST dataset from tf.keras.datasets...")

    (X_train_raw, y_train_raw), (X_test_raw, y_test_raw) = (
        tf.keras.datasets.mnist.load_data()
    )

    logger.info(
        "MNIST loaded — %d training samples, %d test samples.",
        len(X_train_raw),
        len(X_test_raw),
    )

    X_train_processed = _normalize_and_reshape(X_train_raw)
    X_test_processed = _normalize_and_reshape(X_test_raw)

    X_train_split, X_val_split, y_train_split, y_val_split = _split_validation(
        X_train_processed, y_train_raw
    )

    logger.info(
        "Dataset ready — Train: %d | Val: %d | Test: %d",
        len(X_train_split),
        len(X_val_split),
        len(X_test_processed),
    )

    return MNISTData(
        X_train=X_train_split,
        y_train=_one_hot_encode(y_train_split),
        X_val=X_val_split,
        y_val=_one_hot_encode(y_val_split),
        X_test=X_test_processed,
        y_test=_one_hot_encode(y_test_raw),
        y_test_labels=y_test_raw,
    )


def create_augmented_generator(
    X: np.ndarray,
    y: np.ndarray,
    batch_size: int,
) -> tf.keras.preprocessing.image.NumpyArrayIterator:
    """
    Wrap training data in an ImageDataGenerator that applies augmentation.

    Augmentation is applied only during training — the validation and test
    sets are always evaluated without augmentation. This ensures honest
    evaluation metrics.

    Why augment?
        MNIST images are clean and centred. Real handwritten digits on a
        canvas are slightly rotated, shifted, or scaled. Augmentation
        bridges this gap, making the model more robust to how users
        actually draw.

    Augmentations applied (see config.py for values):
        - Rotation:      handles tilted handwriting
        - Zoom:          handles digits drawn at different sizes
        - Width shift:   handles off-centre drawing (horizontal)
        - Height shift:  handles off-centre drawing (vertical)

    Args:
        X: Training images, shape (N, 28, 28, 1), normalized float32.
        y: One-hot encoded labels, shape (N, 10).
        batch_size: Number of samples per batch.

    Returns:
        A fitted NumpyArrayIterator yielding augmented (X, y) batches.
    """
    generator = tf.keras.preprocessing.image.ImageDataGenerator(
        **AUGMENTATION_CONFIG
    )
    generator.fit(X, seed=RANDOM_SEED)

    return generator.flow(X, y, batch_size=batch_size, seed=RANDOM_SEED)


# ─── Private helpers ───────────────────────────────────────────────────────────
# These functions are implementation details of this module.
# The underscore prefix communicates: "don't call these from outside."


def _normalize_and_reshape(images: np.ndarray) -> np.ndarray:
    """
    Scale pixel values to [0, 1] and add the channel dimension.

    Dividing by 255.0 maps the uint8 range [0, 255] to float32 [0.0, 1.0].
    The reshape adds the channel dimension Conv2D requires: (N, 28, 28, 1).
    """
    normalized = images.astype(np.float32) / 255.0
    return normalized.reshape(-1, *INPUT_SHAPE)


def _split_validation(
    X: np.ndarray,
    y: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Carve a reproducible validation split from the training set.

    A fixed random seed means the split is identical across every run.
    This is essential for reproducibility: training the model twice must
    produce the same train/val split.

    Returns:
        (X_train, X_val, y_train, y_val)
    """
    rng = np.random.default_rng(seed=RANDOM_SEED)
    n_total = len(X)
    n_val = int(n_total * VALIDATION_SPLIT)

    shuffled_indices = rng.permutation(n_total)
    val_indices = shuffled_indices[:n_val]
    train_indices = shuffled_indices[n_val:]

    return X[train_indices], X[val_indices], y[train_indices], y[val_indices]


def _one_hot_encode(labels: np.ndarray) -> np.ndarray:
    """Convert integer class labels to one-hot encoded vectors."""
    return tf.keras.utils.to_categorical(labels, num_classes=NUM_CLASSES)
