"""
Dense Neural Network baseline for DigitVision.

This model is deliberately simple. Its purpose is to establish a
performance floor and to demonstrate, by contrast, why convolutional
networks exist.

A dense network treats every pixel as a completely independent feature.
When you flatten a 28×28 image, you lose all spatial structure —
the relationship between adjacent pixels is destroyed. A '2' shifted
three pixels to the right looks like an entirely different input to
this model.

Despite this fundamental limitation, it achieves ~97.5% accuracy on
MNIST — a reminder that the dataset itself is relatively easy. The
Custom CNN's improvement to ~99.3% with fewer parameters illustrates
the efficiency gained by respecting image structure.

Architecture:
    Input (28, 28, 1)
    Flatten → (784,)
    Dense(512) + ReLU + Dropout(0.3)
    Dense(256) + ReLU + Dropout(0.3)
    Dense(128) + ReLU + Dropout(0.2)
    Dense(10)  + Softmax

Parameters: ~530,000
Expected accuracy: ~97.5%

Interview Note:
    "The Dense NN is my intentional baseline. It gets 97.5% accuracy
    with 530K parameters. My Custom CNN gets 99.3% with 75K parameters.
    That's 7× fewer parameters for better performance — because
    convolutions share weights across the entire image."
"""

import tensorflow as tf

from config.config import INPUT_SHAPE, NUM_CLASSES
from src.logger import get_logger

logger = get_logger(__name__)


def build_dense_nn(learning_rate: float = 1e-3) -> tf.keras.Model:
    """
    Build and compile the Dense Neural Network baseline.

    The Sequential API is appropriate here: the architecture is a
    straight pipeline with no branches, skip connections, or shared layers.
    Sequential reads like a recipe — each step flows to the next.

    Args:
        learning_rate: Learning rate for the Adam optimiser.
            The default of 1e-3 is the standard starting point for Adam.

    Returns:
        A compiled tf.keras.Model, ready for training with .fit().
    """
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=INPUT_SHAPE, name="input"),
            # Flatten the (28, 28, 1) image into a 784-element vector.
            # All spatial relationships are discarded from this point forward.
            tf.keras.layers.Flatten(name="flatten"),
            # Three dense blocks: each halves the width, concentrating features.
            # Dropout after each block prevents co-adaptation between neurons
            # and reduces overfitting.
            tf.keras.layers.Dense(512, activation="relu", name="dense_1"),
            tf.keras.layers.Dropout(0.3, name="dropout_1"),
            tf.keras.layers.Dense(256, activation="relu", name="dense_2"),
            tf.keras.layers.Dropout(0.3, name="dropout_2"),
            tf.keras.layers.Dense(128, activation="relu", name="dense_3"),
            tf.keras.layers.Dropout(0.2, name="dropout_3"),
            # Softmax converts raw scores into probabilities that sum to 1.
            # Required for categorical cross-entropy loss.
            tf.keras.layers.Dense(NUM_CLASSES, activation="softmax", name="output"),
        ],
        name="dense_nn",
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    logger.info(
        "Dense NN built — %s trainable parameters.",
        f"{model.count_params():,}",
    )

    return model
