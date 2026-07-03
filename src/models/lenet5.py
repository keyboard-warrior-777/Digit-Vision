"""
LeNet-5 implementation for DigitVision.

LeNet-5 is Yann LeCun's 1998 architecture — the first convolutional
network to achieve practical handwritten digit recognition and one of
the most historically significant neural networks ever published.

This implementation is a faithful modernisation of the original paper:

What is preserved from the 1998 paper:
    - Two convolutional stages, each followed by spatial pooling
    - Filter counts of 6 and 16 (as in the original)
    - 5×5 convolution kernels
    - Fully-connected classifier of sizes 120 → 84 → 10
    - AveragePooling (the original used this, unlike modern MaxPooling)

What is updated for modern practice:
    - ReLU instead of tanh/sigmoid: avoids vanishing gradient and
      trains significantly faster on modern hardware
    - Softmax output instead of Euclidean RBF: standard for multi-class
      classification and compatible with categorical cross-entropy
    - padding='same' on the first conv layer: adapts the original 32×32
      design to MNIST's 28×28 images without manual padding

Why include a 1998 architecture in a 2024 portfolio?
    It gives you a historically grounded comparison point. The leap
    from LeNet-5 to your Custom CNN mirrors 25 years of real research
    progress — BatchNorm, MaxPooling, GlobalAveragePooling — and you
    can explain every improvement in an interview.

Architecture:
    Input (28, 28, 1)
    Conv2D(6,  5×5, same) → ReLU → AveragePooling(2×2)  → (14, 14,  6)
    Conv2D(16, 5×5, valid)→ ReLU → AveragePooling(2×2)  → ( 5,  5, 16)
    Flatten → (400)
    Dense(120) → ReLU
    Dense(84)  → ReLU
    Dense(10)  → Softmax

Parameters: ~61,000
Expected accuracy: ~98.5%

Reference:
    LeCun, Y. et al. "Gradient-Based Learning Applied to Document
    Recognition." Proceedings of the IEEE, 1998.
    http://yann.lecun.com/exdb/publis/pdf/lecun-01a.pdf
"""

import tensorflow as tf

from config.config import INPUT_SHAPE, NUM_CLASSES
from src.logger import get_logger

logger = get_logger(__name__)


def build_lenet5(learning_rate: float = 1e-3) -> tf.keras.Model:
    """
    Build and compile a modernised LeNet-5 architecture.

    Sequential API is appropriate here: LeNet-5 is a strictly linear
    pipeline. Using Functional API would add complexity without benefit.

    Args:
        learning_rate: Learning rate for the Adam optimiser.

    Returns:
        A compiled tf.keras.Model, ready for training with .fit().
    """
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=INPUT_SHAPE, name="input"),
            # ── Stage 1 ──────────────────────────────────────────────────────
            # padding='same' keeps output at 28×28 despite the 5×5 kernel.
            # The original LeNet used 32×32 input; this adaptation makes it
            # compatible with MNIST's 28×28 without adding a padding layer.
            tf.keras.layers.Conv2D(
                filters=6,
                kernel_size=(5, 5),
                padding="same",
                activation="relu",
                name="conv_1",
            ),
            # AveragePooling matches the 1998 original. MaxPooling would
            # perform slightly better, but historical fidelity is the point.
            tf.keras.layers.AveragePooling2D(pool_size=(2, 2), name="pool_1"),
            # ── Stage 2 ──────────────────────────────────────────────────────
            # padding='valid' (no padding) matches the original paper.
            # After pooling: 14×14 → (14-5+1)=10 → 10×10×16.
            # After second pooling: 5×5×16 = 400 features.
            tf.keras.layers.Conv2D(
                filters=16,
                kernel_size=(5, 5),
                padding="valid",
                activation="relu",
                name="conv_2",
            ),
            tf.keras.layers.AveragePooling2D(pool_size=(2, 2), name="pool_2"),
            # ── Classifier ───────────────────────────────────────────────────
            tf.keras.layers.Flatten(name="flatten"),
            tf.keras.layers.Dense(120, activation="relu", name="dense_1"),
            tf.keras.layers.Dense(84, activation="relu", name="dense_2"),
            tf.keras.layers.Dense(NUM_CLASSES, activation="softmax", name="output"),
        ],
        name="lenet5",
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    logger.info(
        "LeNet-5 built — %s trainable parameters.",
        f"{model.count_params():,}",
    )

    return model
