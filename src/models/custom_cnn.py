"""
Custom CNN architecture for DigitVision.

This is the headline model — designed using techniques that have emerged
from research published after LeNet-5, and that are now standard in
all serious convolutional architectures.

Each architectural choice is deliberate and explainable:

1. Two convolutional blocks, doubling filter count (32 → 64).
   Early layers detect simple features (edges, curves). Deeper layers
   recombine these into complex patterns (digit strokes, loops). More
   filters in deeper layers accommodate this growing feature complexity.

2. BatchNormalization after every Conv2D.
   Normalises activations within each mini-batch, which:
   — Allows higher learning rates (faster convergence)
   — Reduces sensitivity to weight initialisation
   — Acts as a mild regulariser, reducing the need for large Dropout rates
   Note: 'use_bias=False' in Conv2D when followed by BatchNorm, because
   BatchNorm's beta parameter already acts as a learned bias.

3. MaxPooling instead of AveragePooling.
   MaxPool retains the strongest feature activations, making it more
   robust to small translations than AveragePooling.

4. GlobalAveragePooling2D instead of Flatten.
   After the final conv block, GAP averages each of the 64 feature maps
   to a single value, producing a (64,) vector. Compared to Flatten:
   — Dramatically fewer parameters (~64 vs ~1600 → Dense connection)
   — Acts as structural regularisation (harder to memorise training data)
   — Used in ResNet, EfficientNet, MobileNet — it is the modern standard

5. Dropout(0.25) after conv blocks, Dropout(0.5) after dense.
   The higher rate after the dense layer reflects its greater capacity
   to overfit compared to the spatially-shared conv filters.

Architecture:
    Input (28, 28, 1)
    Conv2D(32) → BN → ReLU → Conv2D(32) → BN → ReLU → MaxPool → Dropout(0.25)
    Conv2D(64) → BN → ReLU → Conv2D(64) → BN → ReLU → MaxPool → Dropout(0.25)
    GlobalAveragePooling2D
    Dense(128) → BN → ReLU → Dropout(0.5)
    Dense(10)  → Softmax

Parameters: ~75,000
Expected accuracy: ~99.3%

Interview Note:
    "I used the Functional API instead of Sequential because it's how
    all production models are written — ResNet, EfficientNet, transformers.
    It also makes the data flow completely explicit, which makes the
    architecture easier to review and modify."
"""

import tensorflow as tf

from config.config import INPUT_SHAPE, NUM_CLASSES
from src.logger import get_logger

logger = get_logger(__name__)


def build_custom_cnn(learning_rate: float = 1e-3) -> tf.keras.Model:
    """
    Build and compile the Custom CNN using the Keras Functional API.

    The Functional API makes the data flow explicit: each layer receives
    a tensor and returns a tensor. This is how production models are
    built — it supports branching, skip connections, and multi-input/
    output architectures that Sequential cannot express.

    Args:
        learning_rate: Learning rate for the Adam optimiser.

    Returns:
        A compiled tf.keras.Model, ready for training with .fit().
    """
    inputs = tf.keras.layers.Input(shape=INPUT_SHAPE, name="input")

    # ── Convolutional Block 1 — 32 filters ────────────────────────────────────
    # Two consecutive conv layers before pooling allows the network to compose
    # more complex features than a single layer would permit.
    x = _conv_bn_relu_block(inputs, filters=32, name_prefix="block1_conv1")
    x = _conv_bn_relu_block(x,      filters=32, name_prefix="block1_conv2")
    x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), name="block1_pool")(x)
    x = tf.keras.layers.Dropout(0.25, name="block1_dropout")(x)

    # ── Convolutional Block 2 — 64 filters ────────────────────────────────────
    # Doubling the filter count captures more complex spatial patterns.
    # After two MaxPool operations, spatial dimensions are 7×7.
    x = _conv_bn_relu_block(x, filters=64, name_prefix="block2_conv1")
    x = _conv_bn_relu_block(x, filters=64, name_prefix="block2_conv2")
    x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), name="block2_pool")(x)
    x = tf.keras.layers.Dropout(0.25, name="block2_dropout")(x)

    # ── Classifier Head ───────────────────────────────────────────────────────
    # GlobalAveragePooling replaces Flatten. Each of the 64 feature maps
    # (7×7 spatial) is reduced to its average, producing a (64,) vector.
    # This is parameter-efficient and inherently regularising.
    x = tf.keras.layers.GlobalAveragePooling2D(name="global_avg_pool")(x)

    x = tf.keras.layers.Dense(128, use_bias=False, name="dense_1")(x)
    x = tf.keras.layers.BatchNormalization(name="dense_bn")(x)
    x = tf.keras.layers.Activation("relu", name="dense_relu")(x)
    x = tf.keras.layers.Dropout(0.5, name="dense_dropout")(x)

    outputs = tf.keras.layers.Dense(
        NUM_CLASSES, activation="softmax", name="output"
    )(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="custom_cnn")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    logger.info(
        "Custom CNN built — %s trainable parameters.",
        f"{model.count_params():,}",
    )

    return model


# ─── Private builder helper ────────────────────────────────────────────────────


def _conv_bn_relu_block(
    x: tf.Tensor,
    filters: int,
    kernel_size: tuple[int, int] = (3, 3),
    name_prefix: str = "conv",
) -> tf.Tensor:
    """
    Apply Conv2D → BatchNormalization → ReLU activation in sequence.

    This ordering (Conv → BN → ReLU) is the most widely adopted in modern
    CNNs. An alternative (BN before Conv) exists but is less standard
    and harder to explain without deeper theoretical background.

    'use_bias=False' in Conv2D: when BatchNorm follows, its learned
    'beta' parameter subsumes the bias. Keeping Conv's bias would add
    parameters that have no effect, wasting capacity.

    'padding=same': output spatial dimensions match input spatial
    dimensions. This makes the architecture easier to reason about —
    size reduction only happens at the MaxPool layers.

    Args:
        x: Input tensor.
        filters: Number of convolution filters.
        kernel_size: Height and width of each filter. Default 3×3 is
            the most common choice in modern CNNs (smaller than LeNet's 5×5).
        name_prefix: Prefix applied to all layer names for readability
            in model.summary() output.

    Returns:
        Output tensor after Conv → BN → ReLU.
    """
    x = tf.keras.layers.Conv2D(
        filters=filters,
        kernel_size=kernel_size,
        padding="same",
        use_bias=False,  # Redundant when BatchNorm follows — saves parameters
        name=f"{name_prefix}_conv",
    )(x)
    x = tf.keras.layers.BatchNormalization(name=f"{name_prefix}_bn")(x)
    x = tf.keras.layers.Activation("relu", name=f"{name_prefix}_relu")(x)
    return x
