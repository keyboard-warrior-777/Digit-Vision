"""
Grad-CAM (Gradient-weighted Class Activation Mapping) for DigitVision.

Grad-CAM produces visual explanations of what a CNN "looked at" when
making a prediction. It answers: "Which spatial regions of the input
image most influenced this prediction?"

How it works (the intuition):
    1. Run a forward pass through the model up to the final Conv2D layer.
    2. Record the feature map activations at that layer.
    3. Compute gradients of the predicted class score with respect to
       those activations — this tells us how much each activation
       contributed to the final prediction.
    4. Average the gradients across spatial dimensions to get one
       importance weight per feature map (filter).
    5. Take a weighted sum of the feature maps using these importance weights.
    6. Apply ReLU — keeping only features that positively influenced
       the prediction (not features that suppressed it).
    7. Resize to the input image dimensions and normalise to [0, 1].

Why include Grad-CAM in a student portfolio?
    It demonstrates awareness of model interpretability — a topic that
    is increasingly critical in real-world ML deployment. It also proves
    the model is attending to the actual digit structure rather than
    background noise.

Limitation — Dense NN:
    Grad-CAM requires at least one convolutional layer. The Dense NN
    has none, so compute_gradcam() returns None for that model. The
    Streamlit UI handles this case explicitly.

Reference:
    Selvaraju, R.R. et al. "Grad-CAM: Visual Explanations from Deep
    Networks via Gradient-based Localization." ICCV 2017.
    https://arxiv.org/abs/1610.02391
"""

from __future__ import annotations

from typing import Optional

import cv2
import numpy as np
import tensorflow as tf

from src.logger import get_logger

logger = get_logger(__name__)


def compute_gradcam(
    model: tf.keras.Model,
    preprocessed_image: np.ndarray,
    predicted_class_index: int,
) -> Optional[np.ndarray]:
    """
    Compute a Grad-CAM heatmap for the given model, image, and prediction.

    The heatmap highlights which 28×28 pixel regions most influenced the
    model's prediction for the given class.

    Args:
        model: A trained Keras model (any of the three DigitVision architectures).
        preprocessed_image: Model-ready array of shape (1, 28, 28, 1), float32.
        predicted_class_index: The class index to explain. Typically the argmax
            of the model's output probabilities (the predicted digit).

    Returns:
        A normalised heatmap, shape (28, 28), values in [0.0, 1.0].
        Returns None if the model contains no Conv2D layers.
    """
    last_conv_layer = _find_last_conv_layer(model)

    if last_conv_layer is None:
        logger.info(
            "Model '%s' has no Conv2D layers — Grad-CAM is not applicable.",
            model.name,
        )
        return None

    gradient_model = _build_gradient_model(model, last_conv_layer)
    heatmap = _compute_heatmap(
        gradient_model, preprocessed_image, predicted_class_index
    )

    return heatmap


def overlay_heatmap_on_image(
    heatmap: np.ndarray,
    preprocessed_image: np.ndarray,
    alpha: float = 0.4,
) -> np.ndarray:
    """
    Blend a Grad-CAM heatmap onto the input image for visual display.

    The heatmap is colourised with the 'jet' colourmap (blue → green → red),
    then blended with the grayscale digit image. High-activation regions
    appear red/orange; low-activation regions appear blue.

    Args:
        heatmap: Normalised Grad-CAM output, shape (28, 28), values in [0, 1].
        preprocessed_image: Model input, shape (1, 28, 28, 1), float32.
        alpha: Heatmap opacity in the blend. 0.0 = image only, 1.0 = heatmap only.
            Default 0.4 gives a clear overlay without obscuring the digit.

    Returns:
        RGB overlay image, shape (28, 28, 3), dtype uint8.
    """
    # Scale the normalised input back to [0, 255] for display
    gray_uint8 = (preprocessed_image.squeeze() * 255).astype(np.uint8)

    # Convert to 3-channel so it can be blended with the colour heatmap
    gray_rgb = cv2.cvtColor(gray_uint8, cv2.COLOR_GRAY2RGB)

    # Apply jet colourmap to the heatmap (output is BGR — convert to RGB)
    heatmap_uint8 = (heatmap * 255).astype(np.uint8)
    heatmap_coloured_bgr = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    heatmap_coloured_rgb = cv2.cvtColor(heatmap_coloured_bgr, cv2.COLOR_BGR2RGB)

    # Weighted blend: (1 - alpha) × image + alpha × heatmap
    overlay = cv2.addWeighted(gray_rgb, 1 - alpha, heatmap_coloured_rgb, alpha, 0)
    return overlay


# ─── Private implementation ────────────────────────────────────────────────────


def _find_last_conv_layer(
    model: tf.keras.Model,
) -> Optional[tf.keras.layers.Layer]:
    """
    Find the last Conv2D layer in the model by searching backwards.

    The last convolutional layer is the target for Grad-CAM because it
    produces the most semantically rich feature maps. Earlier layers
    detect low-level features (edges, textures); the last conv layer
    captures high-level patterns most relevant to the final prediction.

    Returns:
        The last Conv2D layer, or None if no Conv2D layers exist.
    """
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            logger.debug("Grad-CAM targeting layer: '%s'", layer.name)
            return layer

    return None


def _build_gradient_model(
    model: tf.keras.Model,
    conv_layer: tf.keras.layers.Layer,
) -> tf.keras.Model:
    """
    Build a sub-model with two outputs: conv activations and final predictions.

    We need simultaneous access to the conv layer's activations and the
    model's predictions to compute the gradient of the prediction with
    respect to the activations — which is exactly what tf.GradientTape needs.
    """
    return tf.keras.Model(
        inputs=model.inputs,
        outputs=[conv_layer.output, model.output],
        name="grad_cam_gradient_model",
    )


def _compute_heatmap(
    gradient_model: tf.keras.Model,
    preprocessed_image: np.ndarray,
    class_index: int,
) -> np.ndarray:
    """
    Execute the Grad-CAM gradient computation and produce the final heatmap.

    Step-by-step:
        1. Run the image through the gradient model inside a GradientTape
           to record all operations for automatic differentiation.
        2. Extract the predicted class score (a scalar) — the quantity
           whose gradients we want.
        3. Compute ∂(class_score) / ∂(conv_activations) — the gradient
           of the class score with respect to each activation.
        4. Global average pool the gradients spatially → one weight per filter.
        5. Weight each filter's feature map by its importance weight.
        6. Sum the weighted feature maps → raw heatmap.
        7. Apply ReLU to keep only positively contributing regions.
        8. Resize to 28×28 and normalise to [0, 1].
    """
    image_tensor = tf.cast(preprocessed_image, dtype=tf.float32)

    with tf.GradientTape() as tape:
        tape.watch(image_tensor)
        conv_outputs, predictions = gradient_model(image_tensor, training=False)
        # Scalar class score for the target class
        class_score = predictions[:, class_index]

    # Gradients shape: (1, H, W, num_filters)
    gradients = tape.gradient(class_score, conv_outputs)

    # Average gradients over spatial dimensions → shape: (num_filters,)
    # Each value is the average importance of that filter for this class.
    filter_importance_weights = tf.reduce_mean(gradients, axis=(0, 1, 2))

    # Weight each feature map and sum → raw heatmap, shape (H, W)
    conv_outputs = conv_outputs[0]  # Remove batch dimension
    weighted_activation_sum = tf.reduce_sum(
        conv_outputs * filter_importance_weights, axis=-1
    )
    heatmap = weighted_activation_sum.numpy()

    # ReLU: discard regions that decrease the class score.
    # We only want to highlight what the model found supportive.
    heatmap = np.maximum(heatmap, 0)

    # Resize to match the input image (28×28)
    heatmap = cv2.resize(heatmap, (28, 28))

    # Normalise to [0, 1] for colourmap application
    max_value = heatmap.max()
    if max_value > 0:
        heatmap = heatmap / max_value

    return heatmap
