"""
Model registry for DigitVision.

Provides a single, named entry point for building any supported model.
All other code (training, evaluation, the Streamlit app) calls
build_model(name) — they never import individual model modules directly.

Registry pattern benefits:
    - Adding a new architecture requires only two lines here
    - Training and evaluation scripts need no changes
    - The list of valid model names is always authoritative

Usage:
    from src.models import build_model, list_available_models

    model = build_model("custom_cnn")
    model.summary()

    names = list_available_models()  # ['dense_nn', 'lenet5', 'custom_cnn']
"""

from __future__ import annotations

from collections.abc import Callable

import tensorflow as tf

from config.config import MODEL_TRAINING_CONFIG
from src.models.custom_cnn import build_custom_cnn
from src.models.dense_nn import build_dense_nn
from src.models.lenet5 import build_lenet5

# Each entry maps a string key to a builder function that accepts
# a learning_rate argument and returns a compiled tf.keras.Model.
MODEL_REGISTRY: dict[str, Callable[..., tf.keras.Model]] = {
    "dense_nn": build_dense_nn,
    "lenet5": build_lenet5,
    "custom_cnn": build_custom_cnn,
}


def build_model(name: str) -> tf.keras.Model:
    """
    Build and return a compiled model by its registered name.

    Learning rate is sourced from config.py so there is a single
    source of truth — changing it there affects training, evaluation,
    and any programmatic usage.

    Args:
        name: Registered model name. Must be one of the keys in
              MODEL_REGISTRY ('dense_nn', 'lenet5', 'custom_cnn').

    Returns:
        A compiled tf.keras.Model ready for .fit().

    Raises:
        ValueError: If name is not a registered model, with a helpful
            message listing the valid options.
    """
    if name not in MODEL_REGISTRY:
        valid_names = list(MODEL_REGISTRY.keys())
        raise ValueError(
            f"Unknown model '{name}'. " f"Valid model names are: {valid_names}"
        )

    learning_rate = MODEL_TRAINING_CONFIG[name]["learning_rate"]
    return MODEL_REGISTRY[name](learning_rate=learning_rate)


def list_available_models() -> list[str]:
    """
    Return the names of all registered architectures.

    Used by the CLI argument parser and the Streamlit model selector
    to present valid choices to the user.
    """
    return list(MODEL_REGISTRY.keys())
