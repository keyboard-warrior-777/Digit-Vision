"""
Central configuration for DigitVision.

Design Decision:
    All hyperparameters, file paths, and application constants live here.
    No values are hardcoded anywhere else in the codebase. This means you
    can retrain with different settings by changing one file, not hunting
    through multiple scripts.

Interview Answer:
    "I used a central config module so that every hyperparameter is in one
    place. This makes experiments reproducible — you know exactly what
    settings produced each model."
"""

from pathlib import Path

# ─── Project Root ─────────────────────────────────────────────────────────────
# Path(__file__) = config/config.py → .parent = config/ → .parent = project root
ROOT_DIR: Path = Path(__file__).parent.parent

# ─── Directory Paths ──────────────────────────────────────────────────────────
DATA_DIR: Path = ROOT_DIR / "data"
MODELS_DIR: Path = ROOT_DIR / "models" / "saved"
PERFORMANCE_PLOTS_DIR: Path = ROOT_DIR / "performance_plots"
ASSETS_DIR: Path = ROOT_DIR / "assets"
SAMPLE_IMAGES_DIR: Path = ROOT_DIR / "sample_images"
LOGS_DIR: Path = ROOT_DIR / "logs"

# ─── Saved Model Paths ────────────────────────────────────────────────────────
MODEL_PATHS: dict[str, Path] = {
    "dense_nn": MODELS_DIR / "dense_nn.keras",
    "lenet5": MODELS_DIR / "lenet5.keras",
    "custom_cnn": MODELS_DIR / "custom_cnn.keras",
}

# ─── Training History (accuracy / loss per epoch) ─────────────────────────────
HISTORY_PATHS: dict[str, Path] = {
    "dense_nn": MODELS_DIR / "dense_nn_history.json",
    "lenet5": MODELS_DIR / "lenet5_history.json",
    "custom_cnn": MODELS_DIR / "custom_cnn_history.json",
}

# ─── Model Metadata (generated after training + evaluation) ───────────────────
METADATA_PATHS: dict[str, Path] = {
    "dense_nn": MODELS_DIR / "dense_nn_metadata.json",
    "lenet5": MODELS_DIR / "lenet5_metadata.json",
    "custom_cnn": MODELS_DIR / "custom_cnn_metadata.json",
}

# ─── Human-Readable Model Summaries (.md) ─────────────────────────────────────
MODEL_SUMMARY_PATHS: dict[str, Path] = {
    "dense_nn": MODELS_DIR / "dense_nn_summary.md",
    "lenet5": MODELS_DIR / "lenet5_summary.md",
    "custom_cnn": MODELS_DIR / "custom_cnn_summary.md",
}

# ─── Raw Evaluation Data (read by Streamlit for interactive charts) ───────────
# Stored separately from PNG plots so the frontend can build interactive charts.
RAW_EVAL_DIR: Path = PERFORMANCE_PLOTS_DIR / "raw"

RAW_METRICS_PATHS: dict[str, Path] = {
    "dense_nn": RAW_EVAL_DIR / "dense_nn_metrics.json",
    "lenet5": RAW_EVAL_DIR / "lenet5_metrics.json",
    "custom_cnn": RAW_EVAL_DIR / "custom_cnn_metrics.json",
}

ROC_DATA_PATHS: dict[str, Path] = {
    "dense_nn": RAW_EVAL_DIR / "dense_nn_roc.json",
    "lenet5": RAW_EVAL_DIR / "lenet5_roc.json",
    "custom_cnn": RAW_EVAL_DIR / "custom_cnn_roc.json",
}

CONFUSION_MATRIX_PATHS: dict[str, Path] = {
    "dense_nn": RAW_EVAL_DIR / "dense_nn_confusion_matrix.npy",
    "lenet5": RAW_EVAL_DIR / "lenet5_confusion_matrix.npy",
    "custom_cnn": RAW_EVAL_DIR / "custom_cnn_confusion_matrix.npy",
}

# ─── Sample Predictions + Grad-CAM Samples ────────────────────────────────────
SAMPLE_PREDICTIONS_PATHS: dict[str, Path] = {
    "dense_nn": RAW_EVAL_DIR / "dense_nn_sample_predictions.json",
    "lenet5": RAW_EVAL_DIR / "lenet5_sample_predictions.json",
    "custom_cnn": RAW_EVAL_DIR / "custom_cnn_sample_predictions.json",
}

GRADCAM_SAMPLES_DIR: Path = PERFORMANCE_PLOTS_DIR / "gradcam_samples"
PREDICTION_SAMPLE_IMAGES_DIR: Path = PERFORMANCE_PLOTS_DIR / "prediction_samples"

# ─── Dataset ──────────────────────────────────────────────────────────────────
IMAGE_SIZE: tuple[int, int] = (28, 28)
IMAGE_CHANNELS: int = 1
NUM_CLASSES: int = 10
# Shape format required by TensorFlow Conv2D layers: (height, width, channels)
INPUT_SHAPE: tuple[int, int, int] = (*IMAGE_SIZE, IMAGE_CHANNELS)
CLASS_NAMES: list[str] = [str(i) for i in range(NUM_CLASSES)]

# ─── Training ─────────────────────────────────────────────────────────────────
RANDOM_SEED: int = 42
VALIDATION_SPLIT: float = 0.1  # 10% of training data used for validation
BATCH_SIZE: int = 128
EPOCHS: int = 20
LEARNING_RATE: float = 1e-3

# Per-model overrides — Custom CNN trains longer because it's deeper
MODEL_TRAINING_CONFIG: dict[str, dict] = {
    "dense_nn": {
        "epochs": 20,
        "batch_size": 128,
        "learning_rate": 1e-3,
    },
    "lenet5": {
        "epochs": 20,
        "batch_size": 128,
        "learning_rate": 1e-3,
    },
    "custom_cnn": {
        "epochs": 25,
        "batch_size": 128,
        "learning_rate": 1e-3,
    },
}

# ─── Data Augmentation ────────────────────────────────────────────────────────
# Applied only during training to improve generalisation to hand-drawn digits.
# Interview: "Augmentation makes the model robust to slight rotations and
# shifts — which is exactly how real users draw digits on a canvas."
AUGMENTATION_CONFIG: dict[str, float] = {
    "rotation_range": 10,  # degrees
    "zoom_range": 0.1,  # ±10%
    "width_shift_range": 0.1,  # ±10% of image width
    "height_shift_range": 0.1,  # ±10% of image height
}

# ─── Evaluation & Plotting ────────────────────────────────────────────────────
CONFUSION_MATRIX_FIGSIZE: tuple[int, int] = (10, 8)
CURVES_FIGSIZE: tuple[int, int] = (14, 5)
PLOT_DPI: int = 150
PLOT_STYLE: str = "dark_background"

# ─── Streamlit Application ────────────────────────────────────────────────────
APP_TITLE: str = "DigitVision"
APP_SUBTITLE: str = "Handwritten Digit Recognition · Deep Learning Showcase"
APP_ICON: str = "🔢"

# Canvas size on the Draw page (pixels). 280 = 10× MNIST — clean downscaling.
CANVAS_SIZE: int = 280
CANVAS_STROKE_WIDTH: int = 20

AVAILABLE_MODELS: list[str] = list(MODEL_PATHS.keys())

# Human-readable names shown in the Streamlit UI
MODEL_DISPLAY_NAMES: dict[str, str] = {
    "dense_nn": "Dense Neural Network",
    "lenet5": "LeNet-5 (1998)",
    "custom_cnn": "Custom CNN",
}

# Brief description shown on the Compare page
MODEL_DESCRIPTIONS: dict[str, str] = {
    "dense_nn": (
        "A fully-connected baseline. Treats each pixel independently — "
        "no spatial awareness. Shows why CNNs were invented."
    ),
    "lenet5": (
        "Yann LeCun's 1998 architecture that sparked the deep learning "
        "revolution. Uses convolutional layers for spatial feature extraction."
    ),
    "custom_cnn": (
        "A modern CNN with BatchNormalization, GlobalAveragePooling, and "
        "Dropout. Outperforms the classics with fewer parameters."
    ),
}
