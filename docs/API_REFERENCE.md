# DigitVision — API Reference

> Complete documentation for every public module and function in the DigitVision backend.

---

## Table of Contents

- [`src.dataset`](#srcdataset)
- [`src.preprocessing`](#srcpreprocessing)
- [`src.models`](#srcmodels)
- [`src.train`](#srctrain)
- [`src.evaluate`](#srcevaluate)
- [`src.predict`](#srcpredict)
- [`src.gradcam`](#srcgradcam)
- [`src.artifacts`](#srcartifacts)
- [`src.model_card`](#srcmodel_card)
- [`src.logger`](#srclogger)
- [`components.cards`](#componentscards)
- [`components.charts`](#componentscharts)
- [`config.config`](#configconfig)

---

## `src.dataset`

Handles MNIST loading, normalisation, and train/val/test splitting.

---

### `MNISTData`

```python
@dataclass(frozen=True)
class MNISTData:
    X_train: np.ndarray      # shape (54000, 28, 28, 1), float32, [0, 1]
    y_train: np.ndarray      # shape (54000, 10), one-hot encoded
    X_val: np.ndarray        # shape (6000, 28, 28, 1), float32, [0, 1]
    y_val: np.ndarray        # shape (6000, 10), one-hot encoded
    X_test: np.ndarray       # shape (10000, 28, 28, 1), float32, [0, 1]
    y_test: np.ndarray       # shape (10000, 10), one-hot encoded
    y_test_labels: np.ndarray  # shape (10000,), integer class labels [0–9]
```

An immutable container for the fully-prepared MNIST dataset splits.

---

### `load_and_prepare_mnist`

```python
def load_and_prepare_mnist() -> MNISTData
```

Downloads (if needed) and prepares the MNIST dataset for training.

**Returns:** A `MNISTData` instance with all four splits.

**Side effects:** Downloads MNIST to `~/.keras/datasets/` on first call (11 MB).

**Example:**
```python
from src.dataset import load_and_prepare_mnist

data = load_and_prepare_mnist()
print(data.X_train.shape)  # (54000, 28, 28, 1)
print(data.y_test_labels[:5])  # [7 2 1 0 4]
```

---

## `src.preprocessing`

Converts user input (canvas drawings and uploaded images) into model-ready tensors.

---

### `canvas_image_to_model_input`

```python
def canvas_image_to_model_input(canvas_image_data: np.ndarray) -> np.ndarray
```

Convert an RGBA canvas image (from streamlit-drawable-canvas) to a model-ready tensor.

**Args:**
- `canvas_image_data`: RGBA numpy array, shape `(H, W, 4)`, dtype `uint8`.
  The alpha channel encodes the stroke (255 = stroke, 0 = background).

**Returns:** `float32` array of shape `(1, 28, 28, 1)`, values in `[0.0, 1.0]`.

**Raises:** `ValueError` if the input does not have exactly 4 channels.

**Processing steps:**
1. Extract alpha channel `(H, W)`
2. Invert: `255 - alpha` (stroke → dark, background → bright)
3. Resize to `28×28`
4. Normalise: `÷ 255.0`
5. Reshape: `(1, 28, 28, 1)`

**Example:**
```python
import numpy as np
from src.preprocessing import canvas_image_to_model_input

canvas = np.zeros((280, 280, 4), dtype=np.uint8)
canvas[50:230, 130:150, 3] = 200  # a vertical stroke

tensor = canvas_image_to_model_input(canvas)
print(tensor.shape)   # (1, 28, 28, 1)
print(tensor.dtype)   # float32
```

---

### `uploaded_image_to_model_input`

```python
def uploaded_image_to_model_input(pil_image: Image.Image) -> np.ndarray
```

Convert a user-uploaded PIL Image into a model-ready tensor.

**Args:**
- `pil_image`: Any PIL Image mode (RGB, RGBA, L, etc.) at any resolution.

**Returns:** `float32` array of shape `(1, 28, 28, 1)`, values in `[0.0, 1.0]`.

**Inversion heuristic:** Samples 5×5 pixel patches from each of the four image corners
and computes their mean brightness. If `corner_mean > 127`, the image is assumed to have
a light background and is inverted to match MNIST's white-on-black format. Corners are
used rather than the global mean because they are almost exclusively background pixels,
making them a reliable proxy even when the digit is thick or large.

**Example:**
```python
from PIL import Image
from src.preprocessing import uploaded_image_to_model_input

image = Image.open("my_digit.png")
tensor = uploaded_image_to_model_input(image)
print(tensor.shape)   # (1, 28, 28, 1)
```

---

## `src.models`

Model registry providing a unified interface to all architectures.

---

### `build_model`

```python
def build_model(model_name: str) -> tf.keras.Model
```

Build and compile a model by name.

**Args:**
- `model_name`: One of `"dense_nn"`, `"lenet5"`, `"custom_cnn"`.

**Returns:** Compiled `tf.keras.Model` ready for training.

**Raises:** `ValueError` with list of valid names if `model_name` is unregistered.

**Example:**
```python
from src.models import build_model

model = build_model("custom_cnn")
model.summary()
print(model.count_params())   # ~75,000
```

---

### `list_available_models`

```python
def list_available_models() -> list[str]
```

Return model names in registration order: `["dense_nn", "lenet5", "custom_cnn"]`.

---

## `src.train`

Training engine with callbacks, history serialisation, and result packaging.

---

### `TrainingResult`

```python
@dataclass(frozen=True)
class TrainingResult:
    model_name: str
    final_train_accuracy: float
    final_val_accuracy: float
    epochs_trained: int
    model_path: Path
    history_path: Path
```

Immutable record of a completed training run.

---

### `train_model`

```python
def train_model(model_name: str, data: MNISTData) -> TrainingResult
```

Train a model with early stopping, learning rate reduction, and checkpointing.

**Args:**
- `model_name`: Registered model name.
- `data`: Prepared `MNISTData` instance.

**Returns:** `TrainingResult` with final metrics and file paths.

**Side effects:**
- Saves best checkpoint to `models/saved/<model_name>.keras`
- Saves history JSON to `models/saved/<model_name>_history.json`

---

## `src.evaluate`

Computes evaluation metrics, generates visualisation artifacts, and returns structured results.

---

### `ModelEvaluation`

```python
@dataclass(frozen=True)
class ModelEvaluation:
    model_name: str
    test_accuracy: float
    test_loss: float
    per_class_f1: dict[str, float]   # keys: "0"–"9"
    macro_f1: float
    weighted_f1: float
    classification_report_text: str
    confusion_matrix_path: Path
    f1_chart_path: Path
```

Immutable record of a complete model evaluation.

---

### `evaluate_model`

```python
def evaluate_model(model_name: str) -> ModelEvaluation
```

Load a trained model, evaluate on the MNIST test set, and generate all visualisations.

**Args:**
- `model_name`: Registered model name. The model must have been trained.

**Returns:** `ModelEvaluation` with all metrics and artifact paths.

**Raises:** `FileNotFoundError` with "Train it first" message if model file is absent.

---

### `load_training_history`

```python
def load_training_history(model_name: str) -> Optional[dict[str, list[float]]]
```

Load the saved training history JSON for a model.

**Returns:** Dict with keys `"accuracy"`, `"val_accuracy"`, `"loss"`, `"val_loss"`,
each mapping to a list of per-epoch float values. Returns `None` if no history exists.

---

### `_compute_per_class_f1`

```python
def _compute_per_class_f1(
    true_labels: np.ndarray,
    predicted_labels: np.ndarray,
) -> dict[str, float]
```

Compute per-class F1 scores using sklearn.

**Returns:** Dict mapping class name strings (`"0"`–`"9"`) to Python `float` values.
Python `float` (not numpy float32) is used to ensure JSON serializability.

---

## `src.predict`

Inference engine with model caching and structured result types.

---

### `PredictionResult`

```python
@dataclass(frozen=True)
class PredictionResult:
    predicted_digit: int             # 0–9
    predicted_label: str             # "0"–"9"
    confidence: float                # probability of predicted class [0, 1]
    all_probabilities: dict[str, float]   # keys: "0"–"9"
    model_display_name: str          # e.g. "Custom CNN"
```

Immutable prediction result returned by all inference functions.

---

### `predict_from_canvas`

```python
def predict_from_canvas(
    canvas_image_data: np.ndarray,
    model_name: str,
) -> PredictionResult
```

Preprocess a canvas RGBA array and run inference.

**Args:**
- `canvas_image_data`: RGBA `uint8` array from streamlit-drawable-canvas.
- `model_name`: Registered model name.

**Returns:** `PredictionResult`.

**Raises:** `ValueError` if input is not RGBA.

---

### `predict_from_upload`

```python
def predict_from_upload(
    pil_image: Image.Image,
    model_name: str,
) -> PredictionResult
```

Preprocess an uploaded PIL Image and run inference.

**Args:**
- `pil_image`: Any PIL Image mode, any resolution.
- `model_name`: Registered model name.

**Returns:** `PredictionResult`.

---

### `predict_batch`

```python
def predict_batch(
    preprocessed_images: list[np.ndarray],
    model_name: str,
) -> list[PredictionResult]
```

Run batched inference on a list of pre-processed image tensors.

Internally calls `model.predict(batch)` in a single forward pass — approximately
20× faster than calling `predict` individually on each image.

**Args:**
- `preprocessed_images`: List of `(1, 28, 28, 1)` float32 arrays.
- `model_name`: Registered model name.

**Returns:** List of `PredictionResult`, same length as input. Empty list → empty list.

---

## `src.gradcam`

Grad-CAM heatmap generation and image overlay.

---

### `compute_gradcam`

```python
def compute_gradcam(
    model: tf.keras.Model,
    preprocessed_image: np.ndarray,
    predicted_class_index: int,
) -> Optional[np.ndarray]
```

Compute a Grad-CAM heatmap for the specified class.

**Args:**
- `model`: A compiled Keras model.
- `preprocessed_image`: Shape `(1, 28, 28, 1)`, float32.
- `predicted_class_index`: Class index (0–9) to compute gradients for.

**Returns:** Normalised heatmap of shape `(28, 28)`, float32, values in `[0.0, 1.0]`.
Returns `None` if the model has no Conv2D layers.

**Algorithm:**
1. Find the last Conv2D layer using `_find_last_conv_layer()`
2. Build a gradient model outputting both the conv layer activations and the final predictions
3. Compute gradients of the target class score with respect to the conv layer output
4. Pool gradients across the spatial dimensions
5. Weight the feature maps by pooled gradients and take the ReLU
6. Resize to input resolution and normalise to `[0, 1]`

---

### `overlay_heatmap_on_image`

```python
def overlay_heatmap_on_image(
    heatmap: np.ndarray,
    original_image: np.ndarray,
    alpha: float = 0.4,
) -> np.ndarray
```

Blend a Grad-CAM heatmap over the original digit image.

**Args:**
- `heatmap`: Shape `(28, 28)`, float32, values in `[0.0, 1.0]`.
- `original_image`: Shape `(1, 28, 28, 1)`, float32.
- `alpha`: Heatmap blending weight. `0.0` = image only, `1.0` = heatmap only.

**Returns:** RGB image of shape `(28, 28, 3)`, dtype `uint8`.

---

## `src.artifacts`

Generates all post-evaluation artifacts: confusion matrix, ROC data, sample predictions, Grad-CAM images.

---

### `generate_all_artifacts`

```python
def generate_all_artifacts(
    model_name: str,
    model: tf.keras.Model,
    data: MNISTData,
    y_predicted: np.ndarray,
    probabilities: np.ndarray,
) -> None
```

Generate the complete artifact set for one evaluated model.

**What is generated:**
- Raw confusion matrix as `.npy`
- Per-class metrics JSON (sklearn classification report)
- Per-class ROC curve data JSON
- 20 sample prediction images with metadata JSON
- Grad-CAM sample images for each digit class (CNN models only)

---

## `src.model_card`

Auto-generates structured model cards after training and evaluation.

---

### `generate_model_card`

```python
def generate_model_card(
    model_name: str,
    training_result: TrainingResult,
    evaluation_result: ModelEvaluation,
    training_time_seconds: float,
    model: tf.keras.Model,
) -> None
```

Generate and save a model card in JSON and Markdown formats.

**Saved files:**
- `models/saved/<model_name>_metadata.json` — machine-readable metadata
- `models/saved/<model_name>_summary.md` — human-readable Markdown card

---

## `src.logger`

Centralised logging configuration.

---

### `get_logger`

```python
def get_logger(name: str) -> logging.Logger
```

Get a configured logger for a module.

**Args:**
- `name`: Typically `__name__` from the calling module.

**Returns:** Logger writing to both console (INFO+) and `logs/digitvision.log` (DEBUG+).

**Usage:**
```python
from src.logger import get_logger
logger = get_logger(__name__)
logger.info("Training started for model: %s", model_name)
```

---

## `components.cards`

Reusable HTML string components for the Streamlit frontend.

All functions return HTML strings for use with `st.markdown(..., unsafe_allow_html=True)`.

---

### `page_header`

```python
def page_header(title: str, subtitle: str, icon: str = "") -> str
```

Render a styled page header with title, subtitle, and optional emoji icon.

---

### `metric_card`

```python
def metric_card(
    label: str,
    value: str,
    icon: str = "",
    delta: Optional[str] = None,
    delta_positive: bool = True,
) -> str
```

Render a styled metric card with optional delta indicator (↑/↓).

---

### `status_badge`

```python
def status_badge(text: str, status: str) -> str
```

Render a coloured status badge.

**Args:**
- `status`: One of `"success"`, `"warning"`, `"error"`, `"info"`.

---

### `prediction_result_card`

```python
def prediction_result_card(
    predicted_digit: int,
    confidence: float,
    inference_time_ms: float,
    model_display_name: str,
) -> str
```

Render the primary prediction result card with colour-coded confidence:
- ≥ 90%: green
- ≥ 70%: amber
- < 70%: red

---

### `info_box`

```python
def info_box(content: str, box_type: str = "info") -> str
```

Render a styled callout box.

---

### `pipeline_step`

```python
def pipeline_step(step_number: int, title: str, description: str) -> str
```

Render a numbered pipeline step for the CNN Explainer page.

---

### `section_header`

```python
def section_header(title: str, subtitle: str = "") -> str
```

Render a section separator with optional subtitle.

---

## `components.charts`

Plotly figure builders with consistent styling.

All functions return `plotly.graph_objects.Figure` instances.

---

### `build_confidence_bar_chart`

```python
def build_confidence_bar_chart(
    all_probabilities: dict[str, float],
    predicted_digit: int,
) -> go.Figure
```

Build a horizontal bar chart of class probabilities with the predicted class highlighted.

---

### `build_training_curves_chart`

```python
def build_training_curves_chart(
    histories: dict[str, dict[str, list[float]]],
    metric: str = "accuracy",
) -> go.Figure
```

Build a multi-line training history chart.

**Args:**
- `histories`: Dict mapping model name → history dict.
- `metric`: `"accuracy"` or `"loss"`.

Models with missing keys are silently skipped.

---

### `build_confusion_matrix_chart`

```python
def build_confusion_matrix_chart(
    confusion_matrix: np.ndarray,
    class_labels: list[str],
    title: str = "Confusion Matrix",
) -> go.Figure
```

Build an interactive heatmap confusion matrix.

---

### `build_f1_bar_chart`

```python
def build_f1_bar_chart(
    per_class_f1: dict[str, float],
    model_display_name: str = "",
) -> go.Figure
```

Build a bar chart of per-class F1 scores.

---

### `build_roc_chart`

```python
def build_roc_chart(
    roc_data: dict[str, dict],
    model_display_name: str = "",
) -> go.Figure
```

Build a multi-class ROC curve plot with AUC scores in the legend.

---

### `build_accuracy_comparison_chart`

```python
def build_accuracy_comparison_chart(
    accuracies: dict[str, float],
) -> go.Figure
```

Build a grouped bar chart comparing test accuracy across all models.

---

### `build_class_distribution_chart`

```python
def build_class_distribution_chart(
    counts: dict[str, int],
    title: str = "Class Distribution",
) -> go.Figure
```

Build a bar chart of sample counts per digit class.

---

## `config.config`

Single source of truth for all configurable values.

### Key constants

| Name | Type | Description |
|------|------|-------------|
| `ROOT_DIR` | `Path` | Absolute path to project root |
| `DATA_DIR` | `Path` | `project_root/data/` |
| `MODELS_DIR` | `Path` | `project_root/models/saved/` |
| `LOGS_DIR` | `Path` | `project_root/logs/` |
| `PLOTS_DIR` | `Path` | `project_root/performance_plots/` |
| `IMAGE_SIZE` | `tuple[int, int]` | `(28, 28)` |
| `NUM_CLASSES` | `int` | `10` |
| `CLASS_NAMES` | `list[str]` | `["0", "1", ..., "9"]` |
| `MODEL_DISPLAY_NAMES` | `dict[str, str]` | Human-readable model names |
| `MODEL_PATHS` | `dict[str, Path]` | Paths to `.keras` files |
| `HISTORY_PATHS` | `dict[str, Path]` | Paths to history JSON files |
| `MODEL_TRAINING_CONFIG` | `dict[str, dict]` | Per-model training hyperparameters |
