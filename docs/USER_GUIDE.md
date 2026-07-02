# DigitVision — User Guide

> Everything you need to train, evaluate, and use DigitVision — no programming experience required for the app.

---

## Table of Contents

1. [Getting Started](#getting-started)
2. [Training the Models](#training-the-models)
3. [Evaluating the Models](#evaluating-the-models)
4. [Launching the Application](#launching-the-application)
5. [Using Each Page](#using-each-page)
   - [Home](#home-page)
   - [Recognise](#recognise-page)
   - [Playground](#playground-page)
   - [Analytics](#analytics-page)
   - [CNN Explainer](#cnn-explainer-page)
   - [Dataset Explorer](#dataset-explorer-page)
   - [About](#about-page)
6. [Common Problems](#common-problems)

---

## Getting Started

### System Requirements

| Requirement | Minimum | Recommended |
|-------------|---------|-------------|
| Python | 3.12.x | 3.12.x |
| RAM | 4 GB | 8 GB |
| Disk Space | 2 GB | 4 GB |
| GPU | Not required | Optional (speeds training 5–10×) |

### Installation

```bash
# Clone the project
git clone https://github.com/keyboard-warrior-777/digitvision.git
cd digitvision

# Windows
py -3.12 -m venv .venv
.venv\Scripts\activate

# Linux / macOS
python3.12 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

---

## Training the Models

Training downloads MNIST, builds the model, and saves the best weights automatically.

### Train all three models

```bash
make train
```

### Train a specific model

```bash
python -m src.train --model dense_nn
python -m src.train --model lenet5
python -m src.train --model custom_cnn
```

### What happens during training?

1. MNIST is downloaded to `data/` (only on first run — ~11 MB)
2. The model is built and compiled
3. Training starts (you'll see progress bars with accuracy and loss)
4. The best checkpoint is automatically saved to `models/saved/`
5. Training history (accuracy per epoch) is saved as JSON
6. A completion log is written to `logs/digitvision.log`

### Expected training time (CPU only)

| Model | Expected Time |
|-------|--------------|
| Dense NN | 3–5 minutes |
| LeNet-5 | 5–8 minutes |
| Custom CNN | 8–15 minutes |

### Watching training progress

Training outputs live progress to the terminal:

```
Epoch 1/25
422/422 ━━━━━━━━━━━━━━━━━━━━ 12s 28ms/step
  accuracy: 0.9201 — loss: 0.2571
  val_accuracy: 0.9784 — val_loss: 0.0721
```

Training stops automatically when the model stops improving (early stopping).

---

## Evaluating the Models

Evaluation runs the trained model on the 10,000 MNIST test images and generates all visualisations.

### Evaluate all models

```bash
make evaluate
```

### Evaluate a specific model

```bash
python -m src.evaluate --model custom_cnn
```

### What gets generated?

After evaluation, you will find new files in `performance_plots/`:

| File | Description |
|------|-------------|
| `custom_cnn_confusion_matrix.png` | 10×10 heatmap of correct vs. wrong predictions |
| `custom_cnn_f1_per_class.png` | Bar chart of F1 score per digit class |
| `custom_cnn_history.json` | Epoch-by-epoch accuracy and loss |
| `raw/custom_cnn_roc_data.json` | ROC curve data for all 10 classes |
| `raw/custom_cnn_metrics.json` | Full per-class precision, recall, F1 |
| `prediction_samples/` | 20 example predictions with confidence |
| `gradcam_samples/` | Grad-CAM heatmaps for each digit class |

And in `models/saved/`:

| File | Description |
|------|-------------|
| `custom_cnn_metadata.json` | Machine-readable model card |
| `custom_cnn_summary.md` | Human-readable model card |

---

## Launching the Application

```bash
make run
# or
streamlit run streamlit_app/app.py
```

Open your browser to **http://localhost:8501**

> **Note:** You can launch the app without training — it handles missing models gracefully by showing placeholder content. However, most dashboard features require at least one trained and evaluated model.

### With Docker (no Python setup required)

```bash
docker compose up --build
```

Then open **http://localhost:8501**

---

## Using Each Page

### Home Page

The landing page shows:
- A brief overview of the project
- A model performance summary (requires trained models)
- Quick navigation cards to each feature

Nothing to interact with here — it's your entry point.

---

### Recognise Page

**What it does:** Predict a digit from a drawing or uploaded image.

**Using the Draw tab:**
1. Select a model from the dropdown (Custom CNN is recommended)
2. Draw a digit in the black canvas area — use a thick stroke
3. Click **Predict**
4. The predicted digit and confidence appear on the right
5. The probability bar chart shows how confident the model is for each class
6. Click **Clear Canvas** to draw a new digit

**Tips for the best predictions:**
- Draw thick, centered strokes
- Fill most of the canvas — don't draw tiny digits in a corner
- Draw one digit at a time
- Avoid drawing too slowly (the canvas captures the final stroked image)

**Using the Upload tab:**
1. Click **Browse files** and select a PNG, JPEG, or BMP image
2. The image is automatically preprocessed and predicted
3. If your image has a white background (like a scan), it will be automatically inverted

---

### Playground Page

**What it does:** Draw a digit once and see all three models predict simultaneously.

1. Draw a digit in the canvas
2. Click **Compare Models**
3. Three prediction result cards appear side-by-side — one per model
4. Compare the confidence levels and predicted digits

**Why this is useful:**
- See where the Dense NN and CNN disagree
- Observe which model is most confident
- Find digits that confuse one model but not another

---

### Analytics Page

**What it does:** Full evaluation dashboard for each trained model.

**Controls:**
- Use the **model selector** at the top to switch between models

**Sections:**

| Section | What to look at |
|---------|----------------|
| **Summary Metrics** | Test accuracy, F1, parameter count |
| **Confusion Matrix** | Off-diagonal cells = wrong predictions. Darker = more errors. Hover for counts |
| **ROC Curves** | Area Under Curve (AUC) closer to 1.0 = better. Diagonal line = random guessing |
| **Per-Class F1** | Which digits are hardest? Shorter bars = more errors for that digit |
| **Training Curves** | Did the model overfit? Val loss rising while train loss falls = overfitting |
| **Sample Predictions** | 20 real test examples with the model's prediction and confidence |

---

### CNN Explainer Page

**What it does:** Shows Grad-CAM heatmaps — visualisations of which pixels in your digit influenced the model's prediction.

1. Draw or upload a digit
2. Select a model with Conv2D layers (LeNet-5 or Custom CNN — Dense NN is not supported)
3. The heatmap overlay appears: **red = most influential, blue = least influential**
4. Optionally change the class index to see which regions activate for a different digit

**What you're seeing:**
- The Grad-CAM heatmap is computed by backpropagating gradients from the output class back to the last convolutional layer
- High-activation regions are where the model "looked" most when making its decision
- A well-trained model shows activation concentrated on the actual digit strokes

**Note:** Grad-CAM is not available for the Dense NN because it has no convolutional (spatial) layers.

---

### Dataset Explorer Page

**What it does:** Explore the MNIST dataset your models were trained on.

**Sections:**

| Section | Description |
|---------|-------------|
| **Class Distribution** | How many training examples exist per digit (ideally ~6000 each) |
| **Sample Images** | Random samples from the training set |
| **Augmentation Preview** | See what the augmented versions look like |

This page helps you understand *why* the models behave as they do — for example, if digit 1 has fewer training examples, its F1 score may be lower.

---

### About Page

**What it does:** Educational content about the project architecture, model cards, and references.

**Sections:**

| Section | Description |
|---------|-------------|
| **Model Cards** | JSON and Markdown summaries of each trained model |
| **Architecture** | How the CNN pipeline works, explained visually |
| **References** | Academic papers that inspired the design |

---

## Common Problems

### "The canvas prediction is always wrong"

**Cause:** You may be drawing too small or in a corner.
**Fix:** Draw a large, centered, thick digit. The model expects the digit to fill most of the 28×28 image.

---

### "The app shows 'No model found' on the Recognise page"

**Cause:** The model hasn't been trained yet.
**Fix:** Run `python -m src.train --model custom_cnn` first.

---

### "Training crashed with 'out of memory'"

**Cause:** Your system doesn't have enough RAM for the batch size.
**Fix:** Edit `config/config.py` and reduce `batch_size` from 128 to 64 or 32.

---

### "Grad-CAM heatmap is all one colour"

**Cause:** The model is predicting with very low confidence, and all gradients are near-zero.
**Fix:** Draw a clearer, thicker digit. Ensure you're using a trained LeNet-5 or Custom CNN (not Dense NN).

---

### "Docker build fails"

**Cause:** The TensorFlow package is large (~600 MB) and may time out on slow connections.
**Fix:** Run `docker compose build --no-cache` on a faster network, or increase Docker's memory limit.

---

### "Training accuracy is much lower than expected"

**Possible causes:**
1. Learning rate is too high — try reducing to `0.0001`
2. Augmentation is too aggressive — reduce rotation angle in `config.py`
3. Insufficient training epochs — increase `epochs` in `MODEL_TRAINING_CONFIG`

---

### "pytest fails with 'import error'"

**Cause:** Running pytest from inside a subdirectory.
**Fix:** Always run from the project root:
```bash
cd "path/to/Handwritten Digit Recognizer"
pytest tests/
```
