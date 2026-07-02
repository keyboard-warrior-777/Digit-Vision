# DigitVision — Troubleshooting Guide

A concise, evidence-based guide to the most common failure modes. Each entry
includes the exact error message, the root cause, and a verified fix.

---

## Installation

### `ERROR: No matching distribution found for tensorflow==2.16.2`

**When it appears:** Running `pip install -r requirements.txt` on Python 3.13.

**Root cause:** TensorFlow 2.16.2 does not publish wheels for Python 3.13.
This repository requires Python 3.12.x.

**Fix:**

```bash
# Verify your Python version
python --version      # Must be 3.12.x

# On Windows, if you have multiple versions installed:
py -3.12 --version

# Create a new venv explicitly with Python 3.12
py -3.12 -m venv .venv       # Windows
python3.12 -m venv .venv     # Linux / macOS

# Activate and reinstall
.venv\Scripts\activate        # Windows
source .venv/bin/activate     # Linux / macOS
pip install -r requirements.txt
```

---

### `ModuleNotFoundError: No module named 'src'`

**When it appears:** Running `pytest` or any `python -m src.*` command.

**Root cause:** The command is being run from the wrong directory. The project
expects to be run from the repository root (the directory containing `src/`,
`config/`, and `streamlit_app/`).

**Fix:**

```bash
# Verify your working directory
pwd                               # Linux / macOS
Get-Location                      # Windows PowerShell

# Change to the project root if needed
cd path/to/digitvision

# Then re-run
python -m src.train --model custom_cnn
pytest tests/ -v
```

---

### `ModuleNotFoundError: No module named 'numpy'` (in type checker / Pyrefly)

**When it appears:** In your IDE's Pyrefly/Pyright panel even though the tests pass.

**Root cause:** The IDE is using the system Python interpreter instead of the
`.venv` interpreter, or the `.venv` packages are not installed.

**Fix:**

1. In VS Code: open the Command Palette (`Ctrl+Shift+P`) → **Python: Select Interpreter**
   → choose the `.venv` interpreter (it will show `.venv/Scripts/python.exe` on Windows).
2. If packages are missing, reinstall: `.venv\Scripts\pip install -r requirements.txt`.

---

## Training

### Training stops early with few epochs

**Root cause:** EarlyStopping is working correctly. When `val_loss` does not
improve for 5 consecutive epochs, training halts and restores the best weights.
This is expected behaviour, not a failure.

**Verify it worked:** Check the log output — the final line before "Training
complete" will say:

```
Epoch N: early stopping
Restoring model weights from the end of the best epoch.
```

The saved model at `models/saved/<model_name>.keras` contains the best weights,
not the last epoch's weights.

---

### `WARNING: ImageDataGenerator is deprecated`

**When it appears:** During `make train` or `python -m src.train`.

**Root cause:** TensorFlow 2.16 deprecated `tf.keras.preprocessing.image.ImageDataGenerator`.
This is a soft warning — the augmentation still runs correctly.

**Impact:** None on correctness or accuracy. The warning can be suppressed by setting:

```bash
TF_CPP_MIN_LOG_LEVEL=2 python -m src.train --model custom_cnn
```

This will be addressed in v1.1 by migrating to `tf.data`.

---

### `FileNotFoundError: No trained model found at 'models/saved/custom_cnn.keras'`

**When it appears:** Running `make evaluate` or launching the Streamlit app before training.

**Root cause:** The model has not been trained yet. Evaluation and the Streamlit
prediction pages require trained model files.

**Fix:**

```bash
# Train the missing model
python -m src.train --model custom_cnn

# Or train all three at once
make train
```

---

## Streamlit Application

### Blank page or `ModuleNotFoundError` on first load

**Root cause:** Streamlit is being run from a directory other than the project root,
or the `.venv` is not activated.

**Fix:**

```bash
# Always run from the project root
cd path/to/digitvision

# Ensure the venv is active (you should see (.venv) in your prompt)
.venv\Scripts\activate      # Windows
source .venv/bin/activate   # Linux / macOS

# Then launch
streamlit run streamlit_app/app.py
```

---

### "No model found" message on Draw or Playground pages

**Root cause:** The model `.keras` files have not been generated yet. The app
handles this gracefully with an informational message rather than crashing.

**Fix:** Run `make train` and then `make evaluate` before using prediction features.

---

### Canvas drawing is not recognized correctly

**Symptom:** Drawing a digit produces random or consistently wrong predictions.

**Most common cause:** The canvas stroke width is too thin. A very thin stroke
may not survive the 10× downscale from 280×280 to 28×28 pixels.

**Fix:** Draw with a thicker, bolder stroke. The default canvas stroke width in
`config/config.py` is `CANVAS_STROKE_WIDTH = 20` — if you have customised this,
try increasing it.

**Second common cause:** The digit is not centred. MNIST training images are
centred and padded. Very off-centre digits confuse all three models.

---

### Grad-CAM page shows "Grad-CAM not available for this model"

**Root cause:** This is expected behaviour for the Dense Neural Network. Grad-CAM
requires at least one Conv2D layer. The Dense NN has none.

The Grad-CAM page only works with LeNet-5 and Custom CNN.

---

## Docker

### `docker compose up` fails with `port is already allocated`

**Root cause:** Port 8501 is already in use (possibly another Streamlit instance
or a previous container that did not shut down cleanly).

**Fix:**

```bash
# Stop any running containers
docker compose down

# Or change the host port in docker-compose.yml
ports:
  - "8502:8501"
```

---

### Models are not visible inside the container

**Root cause:** The model files exist on the host but the container started before
they were trained, or the volume mount is not working.

**Verify the mount:**

```bash
docker compose exec digitvision ls /app/models/saved/
```

If empty, check that you are running `docker compose` from the project root (where
`docker-compose.yml` lives) so the relative volume paths resolve correctly.

---

### `curl: (7) Failed to connect` on health check

**Root cause:** The container is still starting up. TensorFlow imports take 20–30
seconds on first launch. The health check has a `start_period: 40s` grace period.

**Fix:** Wait 40–60 seconds after `docker compose up` before checking the health
endpoint. You can monitor startup with:

```bash
docker compose logs -f digitvision
```

---

## Tests

### `ImportError` when running pytest

**Root cause:** The test runner cannot find the `src` or `components` packages.
`conftest.py` adds both the project root and `streamlit_app/` to `sys.path` — but
only if pytest is run from the project root.

**Fix:** Always run pytest from the project root:

```bash
cd path/to/digitvision
pytest tests/ -v
```

---

### Tests pass locally but fail in CI

**Root cause:** The most common causes are:

1. A test depends on a file that exists locally but is in `.gitignore` (e.g., trained models).
   Solution: use the stub model fixtures from `conftest.py`.
2. A test imports from `streamlit_app/components/` but `sys.path` was not set up.
   Solution: `conftest.py` handles this — ensure you are not bypassing it by running
   a single test file directly without pytest discovery.

---

## Performance

### Training is slow (CPU only)

TensorFlow uses CPU by default if no GPU is detected. Expected CPU training times:

| Model | CPU Time |
|---|---|
| Dense NN | 3–5 minutes |
| LeNet-5 | 5–8 minutes |
| Custom CNN | 8–15 minutes |

**To use a GPU (NVIDIA):** Install the CUDA-enabled TensorFlow variant:

```bash
pip install tensorflow[and-cuda]==2.16.2
```

Then set the visible device:

```bash
CUDA_VISIBLE_DEVICES=0 python -m src.train --all
```

---

## Getting Further Help

If your issue is not listed here:

1. Check `logs/digitvision.log` — every module logs to this file at DEBUG level.
2. Search the repository's GitHub Issues.
3. Open a new issue with the exact error message and the output of `python --version`.
