# DigitVision — Changelog

---

## v1.0.1 — Pre-release Fixes

**Release date:** July 2026
**Tag:** `v1.0.1`

### Fixed

- **Python version constraint** (`pyproject.toml`): Changed `requires-python = ">=3.12"` to
  `requires-python = "~=3.12.0"`. The previous constraint allowed Python 3.13, which cannot
  install `tensorflow==2.16.2` (no wheel published for 3.13). This caused a hard installation
  crash for any reviewer using a modern Python version. The constraint now matches the
  actual supported runtime exactly.

- **Pyrefly type analysis target** (`pyproject.toml`): Changed `python-version = "3.13"` to
  `python-version = "3.12"` in the `[tool.pyrefly]` stanza. The previous value caused
  type analysis to target a Python version not used by the project.

- **Uploaded image inversion heuristic** (`src/preprocessing.py`): Replaced the global
  mean-based background detection (`image.mean() > 127`) with a corner-sampling heuristic.
  The new implementation samples four 5×5 pixel patches from the image corners. Since corners
  are almost exclusively background pixels, this correctly detects light backgrounds even when
  a thick or large digit pulls the global mean below the 127 threshold.

- **Editable install support** (`pyproject.toml`): Added `[project.optional-dependencies]`
  with a `dev` extra and `[tool.setuptools.packages.find]` so that `pip install -e ".[dev]"`
  works correctly. Also added `[project.urls]` for GitHub links.

- **Makefile Docker targets** (`Makefile`): Corrected `docker-compose` (v1, deprecated CLI)
  to `docker compose` (v2, current CLI) in the `docker-build`, `docker-run`, and `docker-stop`
  targets. All documentation already referenced `docker compose`; the Makefile was inconsistent.

---

# DigitVision v1.0.0 — Initial Release

**Release date:** July 2026
**Tag:** `v1.0.0`

---

## Overview

DigitVision v1.0.0 is the foundation release of a production-quality handwritten
digit recognition system built as a student AI portfolio project.

This release includes a complete deep learning backend, an interactive 7-page
Streamlit dashboard, 145 automated tests, Docker support, and GitHub Actions CI/CD.

---

## What's Included

### 🧠 Machine Learning Backend

- **Three neural network architectures:**
  - Dense Neural Network — fully-connected baseline
  - LeNet-5 — the 1998 convolutional pioneer
  - Custom CNN — modern architecture with BatchNorm, GlobalAveragePooling, and Dropout
- **Training pipeline** with EarlyStopping, ModelCheckpoint, and ReduceLROnPlateau
- **Data augmentation** — rotation, zoom, and shift for improved generalisation
- **Grad-CAM explainability** — heatmaps for any Conv2D model

### 📊 Streamlit Dashboard (7 pages)

| Page | Feature |
|------|---------|
| Home | Project overview and model performance summary |
| Recognise | Real-time canvas drawing + image upload prediction |
| Playground | Side-by-side comparison of all three models |
| Analytics | Full evaluation dashboard with interactive charts |
| CNN Explainer | Grad-CAM heatmap overlay visualisation |
| Dataset | MNIST explorer with distribution and samples |
| About | Architecture explanation and auto-generated model cards |

### ✅ Testing

- **145 tests, 0 failures** across 8 test files
- Pure Python — no Streamlit server needed, no real model weights needed
- Covers preprocessing, models, Grad-CAM, prediction, evaluation, artifacts, training callbacks, and UI components

### 🐳 Deployment

- Multi-stage Dockerfile with minimal production image
- `docker-compose.yml` for one-command launch
- `.dockerignore` to exclude unnecessary files

### 🔄 CI/CD

- GitHub Actions workflow: lint → format check → test on every push/PR to `main`
- Ruff linting + Black formatting enforced
- Coverage report uploaded as build artifact

### 📁 Documentation

- `README.md` — comprehensive project overview
- `docs/ARCHITECTURE.md` — Mermaid diagrams and design decisions
- `docs/DEVELOPMENT.md` — developer setup and contribution guide
- `docs/USER_GUIDE.md` — end-user documentation
- `docs/API_REFERENCE.md` — full public API documentation
- `ROADMAP.md` — versioned feature plan
- Auto-generated model cards (JSON + Markdown) after training

---

## Performance Benchmarks

*Run after training with `make train && make evaluate`*

| Model | Parameters | Expected Test Accuracy |
|-------|-----------|----------------------|
| Dense NN | ~530K | ~97.5% |
| LeNet-5 | ~61K | ~98.5% |
| Custom CNN | ~75K | ~99.3% |

---

## Installation

```bash
git clone https://github.com/keyboard-warrior-777/digitvision.git
cd digitvision
pip install -r requirements.txt
make train
make evaluate
make run
```

Or with Docker:
```bash
docker compose up --build
```

---

## Known Issues

- `ImageDataGenerator` is deprecated in Keras 3 — a soft warning is shown during training. This will be replaced with `tf.data` in v1.1.
- Grad-CAM heatmaps are low-resolution (28×28) due to MNIST's small image size.
- No GPU auto-detection — add `CUDA_VISIBLE_DEVICES=0` manually if using a GPU.

---

## What's Next — v1.1

- GPU auto-detection
- `tf.data` augmentation pipeline
- Webcam live input
- Pre-commit hooks and Dependabot

See [ROADMAP.md](ROADMAP.md) for the full plan.
