# DigitVision — Roadmap

> Versioned feature roadmap. This is a living document — items may shift
> between versions as priorities evolve.

---

## Current Version: v1.0.0 — Foundation Release

**Status:** ✅ Complete

Core features:
- Three neural network architectures (Dense NN, LeNet-5, Custom CNN)
- Streamlit dashboard with 7 pages
- Interactive canvas drawing and prediction
- Batch image upload and prediction
- Model comparison playground
- Full analytics dashboard (confusion matrix, ROC, F1, training curves)
- Grad-CAM explainability
- 145-test pytest suite
- GitHub Actions CI/CD
- Docker support
- Auto-generated model cards

---

## v1.1 — Performance & Developer Experience

**Theme:** Make training faster and the codebase future-proof.

### Training

- [ ] **GPU auto-detection** — Automatically use GPU if available, with a clear terminal message
- [ ] **`tf.data` pipeline** — Replace deprecated `ImageDataGenerator` with `tf.data.Dataset` for faster, modern augmentation
- [ ] **Mixed precision training** — Enable `tf.keras.mixed_precision` for 2× speed on modern GPUs
- [ ] **Learning rate finder** — Implement the Leslie Smith LR range test to recommend optimal learning rates

### Features

- [ ] **Webcam input** — Live webcam feed with real-time digit detection (OpenCV capture → prediction loop)
- [ ] **Prediction history** — Remember the last 10 drawings in session state with a "History" panel
- [ ] **Confidence threshold alert** — Warn the user when confidence < 60% ("I'm not sure about this one")
- [ ] **Augmentation preview page** — Interactive slider to preview augmentation strength before training

### Developer Experience

- [ ] **Makefile profiling target** — `make profile` runs cProfile on the training loop
- [ ] **Pre-commit hooks** — Auto-run Black and Ruff on `git commit`
- [ ] **Coverage badge** — Add a `coverage.svg` badge to README via `codecov.io`
- [ ] **Dependabot configuration** — Automated dependency update PRs

---

## v1.2 — Extensibility & Deployment

**Theme:** Make DigitVision consumable as a service and deployable to the cloud.

### Backend API

- [ ] **FastAPI REST backend** — Separate inference server exposing:
  - `POST /predict` — Single image prediction
  - `POST /predict/batch` — Batch image prediction
  - `GET /models` — List available models with metadata
  - `GET /health` — Liveness probe for Kubernetes
- [ ] **API authentication** — Simple API key authentication for the REST endpoints
- [ ] **OpenAPI documentation** — Auto-generated Swagger UI from FastAPI

### Model Export

- [ ] **TensorFlow Lite export** — Convert trained models to `.tflite` for edge/mobile deployment
- [ ] **ONNX export** — Export to ONNX format for deployment in non-TF environments
- [ ] **TensorFlow Serving** — Docker image with TF Serving for production inference

### Cloud Deployment

- [ ] **Streamlit Community Cloud** — One-click deployment guide for `share.streamlit.io`
- [ ] **Google Cloud Run** — Containerised deployment with auto-scaling to zero
- [ ] **Terraform IaC** — Infrastructure-as-code for reproducible cloud deployment
- [ ] **CI/CD for deployment** — Add `deploy` job to GitHub Actions that runs on release tags

### Mobile UI

- [ ] **Progressive Web App (PWA)** — Add a `manifest.json` and service worker to make the Streamlit app installable on mobile
- [ ] **Touch-optimised canvas** — Improve the drawing canvas for finger input on tablets and phones
- [ ] **Responsive layout** — CSS media queries to adapt the dashboard for small screens

---

## v2.0 — Next-Generation Capabilities

**Theme:** Expand beyond MNIST to real-world handwriting recognition.

### Extended Recognition

- [ ] **Alphanumeric recognition** — Extend to the EMNIST dataset (letters + digits, 62 classes)
- [ ] **Multi-character input** — Segment a drawn string into individual characters and predict each
- [ ] **Handwritten equation solver** — Detect digits and mathematical operators, evaluate the expression
- [ ] **Language support** — Support for Devanagari (Hindi), Chinese digits, Arabic-Indic numerals

### Advanced ML

- [ ] **Transformer architecture** — Vision Transformer (ViT) as a fourth model option, demonstrating attention-based vision
- [ ] **Continual learning** — Online model updates without forgetting (Elastic Weight Consolidation)
- [ ] **Model distillation** — Distill Custom CNN into a micro-model for TF Lite with < 1% accuracy loss
- [ ] **Uncertainty quantification** — Monte Carlo Dropout for calibrated confidence estimates
- [ ] **Adversarial robustness testing** — Generate FGSM adversarial examples and measure model resistance

### Platform Features

- [ ] **User accounts** — Persist prediction history across sessions (PostgreSQL backend)
- [ ] **Model versioning** — Track model versions with git-like hash and performance regression detection
- [ ] **A/B testing** — Route 50% of predictions to Model A and 50% to Model B, track accuracy
- [ ] **Annotation tool** — Allow users to label misclassified predictions to build a custom fine-tuning dataset
- [ ] **Scheduled retraining** — Cron-triggered retraining on collected user data, with regression gates

### Explainability

- [ ] **LIME explanations** — Local Interpretable Model-agnostic Explanations as an alternative to Grad-CAM
- [ ] **SHAP integration** — SHAP DeepExplainer for pixel-level attribution scores
- [ ] **Attention rollout** — Visualise attention patterns for the ViT model
- [ ] **Counterfactual explanations** — "If pixel X were darker, the prediction would change from 4 to 9"

---

## Community & Open Source

- [ ] **Contributing guide** — `CONTRIBUTING.md` with PR template and code review checklist
- [ ] **Issue templates** — Bug report and feature request templates for GitHub Issues
- [ ] **Discussion forum** — Enable GitHub Discussions for questions and ideas
- [ ] **Hacktoberfest participation** — Tag good-first-issues for open-source contributors
- [ ] **Python package** — Publish `digitvision-core` to PyPI for the inference engine
