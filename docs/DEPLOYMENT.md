# DigitVision — Deployment Guide

This document covers every supported deployment path for DigitVision: local
development, Docker (single machine), and cloud. All commands have been verified
against the actual `Dockerfile`, `docker-compose.yml`, and `Makefile` in this repository.

---

## Prerequisites

| Requirement | Minimum | Notes |
|---|---|---|
| Python | 3.12.x | 3.13 is not supported — `tensorflow==2.16.2` has no 3.13 wheel |
| pip | 23+ | Bundled with Python 3.12 |
| Docker | 24+ | Required only for Docker deployment |
| docker compose | v2+ | Use `docker compose` (v2), not `docker-compose` (v1) |
| RAM | 4 GB | 8 GB recommended for training |
| Disk | 3 GB | Models (~150 MB each), dataset (~200 MB), base image (~1 GB) |

---

## Option 1 — Local Python Environment

This is the recommended path for development and experimentation.

### Step 1: Install Python 3.12

**Windows**

Download the Python 3.12.x installer from [python.org/downloads](https://www.python.org/downloads/).
During installation, check **"Add Python to PATH"**.

Verify:

```powershell
py -3.12 --version
# Expected: Python 3.12.x
```

**Linux (Ubuntu/Debian)**

```bash
sudo apt update
sudo apt install python3.12 python3.12-venv python3.12-pip
python3.12 --version
```

**macOS**

```bash
brew install python@3.12
python3.12 --version
```

### Step 2: Clone and set up

```bash
git clone https://github.com/keyboard-warrior-777/digitvision.git
cd digitvision
```

**Windows (PowerShell):**

```powershell
py -3.12 -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

**Linux / macOS:**

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Expected output ends with:
```
Successfully installed tensorflow-2.16.2 numpy-1.26.4 ...
```

### Step 3: Train the models

```bash
# Train all three models sequentially (~20–35 minutes on CPU)
make train

# Or train individually:
python -m src.train --model dense_nn      # ~3–5 min
python -m src.train --model lenet5        # ~5–8 min
python -m src.train --model custom_cnn   # ~8–15 min
```

Trained models are saved to `models/saved/` as `.keras` files.
Training history (accuracy/loss per epoch) is saved as JSON alongside each model.

### Step 4: Evaluate and generate artifacts

```bash
make evaluate
# Or: python -m src.evaluate --all
```

This generates:

- Confusion matrix PNGs in `performance_plots/`
- Per-class F1 bar charts in `performance_plots/`
- Raw JSON data (`performance_plots/raw/`) for interactive Plotly charts
- Grad-CAM sample images in `performance_plots/gradcam_samples/`
- Model cards (JSON + Markdown) in `models/saved/`

### Step 5: Launch the dashboard

```bash
make run
# Or: streamlit run streamlit_app/app.py
```

Open **http://localhost:8501** in your browser.

> **Important:** Always run `streamlit run` from the project root directory.
> The app resolves all file paths relative to the project root. Running from
> a subdirectory will cause `FileNotFoundError` on model loads.

---

## Option 2 — Docker (Recommended for Deployment)

Docker ensures an identical environment on any machine, regardless of what
Python version is installed on the host.

### Build and run

```bash
docker compose up --build
```

This single command:
1. Builds the image using `python:3.12-slim` as the base
2. Installs all dependencies from `requirements.txt`
3. Copies the application source
4. Starts the Streamlit app on port 8501
5. Waits for the health check to pass before marking the container as ready

The app will be available at **http://localhost:8501**.

### Persist trained models

The `docker-compose.yml` mounts four host directories into the container:

```yaml
volumes:
  - ./models:/app/models
  - ./data:/app/data
  - ./performance_plots:/app/performance_plots
  - ./logs:/app/logs
```

This means models trained on your host machine are immediately available in the
container, and any artifacts generated inside the container are persisted to disk
after the container stops.

### Useful Docker commands

```bash
# Stop the container
docker compose down

# Rebuild after code changes
docker compose up --build

# View live logs
docker compose logs -f digitvision

# Check container health
docker inspect digitvision_app --format='{{.State.Health.Status}}'
```

### Training inside Docker

The Docker image runs only the Streamlit application. If you want to train inside
the container, you must override the entrypoint:

```bash
docker compose run --rm --entrypoint "" digitvision \
  python -m src.train --model custom_cnn
```

Trained models are written to `/app/models/saved/` inside the container, which
is bind-mounted to `./models/saved/` on the host — so the weights are saved locally.

---

## Option 3 — Streamlit Community Cloud

Streamlit Community Cloud (share.streamlit.io) provides free hosting for public
Streamlit apps directly from a GitHub repository.

> **Note:** Because MNIST models are large (~50–150 MB each), trained model weights
> cannot be committed to git (they are in `.gitignore`). For a live demo, you have
> two options:
>
> 1. Commit pre-trained `.keras` files to `models/saved/` (remove them from
>    `.gitignore` for the demo branch).
> 2. Load pre-trained models from cloud storage (S3, GCS) at startup.

**Steps:**

1. Push your repository to GitHub (with pre-trained models committed)
2. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub
3. Click **New app**
4. Select repository, branch, and main file: `streamlit_app/app.py`
5. Set Python version to **3.12** in the Advanced settings
6. Click **Deploy**

The deployment reads `requirements.txt` automatically and installs dependencies.
Allow 5–10 minutes for first-time builds (TensorFlow is large).

---

## Environment Variables

The app reads no required environment variables. The following are optional:

| Variable | Default | Purpose |
|---|---|---|
| `TF_CPP_MIN_LOG_LEVEL` | `0` | Set to `3` to suppress TensorFlow C++ startup logs |
| `PYTHONUNBUFFERED` | (unset) | Set to `1` to disable output buffering (set in docker-compose.yml) |
| `PYTHONDONTWRITEBYTECODE` | (unset) | Set to `1` to prevent `.pyc` file creation (set in docker-compose.yml) |

The `.env.example` file in the repository root documents all optional environment
variables. Copy it to `.env` and customise as needed:

```bash
cp .env.example .env
```

---

## Production Checklist

Before making the application accessible over a network:

- [ ] Run behind a reverse proxy (nginx, Caddy) with TLS
- [ ] Set `--server.address=0.0.0.0` in the Streamlit entrypoint (already set in Dockerfile)
- [ ] Confirm the health check endpoint responds: `curl http://localhost:8501/_stcore/health`
- [ ] Verify model files exist in `models/saved/` before starting the app
- [ ] Review the `TROUBLESHOOTING.md` for common startup failures

---

## Port Reference

| Port | Service | Notes |
|---|---|---|
| 8501 | Streamlit dashboard | Default Streamlit port |

The Dockerfile exposes port 8501. The `docker-compose.yml` maps `8501:8501` (host:container).
To use a different host port, edit `docker-compose.yml`:

```yaml
ports:
  - "9000:8501"   # Access at http://localhost:9000
```
