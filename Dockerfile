# ─────────────────────────────────────────────────────────────────────────────
# DigitVision Dockerfile
#
# Design Decisions:
#   - python:3.12-slim: minimal base image (~50MB vs ~900MB for full Python)
#   - opencv-python-headless: no X11/GUI deps needed inside a container
#   - COPY requirements.txt first: Docker layer caching — only re-installs
#     packages when requirements.txt changes, not on every code change
#   - Non-root user: security best practice for containerised applications
#   - HEALTHCHECK: lets Docker Compose know when the app is actually ready
# ─────────────────────────────────────────────────────────────────────────────

FROM python:3.12-slim

# Metadata labels (visible in Docker Hub / GitHub Container Registry)
LABEL maintainer="DigitVision"
LABEL description="Handwritten digit recognition — Streamlit application"
LABEL version="1.0.0"

# System dependencies required by OpenCV and TensorFlow
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ── Layer 1: Install Python dependencies ──────────────────────────────────────
# Copied before the rest of the source so Docker re-uses this layer when
# only application code changes (not dependencies).
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# ── Layer 2: Copy application source ─────────────────────────────────────────
COPY . .

# Create directories that are gitignored but needed at runtime
RUN mkdir -p data models/saved performance_plots logs sample_images

# ── Security: run as non-root user ───────────────────────────────────────────
RUN useradd --create-home --shell /bin/bash digitvision
RUN chown -R digitvision:digitvision /app
USER digitvision

# Streamlit port
EXPOSE 8501

# Health check — Streamlit exposes a health endpoint
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# Start the Streamlit app
ENTRYPOINT ["streamlit", "run", "streamlit_app/app.py", \
    "--server.port=8501", \
    "--server.address=0.0.0.0", \
    "--server.headless=true", \
    "--browser.gatherUsageStats=false"]
