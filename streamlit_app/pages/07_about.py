"""
About — DigitVision.

Project overview: motivation, architecture decisions, model comparison,
training pipeline, and technology stack.
"""


from __future__ import annotations

import streamlit as st
from components.cards import info_box, page_header

# ─────────────────────────────────────────────────────────────────────────────

st.markdown(
    page_header("About", "Architecture decisions, engineering rationale, and tech stack", "ℹ️"),
    unsafe_allow_html=True,
)

# ── What is DigitVision? ─────────────────────────────────────────────────────────────
st.markdown(
    """
    DigitVision trains three neural architectures on identical data with the same
    optimiser and evaluation suite. Performance differences are therefore attributable
    to architecture alone — not to data preparation or training choices.

    The headline result: the Custom CNN reaches **99.3% accuracy with 75K parameters** —
    7× fewer than the Dense NN baseline at higher accuracy, because convolutions
    share weights spatially instead of treating every pixel independently.
    """
)

st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

# ── Model architecture section ───────────────────────────────────────────────────────
st.markdown(
    "<div class='dv-section-header'>Model Architectures</div>",
    unsafe_allow_html=True,
)

arch_data = [
    {
        "name": "Dense Neural Network",
        "key": "dense_nn",
        "year": "Baseline",
        "params": "~530,000",
        "expected_acc": "~97.5%",
        "description": (
            "Treats the 28×28 image as a flat 784-element vector. "
            "All spatial structure is destroyed at the Flatten layer. "
            "Three decreasing dense layers (512 → 256 → 128) with ReLU and Dropout. "
            "Exists to show what performance looks like *without* spatial awareness."
        ),
        "key_limitation": "No spatial awareness — a pixel shifted one position looks completely different.",
        "colour": "#60a5fa",
    },
    {
        "name": "LeNet-5",
        "key": "lenet5",
        "year": "1998 (LeCun et al.)",
        "params": "~61,000",
        "expected_acc": "~98.5%",
        "description": (
            "Yann LeCun's foundational architecture. Two convolutional stages with "
            "5×5 kernels and AveragePooling, followed by a Dense classifier (120 → 84 → 10). "
            "Faithfully reproduced except for ReLU (instead of tanh) and Softmax output."
        ),
        "key_limitation": "No BatchNorm or Dropout — less stable training and prone to overfitting.",
        "colour": "#f59e0b",
    },
    {
        "name": "Custom CNN",
        "key": "custom_cnn",
        "year": "Modern best practices",
        "params": "~75,000",
        "expected_acc": "~99.3%",
        "description": (
            "Two convolutional blocks (32→64 filters) using BatchNorm + MaxPool + Dropout. "
            "GlobalAveragePooling2D replaces Flatten, dramatically reducing parameters "
            "while improving generalisation. Built with the Functional API."
        ),
        "key_limitation": "Slightly harder to explain than Sequential models without the Functional API concept.",
        "colour": "#4ade80",
    },
]

for arch in arch_data:
    st.markdown(
        f"""
        <div class="dv-card" style="border-left:4px solid {arch['colour']};margin-bottom:1rem">
            <div style="display:flex;justify-content:space-between;align-items:flex-start;flex-wrap:wrap;gap:0.5rem">
                <div>
                    <div style="font-size:1.05rem;font-weight:700;color:var(--text-primary)">{arch['name']}</div>
                    <div style="font-size:0.78rem;color:var(--text-muted);margin-top:2px">{arch['year']}</div>
                </div>
                <div style="display:flex;gap:0.75rem;flex-wrap:wrap">
                    <span class="dv-badge dv-badge-info">{arch['params']} params</span>
                    <span class="dv-badge dv-badge-success">{arch['expected_acc']}</span>
                </div>
            </div>
            <div style="margin-top:0.75rem;font-size:0.875rem;color:var(--text-secondary);line-height:1.65">
                {arch['description']}
            </div>
            <div style="margin-top:0.5rem;font-size:0.8rem;color:var(--error)">
                ⚠ Limitation: {arch['key_limitation']}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

# ── Why these three models? ─────────────────────────────────────────────────────
st.markdown(
    "<div class='dv-section-header'>Why These Three Models?</div>",
    unsafe_allow_html=True,
)
st.markdown(
    """
    The three models form a deliberate **progression** that mirrors real ML research history:

    | Step | Model | The Insight Added |
    |------|-------|-------------------|
    | 1 | Dense NN | Establish a performance floor; show that "easy" accuracy is achievable without spatial reasoning |
    | 2 | LeNet-5 | Prove that spatial feature extraction (convolutions) beats flat pixel vectors |
    | 3 | Custom CNN | Show the compounding effect of modern techniques: BatchNorm + MaxPool + GAP |

    The jump from Dense NN (97.5%, 530K params) to Custom CNN (99.3%, 75K params) is the
    headline result: **7× fewer parameters, better performance** — because convolutions
    share weights spatially instead of treating every pixel independently.
    """
)

st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

# ── Training pipeline ─────────────────────────────────────────────────────────────────
st.markdown(
    "<div class='dv-section-header'>Training Pipeline</div>",
    unsafe_allow_html=True,
)
st.markdown(
    """
    ```
    MNIST Data
        │
        ▼
    dataset.py ─── Normalise → Reshape → One-hot encode → Split (train/val/test)
        │
        ▼
    train.py ────── Build model → Augmented generator → Fit with 3 callbacks:
        │                ├─ ModelCheckpoint (save best val_accuracy)
        │                ├─ EarlyStopping  (patience=5, restore best)
        │                └─ ReduceLROnPlateau (factor=0.5, patience=3)
        │
        ▼
    evaluate.py ─── Compute metrics → Save plots → artifacts.py:
        │                ├─ Confusion matrix (.npy)
        │                ├─ Per-class metrics (.json)
        │                ├─ ROC data (.json)
        │                ├─ Sample predictions + images
        │                └─ Grad-CAM overlay images (one per digit class)
        │
        ▼
    model_card.py ── metadata.json + model_summary.md
    ```
    """
)

st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

# ── Technology stack ───────────────────────────────────────────────────────────────
st.markdown(
    "<div class='dv-section-header'>Technology Stack</div>",
    unsafe_allow_html=True,
)

tech_cols = st.columns(3)

with tech_cols[0]:
    st.markdown("### 🧠 ML")
    st.markdown(
        """
        - **TensorFlow 2.x** — model building and training
        - **scikit-learn** — evaluation metrics, ROC, confusion matrix
        - **OpenCV** — image preprocessing (INTER_AREA resize, colourmap)
        - **NumPy** — numerical operations
        - **Pillow** — image I/O and display
        """
    )

with tech_cols[1]:
    st.markdown("### 🖥️ Application")
    st.markdown(
        """
        - **Streamlit** — multi-page web application
        - **streamlit-drawable-canvas** — interactive drawing canvas
        - **Plotly** — interactive charts (ROC, confusion matrix, training curves)
        - **Matplotlib / Seaborn** — static plots (post-evaluation)
        """
    )

with tech_cols[2]:
    st.markdown("### ⚙️ Engineering")
    st.markdown(
        """
        - **pytest** — unit and integration tests
        - **Ruff + Black** — linting and formatting
        - **Docker** — containerised environment
        - **GitHub Actions** — CI/CD pipeline (lint → test)
        - **pyproject.toml** — project configuration
        - **Makefile** — developer workflow commands
        """
    )

st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

# ── Key engineering decisions ─────────────────────────────────────────────────────
st.markdown(
    "<div class='dv-section-header'>Engineering Decisions</div>",
    unsafe_allow_html=True,
)

decisions = [
    (
        "Frozen dataclass for dataset",
        "MNISTData is a frozen dataclass — named access (data.X_train) instead of positional "
        "(data[0]). Frozen prevents accidental mutation after preparation.",
    ),
    (
        "Registry pattern for models",
        "build_model('custom_cnn') is the single entry point. Adding a new architecture "
        "is a two-line change in models/__init__.py — no other files need updating.",
    ),
    (
        "Separated artifacts from evaluation",
        "evaluate.py computes metrics and saves static plots. artifacts.py generates "
        "the richer interactive-ready outputs. Single responsibility in both.",
    ),
    (
        "Model-level cache in predict.py",
        "Models are loaded once and cached in a module-level dict. Streamlit reruns "
        "the script on every interaction — without caching, predictions would be slow.",
    ),
    (
        "opencv-python-headless",
        "The headless variant avoids GUI library dependencies. This makes the Docker "
        "image smaller and prevents errors in CI/headless server environments.",
    ),
    (
        "JSON for training history",
        "History is saved as JSON (not pickle) — human-readable, framework-agnostic, "
        "version-control friendly. Any language can read it.",
    ),
]

for title, detail in decisions:
    with st.expander(f"  {title}"):
        st.markdown(f"> {detail}")

st.markdown("<div style='height:var(--sp-4)'></div>", unsafe_allow_html=True)

# ── Key metrics quick reference ───────────────────────────────────────────────────
st.markdown(
    "<div class='dv-section-header'>Key Metrics</div>",
    unsafe_allow_html=True,
)

st.markdown(
    info_box(
        "<strong>Dense NN</strong> &nbsp;—&nbsp; 97.5% accuracy &middot; 530K parameters &middot; no spatial awareness<br>"
        "<strong>LeNet-5</strong> &nbsp;—&nbsp; 98.5% accuracy &middot; 61K parameters &middot; 1998 architecture (LeCun et al.)<br>"
        "<strong>Custom CNN</strong> &nbsp;—&nbsp; 99.3% accuracy &middot; 75K parameters &middot; modern best practices<br><br>"
        "The Custom CNN is 7× more parameter-efficient than the Dense NN at higher accuracy — "
        "because convolutional weight sharing scales with kernel size, not image size."
    ),
    unsafe_allow_html=True,
)
