"""
Dataset Explorer — DigitVision.

Browse the MNIST dataset: class distribution, per-class sample grids,
and dataset statistics. All data is loaded once and cached.
"""

from __future__ import annotations

import numpy as np
import streamlit as st
from PIL import Image

from components.cards import info_box, metric_card, page_header
from components.charts import build_class_distribution_chart
from components.styles import get_global_css

from config.config import CLASS_NAMES


@st.cache_data(show_spinner="Loading MNIST dataset...")
def _load_mnist():
    """Load MNIST once; cache for the session."""
    import tensorflow as tf
    (X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
    return X_train, y_train, X_test, y_test


def _make_class_grid(images: np.ndarray, labels: np.ndarray, digit: int, n: int = 10) -> Image.Image:
    """
    Build a horizontal strip of n example images for a given digit class.

    Each image is upscaled to 56×56 with NEAREST interpolation to preserve
    the authentic pixel appearance of MNIST.
    """
    indices = np.where(labels == digit)[0][:n]
    strips = []
    for idx in indices:
        img = Image.fromarray(images[idx], "L").resize((56, 56), Image.NEAREST)
        strips.append(img)

    if not strips:
        return Image.new("L", (56 * n, 56), 0)

    grid = Image.new("L", (56 * len(strips), 56), 0)
    for i, img in enumerate(strips):
        grid.paste(img, (i * 56, 0))
    return grid


# ─────────────────────────────────────────────────────────────────────────────

st.markdown(
    page_header(
        "Dataset Explorer",
        "MNIST at a glance — class distribution, sample images, and pixel statistics",
        "🗂️",
    ),
    unsafe_allow_html=True,
)

X_train, y_train, X_test, y_test = _load_mnist()

# ── Overview statistics ───────────────────────────────────────────────────────
st.markdown("<div class='dv-section-header'>Dataset Overview</div>", unsafe_allow_html=True)

c1, c2, c3, c4 = st.columns(4)
with c1:
    st.markdown(metric_card("Training Samples", "60,000", icon="🏋️"), unsafe_allow_html=True)
with c2:
    st.markdown(metric_card("Test Samples", "10,000", icon="🧪"), unsafe_allow_html=True)
with c3:
    st.markdown(metric_card("Classes", "10", icon="🔢"), unsafe_allow_html=True)
with c4:
    st.markdown(metric_card("Image Size", "28 × 28 px", icon="🖼️"), unsafe_allow_html=True)

st.markdown("<div style='margin-top:0.5rem'></div>", unsafe_allow_html=True)
st.markdown(
    info_box(
        "MNIST (Modified National Institute of Standards and Technology, 1998) is the standard "
        "benchmark for digit recognition. Each image is a grayscale 28×28 scan, normalised and centred."
    ),
    unsafe_allow_html=True,
)

st.markdown("<div style='margin-top:1.5rem'></div>", unsafe_allow_html=True)

# ── Class distribution chart ──────────────────────────────────────────────────
st.markdown(
    "<div class='dv-section-header'>Class Distribution</div>",
    unsafe_allow_html=True,
)

split_choice = st.radio("Dataset split", ["Training", "Test"], horizontal=True)
y_display = y_train if split_choice == "Training" else y_test
X_display = X_train if split_choice == "Training" else X_test

class_counts = {str(d): int(np.sum(y_display == d)) for d in range(10)}

st.plotly_chart(
    build_class_distribution_chart(
        class_counts,
        title=f"Samples per Digit Class — {split_choice} Set",
    ),
    use_container_width=True,
)

# ── Balance analysis ──────────────────────────────────────────────────────────
counts_arr = np.array(list(class_counts.values()))
imbalance_ratio = counts_arr.max() / counts_arr.min()

col_a, col_b = st.columns(2)
col_a.metric("Least represented class", f"{counts_arr.min():,} samples")
col_b.metric("Most represented class", f"{counts_arr.max():,} samples")

if imbalance_ratio < 1.1:
    st.success(f"**Well balanced dataset** — imbalance ratio: {imbalance_ratio:.2f}×")
else:
    st.warning(f"Imbalance ratio: {imbalance_ratio:.2f}×")

st.markdown("<div style='margin-top:2rem'></div>", unsafe_allow_html=True)

# ── Sample image grid ─────────────────────────────────────────────────────────
st.markdown(
    "<div class='dv-section-header'>Sample Images by Class</div>",
    unsafe_allow_html=True,
)

selected_digits = st.multiselect(
    "Select digit classes to display",
    options=list(range(10)),
    default=list(range(10)),
    format_func=lambda d: f"Digit {d}",
)

samples_per_class = st.slider("Samples per class", min_value=5, max_value=20, value=10)

st.markdown("<div style='margin-top:0.75rem'></div>", unsafe_allow_html=True)

for digit in selected_digits:
    grid_image = _make_class_grid(X_display, y_display, digit, n=samples_per_class)
    count = class_counts[str(digit)]

    col_label, col_grid = st.columns([1, 9])
    with col_label:
        st.markdown(
            f"""
            <div style='text-align:center;padding-top:1rem'>
                <div style='font-size:1.5rem;font-weight:800;
                            background:linear-gradient(135deg,#6366f1,#22d3ee);
                            -webkit-background-clip:text;
                            -webkit-text-fill-color:transparent;
                            font-family:"JetBrains Mono",monospace'>{digit}</div>
                <div style='font-size:0.65rem;color:#64748b'>{count:,}<br>samples</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col_grid:
        st.image(grid_image, use_column_width="always")

st.markdown("<div style='margin-top:2rem'></div>", unsafe_allow_html=True)

# ── Pixel statistics ──────────────────────────────────────────────────────────
with st.expander("📐  Pixel Statistics (Training Set)"):
    flat = X_train.flatten().astype(np.float32) / 255.0
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Mean pixel", f"{flat.mean():.4f}")
    col2.metric("Std deviation", f"{flat.std():.4f}")
    col3.metric("Min pixel", "0.0000")
    col4.metric("Max pixel", "1.0000")

    st.markdown(
        info_box(
            "The mean pixel value of MNIST is ~0.1307 and std ~0.3081. "
            "These are the exact values used in the data normalisation step of this project. "
            "Applying the correct mean/std before prediction is critical for good performance."
        ),
        unsafe_allow_html=True,
    )
