"""
Home Dashboard — DigitVision.

The landing page. Shows overall project status, a per-model performance
summary, and key statistics at a glance. Designed to impress on first load.
"""

from __future__ import annotations

import json
from pathlib import Path

import streamlit as st

from components.cards import metric_card, model_comparison_card, page_header, status_badge
from components.styles import get_global_css

# ── Imports that depend on project root being in sys.path ─────────────────────
from config.config import (
    AVAILABLE_MODELS,
    METADATA_PATHS,
    MODEL_DESCRIPTIONS,
    MODEL_DISPLAY_NAMES,
    MODEL_PATHS,
)


def _load_all_metadata() -> dict[str, dict | None]:
    """Load metadata JSON for each model. Returns None if not trained yet."""
    result = {}
    for model_name in AVAILABLE_MODELS:
        path = METADATA_PATHS.get(model_name)
        if path and path.exists():
            with path.open("r", encoding="utf-8") as f:
                result[model_name] = json.load(f)
        else:
            result[model_name] = None
    return result


# ─────────────────────────────────────────────────────────────────────────────

metadata = _load_all_metadata()
trained_models = [name for name, meta in metadata.items() if meta is not None]
total_trained = len(trained_models)

# ── Page header ───────────────────────────────────────────────────────────────
st.markdown(
    page_header(
        "DigitVision",
        "Handwritten digit recognition · Three architectures · Deep Learning showcase",
        "🔢",
    ),
    unsafe_allow_html=True,
)

# ── Status bar ────────────────────────────────────────────────────────────────
col_status, col_spacer = st.columns([3, 5])
with col_status:
    if total_trained == 3:
        st.markdown(
            status_badge("All 3 models trained and ready", "success"),
            unsafe_allow_html=True,
        )
    elif total_trained > 0:
        st.markdown(
            status_badge(f"{total_trained}/3 models trained", "warning"),
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            status_badge("No models trained yet — run: make train", "error"),
            unsafe_allow_html=True,
        )

st.markdown("<div style='margin-top:0.5rem'></div>", unsafe_allow_html=True)

if total_trained == 0:
    st.info(
        "**Getting started:** Train all three models by running `make train` "
        "from the project root, or `python -m src.train --all`. "
        "Training takes approximately 10–20 minutes on CPU."
    )

# ── Quick statistics ──────────────────────────────────────────────────────────
st.markdown("<div class='dv-section-header'>Quick Statistics</div>", unsafe_allow_html=True)

best_accuracy = max(
    (m["test_accuracy"] for m in metadata.values() if m), default=None
)
total_params = sum(
    m["total_parameters"] for m in metadata.values() if m
)
architectures_count = 3

c1, c2, c3, c4 = st.columns(4)
with c1:
    st.markdown(
        metric_card("Models Available", str(architectures_count), icon="🏗️"),
        unsafe_allow_html=True,
    )
with c2:
    val = f"{best_accuracy:.2%}" if best_accuracy else "—"
    st.markdown(
        metric_card("Best Test Accuracy", val, icon="🏆"),
        unsafe_allow_html=True,
    )
with c3:
    val = f"{total_params:,}" if total_params else "—"
    st.markdown(
        metric_card("Total Parameters", val, icon="⚙️"),
        unsafe_allow_html=True,
    )
with c4:
    st.markdown(
        metric_card("Dataset", "MNIST · 70K images", icon="🗂️"),
        unsafe_allow_html=True,
    )

st.markdown("<div style='margin-top:2rem'></div>", unsafe_allow_html=True)

# ── Model performance cards ───────────────────────────────────────────────────
st.markdown(
    "<div class='dv-section-header'>Model Performance Overview</div>",
    unsafe_allow_html=True,
)

best_model = (
    max(trained_models, key=lambda n: metadata[n]["test_accuracy"])
    if trained_models
    else None
)

cols = st.columns(3)
for col, model_name in zip(cols, AVAILABLE_MODELS):
    meta = metadata[model_name]
    display_name = MODEL_DISPLAY_NAMES[model_name]
    description = MODEL_DESCRIPTIONS[model_name]

    with col:
        if meta:
            acc_pct = f"{meta['test_accuracy']:.2%}"
            f1_pct = f"{meta['macro_f1']:.2%}"
            params = f"{meta['total_parameters']:,}"
            time_min = f"{meta['training_time_seconds'] / 60:.1f} min"
            is_best = model_name == best_model

            st.markdown(
                model_comparison_card(
                    title=display_name,
                    subtitle=description[:80] + "...",
                    stats=[
                        ("Test Accuracy", acc_pct),
                        ("Macro F1", f1_pct),
                        ("Parameters", params),
                        ("Training Time", time_min),
                    ],
                    badge_text="Best Accuracy" if is_best else "",
                    badge_status="success",
                    is_best=is_best,
                ),
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"""
                <div class="dv-model-card" style="opacity:0.5">
                    <div class="dv-model-card-title">{display_name}</div>
                    <div style="margin-top:0.5rem">
                        {"Not trained yet"}
                    </div>
                    <div style="margin-top:1rem;font-size:0.8rem;color:#64748b">
                        Run: <code>python -m src.train --model {model_name}</code>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

st.markdown("<div style='margin-top:2rem'></div>", unsafe_allow_html=True)

# ── About section ─────────────────────────────────────────────────────────────
with st.expander("📖  About This Project", expanded=False):
    st.markdown(
        """
        **DigitVision** is a handwritten digit recognition system that demonstrates
        real machine learning engineering practices — not just a notebook demo.

        **What makes it different:**
        - Three architectures compared honestly (Dense NN → LeNet-5 → Custom CNN)
        - Grad-CAM visualisation showing *what* the model attends to
        - Complete evaluation: confusion matrix, per-class F1, ROC curves
        - Production practices: Docker, CI/CD, pytest, logging, type hints

        **Tech stack:** TensorFlow · Streamlit · Plotly · scikit-learn · OpenCV

        Navigate using the sidebar to explore predictions, compare models,
        and understand how convolutional networks process images.
        """
    )
