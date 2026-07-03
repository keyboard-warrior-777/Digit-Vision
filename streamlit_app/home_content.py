"""
home_content.py — shared Home page rendering logic.

Both app.py (root route "/") and pages/01_home.py ("/home") call
render_home_page() from here so that content is defined exactly once.
"""

from __future__ import annotations

import json

import streamlit as st
from components.cards import (
    metric_card,
    model_comparison_card,
    page_header,
    status_badge,
)
from components.styles import get_global_css

from config.config import (
    AVAILABLE_MODELS,
    HISTORY_PATHS,
    MODEL_DESCRIPTIONS,
    MODEL_DISPLAY_NAMES,
    MODEL_PATHS,
    RAW_METRICS_PATHS,
)

# ── Data loading ──────────────────────────────────────────────────────────────

def _load_all_metadata() -> dict[str, dict | None]:
    """Load evaluation results for each model from the artefact files.

    The evaluation pipeline writes per-class classification reports to
    RAW_METRICS_PATHS, not to METADATA_PATHS. This function synthesises a
    metadata-compatible dict so the home page cards are populated correctly.
    Only models whose .keras file exists are considered trained.
    """
    result: dict[str, dict | None] = {}

    for model_name in AVAILABLE_MODELS:
        metrics_path = RAW_METRICS_PATHS.get(model_name)
        history_path = HISTORY_PATHS.get(model_name)
        model_path = MODEL_PATHS.get(model_name)

        if not (model_path and model_path.exists()):
            result[model_name] = None
            continue

        meta: dict = {}

        if metrics_path and metrics_path.exists():
            with metrics_path.open("r", encoding="utf-8") as f:
                raw = json.load(f)
            meta["test_accuracy"] = raw.get("accuracy", 0.0)
            meta["macro_f1"] = raw.get("macro avg", {}).get("f1-score", 0.0)
        else:
            meta["test_accuracy"] = 0.0
            meta["macro_f1"] = 0.0

        if history_path and history_path.exists():
            with history_path.open("r", encoding="utf-8") as f:
                hist = json.load(f)
            meta["epochs_trained"] = len(hist.get("val_accuracy", []))
        else:
            meta["epochs_trained"] = None

        # Parameters and training time are not derived at runtime (too slow).
        meta["total_parameters"] = None
        meta["training_time_seconds"] = None

        result[model_name] = meta

    return result


# ── Page renderer ─────────────────────────────────────────────────────────────

def render_home_page() -> None:
    """Render the full DigitVision Home page content.

    Called by both app.py (for the "/" route) and pages/01_home.py
    (for the "/home" route).  Injects global CSS so the page is styled
    correctly even when accessed directly rather than through app.py.
    """
    # Inject CSS — harmless if called twice (app.py + this function on "/")
    st.markdown(get_global_css(), unsafe_allow_html=True)

    metadata = _load_all_metadata()
    trained_models = [name for name, meta in metadata.items() if meta is not None]
    total_trained = len(trained_models)

    # ── Page header ───────────────────────────────────────────────────────────
    st.markdown(
        page_header(
            "DigitVision",
            "Three architectures, one benchmark — Dense NN → LeNet-5 → Custom CNN",
            image_path="assets/logo.png",
        ),
        unsafe_allow_html=True,
    )

    # ── Status badge ──────────────────────────────────────────────────────────
    col_status, _col_spacer = st.columns([3, 5])
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
            "No trained models found. "
            "Run `make train` or `python -m src.train --all` to get started. "
            "Training takes roughly 10–20 minutes on CPU."
        )

    # ── Quick statistics ──────────────────────────────────────────────────────
    st.markdown(
        "<div class='dv-section-header'>Quick Statistics</div>",
        unsafe_allow_html=True,
    )

    best_accuracy = max(
        (m["test_accuracy"] for m in metadata.values() if m), default=None
    )

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(
            metric_card("Models Available", "3", icon="🏗️"),
            unsafe_allow_html=True,
        )
    with c2:
        val = f"{best_accuracy:.2%}" if best_accuracy else "—"
        st.markdown(
            metric_card("Best Test Accuracy", val, icon="🏆"),
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            metric_card("Models Trained", f"{total_trained} / 3 Trained", icon="⚙️"),
            unsafe_allow_html=True,
        )
    with c4:
        st.markdown(
            metric_card("Dataset", "MNIST · 70K images", icon="🗂️"),
            unsafe_allow_html=True,
        )

    st.markdown("<div style='margin-top:2rem'></div>", unsafe_allow_html=True)

    # ── Model performance cards ───────────────────────────────────────────────
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
    for col, model_name in zip(cols, AVAILABLE_MODELS, strict=False):
        meta = metadata[model_name]
        display_name = MODEL_DISPLAY_NAMES[model_name]
        description = MODEL_DESCRIPTIONS[model_name]

        with col:
            if meta:
                is_best = model_name == best_model
                epochs_str = (
                    f"{meta['epochs_trained']} epochs"
                    if meta.get("epochs_trained")
                    else "N/A"
                )
                st.markdown(
                    model_comparison_card(
                        title=display_name,
                        subtitle=description[:80] + "...",
                        stats=[
                            ("Test Accuracy", f"{meta['test_accuracy']:.2%}"),
                            ("Macro F1", f"{meta['macro_f1']:.2%}"),
                            ("Epochs Trained", epochs_str),
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
                        <div style="margin-top:0.5rem">Not trained yet</div>
                        <div style="margin-top:1rem;font-size:0.8rem;color:#7c8aaa">
                            Run: <code>python -m src.train --model {model_name}</code>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

    st.markdown("<div style='margin-top:2rem'></div>", unsafe_allow_html=True)

    # ── About section ─────────────────────────────────────────────────────────
    with st.expander("📖  About", expanded=False):
        st.markdown(
            """
            **DigitVision** compares three neural architectures on MNIST under identical
            training conditions — same data, same optimiser, same evaluation suite.
            The goal is to isolate the effect of architecture on accuracy and
            parameter efficiency.

            **Stack:** TensorFlow · Streamlit · Plotly · scikit-learn · OpenCV

            Three architectures compared honestly, Grad-CAM explainability, full
            evaluation pipeline (confusion matrix, per-class F1, ROC), Docker,
            CI/CD, and 145 unit tests.
            """
        )
