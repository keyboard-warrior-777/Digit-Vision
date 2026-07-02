"""
Analytics Dashboard — DigitVision.

Interactive Plotly charts for every aspect of model evaluation.
All data is read from JSON/NPY files generated during evaluate.py —
no models are loaded, no computation happens at render time.

Pages:
    Tab 1 — Training Curves (accuracy + loss, all models overlaid)
    Tab 2 — Confusion Matrix (interactive heatmap)
    Tab 3 — Per-Class F1 (bar chart with performance tiers)
    Tab 4 — ROC Curves (one-vs-rest for all 10 digit classes)
    Tab 5 — Accuracy Comparison (bar chart, all three models)
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import streamlit as st

from components.cards import info_box, page_header, status_badge
from components.charts import (
    build_accuracy_comparison_chart,
    build_class_distribution_chart,
    build_confusion_matrix_chart,
    build_f1_bar_chart,
    build_roc_chart,
    build_training_curves_chart,
)
from components.styles import get_global_css

from config.config import (
    AVAILABLE_MODELS,
    CLASS_NAMES,
    CONFUSION_MATRIX_PATHS,
    HISTORY_PATHS,
    METADATA_PATHS,
    MODEL_DISPLAY_NAMES,
    MODEL_PATHS,
    RAW_METRICS_PATHS,
    ROC_DATA_PATHS,
)


# ── Data loaders (cached to avoid re-reading files on every rerun) ────────────

@st.cache_data
def _load_history(model_name: str) -> dict | None:
    path = HISTORY_PATHS.get(model_name)
    if path and path.exists():
        with path.open() as f:
            return json.load(f)
    return None


@st.cache_data
def _load_metadata(model_name: str) -> dict | None:
    path = METADATA_PATHS.get(model_name)
    if path and path.exists():
        with path.open() as f:
            return json.load(f)
    return None


@st.cache_data
def _load_confusion_matrix(model_name: str) -> np.ndarray | None:
    path = CONFUSION_MATRIX_PATHS.get(model_name)
    if path and path.exists():
        return np.load(path)
    return None


@st.cache_data
def _load_raw_metrics(model_name: str) -> dict | None:
    path = RAW_METRICS_PATHS.get(model_name)
    if path and path.exists():
        with path.open() as f:
            return json.load(f)
    return None


@st.cache_data
def _load_roc_data(model_name: str) -> dict | None:
    path = ROC_DATA_PATHS.get(model_name)
    if path and path.exists():
        with path.open() as f:
            return json.load(f)
    return None


# ─────────────────────────────────────────────────────────────────────────────

st.markdown(
    page_header(
        "Analytics Dashboard",
        "Interactive evaluation metrics — confusion matrix, ROC curves, training curves",
        "📊",
    ),
    unsafe_allow_html=True,
)

trained_models = [n for n in AVAILABLE_MODELS if MODEL_PATHS[n].exists()]

if not trained_models:
    st.warning("No trained models found. Run `make train` to begin.")
    st.stop()

# ── Model filter selector ─────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🔧 Filters")
    selected_model = st.selectbox(
        "Focus model",
        options=trained_models,
        format_func=lambda n: MODEL_DISPLAY_NAMES[n],
        help="Charts that show a single model use this selection.",
    )
    st.markdown("<hr style='border-color:#2d3154'>", unsafe_allow_html=True)

    # Quick accuracy summary
    st.markdown("### 🏆 Accuracy")
    for name in trained_models:
        meta = _load_metadata(name)
        if meta:
            acc = meta.get("test_accuracy", 0)
            st.markdown(
                f"**{MODEL_DISPLAY_NAMES[name]}**  \n`{acc:.2%}`"
            )

display_name = MODEL_DISPLAY_NAMES[selected_model]

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab_curves, tab_cm, tab_f1, tab_roc, tab_compare = st.tabs([
    "📈 Training Curves",
    "🔲 Confusion Matrix",
    "📏 Per-Class F1",
    "📉 ROC Curves",
    "⚖️ Model Comparison",
])

# ════════ TAB 1 — Training Curves ════════════════════════════════════════════
with tab_curves:
    histories = {
        name: _load_history(name)
        for name in trained_models
        if _load_history(name) is not None
    }

    if not histories:
        st.info("No training history files found. Ensure training completed successfully.")
    else:
        metric_choice = st.radio(
            "Metric", ["accuracy", "loss"], horizontal=True, label_visibility="collapsed"
        )
        st.plotly_chart(
            build_training_curves_chart(histories, metric=metric_choice),
            use_container_width=True,
        )
        st.markdown(
            info_box(
                "Solid lines = validation · Dotted lines = training. "
                "A large gap between training and validation indicates overfitting."
            ),
            unsafe_allow_html=True,
        )

# ════════ TAB 2 — Confusion Matrix ═══════════════════════════════════════════
with tab_cm:
    cm = _load_confusion_matrix(selected_model)

    if cm is None:
        st.info(
            f"No confusion matrix data for {display_name}. "
            "Run `python -m src.evaluate --all` to generate evaluation artifacts."
        )
    else:
        st.plotly_chart(
            build_confusion_matrix_chart(cm, CLASS_NAMES),
            use_container_width=True,
        )

        # Worst error pairs
        np.fill_diagonal(cm, 0)
        flat_idx = np.argpartition(cm.flatten(), -5)[-5:]
        top_pairs = [(np.unravel_index(i, cm.shape), cm.flatten()[i]) for i in flat_idx]
        top_pairs.sort(key=lambda x: -x[1])

        st.markdown(
            "<div class='dv-section-header'>Top Confusion Pairs</div>",
            unsafe_allow_html=True,
        )
        rows = "".join(
            f"<tr><td>True <strong>{r}</strong> → Predicted <strong>{c}</strong></td>"
            f"<td style='text-align:right'>{count} errors</td></tr>"
            for (r, c), count in top_pairs
            if count > 0
        )
        st.markdown(
            f"<table class='dv-table'><tbody>{rows}</tbody></table>",
            unsafe_allow_html=True,
        )

# ════════ TAB 3 — Per-Class F1 ═══════════════════════════════════════════════
with tab_f1:
    raw_metrics = _load_raw_metrics(selected_model)

    if raw_metrics is None:
        st.info(
            f"No per-class metrics for {display_name}. "
            "Run `python -m src.evaluate --all`."
        )
    else:
        # Extract per-class F1 from sklearn report dict
        f1_scores = {
            label: raw_metrics[label]["f1-score"]
            for label in CLASS_NAMES
            if label in raw_metrics
        }

        st.plotly_chart(
            build_f1_bar_chart(f1_scores, model_display_name=display_name),
            use_container_width=True,
        )

        # Summary metrics row
        macro_f1 = raw_metrics.get("macro avg", {}).get("f1-score", 0)
        weighted_f1 = raw_metrics.get("weighted avg", {}).get("f1-score", 0)

        col1, col2 = st.columns(2)
        col1.metric("Macro F1", f"{macro_f1:.4f}")
        col2.metric("Weighted F1", f"{weighted_f1:.4f}")

        st.markdown(
            info_box(
                "<strong>Green</strong> ≥ 0.990 · <strong style='color:#fbbf24'>Amber</strong> ≥ 0.970 · "
                "<strong style='color:#f87171'>Red</strong> &lt; 0.970. "
                "Digits 4↔9 and 3↔8 are classically hard due to similar stroke patterns."
            ),
            unsafe_allow_html=True,
        )

# ════════ TAB 4 — ROC Curves ═════════════════════════════════════════════════
with tab_roc:
    roc_data = _load_roc_data(selected_model)

    if roc_data is None:
        st.info(
            f"No ROC data for {display_name}. "
            "Run `python -m src.evaluate --all`."
        )
    else:
        selected_roc_classes = st.multiselect(
            "Filter digit classes",
            options=CLASS_NAMES,
            default=CLASS_NAMES,
            label_visibility="collapsed",
        )

        st.plotly_chart(
            build_roc_chart(roc_data, selected_classes=selected_roc_classes or CLASS_NAMES),
            use_container_width=True,
        )

        # AUC table
        auc_rows = "".join(
            f"<tr><td>Digit {digit}</td><td style='text-align:right'>{roc_data[digit]['auc']:.5f}</td></tr>"
            for digit in sorted(roc_data.keys(), key=int)
        )
        with st.expander("AUC Scores by Class"):
            st.markdown(
                f"<table class='dv-table'><thead><tr><th>Class</th><th style='text-align:right'>AUC</th></tr></thead>"
                f"<tbody>{auc_rows}</tbody></table>",
                unsafe_allow_html=True,
            )

# ════════ TAB 5 — Model Comparison ═══════════════════════════════════════════
with tab_compare:
    model_accuracies = {}
    model_f1s = {}

    for name in trained_models:
        meta = _load_metadata(name)
        if meta:
            model_accuracies[name] = meta["test_accuracy"]
            model_f1s[name] = meta.get("macro_f1", 0)

    if not model_accuracies:
        st.info("No metadata available. Run evaluation first.")
    else:
        col_acc, col_f1 = st.columns(2)
        with col_acc:
            st.plotly_chart(
                build_accuracy_comparison_chart(model_accuracies),
                use_container_width=True,
            )
        with col_f1:
            st.plotly_chart(
                build_accuracy_comparison_chart(model_f1s),
                use_container_width=True,
            )
            # Re-title the second chart manually
            st.caption("↑ Macro F1 Comparison (same scale)")

        # Full comparison table
        st.markdown(
            "<div class='dv-section-header'>Full Comparison Table</div>",
            unsafe_allow_html=True,
        )
        header = "<tr><th>Model</th><th>Accuracy</th><th>Macro F1</th><th>Parameters</th><th>Training Time</th></tr>"
        rows_html = ""
        for name in trained_models:
            meta = _load_metadata(name)
            if meta:
                rows_html += (
                    f"<tr>"
                    f"<td><strong>{MODEL_DISPLAY_NAMES[name]}</strong></td>"
                    f"<td>{meta['test_accuracy']:.2%}</td>"
                    f"<td>{meta.get('macro_f1', 0):.2%}</td>"
                    f"<td>{meta['total_parameters']:,}</td>"
                    f"<td>{meta['training_time_seconds'] / 60:.1f} min</td>"
                    f"</tr>"
                )

        st.markdown(
            f"<table class='dv-table'><thead>{header}</thead><tbody>{rows_html}</tbody></table>",
            unsafe_allow_html=True,
        )
