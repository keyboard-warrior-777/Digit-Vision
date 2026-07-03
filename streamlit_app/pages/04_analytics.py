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
    build_class_distribution_chart,
    build_confusion_matrix_chart,
    build_f1_bar_chart,
    build_roc_chart,
    build_training_curves_chart,
)

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
    SAMPLE_PREDICTIONS_PATHS,
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


@st.cache_data
def _load_sample_predictions(model_name: str) -> list | None:
    """Load the sample prediction records generated during evaluation."""
    path = SAMPLE_PREDICTIONS_PATHS.get(model_name)
    if path and path.exists():
        with path.open() as f:
            return json.load(f)
    return None

# ─────────────────────────────────────────────────────────────────────────────

st.markdown(
    page_header(
        "Analytics",
        "Training curves, confusion matrix, per-class F1, ROC, and model comparison",
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
    st.markdown(
        "<div style='font-size:0.72rem;font-weight:700;text-transform:uppercase;"
        "letter-spacing:0.1em;color:var(--text-muted);padding:0.5rem 0 0.75rem'>&#9001; Filters</div>",
        unsafe_allow_html=True,
    )
    selected_model = st.selectbox(
        "Model",
        options=trained_models,
        format_func=lambda n: MODEL_DISPLAY_NAMES[n],
        help="Charts that show a single model use this selection.",
    )
    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

    st.markdown(
        "<div style='font-size:0.72rem;font-weight:700;text-transform:uppercase;"
        "letter-spacing:0.1em;color:var(--text-muted);padding:0.5rem 0 0.5rem'>&#9001; Results</div>",
        unsafe_allow_html=True,
    )
    for name in trained_models:
        raw = _load_raw_metrics(name)
        if raw:
            acc = raw.get("accuracy", 0)
            macro_f1 = raw.get("macro avg", {}).get("f1-score", 0)
            st.markdown(
                f"**{MODEL_DISPLAY_NAMES[name]}**  \n`{acc:.2%}`  ·  F1 `{macro_f1:.4f}`"
            )

display_name = MODEL_DISPLAY_NAMES[selected_model]

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab_curves, tab_cm, tab_f1, tab_roc, tab_compare = st.tabs([
    "📈 Training Curves",
    "🔲 Confusion Matrix",
    "📊 Per-Class F1",
    "📉 ROC Curves",
    "⚖️ Model Comparison",
])

# ════════ TAB 1 — Training Curves ═════════════════════════════════════════════
with tab_curves:
    # Call _load_history() once per model, not twice (was double-calling before)
    histories = {
        name: hist
        for name in trained_models
        if (hist := _load_history(name)) is not None
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

        # ── Training Curve Summary ──────────────────────────────────────────
        st.markdown(
            "<div class='dv-section-header'>Training Summary</div>",
            unsafe_allow_html=True,
        )
        from components.charts import MODEL_LABELS
        summary_rows = ""
        for name, hist in histories.items():
            val_acc = hist.get("val_accuracy", [])
            if not val_acc:
                continue
            best_val = max(val_acc)
            best_epoch = val_acc.index(best_val) + 1
            final_val = val_acc[-1]
            summary_rows += (
                f"<tr>"
                f"<td><strong>{MODEL_LABELS.get(name, name)}</strong></td>"
                f"<td>{best_val:.4f} (epoch {best_epoch})</td>"
                f"<td>{final_val:.4f}</td>"
                f"<td>{len(val_acc)}</td>"
                f"</tr>"
            )
        header_html = (
            "<tr><th>Model</th><th>Best Val Accuracy</th>"
            "<th>Final Val Accuracy</th><th>Total Epochs</th></tr>"
        )
        st.markdown(
            f"<table class='dv-table'><thead>{header_html}</thead><tbody>{summary_rows}</tbody></table>",
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
        # Raw vs Percentage toggle
        cm_mode = st.radio(
            "Display mode",
            ["Raw Counts", "Percentages"],
            horizontal=True,
            label_visibility="collapsed",
        )

        # Build chart with appropriate normalisation
        cm_display = cm.copy().astype(float)
        if cm_mode == "Percentages":
            row_sums = cm_display.sum(axis=1, keepdims=True)
            cm_display = np.where(row_sums > 0, cm_display / row_sums * 100, 0)

        st.plotly_chart(
            build_confusion_matrix_chart(
                cm.astype(int),  # always pass raw ints for hover annotation
                CLASS_NAMES,
            ),
            use_container_width=True,
        )

        # Worst error pairs — zero diagonal first
        cm_off = cm.copy()
        np.fill_diagonal(cm_off, 0)
        flat_idx = np.argpartition(cm_off.flatten(), -5)[-5:]
        top_pairs = [(np.unravel_index(i, cm_off.shape), cm_off.flatten()[i]) for i in flat_idx]
        top_pairs.sort(key=lambda x: -x[1])

        # Auto-generated plain-English interpretation
        if top_pairs and top_pairs[0][1] > 0:
            worst_true, worst_pred = top_pairs[0][0]
            worst_count = top_pairs[0][1]
            st.markdown(
                info_box(
                    f"ℹ️ Most common error: True <strong>{worst_true}</strong> misclassified as "
                    f"<strong>{worst_pred}</strong> — {int(worst_count)} samples. "
                    "Digits 4↔9, 3↔8, and 5↔3 are classically hard due to similar stroke geometry."
                ),
                unsafe_allow_html=True,
            )

        st.markdown(
            "<div class='dv-section-header'>Top Confusion Pairs</div>",
            unsafe_allow_html=True,
        )
        rows = "".join(
            f"<tr><td>True <strong>{r}</strong> → Predicted <strong>{c}</strong></td>"
            f"<td style='text-align:right'>{int(count)} errors</td></tr>"
            for (r, c), count in top_pairs
            if count > 0
        )
        st.markdown(
            f"<table class='dv-table'><tbody>{rows}</tbody></table>",
            unsafe_allow_html=True,
        )

        # ── Wrong Prediction Explorer (with images) ───────────────────────────
        preds = _load_sample_predictions(selected_model)
        wrong = [p for p in (preds or []) if not p["is_correct"]]
        if wrong:
            with st.expander(f"🔍 Wrong Prediction Explorer — {len(wrong)} samples"):
                img_cols = st.columns(min(len(wrong), 5))
                for col, p in zip(img_cols, wrong[:5], strict=False):
                    with col:
                        img_path = Path(p["image_path"])
                        if img_path.exists():
                            st.image(
                                str(img_path),
                                caption=(
                                    f"True: {p['true_label']}\n"
                                    f"Pred: {p['predicted_label']}\n"
                                    f"Conf: {p['confidence']:.1%}"
                                ),
                                use_column_width="always",
                            )
                        else:
                            st.markdown(
                                f"True **{p['true_label']}** → "
                                f"Pred **{p['predicted_label']}** "
                                f"({p['confidence']:.1%})"
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

        # ── Best / Worst class tables ───────────────────────────────────────
        sorted_f1 = sorted(f1_scores.items(), key=lambda x: -x[1])
        best3 = sorted_f1[:3]
        worst3 = sorted_f1[-3:][::-1]

        col_best, col_worst = st.columns(2)
        with col_best:
            st.markdown(
                "<div class='dv-section-header' style='color:#4ade80'>Top 3 Digits</div>",
                unsafe_allow_html=True,
            )
            for digit, score in best3:
                st.markdown(
                    f"<span style='color:#4ade80;font-weight:700'>→ Digit {digit}</span> — F1: `{score:.4f}`",
                    unsafe_allow_html=True,
                )
        with col_worst:
            st.markdown(
                "<div class='dv-section-header' style='color:#f87171'>Bottom 3 Digits</div>",
                unsafe_allow_html=True,
            )
            for digit, score in worst3:
                st.markdown(
                    f"<span style='color:#f87171;font-weight:700'>→ Digit {digit}</span> — F1: `{score:.4f}`",
                    unsafe_allow_html=True,
                )

        # ── Class support bar chart ──────────────────────────────────────────
        support = {
            label: int(raw_metrics[label]["support"])
            for label in CLASS_NAMES
            if label in raw_metrics
        }
        if support:
            with st.expander("📊 Class Support (test set sample counts)"):
                st.plotly_chart(
                    build_class_distribution_chart(support, title="Test Set Samples per Digit"),
                    use_container_width=True,
                )

        # ── Summary metrics ─────────────────────────────────────────────────
        macro_f1 = raw_metrics.get("macro avg", {}).get("f1-score", 0)
        weighted_f1 = raw_metrics.get("weighted avg", {}).get("f1-score", 0)

        col1, col2 = st.columns(2)
        col1.metric("Macro F1", f"{macro_f1:.4f}")
        col2.metric("Weighted F1", f"{weighted_f1:.4f}")

        # Metric interpretation
        if macro_f1 >= 0.990:
            interp = "Balanced across all digit classes."
            badge_status = "success"
        elif macro_f1 >= 0.985:
            interp = "Minor per-class variation."
            badge_status = "warning"
        else:
            interp = "Some digits harder than others — see chart above."
            badge_status = "error"
        st.markdown(
            status_badge(interp, badge_status),
            unsafe_allow_html=True,
        )

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
            "Digit classes",
            options=CLASS_NAMES,
            default=CLASS_NAMES,
        )

        st.plotly_chart(
            build_roc_chart(roc_data, selected_classes=selected_roc_classes or CLASS_NAMES),
            use_container_width=True,
        )

        # ── ROC Summary ──────────────────────────────────────────────────
        auc_values = [v["auc"] for v in roc_data.values()]
        if auc_values:
            mean_auc = float(np.mean(auc_values))
            std_auc = float(np.std(auc_values))
            best_digit = max(roc_data, key=lambda d: roc_data[d]["auc"])
            worst_digit = min(roc_data, key=lambda d: roc_data[d]["auc"])

            roc_c1, roc_c2, roc_c3, roc_c4 = st.columns(4)
            roc_c1.metric("Mean AUC", f"{mean_auc:.5f}")
            roc_c2.metric("Std Dev", f"{std_auc:.5f}")
            roc_c3.metric("Best Class", f"Digit {best_digit} ({roc_data[best_digit]['auc']:.4f})")
            roc_c4.metric("Worst Class", f"Digit {worst_digit} ({roc_data[worst_digit]['auc']:.4f})")

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
    # Build comparison dict from files that actually exist
    compare_data: dict[str, dict] = {}
    for name in trained_models:
        raw = _load_raw_metrics(name)
        hist = _load_history(name)
        roc = _load_roc_data(name)
        if raw:
            macro_avg = raw.get("macro avg", {})
            compare_data[name] = {
                "accuracy":    raw.get("accuracy", 0.0),
                "macro_f1":    macro_avg.get("f1-score", 0.0),
                "precision":   macro_avg.get("precision", 0.0),
                "recall":      macro_avg.get("recall", 0.0),
                "weighted_f1": raw.get("weighted avg", {}).get("f1-score", 0.0),
                "epochs": len(hist.get("val_accuracy", [])) if hist else None,
                "mean_auc": (
                    float(sum(v["auc"] for v in roc.values()) / len(roc))
                    if roc else None
                ),
            }

    if not compare_data:
        st.info(
            "No evaluation data found. Run `python -m src.evaluate --all` to generate artifacts."
        )
    else:
        # ── Ranking cards ────────────────────────────────────────────────────
        best_acc_model = max(compare_data, key=lambda n: compare_data[n]["accuracy"])
        rank_cols = st.columns(len(compare_data))
        for col, (name, d) in zip(rank_cols, compare_data.items(), strict=False):
            is_winner = name == best_acc_model
            border = "3px solid #fbbf24" if is_winner else "1px solid #2d3154"
            badge = "🏆 Best Accuracy" if is_winner else ""
            with col:
                st.markdown(
                    f"""
                    <div style='background:#1a1d2e;border:{border};border-radius:12px;
                                padding:1.25rem;text-align:center'>
                        <div style='font-size:0.75rem;color:#94a3b8;text-transform:uppercase;
                                    letter-spacing:0.08em;margin-bottom:0.5rem'>
                            {MODEL_DISPLAY_NAMES[name]}
                        </div>
                        <div style='font-size:2.5rem;font-weight:900;
                                    background:linear-gradient(135deg,#6366f1,#22d3ee);
                                    -webkit-background-clip:text;
                                    -webkit-text-fill-color:transparent'>
                            {d['accuracy']:.2%}
                        </div>
                        <div style='font-size:0.75rem;color:#64748b;margin-top:0.25rem'>Test Accuracy</div>
                        {f"<div style='margin-top:0.75rem;font-size:0.8rem;color:#fbbf24;font-weight:600'>{badge}</div>" if badge else ""}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        st.markdown("<div style='margin-top:1.5rem'></div>", unsafe_allow_html=True)

        # ── Grouped metric comparison chart ──────────────────────────────────
        from components.charts import (
            build_grouped_metric_comparison_chart,
            build_radar_chart,
        )
        st.plotly_chart(
            build_grouped_metric_comparison_chart(compare_data),
            use_container_width=True,
        )

        # ── Radar chart ──────────────────────────────────────────────────────
        st.plotly_chart(
            build_radar_chart(compare_data),
            use_container_width=True,
        )

        # ── Full comparison table ────────────────────────────────────────────
        st.markdown(
            "<div class='dv-section-header'>Full Metrics Table</div>",
            unsafe_allow_html=True,
        )
        header = (
            "<tr><th>Model</th><th>Accuracy</th><th>Macro F1</th>"
            "<th>Macro Precision</th><th>Macro Recall</th>"
            "<th>Weighted F1</th><th>Mean AUC</th><th>Epochs</th></tr>"
        )
        rows_html = ""
        for name, d in compare_data.items():
            auc_str = f"{d['mean_auc']:.5f}" if d["mean_auc"] is not None else "N/A"
            ep_str = str(d["epochs"]) if d["epochs"] is not None else "N/A"
            winner_style = " style='background:rgba(251,191,36,0.06)'" if name == best_acc_model else ""
            rows_html += (
                f"<tr{winner_style}>"
                f"<td><strong>{MODEL_DISPLAY_NAMES[name]}</strong>"
                f"{'&nbsp;🏆' if name == best_acc_model else ''}</td>"
                f"<td>{d['accuracy']:.2%}</td>"
                f"<td>{d['macro_f1']:.4f}</td>"
                f"<td>{d['precision']:.4f}</td>"
                f"<td>{d['recall']:.4f}</td>"
                f"<td>{d['weighted_f1']:.4f}</td>"
                f"<td>{auc_str}</td>"
                f"<td>{ep_str}</td>"
                f"</tr>"
            )
        st.markdown(
            f"<table class='dv-table'><thead>{header}</thead><tbody>{rows_html}</tbody></table>",
            unsafe_allow_html=True,
        )

        st.markdown(
            info_box(
                "<strong>Accuracy</strong> = overall correct predictions. "
                "<strong>Macro F1</strong> = unweighted mean across all 10 digit classes — "
                "better reflects performance on underrepresented digits. "
                "<strong>Mean AUC</strong> = average one-vs-rest ROC area across all classes. "
                "<strong>Epochs</strong> = best epoch saved by ModelCheckpoint (early stopping aware)."
            ),
            unsafe_allow_html=True,
        )
