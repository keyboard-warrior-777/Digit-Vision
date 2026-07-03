"""
Reusable Plotly chart builders for DigitVision.

All charts share a consistent dark theme defined by PLOTLY_LAYOUT_DEFAULTS.
Each function accepts raw data and returns a ``plotly.graph_objects.Figure``
ready to be passed to ``st.plotly_chart()``.

Design:
    Charts are built separately from page logic. A page imports the chart
    function it needs, provides data, and renders the result. This means
    charts are independently testable and reusable across pages.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots


# ─── Shared theme ────────────────────────────────────────────────────────────
# Applied to every figure via fig.update_layout(**PLOTLY_LAYOUT_DEFAULTS)

PLOTLY_LAYOUT_DEFAULTS: dict = {
    "paper_bgcolor": "#1a1d2e",
    "plot_bgcolor": "#1a1d2e",
    "font": {
        "family": "Inter, -apple-system, sans-serif",
        "color": "#e2e8f0",
        "size": 13,
    },
    "legend": {
        "bgcolor": "rgba(26,29,46,0.8)",
        "bordercolor": "#2d3154",
        "borderwidth": 1,
    },
    "margin": {"l": 50, "r": 20, "t": 50, "b": 50},
    "hoverlabel": {
        "bgcolor": "#252842",
        "bordercolor": "#6366f1",
        "font_color": "#e2e8f0",
    },
}

# Base axis style — merged into per-chart axis dicts
AXIS_STYLE: dict = {
    "gridcolor": "#2d3154",
    "linecolor": "#2d3154",
    "tickcolor": "#64748b",
}

# Model colour palette — consistent across all charts
MODEL_COLOURS: dict[str, str] = {
    "dense_nn": "#60a5fa",   # blue
    "lenet5": "#f59e0b",     # amber
    "custom_cnn": "#4ade80", # green
}

MODEL_LABELS: dict[str, str] = {
    "dense_nn": "Dense NN",
    "lenet5": "LeNet-5",
    "custom_cnn": "Custom CNN",
}


# ─── Confidence chart ─────────────────────────────────────────────────────────

def build_confidence_bar_chart(
    probabilities: dict[str, float],
    predicted_digit: int,
) -> go.Figure:
    """
    Build a horizontal bar chart showing prediction confidence for all 10 classes.

    The predicted class bar is highlighted in indigo. All other bars use a
    muted colour. This immediately communicates both the prediction and the
    model's uncertainty about other classes.

    Args:
        probabilities: Dict mapping digit label → probability (e.g., {'0': 0.02, '1': 0.95, ...}).
        predicted_digit: The argmax class index (0–9).

    Returns:
        Plotly Figure with horizontal bars ordered by digit class.
    """
    digits = list(probabilities.keys())
    scores = list(probabilities.values())

    bar_colours = [
        "#6366f1" if int(d) == predicted_digit else "#2d3154"
        for d in digits
    ]

    fig = go.Figure(
        go.Bar(
            x=scores,
            y=digits,
            orientation="h",
            marker_color=bar_colours,
            marker_line_color="#0f1117",
            marker_line_width=1,
            text=[f"{s:.1%}" for s in scores],
            textposition="outside",
            textfont={"size": 11, "color": "#94a3b8"},
            hovertemplate="Digit %{y}: %{x:.2%}<extra></extra>",
        )
    )

    fig.update_layout(
        **PLOTLY_LAYOUT_DEFAULTS,
        title={"text": "Confidence by Class", "font": {"size": 14}},
        xaxis={"range": [0, 1.15], "tickformat": ".0%", **AXIS_STYLE},
        yaxis=AXIS_STYLE,
        height=320,
        showlegend=False,
    )

    return fig


# ─── Training curves ─────────────────────────────────────────────────────────

def build_training_curves_chart(
    histories: dict[str, dict[str, list[float]]],
    metric: str = "accuracy",
) -> go.Figure:
    """
    Build a line chart of training and validation curves for all loaded models.

    Args:
        histories: Dict mapping model_name → history dict
                   (keys: 'accuracy', 'val_accuracy', 'loss', 'val_loss').
        metric: 'accuracy' or 'loss'.

    Returns:
        Plotly Figure with one line per model.
    """
    val_key = f"val_{metric}"
    y_title = "Accuracy" if metric == "accuracy" else "Loss"

    fig = go.Figure()

    for model_name, history in histories.items():
        if val_key not in history:
            continue

        colour = MODEL_COLOURS.get(model_name, "#ffffff")
        label = MODEL_LABELS.get(model_name, model_name)
        epochs = list(range(1, len(history[val_key]) + 1))

        # Training line (dashed, same colour, lower opacity)
        if metric in history:
            fig.add_trace(
                go.Scatter(
                    x=epochs,
                    y=history[metric],
                    name=f"{label} (train)",
                    line={"color": colour, "width": 1.5, "dash": "dot"},
                    opacity=0.5,
                    hovertemplate=f"{label} train: %{{y:.4f}}<extra></extra>",
                )
            )

        # Validation line (solid)
        fig.add_trace(
            go.Scatter(
                x=epochs,
                y=history[val_key],
                name=f"{label} (val)",
                line={"color": colour, "width": 2.5},
                hovertemplate=f"{label} val: %{{y:.4f}}<extra></extra>",
            )
        )

    fig.update_layout(
        **PLOTLY_LAYOUT_DEFAULTS,
        title={"text": f"Validation {y_title} over Epochs", "font": {"size": 15}},
        xaxis={"title": "Epoch", **AXIS_STYLE},
        yaxis={"title": y_title, **AXIS_STYLE},
        height=400,
        hovermode="x unified",
    )

    return fig


# ─── Confusion matrix ─────────────────────────────────────────────────────────

def build_confusion_matrix_chart(
    confusion_matrix: np.ndarray,
    class_labels: list[str],
) -> go.Figure:
    """
    Build an interactive confusion matrix heatmap.

    Hover tooltips show both raw counts and the row-normalised percentage,
    making it easy to see both volume and proportion of errors.

    Args:
        confusion_matrix: Raw integer count matrix, shape (10, 10).
        class_labels: Labels for each class (e.g., ['0', '1', ..., '9']).

    Returns:
        Plotly Figure with an annotated heatmap.
    """
    # Normalise row-wise to get proportions for colour scale
    row_sums = confusion_matrix.sum(axis=1, keepdims=True)
    cm_normalised = np.where(row_sums > 0, confusion_matrix / row_sums, 0)

    # Custom hover text: "Predicted 4 | True 9\n42 samples (8.4%)"
    hover_text = [
        [
            f"True: {class_labels[r]}, Predicted: {class_labels[c]}<br>"
            f"Count: {confusion_matrix[r, c]}<br>"
            f"Rate: {cm_normalised[r, c]:.1%}"
            for c in range(len(class_labels))
        ]
        for r in range(len(class_labels))
    ]

    fig = go.Figure(
        go.Heatmap(
            z=cm_normalised,
            x=class_labels,
            y=class_labels,
            text=[[f"{confusion_matrix[r, c]}" for c in range(10)] for r in range(10)],
            texttemplate="%{text}",
            hovertext=hover_text,
            hoverinfo="text",
            colorscale=[
                [0.0, "#1a1d2e"],
                [0.3, "#2d3154"],
                [0.7, "#4f46e5"],
                [1.0, "#6366f1"],
            ],
            showscale=True,
            colorbar={"title": "Rate", "tickformat": ".0%"},
        )
    )

    fig.update_layout(
        **PLOTLY_LAYOUT_DEFAULTS,
        title={"text": "Confusion Matrix (count annotated, colour = row rate)", "font": {"size": 14}},
        xaxis={"title": "Predicted", **AXIS_STYLE},
        yaxis={"title": "True Label", "autorange": "reversed", **AXIS_STYLE},
        height=520,
    )

    return fig


# ─── Per-class F1 chart ───────────────────────────────────────────────────────

def build_f1_bar_chart(
    f1_scores: dict[str, float],
    model_display_name: str = "",
) -> go.Figure:
    """
    Build a horizontal bar chart of per-class F1 scores.

    Colour encodes performance tier: green ≥ 0.99, amber ≥ 0.97, red < 0.97.

    Args:
        f1_scores: Dict mapping digit label → F1 score.
        model_display_name: Used in the chart title.

    Returns:
        Plotly Figure.
    """
    digits = sorted(f1_scores.keys(), key=int)
    scores = [f1_scores[d] for d in digits]
    colours = [
        "#4ade80" if s >= 0.990 else "#fbbf24" if s >= 0.970 else "#f87171"
        for s in scores
    ]

    fig = go.Figure(
        go.Bar(
            x=scores,
            y=digits,
            orientation="h",
            marker_color=colours,
            marker_line_width=0,
            text=[f"{s:.4f}" for s in scores],
            textposition="outside",
            textfont={"size": 11},
            hovertemplate="Digit %{y}: F1 = %{x:.4f}<extra></extra>",
        )
    )

    title = f"Per-Class F1 Score — {model_display_name}" if model_display_name else "Per-Class F1 Score"
    fig.update_layout(
        **PLOTLY_LAYOUT_DEFAULTS,
        title={"text": title, "font": {"size": 14}},
        xaxis={"range": [0.92, 1.01], **AXIS_STYLE},
        yaxis=AXIS_STYLE,
        height=380,
        showlegend=False,
    )

    return fig


# ─── Model accuracy comparison ────────────────────────────────────────────────

def build_accuracy_comparison_chart(
    model_accuracies: dict[str, float],
) -> go.Figure:
    """
    Build a grouped bar chart comparing test accuracy across all three models.

    Args:
        model_accuracies: Dict mapping model_name → test accuracy float.

    Returns:
        Plotly Figure.
    """
    names = [MODEL_LABELS.get(k, k) for k in model_accuracies]
    values = list(model_accuracies.values())
    colours = [MODEL_COLOURS.get(k, "#94a3b8") for k in model_accuracies]

    fig = go.Figure(
        go.Bar(
            x=names,
            y=values,
            marker_color=colours,
            marker_line_width=0,
            text=[f"{v:.2%}" for v in values],
            textposition="outside",
            textfont={"size": 13, "color": "#e2e8f0"},
            hovertemplate="%{x}: %{y:.4%}<extra></extra>",
        )
    )

    y_min = max(0, min(values) - 0.02)
    fig.update_layout(
        **PLOTLY_LAYOUT_DEFAULTS,
        title={"text": "Test Accuracy Comparison", "font": {"size": 14}},
        xaxis=AXIS_STYLE,
        yaxis={"range": [y_min, 1.0], "tickformat": ".1%", **AXIS_STYLE},
        height=360,
        showlegend=False,
    )

    return fig


# ─── ROC chart ────────────────────────────────────────────────────────────────

def build_roc_chart(
    roc_data: dict[str, dict],
    selected_classes: Optional[list[str]] = None,
) -> go.Figure:
    """
    Build a multi-line ROC chart (one line per selected digit class).

    Args:
        roc_data: Dict mapping digit label → {'fpr': [...], 'tpr': [...], 'auc': float}.
        selected_classes: List of digit labels to plot. Defaults to all 10.

    Returns:
        Plotly Figure.
    """
    classes_to_plot = selected_classes or sorted(roc_data.keys(), key=int)

    # Colour palette for 10 classes
    palette = px.colors.qualitative.Plotly

    fig = go.Figure()

    # Diagonal reference line (random classifier)
    fig.add_trace(
        go.Scatter(
            x=[0, 1], y=[0, 1],
            mode="lines",
            line={"color": "#2d3154", "dash": "dash", "width": 1.5},
            name="Random (AUC = 0.50)",
            hoverinfo="skip",
        )
    )

    for i, digit in enumerate(classes_to_plot):
        if digit not in roc_data:
            continue
        entry = roc_data[digit]
        colour = palette[int(digit) % len(palette)]

        fig.add_trace(
            go.Scatter(
                x=entry["fpr"],
                y=entry["tpr"],
                mode="lines",
                name=f"Digit {digit} (AUC={entry['auc']:.3f})",
                line={"color": colour, "width": 2},
                hovertemplate=f"Digit {digit} — FPR: %{{x:.3f}}, TPR: %{{y:.3f}}<extra></extra>",
            )
        )

    fig.update_layout(
        **PLOTLY_LAYOUT_DEFAULTS,
        title={"text": "ROC Curves — One-vs-Rest (per digit class)", "font": {"size": 14}},
        xaxis={"title": "False Positive Rate", "range": [0, 1], **AXIS_STYLE},
        yaxis={"title": "True Positive Rate", "range": [0, 1.02], **AXIS_STYLE},
        height=480,
    )

    return fig


# ─── Dataset distribution ─────────────────────────────────────────────────────

def build_class_distribution_chart(
    class_counts: dict[str, int],
    title: str = "Class Distribution",
) -> go.Figure:
    """
    Build a bar chart of sample counts per digit class.

    Args:
        class_counts: Dict mapping digit label → sample count.
        title: Chart title.

    Returns:
        Plotly Figure.
    """
    digits = sorted(class_counts.keys(), key=int)
    counts = [class_counts[d] for d in digits]
    mean_count = np.mean(counts)

    colours = [
        "#6366f1" if abs(c - mean_count) / mean_count < 0.03 else "#f59e0b"
        for c in counts
    ]

    fig = go.Figure(
        go.Bar(
            x=digits,
            y=counts,
            marker_color=colours,
            marker_line_width=0,
            text=counts,
            textposition="outside",
            hovertemplate="Digit %{x}: %{y:,} samples<extra></extra>",
        )
    )

    fig.add_hline(
        y=mean_count,
        line_dash="dot",
        line_color="#94a3b8",
        annotation_text=f"Mean: {mean_count:.0f}",
        annotation_font_color="#94a3b8",
    )

    fig.update_layout(
        **PLOTLY_LAYOUT_DEFAULTS,
        title={"text": title, "font": {"size": 14}},
        xaxis={"title": "Digit Class", **AXIS_STYLE},
        yaxis={"title": "Sample Count", **AXIS_STYLE},
        height=360,
        showlegend=False,
    )

    return fig


# ─── Grouped metric comparison (Model Comparison tab) ─────────────────────────

def build_grouped_metric_comparison_chart(
    compare_data: dict[str, dict],
) -> go.Figure:
    """
    Build a grouped bar chart comparing Accuracy, Macro F1, Precision and Recall
    across all models simultaneously.

    Args:
        compare_data: Dict mapping model_name → metric dict with keys
                      'accuracy', 'macro_f1', 'precision', 'recall'.

    Returns:
        Plotly Figure.
    """
    metrics = [
        ("accuracy",  "Accuracy"),
        ("macro_f1",  "Macro F1"),
        ("precision", "Macro Precision"),
        ("recall",    "Macro Recall"),
    ]

    fig = go.Figure()

    for model_name, d in compare_data.items():
        colour = MODEL_COLOURS.get(model_name, "#94a3b8")
        label  = MODEL_LABELS.get(model_name, model_name)
        fig.add_trace(
            go.Bar(
                name=label,
                x=[m_label for _, m_label in metrics],
                y=[d.get(m_key, 0) for m_key, _ in metrics],
                marker_color=colour,
                marker_line_width=0,
                text=[f"{d.get(m_key, 0):.3f}" for m_key, _ in metrics],
                textposition="outside",
                textfont={"size": 11},
                hovertemplate=f"{label} — %{{x}}: %{{y:.4f}}<extra></extra>",
            )
        )

    fig.update_layout(
        **PLOTLY_LAYOUT_DEFAULTS,
        title={"text": "Metric Comparison — All Models", "font": {"size": 14}},
        barmode="group",
        xaxis=AXIS_STYLE,
        yaxis={"range": [0.97, 1.005], "tickformat": ".2%", **AXIS_STYLE},
        height=400,
    )

    return fig


# ─── Radar chart (Model Comparison tab) ──────────────────────────────────────

def build_radar_chart(
    compare_data: dict[str, dict],
) -> go.Figure:
    """
    Build a radar / spider chart visualising five model metrics simultaneously.

    Each axis runs from 0 to 1. The chart makes architectural trade-offs
    immediately visible (e.g. one model may have higher recall but lower
    precision).

    Args:
        compare_data: Dict mapping model_name → metric dict.

    Returns:
        Plotly Figure.
    """
    dimensions = ["accuracy", "macro_f1", "precision", "recall", "weighted_f1"]
    dim_labels = ["Accuracy", "Macro F1", "Precision", "Recall", "Weighted F1"]

    fig = go.Figure()

    for model_name, d in compare_data.items():
        colour = MODEL_COLOURS.get(model_name, "#94a3b8")
        label  = MODEL_LABELS.get(model_name, model_name)
        values = [d.get(k, 0) for k in dimensions]
        # Close the polygon
        values_closed = values + [values[0]]
        labels_closed = dim_labels + [dim_labels[0]]

        fig.add_trace(
            go.Scatterpolar(
                r=values_closed,
                theta=labels_closed,
                name=label,
                line={"color": colour, "width": 2.5},
                fill="toself",
                fillcolor=f"rgba({int(colour[1:3],16)},{int(colour[3:5],16)},{int(colour[5:7],16)},0.12)",
                hovertemplate=f"{label}<br>%{{theta}}: %{{r:.4f}}<extra></extra>",
            )
        )

    fig.update_layout(
        **PLOTLY_LAYOUT_DEFAULTS,
        title={"text": "Architecture Performance Radar", "font": {"size": 14}},
        polar={
            "bgcolor": "#1a1d2e",
            "radialaxis": {
                "visible": True,
                "range": [0.97, 1.0],
                "tickformat": ".2%",
                "gridcolor": "#2d3154",
                "linecolor": "#2d3154",
            },
            "angularaxis": {
                "gridcolor": "#2d3154",
                "linecolor": "#2d3154",
            },
        },
        height=420,
    )

    return fig
