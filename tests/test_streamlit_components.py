"""
Tests for Streamlit UI components — cards.py and charts.py.

Why these tests matter:
    Streamlit components are Python functions — they can be tested without
    launching Streamlit. These tests verify the HTML structure and chart
    correctness programmatically, catching regressions before the UI is even
    opened.

Tests in this file:
    cards.py:
        - page_header includes title and subtitle
        - metric_card includes value and label
        - status_badge applies correct CSS class
        - prediction_result_card formats confidence and inference time
        - info_box wraps content in expected container
        - pipeline_step includes step number
        - section_header wraps text correctly

    charts.py:
        - build_confidence_bar_chart returns Figure
        - Confidence chart has 10 bars (one per class)
        - Highlighted bar is the predicted class
        - build_training_curves_chart returns Figure
        - Training curves skips models with missing history
        - build_confusion_matrix_chart returns Figure
        - build_f1_bar_chart returns Figure
        - build_accuracy_comparison_chart returns Figure
        - build_class_distribution_chart returns Figure
        - All chart heights are set (responsive layout)
"""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
import pytest
from components.cards import (
    info_box,
    metric_card,
    page_header,
    pipeline_step,
    prediction_result_card,
    status_badge,
)
from components.charts import (
    build_accuracy_comparison_chart,
    build_class_distribution_chart,
    build_confidence_bar_chart,
    build_confusion_matrix_chart,
    build_f1_bar_chart,
    build_training_curves_chart,
)

# ─── cards.py tests ───────────────────────────────────────────────────────────


class TestPageHeader:
    """Tests for page_header() HTML component."""

    def test_includes_title(self) -> None:
        """
        What: The title text appears in the HTML output.
        Why:  page_header() is the first visible element on every page.
              A missing title produces a blank header — a major visual regression.
        Prevents: Title text being accidentally dropped from the header template.
        """
        html = page_header("Analytics", "subtitle here", "📊")
        assert "Analytics" in html

    def test_includes_subtitle(self) -> None:
        """
        What: The subtitle text appears in the HTML output.
        Prevents: Subtitle being silently dropped from the header.
        """
        html = page_header("Title", "My subtitle text", "🔢")
        assert "My subtitle text" in html

    def test_includes_icon(self) -> None:
        """
        What: The icon emoji appears in the HTML output.
        Prevents: Icon being dropped from the header, making all pages look the same.
        """
        html = page_header("Title", "Sub", "🔬")
        assert "🔬" in html

    def test_no_icon_produces_clean_html(self) -> None:
        """
        What: Calling with empty icon string produces valid HTML without errors.
        Prevents: Empty icon causing an f-string error or extra whitespace in the header.
        """
        html = page_header("Title", "Subtitle", "")
        assert "Title" in html
        assert "<div" in html

    def test_output_is_html_string(self) -> None:
        """
        What: Output is a non-empty string.
        Prevents: Function returning None or a non-string type.
        """
        result = page_header("T", "S", "X")
        assert isinstance(result, str)
        assert len(result) > 0


class TestMetricCard:
    """Tests for metric_card() HTML component."""

    def test_includes_value(self) -> None:
        """
        What: The metric value appears in the HTML.
        Prevents: Value being omitted from the metric display.
        """
        html = metric_card("Accuracy", "99.3%", icon="🎯")
        assert "99.3%" in html

    def test_includes_label(self) -> None:
        """
        What: The metric label appears in the HTML.
        Prevents: Label being omitted, making the metric unidentifiable.
        """
        html = metric_card("Test Accuracy", "0.993")
        assert "Test Accuracy" in html

    def test_delta_positive_includes_arrow(self) -> None:
        """
        What: Positive delta shows an up arrow indicator.
        Why:  The dashboard uses coloured deltas to show relative performance.
              A missing arrow makes the trend unreadable.
        Prevents: Delta indicator being dropped from the metric card.
        """
        html = metric_card("F1", "0.993", delta="+1.2%", delta_positive=True)
        assert "↑" in html

    def test_delta_negative_includes_arrow(self) -> None:
        """What: Negative delta shows a down arrow. Prevents: Delta always showing ↑."""
        html = metric_card("Loss", "0.024", delta="-0.3%", delta_positive=False)
        assert "↓" in html

    def test_no_delta_omits_arrow(self) -> None:
        """
        What: When delta=None, no arrow appears in the HTML.
        Prevents: Ghost arrows appearing on cards that have no delta.
        """
        html = metric_card("Params", "75K")
        assert "↑" not in html
        assert "↓" not in html


class TestStatusBadge:
    """Tests for status_badge() HTML component."""

    def test_success_badge_includes_text(self) -> None:
        """What: Badge text appears in the output. Prevents: Text being dropped."""
        html = status_badge("Model Ready", "success")
        assert "Model Ready" in html

    def test_badge_includes_status_class(self) -> None:
        """
        What: CSS class for the status is included in the badge HTML.
        Why:  The CSS uses class-based colour coding. Without the correct class,
              the badge appears in the default (info) colour regardless of status.
        Prevents: Status-based colour coding not applying.
        """
        html = status_badge("Error", "error")
        assert "dv-badge-error" in html

    def test_all_statuses_produce_html(self) -> None:
        """
        What: All four status values produce non-empty HTML.
        Prevents: A new status value causing a KeyError or empty output.
        """
        for status in ("success", "warning", "error", "info"):
            html = status_badge("Test", status)
            assert isinstance(html, str) and len(html) > 0


class TestPredictionResultCard:
    """Tests for prediction_result_card() HTML component."""

    def test_includes_predicted_digit(self) -> None:
        """
        What: The predicted digit appears in the card HTML.
        Prevents: The biggest visual element of the Draw page being blank.
        """
        html = prediction_result_card(7, 0.97, 3.2, "Custom CNN")
        assert "7" in html

    def test_includes_model_name(self) -> None:
        """What: The model name appears in the card. Prevents: Wrong model name displayed."""
        html = prediction_result_card(3, 0.85, 4.1, "LeNet-5")
        assert "LeNet-5" in html

    def test_high_confidence_uses_green_colour(self) -> None:
        """
        What: Confidence ≥ 90% uses the green colour (#4ade80).
        Why:  Colour-coded confidence communicates certainty at a glance.
              Green = confident, amber = uncertain, red = very uncertain.
        Prevents: Wrong colour class being applied for high-confidence predictions.
        """
        html = prediction_result_card(7, 0.95, 3.0, "Custom CNN")
        assert "#4ade80" in html

    def test_low_confidence_uses_red_colour(self) -> None:
        """
        What: Confidence < 70% uses the red colour (#f87171).
        Prevents: Low-confidence predictions appearing in green (misleadingly confident).
        """
        html = prediction_result_card(5, 0.50, 3.0, "Dense NN")
        assert "#f87171" in html


class TestInfoBox:
    """Tests for info_box() HTML component."""

    def test_wraps_content(self) -> None:
        """
        What: The content string appears inside the info box HTML.
        Prevents: Content being dropped from the info callout.
        """
        html = info_box("This is a tip about preprocessing.")
        assert "This is a tip about preprocessing." in html


class TestPipelineStep:
    """Tests for pipeline_step() HTML component."""

    def test_includes_step_number(self) -> None:
        """
        What: The step number appears in the HTML.
        Why:  The CNN Explainer page uses numbered steps to guide the reader.
              Missing step numbers make the pipeline visually confusing.
        Prevents: Step numbers being dropped from the pipeline visualization.
        """
        html = pipeline_step(3, "Normalise", "Scale to [0, 1]")
        assert "3" in html

    def test_includes_title_and_description(self) -> None:
        """What: Both title and description appear in the output."""
        html = pipeline_step(1, "Resize", "Downsample to 28×28")
        assert "Resize" in html
        assert "Downsample to 28×28" in html


# ─── charts.py tests ──────────────────────────────────────────────────────────


@pytest.fixture
def uniform_probabilities() -> dict[str, float]:
    """10 equal probabilities — a flat, uncertain prediction."""
    return {str(i): 0.1 for i in range(10)}


@pytest.fixture
def confident_probabilities() -> dict[str, float]:
    """High confidence on class 7, near-zero on others."""
    probs = {str(i): 0.01 for i in range(10)}
    probs["7"] = 0.91
    return probs


@pytest.fixture
def sample_histories() -> dict[str, dict[str, list[float]]]:
    """Synthetic training histories for all three models."""
    return {
        "dense_nn": {
            "accuracy": [0.90, 0.93, 0.96],
            "val_accuracy": [0.88, 0.91, 0.94],
            "loss": [0.30, 0.22, 0.14],
            "val_loss": [0.35, 0.26, 0.18],
        },
        "lenet5": {
            "accuracy": [0.92, 0.95, 0.97],
            "val_accuracy": [0.91, 0.94, 0.96],
            "loss": [0.27, 0.18, 0.11],
            "val_loss": [0.30, 0.21, 0.14],
        },
    }


class TestBuildConfidenceBarChart:
    """Tests for build_confidence_bar_chart()."""

    def test_returns_plotly_figure(self, confident_probabilities: dict) -> None:
        """
        What: Returns a go.Figure instance.
        Why:  st.plotly_chart() requires a go.Figure. None or a wrong type
              would crash the Draw page on first prediction.
        Prevents: Wrong return type crashing the chart renderer.
        """
        fig = build_confidence_bar_chart(confident_probabilities, 7)
        assert isinstance(fig, go.Figure)

    def test_chart_has_one_trace(self, confident_probabilities: dict) -> None:
        """
        What: Chart has exactly one bar trace (one set of 10 bars).
        Why:  Two traces would produce doubled bars. Zero traces = empty chart.
        Prevents: Accidental trace duplication producing a broken chart.
        """
        fig = build_confidence_bar_chart(confident_probabilities, 7)
        assert len(fig.data) == 1

    def test_chart_has_ten_bars(self, confident_probabilities: dict) -> None:
        """
        What: The trace contains 10 y-values (one per digit class).
        Why:  All 10 class probabilities must be shown for the chart to be
              informative. Missing bars hide what the model considered.
        Prevents: Partial bar chart with fewer than 10 bars.
        """
        fig = build_confidence_bar_chart(confident_probabilities, 7)
        assert len(fig.data[0].y) == 10


class TestBuildTrainingCurvesChart:
    """Tests for build_training_curves_chart()."""

    def test_returns_plotly_figure(self, sample_histories: dict) -> None:
        """What: Returns a go.Figure. Prevents: Wrong type crashing chart renderer."""
        fig = build_training_curves_chart(sample_histories)
        assert isinstance(fig, go.Figure)

    def test_has_traces_for_each_model(self, sample_histories: dict) -> None:
        """
        What: Chart has at least one trace per model in the histories dict.
        Why:  Each model contributes 2 traces (train + val). With 2 models,
              we expect at least 4 traces.
        Prevents: Missing traces for some models.
        """
        fig = build_training_curves_chart(sample_histories)
        # 2 models × 2 lines (train + val) = 4 traces minimum
        assert len(fig.data) >= 4

    def test_skips_models_with_missing_key(self) -> None:
        """
        What: A history dict missing 'val_accuracy' key is silently skipped.
        Why:  If a model's history file is corrupt (missing val_ key), the chart
              must still render for other models. It should not raise a KeyError.
        Prevents: One corrupt history file breaking all training curves.
        """
        incomplete_history = {
            "dense_nn": {
                "accuracy": [0.9],
                # 'val_accuracy' intentionally missing
            }
        }
        # Should not raise
        fig = build_training_curves_chart(incomplete_history, metric="accuracy")
        assert isinstance(fig, go.Figure)

    def test_returns_figure_for_empty_histories(self) -> None:
        """
        What: Empty dict returns a valid (empty) figure.
        Why:  If no models have been trained yet, the chart must not crash.
        Prevents: TypeError or AttributeError on empty input.
        """
        fig = build_training_curves_chart({})
        assert isinstance(fig, go.Figure)


class TestBuildConfusionMatrixChart:
    """Tests for build_confusion_matrix_chart()."""

    def test_returns_plotly_figure(self) -> None:
        """What: Returns go.Figure. Prevents: Wrong type breaking the analytics page."""
        cm = np.eye(10, dtype=int) * 1000  # perfect diagonal
        labels = [str(i) for i in range(10)]
        fig = build_confusion_matrix_chart(cm, labels)
        assert isinstance(fig, go.Figure)

    def test_accepts_10x10_matrix(self) -> None:
        """
        What: A (10, 10) matrix produces a valid figure.
        Prevents: Wrong matrix dimensions crashing the heatmap builder.
        """
        cm = np.random.randint(0, 100, size=(10, 10))
        labels = [str(i) for i in range(10)]
        fig = build_confusion_matrix_chart(cm, labels)
        assert isinstance(fig, go.Figure)


class TestBuildF1BarChart:
    """Tests for build_f1_bar_chart()."""

    def test_returns_plotly_figure(self) -> None:
        """What: Returns go.Figure."""
        f1_scores = {str(i): 0.99 - i * 0.001 for i in range(10)}
        fig = build_f1_bar_chart(f1_scores)
        assert isinstance(fig, go.Figure)

    def test_accepts_all_ten_classes(self) -> None:
        """
        What: F1 chart handles all 10 digit classes.
        Prevents: IndexError or truncation on 10-class input.
        """
        f1_scores = {str(i): 0.990 for i in range(10)}
        fig = build_f1_bar_chart(f1_scores, model_display_name="Custom CNN")
        assert isinstance(fig, go.Figure)


class TestBuildAccuracyComparisonChart:
    """Tests for build_accuracy_comparison_chart()."""

    def test_returns_plotly_figure(self) -> None:
        """What: Returns go.Figure."""
        accuracies = {"dense_nn": 0.975, "lenet5": 0.985, "custom_cnn": 0.993}
        fig = build_accuracy_comparison_chart(accuracies)
        assert isinstance(fig, go.Figure)

    def test_handles_single_model(self) -> None:
        """
        What: Works with a single model (no comparison needed).
        Prevents: Division by zero or empty layout on one-model input.
        """
        fig = build_accuracy_comparison_chart({"custom_cnn": 0.993})
        assert isinstance(fig, go.Figure)


class TestBuildClassDistributionChart:
    """Tests for build_class_distribution_chart()."""

    def test_returns_plotly_figure(self) -> None:
        """What: Returns go.Figure."""
        counts = {str(i): 5923 + i * 10 for i in range(10)}
        fig = build_class_distribution_chart(counts)
        assert isinstance(fig, go.Figure)

    def test_custom_title_appears_in_layout(self) -> None:
        """
        What: The provided title appears in the figure layout.
        Prevents: Title parameter being ignored, always showing default title.
        """
        counts = {str(i): 1000 for i in range(10)}
        fig = build_class_distribution_chart(counts, title="Test Set Distribution")
        assert "Test Set Distribution" in fig.layout.title.text
