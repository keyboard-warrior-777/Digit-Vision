"""
Reusable HTML/CSS UI components for DigitVision.

Each function returns an HTML string rendered via st.markdown(unsafe_allow_html=True).
Keeping component HTML here prevents inline HTML from cluttering page logic,
and ensures consistent styling across all pages.

Design tokens are defined in components/styles.py and reused via CSS classes here.

Usage:
    import streamlit as st
    from components.cards import page_header, metric_card

    st.markdown(page_header("Analytics", "Evaluation metrics and charts", "📊"),
                unsafe_allow_html=True)
"""

from __future__ import annotations

from typing import Optional


import base64

def page_header(title: str, subtitle: str, icon: str = "", image_path: str = "") -> str:
    """
    Render the standard DigitVision page header with gradient title.

    Args:
        title:      Main page title.
        subtitle:   Descriptive subtitle shown below the title.
        icon:       Optional emoji displayed before the title.
        image_path: Optional path to an image to use as the icon.

    Returns:
        HTML string for ``st.markdown(unsafe_allow_html=True)``.
    """
    icon_html = ""
    if image_path:
        with open(image_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("utf-8")
        icon_html = f"<img src='data:image/png;base64,{b64}' style='height:1.2em; margin-right:0.5rem; vertical-align:middle; border-radius:8px;' alt='Logo'/>"
    elif icon:
        icon_html = f"<span style='margin-right:0.5rem'>{icon}</span>"
        
    return f"""
<div class="dv-page-header">
    <div class="dv-page-title" style="display:flex; align-items:center;">{icon_html}{title}</div>
    <div class="dv-page-subtitle">{subtitle}</div>
</div>
"""


def metric_card(
    label: str,
    value: str,
    icon: str = "",
    delta: Optional[str] = None,
    delta_positive: Optional[bool] = None,
) -> str:
    """
    Render a single metric display card with an optional delta indicator.

    Args:
        label:          Short label shown below the value (e.g. "Test Accuracy").
        value:          The primary value to display (e.g. "99.3%").
        icon:           Optional emoji icon shown above the value.
        delta:          Optional comparison string (e.g. "+1.8% vs LeNet-5").
        delta_positive: True = green, False = red, None = no delta shown.

    Returns:
        HTML string.
    """
    icon_html = f'<span class="dv-metric-icon">{icon}</span>' if icon else ""
    delta_html = ""
    if delta is not None:
        css_class = "positive" if delta_positive else "negative"
        arrow = "↑" if delta_positive else "↓"
        delta_html = f'<div class="dv-metric-delta {css_class}">{arrow} {delta}</div>'

    return f"""
<div class="dv-metric-card">
    {icon_html}
    <div class="dv-metric-value">{value}</div>
    <div class="dv-metric-label">{label}</div>
    {delta_html}
</div>
"""


def status_badge(text: str, status: str = "info") -> str:
    """
    Render a coloured status badge.

    Args:
        text:   Badge label text.
        status: One of 'success', 'warning', 'error', 'info'.

    Returns:
        HTML string.
    """
    dot_colours = {
        "success": "#4ade80",
        "warning": "#fbbf24",
        "error": "#f87171",
        "info": "#6366f1",
    }
    dot_colour = dot_colours.get(status, "#94a3b8")
    return (
        f'<span class="dv-badge dv-badge-{status}">'
        f'<span style="width:6px;height:6px;background:{dot_colour};'
        f'border-radius:50%;display:inline-block;"></span>'
        f"{text}</span>"
    )


def model_stat_row(label: str, value: str) -> str:
    """Render a single stat row inside a model card."""
    return f"""
<div class="dv-model-stat">
    <span class="dv-model-stat-label">{label}</span>
    <span class="dv-model-stat-value">{value}</span>
</div>
"""


def model_comparison_card(
    title: str,
    subtitle: str,
    stats: list[tuple[str, str]],
    badge_text: str = "",
    badge_status: str = "info",
    is_best: bool = False,
) -> str:
    """
    Render a model summary card for the comparison page.

    Args:
        title:        Model display name.
        subtitle:     One-line architecture description.
        stats:        List of (label, value) tuples for the stat rows.
        badge_text:   Optional badge text (e.g. "Best Accuracy").
        badge_status: Badge colour class.
        is_best:      If True, adds a highlighted border to the card.

    Returns:
        HTML string.

    Note:
        The divider between the header and stats is rendered as a CSS-styled
        <div> rather than a bare <hr> tag.  Bare <hr> tags inside complex HTML
        strings can be re-escaped by some Markdown renderers, causing the tag
        to appear as literal text.  A <div> with border-top is immune to this.
    """
    best_class = " best-model" if is_best else ""
    badge_html = (
        f'<div style="margin-top:0.75rem">{status_badge(badge_text, badge_status)}</div>'
        if badge_text
        else ""
    )
    stats_html = "".join(model_stat_row(label, value) for label, value in stats)

    # Build as adjacent f-strings — no blank lines in the output HTML.
    # Triple-quoted f-strings with an empty {badge_html} insert a blank line,
    # which causes Python-Markdown to exit HTML-block mode and render the
    # subsequent <div> as escaped text rather than HTML.
    return (
        f'<div class="dv-model-card{best_class}">'
        f'<div class="dv-model-card-title">{title}</div>'
        f'<div class="dv-model-card-subtitle">{subtitle}</div>'
        f'{badge_html}'
        f'<div class="dv-model-card-divider"></div>'
        f'{stats_html}'
        f'</div>'
    )


def prediction_result_card(
    predicted_digit: int,
    confidence: float,
    inference_ms: float,
    model_name: str,
) -> str:
    """
    Render the large prediction result display card.

    Args:
        predicted_digit: Predicted class index (0–9).
        confidence:      Softmax probability of the top class.
        inference_ms:    Inference time in milliseconds.
        model_name:      Display name of the model used.

    Returns:
        HTML string.
    """
    confidence_colour = (
        "#4ade80" if confidence >= 0.90 else
        "#fbbf24" if confidence >= 0.70 else
        "#f87171"
    )

    return f"""
<div class="dv-prediction-result">
    <div class="dv-predicted-digit">{predicted_digit}</div>
    <div style="font-size:0.85rem;color:var(--text-muted);margin-bottom:0.75rem">Predicted Digit</div>
    <div style="display:flex;justify-content:center;gap:2rem;flex-wrap:wrap">
        <div style="text-align:center">
            <div style="font-size:1.75rem;font-weight:800;color:{confidence_colour}">
                {confidence:.1%}
            </div>
            <div style="font-size:0.75rem;color:var(--text-muted);text-transform:uppercase;letter-spacing:0.06em">
                Confidence
            </div>
        </div>
        <div style="text-align:center">
            <div style="font-size:1.75rem;font-weight:800;color:var(--text-secondary)">
                {inference_ms:.1f}ms
            </div>
            <div style="font-size:0.75rem;color:var(--text-muted);text-transform:uppercase;letter-spacing:0.06em">
                Inference
            </div>
        </div>
    </div>
    <div style="margin-top:1rem;font-size:0.8rem;color:var(--text-muted)">
        Model: <span style="color:var(--text-secondary);font-weight:600">{model_name}</span>
    </div>
</div>
"""


def info_box(content: str) -> str:
    """Render a styled information callout box."""
    return f'<div class="dv-info-box">{content}</div>'


def pipeline_step(number: int, title: str, description: str) -> str:
    """Render a single step in a visual processing pipeline."""
    return f"""
<div class="dv-pipeline-step">
    <div class="dv-pipeline-step-number">Step {number}</div>
    <div class="dv-pipeline-step-title" style="margin-top:0.75rem">{title}</div>
    <div class="dv-pipeline-step-desc">{description}</div>
</div>
"""


def section_header(text: str) -> str:
    """Render a section heading with a bottom border."""
    return f'<div class="dv-section-header">{text}</div>'
