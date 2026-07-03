"""
Model Playground — DigitVision.

Draw one digit and see all three models predict simultaneously.
Side-by-side comparison of predicted digit, confidence, inference time,
and Grad-CAM (where available). Makes architectural differences tangible.
"""

from __future__ import annotations

import time

import streamlit as st
from components.cards import info_box, page_header
from components.charts import build_confidence_bar_chart
from components.styles import (
    get_global_css,  # noqa: F401 — imported for side-effect in app.py
)
from PIL import Image
from streamlit_drawable_canvas import st_canvas

from config.config import (
    AVAILABLE_MODELS,
    CANVAS_SIZE,
    CANVAS_STROKE_WIDTH,
    MODEL_DISPLAY_NAMES,
    MODEL_PATHS,
)
from src.gradcam import compute_gradcam, overlay_heatmap_on_image
from src.predict import predict_from_canvas
from src.preprocessing import canvas_image_to_model_input


@st.cache_resource
def _load_model(model_name: str):
    import tensorflow as tf
    return tf.keras.models.load_model(MODEL_PATHS[model_name])


def _models_available() -> list[str]:
    return [n for n in AVAILABLE_MODELS if MODEL_PATHS[n].exists()]


# ─────────────────────────────────────────────────────────────────────────────

st.markdown(
    page_header(
        "Model Playground",
        "One drawing, three predictions — compare architectures side by side",
        "🔬",
    ),
    unsafe_allow_html=True,
)

available = _models_available()

if not available:
    st.warning("No trained models found. Run `make train` first.")
    st.stop()

if len(available) < 3:
    missing = [MODEL_DISPLAY_NAMES[n] for n in AVAILABLE_MODELS if n not in available]
    st.markdown(
        info_box(
            f"Some models are not trained yet: <strong>{', '.join(missing)}</strong>. "
            "Only trained models will appear in the comparison."
        ),
        unsafe_allow_html=True,
    )

# ── Canvas ────────────────────────────────────────────────────────────────────

# Brush size slider lives here so it renders before the canvas
brush_size = st.sidebar.slider(
    "Brush size",
    min_value=10,
    max_value=40,
    value=CANVAS_STROKE_WIDTH,
    step=2,
    help="Adjust the stroke width for drawing.",
)

col_canvas, col_spacer = st.columns([1, 2])
with col_canvas:
    st.markdown("**Draw a digit:**")
    canvas_result = st_canvas(
        fill_color="rgba(0, 0, 0, 0)",
        stroke_width=brush_size,
        stroke_color="#FFFFFF",
        background_color="#000000",
        height=CANVAS_SIZE,
        width=CANVAS_SIZE,
        drawing_mode="freedraw",
        key="playground_canvas",
    )
    compare_clicked = st.button(
        "⚡  Compare All Models", type="primary", use_container_width=True
    )

st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

# ── Comparison results ────────────────────────────────────────────────────────
has_drawing = (
    canvas_result.image_data is not None
    and canvas_result.image_data[:, :, :3].sum() > 0
)

if compare_clicked and has_drawing:
    model_input = canvas_image_to_model_input(canvas_result.image_data)

    columns = st.columns(len(available), gap="medium")

    # Collect results here so the summary table can reuse them without
    # running predict_from_canvas() a second time for each model.
    comparison_results: dict[str, object] = {}

    for col, model_name in zip(columns, available, strict=False):
        display_name = MODEL_DISPLAY_NAMES[model_name]

        with col:
            with st.spinner(f"Running {display_name}..."):
                t0 = time.perf_counter()
                result = predict_from_canvas(canvas_result.image_data, model_name)
                inference_ms = (time.perf_counter() - t0) * 1000

            comparison_results[model_name] = result

            # Confidence colour
            conf_colour = (
                "#4ade80" if result.confidence >= 0.90 else
                "#fbbf24" if result.confidence >= 0.70 else
                "#f87171"
            )

            # Model result card
            st.markdown(
                f"""
                <div class="dv-card-accent" style="text-align:center">
                    <div style="font-size:0.8rem;color:#94a3b8;font-weight:600;
                                text-transform:uppercase;letter-spacing:0.06em;
                                margin-bottom:0.5rem">
                        {display_name}
                    </div>
                    <div style="font-size:4rem;font-weight:900;
                                background:linear-gradient(135deg,#6366f1,#22d3ee);
                                -webkit-background-clip:text;
                                -webkit-text-fill-color:transparent;
                                line-height:1;font-family:'JetBrains Mono',monospace">
                        {result.predicted_digit}
                    </div>
                    <div style="font-size:1.5rem;font-weight:700;color:{conf_colour};margin-top:0.5rem">
                        {result.confidence:.1%}
                    </div>
                    <div style="font-size:0.72rem;color:#64748b;text-transform:uppercase;
                                letter-spacing:0.06em">Confidence</div>
                    <div style="font-size:0.85rem;color:#94a3b8;margin-top:0.5rem">
                        {inference_ms:.1f} ms
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.markdown("<div style='height:var(--sp-2)'></div>", unsafe_allow_html=True)

            # Confidence chart
            st.plotly_chart(
                build_confidence_bar_chart(result.all_probabilities, result.predicted_digit),
                use_container_width=True,
                key=f"chart_{model_name}",
            )

            # Grad-CAM
            if model_name != "dense_nn":
                model = _load_model(model_name)
                heatmap = compute_gradcam(model, model_input, result.predicted_digit)
                if heatmap is not None:
                    overlay = overlay_heatmap_on_image(heatmap, model_input, alpha=0.45)
                    st.image(
                        Image.fromarray(overlay).resize((140, 140), Image.NEAREST),
                        caption="Grad-CAM",
                        width=140,
                    )
            else:
                st.markdown(
                    "<div style='font-size:0.75rem;color:#64748b;text-align:center;"
                    "padding:0.5rem;'>No Grad-CAM (no Conv layers)</div>",
                    unsafe_allow_html=True,
                )

    # ── Quick comparison summary table ─────────────────────────────────────
    st.markdown("<div style='height:var(--sp-4)'></div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='dv-section-header'>Comparison Summary</div>",
        unsafe_allow_html=True,
    )

    rows_html = ""
    for model_name, result in comparison_results.items():
        rows_html += f"""
        <tr>
            <td><strong>{MODEL_DISPLAY_NAMES[model_name]}</strong></td>
            <td style="text-align:center;font-size:1.1rem;font-weight:700">{result.predicted_digit}</td>
            <td style="text-align:center">{result.confidence:.1%}</td>
        </tr>
        """

    st.markdown(
        f"""
        <table class="dv-table">
            <thead>
                <tr>
                    <th>Model</th>
                    <th style="text-align:center">Prediction</th>
                    <th style="text-align:center">Confidence</th>
                </tr>
            </thead>
            <tbody>{rows_html}</tbody>
        </table>
        """,
        unsafe_allow_html=True,
    )

elif compare_clicked and not has_drawing:
    st.warning("Please draw a digit in the canvas before comparing.")
else:
    st.markdown(
        """
        <div style='text-align:center;color:#7c8aaa;padding:3rem 0'>
            <div style='font-size:0.95rem;font-weight:500;color:#94a3b8'>
                Draw a digit above and click Compare All Models
            </div>
            <div style='font-size:0.82rem;margin-top:0.4rem'>
                All three architectures will predict simultaneously
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ── Parameter count sidebar ───────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 📊 Architecture Stats")
    param_counts = {
        "dense_nn": "~530K",
        "lenet5": "~61K",
        "custom_cnn": "~75K",
    }
    for name in AVAILABLE_MODELS:
        st.markdown(
            f"**{MODEL_DISPLAY_NAMES[name]}**  \n"
            f"`{param_counts.get(name, '?')} parameters`"
        )
        st.markdown("<div style='margin-bottom:0.5rem'></div>", unsafe_allow_html=True)
