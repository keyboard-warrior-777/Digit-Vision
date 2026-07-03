"""
Digit Recognition Page — DigitVision.

Allows the user to either draw a digit on an interactive canvas or
upload an image file. Displays the predicted digit, confidence score,
full probability distribution chart, Grad-CAM heatmap (for CNN models),
and inference time.
"""

from __future__ import annotations

import time

import numpy as np
import streamlit as st
from components.cards import info_box, page_header, prediction_result_card
from components.charts import build_confidence_bar_chart
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
from src.predict import predict_from_canvas, predict_from_upload
from src.preprocessing import uploaded_image_to_model_input

# ── Helpers ───────────────────────────────────────────────────────────────────

def _models_available() -> list[str]:
    return [name for name in AVAILABLE_MODELS if MODEL_PATHS[name].exists()]


@st.cache_resource
def _load_model(model_name: str):
    """Load and cache a Keras model. Cached per-process — survives reruns."""
    import tensorflow as tf
    return tf.keras.models.load_model(MODEL_PATHS[model_name])


# ─────────────────────────────────────────────────────────────────────────────

st.markdown(
    page_header("Digit Recognition", "Draw or upload a digit — select a model and run inference", "🎨"),
    unsafe_allow_html=True,
)

available = _models_available()

if not available:
    st.warning(
        "No trained models found. Train at least one model before using this page.\n\n"
        "```bash\nmake train\n```"
    )
    st.stop()

# ── Sidebar — model selector ──────────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        "<div style='font-size:0.72rem;font-weight:700;text-transform:uppercase;"
        "letter-spacing:0.1em;color:var(--text-muted);padding:0.5rem 0 0.75rem'>⚙ Settings</div>",
        unsafe_allow_html=True,
    )
    selected_model = st.selectbox(
        "Model",
        options=available,
        format_func=lambda n: MODEL_DISPLAY_NAMES[n],
        help="Select which trained model to use for prediction.",
    )
    st.markdown(
        "<div class='divider'></div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        f"<div style='font-size:0.8rem;color:var(--text-muted)'>"
        f"Active: <span style='color:var(--text-primary);font-weight:600'>"
        f"{MODEL_DISPLAY_NAMES[selected_model]}</span></div>",
        unsafe_allow_html=True,
    )

# ── Input tabs ────────────────────────────────────────────────────────────────
tab_draw, tab_upload = st.tabs(["✏️  Draw a Digit", "📁  Upload an Image"])

# ═══════════════════ TAB 1 — DRAW ════════════════════════════════════════════
with tab_draw:
    st.markdown(
        info_box(
            "Draw a digit (0–9) on the canvas, then click <strong>Predict</strong>."
        ),
        unsafe_allow_html=True,
    )

    col_canvas, col_result = st.columns([1, 1], gap="large")

    with col_canvas:
        st.markdown("**Draw here ↓**")
        canvas_result = st_canvas(
            fill_color="rgba(0, 0, 0, 0)",
            stroke_width=CANVAS_STROKE_WIDTH,
            stroke_color="#FFFFFF",
            background_color="#000000",
            height=CANVAS_SIZE,
            width=CANVAS_SIZE,
            drawing_mode="freedraw",
            key="digit_canvas",
            display_toolbar=True,
        )

        if st.button("🔮  Predict", type="primary", use_container_width=True):
            st.session_state["trigger_canvas_predict"] = True

    with col_result:
        should_predict = (
            canvas_result.image_data is not None
            and canvas_result.image_data[:, :, :3].sum() > 0
            and st.session_state.get("trigger_canvas_predict", False)
        )

        if should_predict:
            st.session_state["trigger_canvas_predict"] = False
            with st.spinner("Classifying..."):
                t0 = time.perf_counter()
                result = predict_from_canvas(canvas_result.image_data, selected_model)
                inference_ms = (time.perf_counter() - t0) * 1000

            st.markdown(
                prediction_result_card(
                    result.predicted_digit,
                    result.confidence,
                    inference_ms,
                    result.model_display_name,
                ),
                unsafe_allow_html=True,
            )

            st.markdown("<div style='height:var(--sp-3)'></div>", unsafe_allow_html=True)
            st.plotly_chart(
                build_confidence_bar_chart(result.all_probabilities, result.predicted_digit),
                use_container_width=True,
            )

            # Top-3 predictions
            sorted_probs = sorted(
                result.all_probabilities.items(),
                key=lambda x: -x[1],
            )[:3]
            top3_cols = st.columns(3)
            for tcol, (digit, prob) in zip(top3_cols, sorted_probs, strict=False):
                tcol.metric(f"#{sorted_probs.index((digit, prob)) + 1} Digit {digit}", f"{prob:.1%}")

            # Grad-CAM — only for CNN models
            if selected_model != "dense_nn":
                st.markdown("**Grad-CAM — attended regions:**")
                model = _load_model(selected_model)
                from src.preprocessing import canvas_image_to_model_input
                model_input = canvas_image_to_model_input(canvas_result.image_data)
                heatmap = compute_gradcam(model, model_input, result.predicted_digit)
                if heatmap is not None:
                    overlay = overlay_heatmap_on_image(heatmap, model_input, alpha=0.45)
                    col_orig, col_heat = st.columns(2)
                    with col_orig:
                        display_img = (model_input.squeeze() * 255).astype(np.uint8)
                        st.image(
                            Image.fromarray(display_img, "L").resize((140, 140), Image.NEAREST),
                            caption="Preprocessed Input",
                            width=140,
                        )
                    with col_heat:
                        st.image(
                            Image.fromarray(overlay).resize((140, 140), Image.NEAREST),
                            caption="Grad-CAM Heatmap",
                            width=140,
                        )
            else:
                st.markdown(
                    info_box(
                        "Grad-CAM is not available for the Dense NN — it has no convolutional layers. "
                        "Switch to <strong>LeNet-5</strong> or <strong>Custom CNN</strong> to see visualisations."
                    ),
                    unsafe_allow_html=True,
                )
        else:
            st.markdown(
                """
                <div style='display:flex;flex-direction:column;align-items:center;
                            justify-content:center;height:280px;
                            border:2px dashed #2d3154;border-radius:12px;
                            color:#64748b;text-align:center;padding:2rem'>
                    <div style='font-size:3rem;margin-bottom:0.75rem'>✏️</div>
                    <div style='font-weight:600;font-size:0.95rem;color:#94a3b8'>
                        Draw a digit, then click Predict
                    </div>
                    <div style='font-size:0.8rem;margin-top:0.35rem'>
                        Supported: digits 0 through 9
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

# ═══════════════════ TAB 2 — UPLOAD ══════════════════════════════════════════
with tab_upload:
    st.markdown(
        info_box(
            "Upload any PNG, JPEG, or BMP containing a handwritten digit. "
            "Any resolution is accepted — it will be resized to 28×28 and converted to MNIST format."
        ),
        unsafe_allow_html=True,
    )

    uploaded_file = st.file_uploader(
        "Choose an image file",
        type=["png", "jpg", "jpeg", "bmp"],
        label_visibility="collapsed",
    )

    if uploaded_file is not None:
        pil_image = Image.open(uploaded_file)

        col_img, col_res = st.columns([1, 1], gap="large")

        with col_img:
            st.markdown("**Uploaded Image:**")
            # Display at a comfortable size regardless of actual dimensions
            display_size = 200
            st.image(pil_image.resize((display_size, display_size)), width=display_size)

        with col_res:
            with st.spinner("Classifying..."):
                t0 = time.perf_counter()
                result = predict_from_upload(pil_image, selected_model)
                inference_ms = (time.perf_counter() - t0) * 1000

            st.markdown(
                prediction_result_card(
                    result.predicted_digit,
                    result.confidence,
                    inference_ms,
                    result.model_display_name,
                ),
                unsafe_allow_html=True,
            )

        st.markdown("<div style='height:var(--sp-2)'></div>", unsafe_allow_html=True)
        st.plotly_chart(
            build_confidence_bar_chart(result.all_probabilities, result.predicted_digit),
            use_container_width=True,
        )

        # Top-3 predictions for upload tab
        sorted_probs_up = sorted(
            result.all_probabilities.items(),
            key=lambda x: -x[1],
        )[:3]
        top3u_cols = st.columns(3)
        for tcol, (digit, prob) in zip(top3u_cols, sorted_probs_up, strict=False):
            tcol.metric(f"#{sorted_probs_up.index((digit, prob)) + 1} Digit {digit}", f"{prob:.1%}")

        # Grad-CAM for CNN models
        if selected_model != "dense_nn":
            model = _load_model(selected_model)
            model_input = uploaded_image_to_model_input(pil_image)
            heatmap = compute_gradcam(model, model_input, result.predicted_digit)
            if heatmap is not None:
                st.markdown("**Grad-CAM Explanation:**")
                overlay = overlay_heatmap_on_image(heatmap, model_input, alpha=0.45)
                col_o, col_h, col_sp = st.columns([1, 1, 2])
                with col_o:
                    display = (model_input.squeeze() * 255).astype(np.uint8)
                    st.image(
                        Image.fromarray(display, "L").resize((140, 140), Image.NEAREST),
                        caption="28×28 Input",
                        width=140,
                    )
                with col_h:
                    st.image(
                        Image.fromarray(overlay).resize((140, 140), Image.NEAREST),
                        caption="Grad-CAM",
                        width=140,
                    )
