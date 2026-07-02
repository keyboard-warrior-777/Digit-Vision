"""
How CNN Thinks — DigitVision.

An educational, step-by-step visual walkthrough of how a convolutional
neural network processes a handwritten digit image.

Pipeline visualised:
    Step 1 — Raw canvas input (or uploaded image)
    Step 2 — Preprocessed: 28×28, white-on-black, normalised
    Step 3 — Conv Block 1 feature maps (first 8 shown)
    Step 4 — After MaxPooling (spatial downsampling)
    Step 5 — Conv Block 2 feature maps (first 8 shown)
    Step 6 — Grad-CAM heatmap (attention overlay)
    Step 7 — Final prediction with probability

This page is designed to help you explain CNNs confidently in interviews.
Each step includes a plain-language explanation of what is happening and why.
"""

from __future__ import annotations

import numpy as np
import streamlit as st
from PIL import Image
from streamlit_drawable_canvas import st_canvas

from components.cards import info_box, page_header, pipeline_step
from components.charts import build_confidence_bar_chart
from components.styles import get_global_css

from config.config import CANVAS_SIZE, CANVAS_STROKE_WIDTH, MODEL_PATHS
from src.gradcam import compute_gradcam, overlay_heatmap_on_image
from src.predict import predict_from_canvas
from src.preprocessing import canvas_image_to_model_input


@st.cache_resource
def _load_custom_cnn():
    """Load the Custom CNN — this is the model used on the explainer page."""
    import tensorflow as tf
    return tf.keras.models.load_model(MODEL_PATHS["custom_cnn"])


def _extract_feature_maps(
    model,
    model_input: np.ndarray,
    layer_name: str,
) -> np.ndarray | None:
    """Extract activations from a specific layer as a (H, W, C) numpy array."""
    import tensorflow as tf
    try:
        layer = model.get_layer(layer_name)
    except ValueError:
        return None
    feat_model = tf.keras.Model(inputs=model.inputs, outputs=layer.output)
    return feat_model.predict(model_input, verbose=0)[0]


def _render_feature_maps_grid(feature_maps: np.ndarray, max_maps: int = 8) -> None:
    """Render up to max_maps feature maps as a row of small images."""
    num_filters = min(feature_maps.shape[-1], max_maps)
    cols = st.columns(num_filters)
    for i, col in enumerate(cols):
        fm = feature_maps[:, :, i]
        # Normalise to [0, 255]
        fm_min, fm_max = fm.min(), fm.max()
        if fm_max > fm_min:
            fm_norm = ((fm - fm_min) / (fm_max - fm_min) * 255).astype(np.uint8)
        else:
            fm_norm = np.zeros_like(fm, dtype=np.uint8)
        pil_fm = Image.fromarray(fm_norm, "L").resize((80, 80), Image.NEAREST)
        with col:
            st.image(pil_fm, caption=f"Filter {i + 1}", use_container_width=False, width=80)


# ─────────────────────────────────────────────────────────────────────────────

st.markdown(
    page_header(
        "How CNN Thinks",
        "A visual walkthrough of convolutional feature extraction, pooling, and attention",
        "🧠",
    ),
    unsafe_allow_html=True,
)

cnn_available = MODEL_PATHS["custom_cnn"].exists()

if not cnn_available:
    st.warning(
        "The Custom CNN model is not trained yet. This page requires it.\n\n"
        "```bash\npython -m src.train --model custom_cnn\n```"
    )
    st.stop()

# ── Introduction callout ──────────────────────────────────────────────────────
st.markdown(
    info_box(
        "<strong>Interview-ready explanation:</strong> A convolutional network processes images "
        "in stages. Early layers detect simple features (edges, curves). Later layers combine "
        "these into complex patterns (loops, strokes). Grad-CAM shows <em>which pixels</em> "
        "the network found most informative for its final decision."
    ),
    unsafe_allow_html=True,
)

# ── Canvas input ──────────────────────────────────────────────────────────────
st.markdown("<div class='dv-section-header'>Input</div>", unsafe_allow_html=True)

col_input, col_btn = st.columns([1, 2])
with col_input:
    canvas_result = st_canvas(
        fill_color="rgba(0, 0, 0, 0)",
        stroke_width=CANVAS_STROKE_WIDTH,
        stroke_color="#FFFFFF",
        background_color="#000000",
        height=CANVAS_SIZE,
        width=CANVAS_SIZE,
        drawing_mode="freedraw",
        key="explainer_canvas",
    )
    analyse_clicked = st.button("🔍  Analyse", type="primary", use_container_width=True)

# ── Processing pipeline ───────────────────────────────────────────────────────
has_drawing = (
    canvas_result.image_data is not None
    and canvas_result.image_data.sum() > 0
)

if analyse_clicked and has_drawing:
    model = _load_custom_cnn()
    model_input = canvas_image_to_model_input(canvas_result.image_data)
    result = predict_from_canvas(canvas_result.image_data, "custom_cnn")

    st.markdown(
        "<div style='margin-top:2rem'></div>",
        unsafe_allow_html=True,
    )

    # ── Step 1: Raw input ─────────────────────────────────────────────────────
    st.markdown(
        pipeline_step(
            1,
            "Raw Canvas Input",
            "A 280×280 RGBA image from the drawing canvas. "
            "Black background, white stroke — the opposite of MNIST format.",
        ),
        unsafe_allow_html=True,
    )
    raw_display = (canvas_result.image_data[:, :, 3]).astype(np.uint8)  # alpha channel
    st.image(Image.fromarray(raw_display, "L").resize((140, 140), Image.NEAREST), width=140)

    st.markdown("<div style='color:#64748b;text-align:center;font-size:0.8rem;margin:0.5rem 0'>↓</div>", unsafe_allow_html=True)

    # ── Step 2: Preprocessed ─────────────────────────────────────────────────
    st.markdown(
        pipeline_step(
            2,
            "Preprocessed: 28×28, Inverted, Normalised",
            "Alpha channel extracted → colours inverted (now white-on-black, matching MNIST) "
            "→ resized to 28×28 with INTER_AREA interpolation → normalised to [0, 1].",
        ),
        unsafe_allow_html=True,
    )
    preprocessed_display = (model_input.squeeze() * 255).astype(np.uint8)
    st.image(Image.fromarray(preprocessed_display, "L").resize((140, 140), Image.NEAREST), width=140)
    st.markdown(
        info_box(
            "The inversion step is the most commonly missed detail in digit recognition demos. "
            "The model was trained on white-on-black digits (MNIST). Feeding it black-on-white "
            "would cause random predictions."
        ),
        unsafe_allow_html=True,
    )

    st.markdown("<div style='color:#64748b;text-align:center;font-size:0.8rem;margin:0.5rem 0'>↓</div>", unsafe_allow_html=True)

    # ── Step 3: Conv Block 1 feature maps ────────────────────────────────────
    st.markdown(
        pipeline_step(
            3,
            "Convolutional Block 1 — 32 Filters (3×3 kernels)",
            "32 learned filters slide across the image. Each produces a 'feature map' — "
            "a grid showing where in the image that filter's pattern was detected. "
            "Early filters detect edges, curves, and stroke endpoints.",
        ),
        unsafe_allow_html=True,
    )
    fm1 = _extract_feature_maps(model, model_input, "block1_conv1_relu")
    if fm1 is not None:
        st.caption("First 8 of 32 feature maps after Conv Block 1:")
        _render_feature_maps_grid(fm1, max_maps=8)

    st.markdown("<div style='color:#64748b;text-align:center;font-size:0.8rem;margin:0.5rem 0'>↓</div>", unsafe_allow_html=True)

    # ── Step 4: After MaxPooling ──────────────────────────────────────────────
    st.markdown(
        pipeline_step(
            4,
            "MaxPooling2D — Spatial Downsampling (28→14)",
            "A 2×2 sliding window keeps only the maximum activation in each region. "
            "This halves the spatial dimensions: 28×28 → 14×14. "
            "The strongest feature activations are preserved; irrelevant detail is discarded.",
        ),
        unsafe_allow_html=True,
    )
    fm_pooled = _extract_feature_maps(model, model_input, "block1_pool")
    if fm_pooled is not None:
        st.caption("After MaxPooling — 14×14 feature maps:")
        _render_feature_maps_grid(fm_pooled, max_maps=8)

    st.markdown("<div style='color:#64748b;text-align:center;font-size:0.8rem;margin:0.5rem 0'>↓</div>", unsafe_allow_html=True)

    # ── Step 5: Conv Block 2 feature maps ────────────────────────────────────
    st.markdown(
        pipeline_step(
            5,
            "Convolutional Block 2 — 64 Filters",
            "A deeper layer with 64 filters. At this stage, the 14×14 inputs to this "
            "layer already contain edge/curve information. The 64 filters recombine "
            "these into more complex patterns: loops (for 0, 8), angles (for 7), etc.",
        ),
        unsafe_allow_html=True,
    )
    fm2 = _extract_feature_maps(model, model_input, "block2_conv1_relu")
    if fm2 is not None:
        st.caption("First 8 of 64 feature maps after Conv Block 2:")
        _render_feature_maps_grid(fm2, max_maps=8)

    st.markdown("<div style='color:#64748b;text-align:center;font-size:0.8rem;margin:0.5rem 0'>↓</div>", unsafe_allow_html=True)

    # ── Step 6: Grad-CAM ──────────────────────────────────────────────────────
    st.markdown(
        pipeline_step(
            6,
            "Grad-CAM — Where Did the Model Look?",
            "Grad-CAM computes the gradient of the predicted class score with respect to "
            "the last conv layer's activations. Regions with high gradient (red/orange) "
            "contributed most to the prediction. Low-gradient regions (blue) were ignored.",
        ),
        unsafe_allow_html=True,
    )
    heatmap = compute_gradcam(model, model_input, result.predicted_digit)
    if heatmap is not None:
        overlay = overlay_heatmap_on_image(heatmap, model_input, alpha=0.5)
        col_orig, col_heat, col_sp = st.columns([1, 1, 2])
        with col_orig:
            st.image(
                Image.fromarray(preprocessed_display, "L").resize((140, 140), Image.NEAREST),
                caption="Input",
                width=140,
            )
        with col_heat:
            st.image(
                Image.fromarray(overlay).resize((140, 140), Image.NEAREST),
                caption="Grad-CAM Overlay",
                width=140,
            )

    st.markdown("<div style='color:#64748b;text-align:center;font-size:0.8rem;margin:0.5rem 0'>↓</div>", unsafe_allow_html=True)

    # ── Step 7: Prediction ────────────────────────────────────────────────────
    st.markdown(
        pipeline_step(
            7,
            "Final Prediction",
            "GlobalAveragePooling reduces each of the 64 final feature maps to a single value. "
            "A Dense layer maps these 64 values to 10 class scores. "
            "Softmax converts the scores to probabilities that sum to 1.",
        ),
        unsafe_allow_html=True,
    )

    st.markdown(
        f"""
        <div class="dv-prediction-result" style="max-width:400px">
            <div class="dv-predicted-digit">{result.predicted_digit}</div>
            <div style="color:#64748b;font-size:0.85rem">Predicted Digit</div>
            <div style="font-size:1.4rem;font-weight:700;color:#4ade80;margin-top:0.5rem">
                {result.confidence:.1%}
            </div>
            <div style="font-size:0.75rem;color:#64748b">Confidence</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("<div style='margin-top:1rem'></div>", unsafe_allow_html=True)
    st.plotly_chart(
        build_confidence_bar_chart(result.all_probabilities, result.predicted_digit),
        use_container_width=True,
    )

else:
    if analyse_clicked and not has_drawing:
        st.warning("Draw a digit first, then click Analyse.")

    # Placeholder instruction
    st.markdown(
        """
        <div style='text-align:center;color:#64748b;padding:4rem 0'>
            <div style='font-size:2.5rem;margin-bottom:0.75rem'>🧠</div>
            <div style='font-size:1rem;color:#94a3b8;font-weight:500'>
                Draw a digit in the canvas above, then click Analyse
            </div>
            <div style='font-size:0.85rem;margin-top:0.35rem'>
                The full processing pipeline will be visualised step by step
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ── Interview cheatsheet ──────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 📖 Interview Answers")
    with st.expander("What is a conv filter?"):
        st.markdown(
            "A small matrix (3×3 here) of learned weights that slides across the "
            "image computing a dot product at each position. It detects one specific "
            "local pattern wherever it appears."
        )
    with st.expander("What does pooling do?"):
        st.markdown(
            "Reduces spatial dimensions (halves width and height). MaxPool keeps the "
            "strongest activation in each 2×2 region, making the representation "
            "robust to small translations."
        )
    with st.expander("What is Grad-CAM?"):
        st.markdown(
            "Computes gradients of the class score w.r.t. conv activations. "
            "High-gradient filters are important — their spatial maps are averaged "
            "and upsampled to produce the heatmap."
        )
    with st.expander("Why GlobalAveragePooling?"):
        st.markdown(
            "Replaces Flatten. Reduces each of the 64 (7×7) feature maps to one "
            "number — much fewer parameters than flattening (3136 vs 64 inputs to Dense). "
            "Also acts as a regulariser."
        )
