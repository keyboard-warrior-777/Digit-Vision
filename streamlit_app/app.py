"""
DigitVision — Streamlit Application Entry Point.

Configures the app-wide Streamlit settings, injects the global CSS,
adds the project root to sys.path (so all src/ imports work from any page),
and defines the multi-page navigation using st.navigation().

Running:
    streamlit run streamlit_app/app.py

    From the project root — this ensures the working directory is correct
    for all relative paths in config.py.
"""

from __future__ import annotations

import sys
from pathlib import Path

# ── Path setup ────────────────────────────────────────────────────────────────
# Streamlit adds streamlit_app/ to sys.path automatically (the script directory).
# We also need the project root so that `from src.xxx import yyy` works in pages.
# This modification persists for the lifetime of the process, so every page
# file benefits from it without repeating the setup.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import streamlit as st

from components.styles import get_global_css

# ── Page configuration ────────────────────────────────────────────────────────
# Must be called before any other Streamlit command.
st.set_page_config(
    page_title="DigitVision",
    page_icon="🔢",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Get Help": "https://github.com",
        "Report a bug": "https://github.com",
        "About": "DigitVision — Handwritten Digit Recognition with Deep Learning",
    },
)

# ── Global styles ─────────────────────────────────────────────────────────────
# Injected once here; persists across all pages in the same session.
st.markdown(get_global_css(), unsafe_allow_html=True)

# ── Navigation ────────────────────────────────────────────────────────────────
# st.navigation() gives full control over sidebar labels and icons,
# unlike the auto-discovery approach which derives names from filenames.
pages = [
    st.Page("pages/01_home.py",          title="Home",            icon="🏠", default=True),
    st.Page("pages/02_recognize.py",     title="Digit Recognition", icon="🎨"),
    st.Page("pages/03_playground.py",    title="Model Playground",  icon="🔬"),
    st.Page("pages/04_analytics.py",     title="Analytics",         icon="📊"),
    st.Page("pages/05_cnn_explainer.py", title="How CNN Thinks",    icon="🧠"),
    st.Page("pages/06_dataset.py",       title="Dataset Explorer",  icon="🗂️"),
    st.Page("pages/07_about.py",         title="About",             icon="ℹ️"),
]

pg = st.navigation(pages)

# ── Sidebar footer ────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("<hr style='border-color:#2d3154;margin:1.5rem 0'>", unsafe_allow_html=True)
    st.markdown(
        """
        <div style='text-align:center;padding:0.5rem'>
            <div style='font-size:0.7rem;color:#64748b;line-height:1.6'>
                DigitVision v1.0.0<br>
                Built with TensorFlow &amp; Streamlit<br>
                <span style='color:#6366f1'>★</span> Star on GitHub
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

pg.run()
