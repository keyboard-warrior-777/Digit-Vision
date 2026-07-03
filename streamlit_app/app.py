"""
DigitVision — Application Entry Point & Home Page.

Sets up Streamlit page config, injects global CSS, renders the sidebar
footer, patches the sidebar nav labels via JavaScript, then renders the
Home page content.

Running:
    streamlit run streamlit_app/app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components
from components.styles import get_global_css
from home_content import render_home_page

# ── Project root on sys.path so 'from src.xxx import yyy' works everywhere ───
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# ── Page configuration ────────────────────────────────────────────────────────
st.set_page_config(
    page_title="DigitVision",
    page_icon="assets/logo.png",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Get Help": "https://github.com/keyboard-warrior-777",
        "Report a bug": "https://github.com/keyboard-warrior-777",
        "About": "DigitVision — Handwritten Digit Recognition with Deep Learning",
    },
)

# Set the logo for the sidebar (requires Streamlit >= 1.35)
st.logo("assets/logo.png")

# ── Global CSS ────────────────────────────────────────────────────────────────
st.markdown(get_global_css(), unsafe_allow_html=True)

# ── Sidebar footer ────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("<hr style='border-color:#2d3154;margin:1.5rem 0'>", unsafe_allow_html=True)
    st.markdown(
        """
        <div style='text-align:center;padding:0.5rem'>
            <div style='font-size:0.75rem;color:#7c8aaa;line-height:1.8'>
                DigitVision &nbsp;&middot;&nbsp; v1.0.0<br>
                TensorFlow &amp; Streamlit<br>
                <a href='https://github.com/keyboard-warrior-777'
                   target='_blank'
                   style='color:#6366f1;text-decoration:none;font-weight:500'>
                    github.com/keyboard-warrior-777
                </a>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ── Sidebar nav patch (JavaScript) ───────────────────────────────────────────
# In Streamlit 1.35 classic multipage the entry script always appears as the
# first sidebar nav item labelled "app".  CSS selectors are unreliable because
# the exact href / data-testid varies by build.  This script targets by TEXT
# CONTENT instead and is therefore version-agnostic.
#
# What it does:
#   1. Hides the "app" nav link (the entry-point item).
#   2. Capitalises "home" → "Home" so the sidebar reads naturally.
#   3. Installs a MutationObserver so the patch survives Streamlit's React
#      rerenders even when the user navigates to other pages.
components.html(
    """
    <script>
    (function () {
        function patchNav() {
            try {
                var doc = window.parent.document;

                // Walk every anchor/button/span in the document.
                // Streamlit 1.35 renders nav links as <a> tags inside <li> items.
                var els = doc.querySelectorAll('a, button, span');
                els.forEach(function (el) {
                    var raw = el.childNodes.length === 1
                        ? (el.firstChild.nodeValue || el.textContent || '')
                        : (el.textContent || '');
                    var text = raw.trim();

                    if (text === 'app') {
                        // Hide the "app" entrypoint entry and its parent <li>.
                        var li = el.closest('li');
                        if (li) {
                            li.style.cssText += ';display:none!important;';
                        } else {
                            el.style.cssText += ';display:none!important;';
                        }
                    }

                    if (text === 'home') {
                        // Capitalise "home" → "Home" so sidebar reads naturally.
                        if (el.firstChild && el.firstChild.nodeType === Node.TEXT_NODE) {
                            el.firstChild.nodeValue = 'Home';
                        } else {
                            el.textContent = 'Home';
                        }
                    }
                });
            } catch (e) {
                // Cross-frame access may be blocked in some environments — fail silently.
            }
        }

        // Run immediately (catches elements already in DOM).
        patchNav();

        // Re-run whenever React updates the DOM.
        try {
            new MutationObserver(patchNav).observe(
                window.parent.document.body,
                { childList: true, subtree: true }
            );
        } catch (e) {}
    })();
    </script>
    """,
    height=0,
    scrolling=False,
)

# ── Home page content ─────────────────────────────────────────────────────────
render_home_page()
