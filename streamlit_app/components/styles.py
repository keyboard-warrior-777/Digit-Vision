"""
Global CSS styles for DigitVision.

Injected once in app.py before page content so the styles apply to every
page in the same browser session.

Design System
─────────────
    Primary:   #6366f1  (indigo)    — interactive elements, headings
    Secondary: #22d3ee  (cyan)      — accents, gradients
    Surface:   #1a1d2e              — card backgrounds
    Elevated:  #252842              — raised elements
    Border:    #2d3154              — subtle dividers

Spacing (8 px grid)
────────────────────
    --sp-1  =  8px    --sp-2  = 16px    --sp-3  = 24px
    --sp-4  = 32px    --sp-6  = 48px    --sp-8  = 64px

Typography
──────────
    Inter is loaded from Google Fonts. It is the standard typeface in
    modern SaaS products and reads exceptionally well at small sizes.
"""

from __future__ import annotations


def get_global_css() -> str:
    """
    Return the complete CSS stylesheet as a Streamlit-injectable string.

    Usage:
        import streamlit as st
        from components.styles import get_global_css
        st.markdown(get_global_css(), unsafe_allow_html=True)
    """
    return """
<style>
/* ── Google Fonts ─────────────────────────────────────────────────────────── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

/* ── Design tokens ──────────────────────────────────────────────────────────*/
:root {
    /* Colours */
    --bg-primary:     #0f1117;
    --bg-surface:     #1a1d2e;
    --bg-elevated:    #252842;
    --bg-input:       #1e2135;
    --accent-primary: #6366f1;
    --accent-hover:   #4f46e5;
    --accent-cyan:    #22d3ee;
    --success:        #4ade80;
    --warning:        #fbbf24;
    --error:          #f87171;
    --text-primary:   #e2e8f0;
    --text-secondary: #94a3b8;
    --text-muted:     #7c8aaa;
    --border:         #2d3154;
    --border-accent:  rgba(99, 102, 241, 0.35);
    --shadow:         0 4px 24px rgba(0, 0, 0, 0.45);
    --shadow-hover:   0 8px 40px rgba(99, 102, 241, 0.2);
    --radius:         12px;
    --radius-sm:      8px;
    --radius-lg:      16px;
    --gradient:       linear-gradient(135deg, #6366f1 0%, #22d3ee 100%);

    /* 8-px spacing grid */
    --sp-1: 8px;   --sp-2: 16px;  --sp-3: 24px;
    --sp-4: 32px;  --sp-6: 48px;  --sp-8: 64px;
}

/* ── Base ───────────────────────────────────────────────────────────────────*/
html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont,
                 'Segoe UI Emoji', 'Apple Color Emoji', 'Noto Color Emoji',
                 sans-serif !important;
}

/* ── Hide Streamlit chrome (preserve sidebar collapse/expand button) ─────── */
/* 'header' is intentionally NOT hidden — the sidebar toggle lives inside it. */
#MainMenu { visibility: hidden; }
footer    { visibility: hidden; }

/* ── Hide app.py entry-point from sidebar navigation ───────────────────────*/
/* Target by href rather than data-testid (varies across Streamlit builds).   */
a[href="/"] { display: none !important; }
li:has(a[href="/"]) { display: none !important; }
li:has(> div > a[href="/"]) { display: none !important; }

/* ── Main content area ──────────────────────────────────────────────────────*/
.stApp { background-color: var(--bg-primary); }
.main .block-container {
    padding: 2rem 2.5rem 5rem 2.5rem;
    max-width: 1400px;
}

/* ── Page header ────────────────────────────────────────────────────────────*/
.dv-page-header {
    padding: 1.5rem 0 1.5rem 0;
    margin-bottom: var(--sp-4);
    border-bottom: 1px solid var(--border);
}
.dv-page-title {
    font-size: 2.25rem;
    font-weight: 800;
    background: var(--gradient);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin: 0 0 0.4rem 0;
    line-height: 1.15;
    letter-spacing: -0.02em;
}
.dv-page-subtitle {
    font-size: 1rem;
    color: var(--text-secondary);
    font-weight: 400;
    margin: 0;
    line-height: 1.5;
}

/* ── Generic cards ──────────────────────────────────────────────────────────*/
.dv-card {
    background: var(--bg-surface);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 1.5rem;
    transition: border-color 0.2s ease, box-shadow 0.2s ease, transform 0.15s ease;
}
.dv-card:hover {
    border-color: var(--border-accent);
    box-shadow: var(--shadow-hover);
    transform: translateY(-1px);
}
.dv-card-accent {
    background: var(--bg-surface);
    border: 1px solid var(--border-accent);
    border-top: 3px solid var(--accent-primary);
    border-radius: var(--radius);
    padding: 1.5rem;
}

/* ── Metric cards ───────────────────────────────────────────────────────────*/
.dv-metric-card {
    background: var(--bg-surface);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: var(--sp-3) var(--sp-2);
    text-align: center;
    min-height: 148px;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    transition: all 0.2s ease;
    position: relative;
    overflow: hidden;
}
.dv-metric-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
    background: var(--gradient);
}
.dv-metric-card:hover {
    border-color: var(--border-accent);
    box-shadow: var(--shadow-hover);
    transform: translateY(-2px);
}
.dv-metric-icon {
    font-size: 1.75rem;
    margin-bottom: var(--sp-1);
    display: block;
    line-height: 1;
}
.dv-metric-value {
    font-size: 1.875rem;
    font-weight: 800;
    color: var(--text-primary);
    line-height: 1;
    margin-bottom: 4px;
    letter-spacing: -0.02em;
}
.dv-metric-label {
    font-size: 0.72rem;
    color: var(--text-muted);
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.09em;
}
.dv-metric-delta {
    font-size: 0.8rem;
    font-weight: 600;
    margin-top: 6px;
}
.dv-metric-delta.positive { color: var(--success); }
.dv-metric-delta.negative { color: var(--error); }

/* ── Status badges ──────────────────────────────────────────────────────────*/
.dv-badge {
    display: inline-flex;
    align-items: center;
    gap: 0.35rem;
    padding: 0.2rem 0.65rem;
    border-radius: 999px;
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 0.04em;
}
.dv-badge-success {
    background: rgba(74, 222, 128, 0.12);
    color: var(--success);
    border: 1px solid rgba(74, 222, 128, 0.3);
}
.dv-badge-warning {
    background: rgba(251, 191, 36, 0.12);
    color: var(--warning);
    border: 1px solid rgba(251, 191, 36, 0.3);
}
.dv-badge-error {
    background: rgba(248, 113, 113, 0.12);
    color: var(--error);
    border: 1px solid rgba(248, 113, 113, 0.3);
}
.dv-badge-info {
    background: rgba(99, 102, 241, 0.12);
    color: var(--accent-primary);
    border: 1px solid var(--border-accent);
}

/* ── Model comparison cards ─────────────────────────────────────────────────*/
.dv-model-card {
    background: var(--bg-surface);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 1.5rem;
    height: 100%;
    display: flex;
    flex-direction: column;
    transition: all 0.2s ease;
}
.dv-model-card:hover {
    border-color: var(--border-accent);
    box-shadow: var(--shadow-hover);
    transform: translateY(-2px);
}
.dv-model-card.best-model {
    border-color: var(--accent-primary);
    box-shadow: 0 0 0 1px var(--border-accent), var(--shadow-hover);
}
.dv-model-card-title {
    font-size: 1.05rem;
    font-weight: 700;
    color: var(--text-primary);
    margin-bottom: 0.4rem;
    line-height: 1.3;
}
.dv-model-card-subtitle {
    font-size: 0.8rem;
    color: var(--text-muted);
    line-height: 1.55;
    flex: 1;
    min-height: 2.5rem;
}
/* CSS divider — replaces bare <hr> which can render as literal text */
.dv-model-card-divider {
    border: none;
    border-top: 1px solid var(--border);
    margin: var(--sp-2) 0;
    flex-shrink: 0;
}
.dv-model-stat {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 0.45rem 0;
    border-bottom: 1px solid rgba(45, 49, 84, 0.6);
    font-size: 0.875rem;
}
.dv-model-stat:last-child { border-bottom: none; }
.dv-model-stat-label { color: var(--text-secondary); }
.dv-model-stat-value { font-weight: 600; color: var(--text-primary); }

/* ── Prediction result display ──────────────────────────────────────────────*/
.dv-prediction-result {
    background: linear-gradient(135deg, rgba(99,102,241,0.12) 0%, rgba(34,211,238,0.08) 100%);
    border: 1px solid var(--border-accent);
    border-radius: var(--radius-lg);
    padding: var(--sp-4);
    text-align: center;
}
.dv-predicted-digit {
    font-size: 5rem;
    font-weight: 900;
    background: var(--gradient);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    line-height: 1;
    margin-bottom: var(--sp-1);
    font-family: 'JetBrains Mono', monospace;
}
.dv-confidence-value {
    font-size: 1.5rem;
    font-weight: 700;
    color: var(--text-primary);
}
.dv-confidence-label {
    font-size: 0.875rem;
    color: var(--text-muted);
}

/* ── Section headers ────────────────────────────────────────────────────────*/
.dv-section-header {
    font-size: 0.72rem;
    font-weight: 700;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin: var(--sp-3) 0 var(--sp-2) 0;
    padding-bottom: var(--sp-1);
    border-bottom: 1px solid var(--border);
}

/* ── Info / callout boxes ───────────────────────────────────────────────────*/
.dv-info-box {
    background: rgba(99, 102, 241, 0.07);
    border: 1px solid var(--border-accent);
    border-left: 3px solid var(--accent-primary);
    border-radius: var(--radius-sm);
    padding: var(--sp-2) var(--sp-3);
    margin: var(--sp-2) 0;
    font-size: 0.9rem;
    color: var(--text-secondary);
    line-height: 1.65;
}
.dv-info-box strong { color: var(--text-primary); }

/* ── Pipeline steps ─────────────────────────────────────────────────────────*/
.dv-pipeline-step {
    background: var(--bg-surface);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: var(--sp-3) var(--sp-2);
    text-align: center;
    position: relative;
    transition: border-color 0.2s ease;
}
.dv-pipeline-step:hover { border-color: var(--border-accent); }
.dv-pipeline-step-number {
    position: absolute;
    top: -10px; left: 50%; transform: translateX(-50%);
    background: var(--accent-primary);
    color: white;
    font-size: 0.68rem;
    font-weight: 700;
    padding: 2px 10px;
    border-radius: 999px;
    letter-spacing: 0.04em;
}
.dv-pipeline-step-title {
    font-weight: 600;
    color: var(--text-primary);
    margin: 0.5rem 0 0.25rem 0;
    font-size: 0.9rem;
}
.dv-pipeline-step-desc {
    font-size: 0.78rem;
    color: var(--text-muted);
    line-height: 1.55;
}

/* ── Sidebar ────────────────────────────────────────────────────────────────*/
.css-1d391kg, [data-testid="stSidebar"] {
    background: var(--bg-surface) !important;
}
[data-testid="stSidebar"] .dv-sidebar-section {
    padding: var(--sp-2);
    border-bottom: 1px solid var(--border);
}

/* ── Tables ─────────────────────────────────────────────────────────────────*/
.dv-table {
    width: 100%;
    border-collapse: collapse;
    font-size: 0.875rem;
}
.dv-table th {
    background: var(--bg-elevated);
    color: var(--text-muted);
    font-weight: 600;
    font-size: 0.72rem;
    text-transform: uppercase;
    letter-spacing: 0.07em;
    padding: 0.75rem 1rem;
    text-align: left;
    border-bottom: 1px solid var(--border);
}
.dv-table td {
    padding: 0.65rem 1rem;
    border-bottom: 1px solid var(--border);
    color: var(--text-primary);
}
.dv-table tr:last-child td { border-bottom: none; }
.dv-table tr:hover td { background: rgba(99, 102, 241, 0.04); }

/* ── Streamlit native overrides ─────────────────────────────────────────────*/

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, var(--accent-primary) 0%, #4f46e5 100%) !important;
    color: white !important;
    border: none !important;
    border-radius: var(--radius-sm) !important;
    font-weight: 600 !important;
    font-family: 'Inter', sans-serif !important;
    letter-spacing: 0.02em !important;
    padding: 0.5rem 1.5rem !important;
    transition: all 0.2s ease !important;
    box-shadow: 0 2px 8px rgba(99, 102, 241, 0.25) !important;
}
.stButton > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 16px rgba(99, 102, 241, 0.4) !important;
    filter: brightness(1.1) !important;
}
.stButton > button:active { transform: translateY(0) !important; }

/* Selectbox / multiselect */
.stSelectbox > div > div,
.stMultiSelect > div > div {
    background: var(--bg-input) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius-sm) !important;
}
.stSelectbox > div > div:focus-within,
.stMultiSelect > div > div:focus-within {
    border-color: var(--accent-primary) !important;
    box-shadow: 0 0 0 2px var(--border-accent) !important;
}

/* Slider */
[data-testid="stSlider"] > div > div > div > div {
    background: var(--gradient) !important;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    background: transparent !important;
    border-bottom: 1px solid var(--border) !important;
    gap: 0 !important;
    padding: 0 !important;
}
.stTabs [data-baseweb="tab"] {
    background: transparent !important;
    color: var(--text-muted) !important;
    font-weight: 500 !important;
    font-family: 'Inter', sans-serif !important;
    border-radius: 0 !important;
    border-bottom: 2px solid transparent !important;
    padding: 0.65rem 1.25rem !important;
    transition: color 0.15s ease, border-color 0.15s ease !important;
}
.stTabs [data-baseweb="tab"]:hover {
    color: var(--text-primary) !important;
    border-bottom-color: var(--border-accent) !important;
}
.stTabs [aria-selected="true"] {
    background: transparent !important;
    color: var(--accent-primary) !important;
    border-bottom: 2px solid var(--accent-primary) !important;
    font-weight: 600 !important;
}
.stTabs [data-baseweb="tab-panel"] {
    padding-top: var(--sp-3) !important;
}

/* Expanders */
.streamlit-expanderHeader {
    background: var(--bg-surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius-sm) !important;
    color: var(--text-primary) !important;
    font-weight: 600 !important;
    font-size: 0.925rem !important;
    padding: 0.75rem 1rem !important;
    transition: background 0.2s ease, border-color 0.2s ease !important;
}
.streamlit-expanderHeader:hover {
    border-color: var(--border-accent) !important;
    background: var(--bg-elevated) !important;
}
.streamlit-expanderContent {
    border: 1px solid var(--border) !important;
    border-top: none !important;
    border-radius: 0 0 var(--radius-sm) var(--radius-sm) !important;
    background: var(--bg-surface) !important;
    padding: var(--sp-2) !important;
}

/* Dataframe */
[data-testid="stDataFrame"] {
    border: 1px solid var(--border) !important;
    border-radius: var(--radius-sm) !important;
    overflow: hidden !important;
}

/* Radio buttons */
.stRadio > label {
    color: var(--text-primary) !important;
    font-weight: 500 !important;
}

/* st.info / st.warning / st.error / st.success callouts */
[data-testid="stNotification"] {
    border-radius: var(--radius-sm) !important;
}

/* Input / text_input */
.stTextInput > div > div > input {
    background: var(--bg-input) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius-sm) !important;
    color: var(--text-primary) !important;
}
.stTextInput > div > div > input:focus {
    border-color: var(--accent-primary) !important;
    box-shadow: 0 0 0 2px var(--border-accent) !important;
}

/* Plotly chart container */
[data-testid="stPlotlyChart"] {
    border-radius: var(--radius) !important;
    overflow: hidden !important;
}

/* ── Utility classes ────────────────────────────────────────────────────────*/
.text-gradient {
    background: var(--gradient);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}
.text-muted    { color: var(--text-muted) !important; }
.text-secondary{ color: var(--text-secondary) !important; }
.font-mono     { font-family: 'JetBrains Mono', monospace !important; }
.divider       { border: none; border-top: 1px solid var(--border); margin: var(--sp-3) 0; }
</style>
"""
