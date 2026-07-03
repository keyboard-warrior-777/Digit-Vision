"""
Home — DigitVision.

All content is defined in home_content.py and shared with app.py (the
root route).  This file exists so that the "/home" URL remains reachable
via the sidebar navigation link.
"""

from home_content import render_home_page

render_home_page()
