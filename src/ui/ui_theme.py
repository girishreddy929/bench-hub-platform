"""UI styling only: theme CSS, logo, header, status chips, footer banner.
No application logic lives here. To get the plain look back, remove the calls to
these functions from app.py."""

import html
import os

import streamlit as st

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ASSETS_DIR = os.path.join(PROJECT_ROOT, "assets")  # src/ui/ui_theme.py -> project root
LOGO_PATH = os.path.join(ASSETS_DIR, "logo.png")
BANNER_PATH = os.path.join(ASSETS_DIR, "banner.png")

BRAND = "#1F4E9C"
BRAND_LIGHT = "#2F6FD0"

CSS = f"""
<style>
/* spacing */
.block-container {{ padding-top: 2.5rem; padding-bottom: 2rem; }}

/* hide Streamlit's default footer and deploy button */
footer {{ visibility: hidden; }}
[data-testid="stAppDeployButton"], .stDeployButton {{ display: none; }}

/* header banner */
.app-header {{
    background: linear-gradient(90deg, {BRAND} 0%, {BRAND_LIGHT} 100%);
    padding: 18px 26px;
    border-radius: 14px;
    margin-bottom: 18px;
}}
.app-title {{ color: #FFFFFF; font-size: 1.9rem; font-weight: 700; line-height: 1.2; }}
.app-subtitle {{ color: #DCE8FA; font-size: 0.95rem; margin-top: 2px; }}

/* headings */
h2, h3, h4 {{ color: #16305F; }}

/* tabs */
button[data-baseweb="tab"] {{ font-weight: 600; padding: 0.6rem 1.2rem; }}

/* buttons */
.stButton > button,
.stDownloadButton > button,
[data-testid="stFormSubmitButton"] > button,
[data-testid="stLinkButton"] a {{
    border-radius: 8px;
    font-weight: 600;
}}

/* cards: forms, expanders, bordered containers */
[data-testid="stForm"],
[data-testid="stExpander"] {{
    border-radius: 12px;
    box-shadow: 0 1px 4px rgba(22, 48, 95, 0.10);
}}
[data-testid="stVerticalBlockBorderWrapper"] {{ border-radius: 12px; }}
[data-testid="stExpander"] summary {{ font-weight: 600; }}

/* status chips */
.chip {{
    display: inline-block;
    padding: 2px 12px;
    border-radius: 999px;
    font-size: 0.82rem;
    font-weight: 600;
    white-space: nowrap;
}}
.chip-available {{ background: #E3F5EA; color: #1B7F3B; }}
.chip-interviewing {{ background: #FFF1D6; color: #9A6400; }}
.chip-allocated {{ background: #E2EBFA; color: {BRAND}; }}
.chip-other {{ background: #ECEEF2; color: #444B5A; }}

/* footer banner spacing */
.footer-space {{ height: 24px; }}
</style>
"""


def page_icon():
    """Logo as the browser-tab icon if it exists, otherwise Streamlit's default."""
    return LOGO_PATH if os.path.exists(LOGO_PATH) else None


def apply_theme() -> None:
    """Inject the CSS and show the logo at the top-left. Call once per run."""
    st.markdown(CSS, unsafe_allow_html=True)
    if os.path.exists(LOGO_PATH):
        st.logo(LOGO_PATH, size="large")


def render_header(title: str, subtitle: str = "") -> None:
    sub = f'<div class="app-subtitle">{html.escape(subtitle)}</div>' if subtitle else ""
    st.markdown(
        f'<div class="app-header"><div class="app-title">{html.escape(title)}</div>{sub}</div>',
        unsafe_allow_html=True,
    )


def status_chip(status: str) -> str:
    """HTML for a colored pill. Use with st.markdown(..., unsafe_allow_html=True)."""
    css_class = {
        "Available": "chip-available",
        "Interviewing": "chip-interviewing",
        "Allocated": "chip-allocated",
    }.get(status, "chip-other")
    return f'<span class="chip {css_class}">{html.escape(status)}</span>'


def render_footer() -> None:
    """Full-width banner at the bottom of the page (skipped if the file is missing)."""
    if not os.path.exists(BANNER_PATH):
        return
    st.markdown('<div class="footer-space"></div>', unsafe_allow_html=True)
    try:
        st.image(BANNER_PATH, use_container_width=True)
    except TypeError:  # older Streamlit versions
        st.image(BANNER_PATH, use_column_width=True)