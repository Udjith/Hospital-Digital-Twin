"""Shared presentation only: original dark theme for final and historical UI."""
import streamlit as st

COLORS = {
    "bg": "#090c10",
    "bg2": "#0e1319",
    "surface": "#151b22",
    "surface2": "#1a222b",
    "surface3": "#202a34",
    "text": "#f3f6f8",
    "muted": "#aab4be",
    "border": "#33404c",
    "accent": "#82c0ea",
    "accent_hover": "#96ccef",
    "accent_text": "#071018",
    "shadow": "rgba(0,0,0,0.58)",
    "shadow_soft": "rgba(0,0,0,0.32)",
    "slider": "#82c0ea",
    "slider_track": "#34414d",
    "input": "#111820",
    "success": "#82c0ea",
}


def apply_dark_theme():
    st.markdown(
        f"""
        <style>
        :root {{
            --app-bg: {COLORS["bg"]};
            --app-bg-2: {COLORS["bg2"]};
            --surface: {COLORS["surface"]};
            --surface-2: {COLORS["surface2"]};
            --surface-3: {COLORS["surface3"]};
            --text: {COLORS["text"]};
            --muted: {COLORS["muted"]};
            --border: {COLORS["border"]};
            --accent: {COLORS["accent"]};
            --accent-hover: {COLORS["accent_hover"]};
            --accent-text: {COLORS["accent_text"]};
            --shadow: {COLORS["shadow"]};
            --shadow-soft: {COLORS["shadow_soft"]};
            --slider: {COLORS["slider"]};
            --slider-track: {COLORS["slider_track"]};
            --input: {COLORS["input"]};
        }}

        html, body, .stApp, [data-testid="stAppViewContainer"] {{
            background:
                radial-gradient(circle at 10% 0%, var(--app-bg-2) 0%, transparent 34%),
                linear-gradient(180deg, var(--app-bg) 0%, var(--app-bg-2) 100%);
            color: var(--text) !important;
        }}

        [data-testid="stHeader"] {{ background: transparent; }}

        .block-container {{
            max-width: 1500px;
            padding-top: 1.25rem;
            padding-bottom: 3rem;
        }}

        [data-testid="stSidebar"] {{
            background: linear-gradient(180deg, var(--surface), var(--surface-2));
            border-right: 1px solid var(--border);
            box-shadow: 10px 0 30px var(--shadow-soft);
        }}

        [data-testid="stSidebar"] * {{
            color: var(--text);
        }}

        .hero {{
            padding: 1.6rem 1.8rem;
            border: 1px solid var(--border);
            border-radius: 22px;
            background: linear-gradient(145deg, var(--surface), var(--surface-2));
            box-shadow:
                0 16px 34px var(--shadow-soft),
                inset 0 1px 0 rgba(255,255,255,0.08);
            margin-bottom: 1rem;
        }}

        .hero h1 {{
            margin: 0.15rem 0 0;
            font-size: 2.08rem;
            letter-spacing: -0.035em;
            color: var(--text) !important;
        }}

        .hero p {{
            margin: 0.5rem 0 0;
            max-width: 920px;
            line-height: 1.55;
            color: var(--muted) !important;
        }}

        .section-kicker {{
            color: var(--muted);
            font-size: 0.78rem;
            font-weight: 800;
            letter-spacing: 0.11em;
            text-transform: uppercase;
        }}

        .status-strip {{
            display: grid;
            grid-template-columns: repeat(4, minmax(0,1fr));
            gap: 0.75rem;
            margin: 0 0 1rem;
        }}

        .status-box {{
            border: 1px solid var(--border);
            background: linear-gradient(145deg, var(--surface), var(--surface-2));
            border-radius: 15px;
            padding: 0.85rem 1rem;
            box-shadow: 0 8px 18px var(--shadow-soft);
            min-width: 0;
            min-height: 88px;
        }}

        .status-label {{
            color: var(--muted);
            font-size: 0.74rem;
            text-transform: uppercase;
            letter-spacing: 0.075em;
            font-weight: 750;
        }}

        .status-value {{
            color: var(--text);
            font-size: 1rem;
            font-weight: 760;
            margin-top: 0.15rem;
            overflow-wrap: anywhere;
            line-height: 1.3;
        }}

        [data-testid="stVerticalBlockBorderWrapper"] {{
            background: linear-gradient(145deg, var(--surface), var(--surface-2));
            border: 1px solid var(--border) !important;
            border-radius: 20px !important;
            box-shadow:
                0 15px 30px var(--shadow-soft),
                0 3px 8px var(--shadow-soft),
                inset 0 1px 0 rgba(255,255,255,0.07);
            padding: 0.45rem 0.55rem;
            margin-bottom: 0.7rem;
        }}

        [data-testid="stMetric"] {{
            background: linear-gradient(150deg, var(--surface), var(--surface-3));
            border: 1px solid var(--border);
            border-radius: 16px;
            padding: 0.95rem 1rem;
            min-height: 106px;
            box-shadow:
                0 9px 20px var(--shadow-soft),
                inset 0 1px 0 rgba(255,255,255,0.07);
            transition: transform 140ms ease, box-shadow 140ms ease;
        }}

        [data-testid="stMetric"]:hover {{
            transform: translateY(-2px);
            box-shadow: 0 14px 26px var(--shadow);
        }}

        [data-testid="stMetricLabel"] *,
        [data-testid="stMetricLabel"] {{
            color: var(--muted) !important;
            font-weight: 650 !important;
            white-space: normal !important;
            overflow-wrap: anywhere;
            line-height: 1.25 !important;
        }}

        [data-testid="stMetricValue"] *,
        [data-testid="stMetricValue"] {{
            color: var(--text) !important;
            font-size: clamp(1.35rem, 1.8vw, 2rem) !important;
            line-height: 1.15 !important;
            white-space: normal !important;
            overflow-wrap: anywhere;
        }}

        [data-testid="stMetricDelta"] *,
        [data-testid="stMetricDelta"] {{
            color: var(--muted) !important;
            white-space: normal !important;
            overflow-wrap: anywhere;
        }}

        [data-baseweb="tab-list"] {{
            gap: 0.35rem;
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: 14px;
            padding: 0.35rem;
            box-shadow: 0 8px 18px var(--shadow-soft);
        }}

        [data-baseweb="tab"] {{
            border-radius: 10px;
            padding: 0.55rem 0.9rem;
            color: var(--muted) !important;
        }}

        [aria-selected="true"][data-baseweb="tab"] {{
            background: var(--surface-3);
            color: var(--text) !important;
            font-weight: 750;
            box-shadow: inset 0 -2px 0 var(--accent);
        }}

        input, textarea,
        [data-baseweb="select"] > div,
        [data-baseweb="input"] > div {{
            background: var(--input) !important;
            color: var(--text) !important;
            border-color: var(--border) !important;
            border-radius: 11px !important;
        }}

        input::placeholder, textarea::placeholder {{
            color: var(--muted) !important;
            opacity: 0.9;
        }}

        [data-testid="stSlider"] [data-baseweb="slider"] > div {{
            background: var(--slider-track) !important;
        }}

        [data-testid="stSlider"] [data-baseweb="slider"] > div > div {{
            background: var(--accent) !important;
        }}

        [data-testid="stSlider"] [role="slider"] {{
            background: var(--slider) !important;
            border: 3px solid var(--surface) !important;
            box-shadow: 0 0 0 1px var(--accent), 0 3px 10px var(--shadow) !important;
        }}

        .stButton > button {{
            border-radius: 12px;
            border: 1px solid var(--border);
            background: var(--surface-3);
            color: var(--text) !important;
            min-height: 2.75rem;
            font-weight: 750;
            box-shadow: 0 8px 17px var(--shadow-soft);
            transition: transform 140ms ease, box-shadow 140ms ease;
        }}

        .stButton > button * {{
            color: inherit !important;
        }}

        .stButton > button[kind="primary"] {{
            background: var(--accent);
            color: var(--accent-text) !important;
            border-color: var(--accent);
        }}

        .stButton > button:hover {{
            transform: translateY(-1px);
            box-shadow: 0 12px 22px var(--shadow);
            border-color: var(--accent);
            color: var(--accent-hover) !important;
        }}

        .stButton > button[kind="primary"]:hover {{
            background: var(--accent-hover);
            color: var(--accent-text) !important;
        }}

        [data-testid="stSidebarCollapseButton"],
        [data-testid="stSidebarCollapseButton"] *,
        [data-testid="stHeader"] button[kind="header"],
        [data-testid="stHeader"] button[kind="header"] * {{
            color: var(--muted) !important;
            fill: var(--muted) !important;
            stroke: var(--muted) !important;
        }}

        [data-testid="stSidebar"] [data-testid="stTooltipIcon"] {{
            display: inline-flex !important;
            align-items: center !important;
            justify-content: center !important;
            position: static !important;
            flex: 0 0 1.1rem !important;
            width: 1.1rem !important;
            height: 1.1rem !important;
            min-width: 1.1rem !important;
            margin-left: 0.3rem !important;
            padding: 0 !important;
            border: 0 !important;
            border-radius: 999px !important;
            background: transparent !important;
            color: var(--muted) !important;
            opacity: 0.9;
            vertical-align: middle;
            transition: color 120ms ease, background 120ms ease, opacity 120ms ease;
        }}

        [data-testid="stSidebar"] [data-testid="stTooltipIcon"] svg {{
            width: 0.82rem !important;
            height: 0.82rem !important;
            color: inherit !important;
        }}

        [data-testid="stSidebar"] [data-testid="stTooltipIcon"]:hover {{
            color: var(--accent) !important;
            background: rgba(130, 192, 234, 0.12) !important;
            opacity: 1;
        }}

        .sidebar-section {{
            margin: 1.25rem 0 0.65rem;
            padding-top: 0.85rem;
            border-top: 1px solid var(--border);
        }}

        .sidebar-section-title {{
            color: var(--accent) !important;
            font-size: 0.76rem;
            font-weight: 800;
            letter-spacing: 0.075em;
            line-height: 1.25;
            text-transform: uppercase;
        }}

        [data-testid="stSidebar"] [data-testid="stWidgetLabel"] {{
            display: flex !important;
            align-items: center !important;
            margin-bottom: 0.22rem;
        }}

        [data-testid="stSidebar"] [data-testid="stWidgetLabel"] p {{
            font-size: 0.88rem;
            font-weight: 650;
            line-height: 1.3;
        }}

        /* Keep the sidebar CTA area independent from Streamlit's column sizing. */
        [data-testid="stSidebar"] .st-key-optimization_actions,
        [data-testid="stSidebar"] .st-key-optimization_actions [data-testid="stVerticalBlock"],
        [data-testid="stSidebar"] .st-key-optimization_actions .stButton {{
            width: 100% !important;
            min-width: 0 !important;
        }}

        [data-testid="stSidebar"] .st-key-optimization_actions .stButton {{
            margin-bottom: 0.5rem;
        }}

        [data-testid="stSidebar"] .st-key-optimization_actions .stButton > button {{
            display: flex !important;
            align-items: center !important;
            justify-content: center !important;
            width: 100% !important;
            min-width: 100% !important;
            min-height: 2.8rem !important;
            height: auto !important;
            padding: 0.65rem 0.9rem !important;
            white-space: nowrap !important;
            word-break: normal !important;
            overflow-wrap: normal !important;
        }}

        [data-testid="stSidebar"] .st-key-optimization_actions .stButton > button * {{
            white-space: nowrap !important;
            word-break: normal !important;
            overflow-wrap: normal !important;
            text-align: center !important;
        }}

        [data-testid="stExpander"],
        [data-testid="stStatusWidget"],
        [data-testid="stAlert"] {{
            background: var(--surface) !important;
            color: var(--text) !important;
            border: 1px solid var(--border);
            border-radius: 14px;
            box-shadow: 0 7px 16px var(--shadow-soft);
        }}

        [data-testid="stDataFrame"] {{
            border: 1px solid var(--border);
            border-radius: 14px;
            overflow: hidden;
            box-shadow: 0 7px 16px var(--shadow-soft);
        }}

        h1, h2, h3, h4, h5, h6, p, label, span {{
            color: var(--text);
        }}

        [data-testid="stCaptionContainer"],
        [data-testid="stCaptionContainer"] * {{
            color: var(--muted) !important;
        }}

        hr {{ border-color: var(--border) !important; }}

        .rule-flow {{
            display: flex;
            align-items: stretch;
            gap: 0.55rem;
            flex-wrap: wrap;
            margin: 0.6rem 0 1rem;
        }}

        .rule-step {{
            flex: 1 1 220px;
            padding: 0.85rem 1rem;
            border-radius: 14px;
            border: 1px solid var(--border);
            border-left: 3px solid var(--accent);
            background: var(--surface-3);
            box-shadow: 0 7px 16px var(--shadow-soft);
            color: var(--text);
        }}

        .rule-num {{
            display: inline-flex;
            min-width: 1.55rem;
            padding: 0 0.3rem;
            height: 1.55rem;
            border-radius: 999px;
            align-items: center;
            justify-content: center;
            background: var(--accent);
            color: var(--accent-text) !important;
            font-size: 0.78rem;
            font-weight: 800;
            margin-right: 0.4rem;
        }}

        .change-box {{
            padding: 0.9rem 1rem;
            border-radius: 14px;
            border: 1px solid var(--border);
            background: var(--surface-3);
            color: var(--text);
            margin-bottom: 0.45rem;
        }}

        code, pre {{
            color: var(--text) !important;
        }}

        [data-testid="stDecoration"] {{ display: none; }}

        @media (max-width: 900px) {{
            .status-strip {{
                grid-template-columns: repeat(2, minmax(0,1fr));
            }}

            [data-testid="stMetric"] {{
                min-height: 96px;
            }}
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )
