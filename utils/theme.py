#!/usr/bin/env python
"""
TechPulse — Centralized Design System & Theme Engine (Red, Black and Gold).

Single source of truth for the visual language:
- Background: #080808 (Pitch Black)
- Secondary background: #111111 (Jet Black)
- Card background: #151515 (Card Black)
- Primary red: #E32636
- Deep red: #8B0000
- Gold: #D4AF37
- Light gold: #FFD76A
- Primary text: #FFFFFF
- Secondary text: #B8B8B8
- Borders: #3A2020
"""
from __future__ import annotations

import streamlit as st

# ── Design tokens ────────────────────────────────────────────────────────────
BG_DARK = "#080808"
BG_SECONDARY = "#111111"
BG_CARD = "#151515"
PRIMARY = "#E32636"          # Crimson / Primary Red
PRIMARY_DARK = "#8B0000"     # Deep Red
GOLD = "#D4AF37"             # Metallic Gold
LIGHT_GOLD = "#FFD76A"       # Bright Light Gold
TEXT_PRIMARY = "#FFFFFF"     # Crisp White
TEXT_SECONDARY = "#B8B8B8"   # Secondary Gray
BORDER_COLOR = "#3A2020"     # Subtle Red Border
SUCCESS = "#22C55E"          # Green
WARNING = "#D4AF37"          # Gold

# High-contrast categorical palette used by Plotly charts
PALETTE = [
    PRIMARY, GOLD, LIGHT_GOLD, PRIMARY_DARK, SUCCESS,
    "#FF8C00", "#F43F5E", TEXT_SECONDARY,
]

# Plotly dark template tuned for Red, Black and Gold theme
PLOTLY_DARK_TEMPLATE = {
    "layout": {
        "font": {
            "family": "Inter, 'Segoe UI', sans-serif",
            "size": 13,
            "color": TEXT_PRIMARY,
        },
        "paper_bgcolor": "rgba(0,0,0,0)",
        "plot_bgcolor": "rgba(0,0,0,0)",
        "margin": {"l": 15, "r": 15, "t": 35, "b": 15},
        "hoverlabel": {
            "bgcolor": BG_CARD,
            "bordercolor": GOLD,
            "font": {"color": LIGHT_GOLD, "family": "Inter, sans-serif"},
        },
        "xaxis": {
            "gridcolor": "rgba(58, 32, 32, 0.4)",
            "zerolinecolor": BORDER_COLOR,
            "showline": False,
            "tickfont": {"color": TEXT_SECONDARY},
            "title": {"font": {"color": LIGHT_GOLD, "size": 13}},
        },
        "yaxis": {
            "gridcolor": "rgba(58, 32, 32, 0.4)",
            "zerolinecolor": BORDER_COLOR,
            "showline": False,
            "tickfont": {"color": TEXT_SECONDARY},
            "title": {"font": {"color": LIGHT_GOLD, "size": 13}},
        },
        "colorway": PALETTE,
        "legend": {
            "orientation": "h",
            "y": -0.18,
            "font": {"color": TEXT_PRIMARY},
        },
    }
}

PLOTLY_TEMPLATE = PLOTLY_DARK_TEMPLATE


def apply_base_css() -> None:
    """Inject the global stylesheet with Red, Black and Gold SaaS styling."""
    st.markdown(
        """
        <style>
        /* ── Global typography & layout ─────────────────────────────── */
        .block-container {
            padding-top: 1.5rem;
            padding-bottom: 3.5rem;
            max-width: 1250px;
        }
        h1, h2, h3 {
            font-weight: 700;
            letter-spacing: -0.02em;
            color: #FFFFFF !important;
        }
        h1 { color: #FFD76A !important; }
        h2 { color: #FFFFFF !important; }
        h3 { color: #D4AF37 !important; }

        /* ── Card containers ─────────────────────────────────────────── */
        div[data-testid="stVerticalBlockBorderWrapper"] {
            border-radius: 12px;
            border: 1px solid #3A2020 !important;
            background: #151515 !important;
            box-shadow: 0 4px 10px rgba(0, 0, 0, 0.6) !important;
        }

        /* ── KPI Metric cards ───────────────────────────────────────── */
        div[data-testid="stMetric"] {
            background: linear-gradient(180deg, #151515 0%, #111111 100%) !important;
            border: 1px solid #3A2020 !important;
            border-radius: 12px !important;
            padding: 14px 18px !important;
            box-shadow: 0 2px 6px rgba(0, 0, 0, 0.5) !important;
        }
        div[data-testid="stMetric"] label {
            font-size: .8rem !important;
            font-weight: 600 !important;
            text-transform: uppercase !important;
            letter-spacing: .05em !important;
            color: #D4AF37 !important; /* Gold label */
        }
        div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
            color: #FFFFFF !important;
            font-weight: 700 !important;
        }

        /* ── Primary & Secondary Buttons ─────────────────────────────── */
        .stButton > button {
            border-radius: 8px;
            font-weight: 600;
            transition: all 0.15s ease-in-out;
            border: 1px solid #3A2020 !important;
        }
        .stButton > button[kind="primary"] {
            background: linear-gradient(135deg, #E32636 0%, #8B0000 100%) !important;
            color: #FFFFFF !important;
            border: 1px solid #E32636 !important;
            box-shadow: 0 2px 8px rgba(227, 38, 54, 0.4);
        }
        .stButton > button[kind="primary"]:hover {
            filter: brightness(1.15);
            box-shadow: 0 3px 12px rgba(227, 38, 54, 0.6);
            transform: translateY(-1px);
        }
        .stButton > button[kind="secondary"] {
            background: #151515 !important;
            color: #D4AF37 !important;
            border: 1px solid #3A2020 !important;
        }
        .stButton > button[kind="secondary"]:hover {
            border-color: #D4AF37 !important;
            color: #FFD76A !important;
        }

        /* ── Badges & Tags ───────────────────────────────────────────── */
        span[data-testid="stBadge"] {
            background: #1A1010 !important;
            border: 1px solid #3A2020 !important;
            color: #FFD76A !important;
        }

        /* ── Sidebar ─────────────────────────────────────────────────── */
        section[data-testid="stSidebar"] {
            background-color: #080808 !important;
            border-right: 1px solid #3A2020 !important;
        }
        section[data-testid="stSidebar"] > div:first-child {
            padding-top: 1.2rem;
        }

        /* ── Chat Messages ───────────────────────────────────────────── */
        div[data-testid="stChatMessage"] {
            border: 1px solid #3A2020 !important;
            border-radius: 12px !important;
            background: #111111 !important;
            color: #FFFFFF !important;
        }

        /* ── Dataframes & Tables ─────────────────────────────────────── */
        div[data-testid="stDataFrame"] {
            border: 1px solid #3A2020 !important;
            border-radius: 10px !important;
            background: #111111 !important;
        }

        /* ── Expanders ───────────────────────────────────────────────── */
        details[data-testid="stExpander"] {
            border: 1px solid #3A2020 !important;
            border-radius: 10px !important;
            background: #111111 !important;
        }
        details[data-testid="stExpander"] summary {
            color: #FFD76A !important;
        }

        /* ── Hero / Page Header (Red, Black and Gold) ────────────────── */
        .tp-page-header {
            display: flex;
            align-items: flex-start;
            gap: 16px;
            padding: 20px 24px;
            margin-bottom: 22px;
            background: linear-gradient(135deg, #180B0B 0%, #111111 60%, #151515 100%);
            border: 1px solid #3A2020;
            border-left: 5px solid #E32636;
            border-radius: 14px;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.7);
        }
        .tp-page-header .tp-ico {
            font-size: 2rem;
            line-height: 1.2;
            color: #FFD76A;
        }
        .tp-page-header h1 {
            margin: 0;
            font-size: 1.6rem;
            color: #FFFFFF !important;
            font-weight: 700;
        }
        .tp-page-header p {
            margin: 6px 0 0;
            color: #B8B8B8;
            font-size: .95rem;
            line-height: 1.4;
        }

        /* ── Hero (Dashboard overview) ───────────────────────────────── */
        .tp-hero {
            padding: 28px 32px;
            border-radius: 16px;
            margin-bottom: 24px;
            background: linear-gradient(135deg, #8B0000 0%, #E32636 50%, #151515 100%);
            color: #FFFFFF;
            border: 1px solid #D4AF37;
            box-shadow: 0 8px 24px rgba(227, 38, 54, 0.3);
        }
        .tp-hero h1 {
            color: #FFFFFF !important;
            margin: 0;
            font-size: 2rem;
            font-weight: 800;
        }
        .tp-hero p {
            color: #FFD76A;
            margin: 6px 0 0;
            font-size: 1.05rem;
        }

        /* ── Empty State Card ────────────────────────────────────────── */
        .tp-empty {
            text-align: center;
            padding: 38px 20px;
            color: #B8B8B8;
            border: 1px dashed #3A2020;
            border-radius: 12px;
            background: #111111;
        }
        .tp-empty .tp-eico {
            font-size: 2.2rem;
            margin-bottom: 8px;
            color: #D4AF37;
        }
        .tp-empty .tp-etitle {
            font-weight: 600;
            color: #FFFFFF;
            font-size: 1.05rem;
            margin-bottom: 4px;
        }

        /* ── Quick Action Cards ──────────────────────────────────────── */
        .tp-action-card {
            background: #151515;
            border: 1px solid #3A2020;
            border-radius: 12px;
            padding: 16px;
            transition: all 0.2s ease;
        }
        .tp-action-card:hover {
            border-color: #D4AF37;
            box-shadow: 0 4px 12px rgba(212, 175, 55, 0.15);
        }

        /* ── Progress bar ────────────────────────────────────────────── */
        div[data-testid="stProgress"] > div > div > div > div {
            background: linear-gradient(90deg, #E32636 0%, #D4AF37 100%) !important;
        }

        /* ── Responsive: tighter layout on mobile devices ────────────── */
        @media (max-width: 768px) {
            .block-container { padding-left: 0.8rem; padding-right: 0.8rem; }
            .tp-page-header { padding: 14px 16px; }
            .tp-page-header h1 { font-size: 1.25rem; }
            .tp-hero { padding: 20px 18px; }
            .tp-hero h1 { font-size: 1.5rem; }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def page_header(icon: str, title: str, subtitle: str) -> None:
    """Consistent Red, Black & Gold page header."""
    st.markdown(
        f"""
        <div class="tp-page-header">
            <div class="tp-ico">{icon}</div>
            <div>
                <h1>{title}</h1>
                <p>{subtitle}</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def hero(title: str, subtitle: str) -> None:
    """Hero welcome container with red and gold highlights."""
    st.markdown(
        f"""
        <div class="tp-hero">
            <h1>{title}</h1>
            <p>{subtitle}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def section_title(text: str, icon: str = "") -> None:
    """Gold/White section header."""
    st.markdown(f"### <span style='color:#D4AF37'>{icon}</span> {text}".strip(), unsafe_allow_html=True)


def empty_state(icon: str, title: str, message: str) -> None:
    """Card for empty queries / initial states."""
    st.markdown(
        f"""
        <div class="tp-empty">
            <div class="tp-eico">{icon}</div>
            <div class="tp-etitle">{title}</div>
            <div>{message}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def kpi_row(metrics: list[dict]) -> None:
    """Render a row of KPI cards. Each item: {label, value, delta?, help?}."""
    cols = st.columns(len(metrics))
    for col, m in zip(cols, metrics):
        with col:
            st.metric(
                label=m["label"],
                value=m["value"],
                delta=m.get("delta"),
                help=m.get("help"),
                border=True,
            )


def px_fig(fig, height: int | None = None, key: str | None = None):
    """
    Standardized Plotly render: theme-adaptive layout + responsive width.
    Applies the Red, Black and Gold dark template.
    """
    fig.update_layout(**PLOTLY_DARK_TEMPLATE["layout"])
    if height:
        fig.update_layout(height=height)
    st.plotly_chart(
        fig,
        width="stretch",
        config={"displayModeBar": False},
        key=key,
    )
