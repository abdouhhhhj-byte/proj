# ======================================================================
# Fonctions utilitaires générales
# ======================================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import random

from config import COLORS, FEATURES, TARGET, TARGET_OPTIMAL, TARGET_TOLERANCE, LOGO_URL

def inject_css():
    """Injecte le CSS global (avec étoiles et animations)"""
    # Pour éviter de surcharger, on peut charger un fichier CSS externe, mais ici on le met en dur.
    # (Le CSS complet est dans le fichier style.css, nous l'importerons via un fichier)
    pass

def inject_auth_css(colors, logo_url):
    st.markdown(f"""
    <style>
    .auth-wrapper {{
        max-width: 460px;
        margin: 1.5rem auto 0 auto;
        background: {colors['glass_bg']};
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        border: 1px solid {colors['border']};
        border-radius: 22px;
        padding: 2rem 2.2rem 1.6rem 2.2rem;
        box-shadow: 0 10px 40px rgba(0,0,0,0.6), inset 0 0 50px rgba(0,255,127,0.03);
    }}
    .auth-logo-wrap {{
        display:flex; justify-content:center; margin-bottom: 0.6rem;
    }}
    .auth-logo-wrap img {{
        height: 64px; object-fit:contain;
        filter: drop-shadow(0 0 18px rgba(0,255,127,0.55));
        animation: authLogoPulse 3s ease-in-out infinite;
    }}
    @keyframes authLogoPulse {{
        0%, 100% {{ filter: drop-shadow(0 0 12px rgba(0,255,127,0.4)); }}
        50% {{ filter: drop-shadow(0 0 26px rgba(57,255,20,0.75)); }}
    }}
    .auth-title {{
        text-align:center; font-family:'Orbitron', sans-serif; font-size:1.3rem;
        letter-spacing:4px; color:{colors['emerald']};
        text-shadow: 0 0 22px rgba(0,255,127,0.35);
        margin-bottom: 0.15rem;
    }}
    .auth-subtitle {{
        text-align:center; font-size:0.72rem; letter-spacing:3px; text-transform:uppercase;
        color:#7fd8a0; margin-bottom: 1.3rem;
    }}
    .auth-clock {{
        text-align:center; font-family:'Orbitron', sans-serif; font-size:0.78rem;
        color:#5f9c78; letter-spacing:2px; margin-bottom:1rem;
    }}
    .auth-divider {{
        display:flex; align-items:center; text-align:center; color:#5f9c78;
        font-size:0.75rem; margin: 1rem 0; letter-spacing:1px;
    }}
    .auth-divider::before, .auth-divider::after {{
        content:''; flex:1; border-bottom: 1px solid {colors['border_soft']};
    }}
    .auth-divider span {{ padding: 0 0.8rem; }}
    .role-badge-select {{
        font-size:0.72rem; color:#8fd6ab; letter-spacing:1px; margin-bottom:0.2rem;
    }}
    </style>
    """, unsafe_allow_html=True)

def render_header(colors, logo_url, title="OCP STRATEGIC INTELLIGENCE", subtitle="Digital Twin Manufacturing"):
    st.markdown(f"""
    <div class="auth-logo-wrap"><img src="{logo_url}" alt="OCP"></div>
    <div class="auth-title">{title}</div>
    <div class="auth-subtitle">{subtitle}</div>
    <div class="auth-clock">⏱ {datetime.now().strftime('%A %d %B %Y — %H:%M:%S')}</div>
    """, unsafe_allow_html=True)

def render_page_title(title, subtitle=None):
    sub_html = f'<div class="section-subtitle">{subtitle}</div>' if subtitle else ""
    st.markdown(f"""
    <div style="margin-bottom: 1rem;">
        <div class="section-title">{title}</div>
        {sub_html}
    </div>
    """, unsafe_allow_html=True)

def render_kpi_card(label, value, icon, delta=None, delta_color="up"):
    delta_html = ""
    if delta is not None:
        cls = "kpi-delta-up" if delta_color == "up" else "kpi-delta-down"
        delta_html = f'<div class="kpi-delta {cls}">{delta}</div>'
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-icon">{icon}</div>
        <div class="kpi-label">{label}</div>
        <div class="kpi-value">{value}</div>
        {delta_html}
    </div>
    """, unsafe_allow_html=True)

def render_sensor_card(row_val, prev_val, cfg):
    delta = row_val - prev_val
    arrow = "▲" if delta >= 0 else "▼"
    cls = "kpi-delta-up" if delta >= 0 else "kpi-delta-down" if delta < 0 else "kpi-delta-neutral"
    pct = (delta / prev_val * 100) if prev_val != 0 else 0
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-icon">{cfg['icon']}</div>
        <div class="kpi-label">{cfg['label']}</div>
        <div class="kpi-value">{row_val:.1f} <span style="font-size:0.7rem;color:rgba(255,255,255,0.3);">{cfg['unit']}</span></div>
        <div class="kpi-delta {cls}">{arrow} {abs(pct):.1f}%</div>
    </div>
    """, unsafe_allow_html=True)

def sparkline(data, color=None):
    color = color or COLORS["primary"]
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        y=data, mode="lines", line=dict(color=color, width=2.4),
        fill="tozeroy", fillcolor="rgba(0,255,127,0.06)", showlegend=False
    ))
    fig.update_layout(
        height=50, margin=dict(l=0, r=0, t=0, b=0),
        xaxis=dict(visible=False), yaxis=dict(visible=False),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig

def status_from_gap(val):
    gap = abs(val - TARGET_OPTIMAL)
    if gap <= TARGET_TOLERANCE:
        return "✅ Stable", "ok"
    if gap <= 0.05:
        return "⚠️ Surveillance", "warning"
    return "🔴 Critique", "danger"

def compute_process_stability(df):
    stability = {}
    for col in FEATURES + [TARGET]:
        if col in df.columns and len(df) >= 30:
            window = df[col].iloc[-30:]
            stability[f"{col}_rolling_std"] = window.std()
            stability[f"{col}_range"] = window.max() - window.min()
            mean = window.mean()
            stability[f"{col}_cv"] = (window.std() / mean) if mean != 0 else 0
    stability_score = 100
    for col in FEATURES + [TARGET]:
        cv = stability.get(f"{col}_cv", 0)
        if cv > 0.15:
            stability_score -= 5
        if cv > 0.25:
            stability_score -= 8
    stability_score = max(0, stability_score)
    stability["global_stability_score"] = stability_score
    return stability

def generate_recommendations(latest):
    recos = []
    pn_rm = latest[TARGET]
    if pn_rm < TARGET_OPTIMAL - 0.03:
        recos.append("🔺 Augmenter <b>NH3 Flow</b> de ~4% — impact estimé PN RM +0.05")
    if pn_rm > TARGET_OPTIMAL + 0.03:
        recos.append("🔻 Réduire <b>NH3 Flow</b> de ~3% — impact estimé PN RM -0.04")
    if "SLURRY_TEMPERATURE" in latest and latest["SLURRY_TEMPERATURE"] > 130:
        recos.append("🌡️ Stabiliser la température de slurry vers 120°C")
    if "WASHING_LIQUID_FLOW" in latest and latest["WASHING_LIQUID_FLOW"] < 40:
        recos.append("💧 Augmenter le débit d'eau de lavage de ~15%")
    if "INPUT_PHOS_ACID_FLOW_54" in latest and latest["INPUT_PHOS_ACID_FLOW_54"] > 150:
        recos.append("🧪 Réduire le débit d'acide 54% de ~2% (risque de surchauffe)")
    return recos