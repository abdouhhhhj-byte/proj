# ======================================================================
# Page Dashboard (Command Center, BI, World Map)
# ======================================================================

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

from config import FEATURES, TARGET, TARGET_OPTIMAL, TARGET_TOLERANCE, CARDS_CONFIG
from utils.helpers import (
    render_page_title, render_kpi_card, render_sensor_card, sparkline,
    status_from_gap, compute_process_stability, generate_recommendations
)
from models.training import train_random_forest

def section_command_center(df, model):
    render_page_title("🏭 PANORAMA PROCÉDÉ", "Pilotage intelligent en temps réel")
    latest = df.iloc[-1]
    prev = df.iloc[-2]
    pn_rm = latest[TARGET]
    pred = model.predict(latest[FEATURES].values.reshape(1, -1))[0]
    ecart = pn_rm - pred
    confiance = max(0.0, 1.0 - min(abs(ecart) / 0.1, 1.0))
    etat_label, etat_class = status_from_gap(pn_rm)
    risk_score = min(100, abs(ecart) * 1000 + (100 - confiance * 100) * 0.3)
    r2 = st.session_state.get("model_r2")
    
    st.markdown("""
    <div class="section-title">🏭 PANORAMA PROCÉDÉ <span class="badge">LIVE</span></div>
    <div style="margin-bottom: 0.5rem;"></div>
    """, unsafe_allow_html=True)
    
    col_k1, col_k2, col_k3, col_k4, col_k5 = st.columns(5)
    with col_k1:
        render_kpi_card("PN RM", f"{pn_rm:.3f}", "🎯", 
                       delta=f"{pn_rm - TARGET_OPTIMAL:+.3f} vs cible",
                       delta_color="up" if pn_rm >= TARGET_OPTIMAL else "down")
    with col_k2:
        status_icon = "✅" if etat_class == "ok" else "⚠️" if etat_class == "warning" else "🔴"
        render_kpi_card("État", f"{status_icon} {etat_label}", "📊")
    with col_k3:
        render_kpi_card("Confiance IA", f"{confiance:.0%}", "🤖")
    with col_k4:
        render_kpi_card("Score Risque", f"{risk_score:.0f}/100", "📈",
                       delta_color="up" if risk_score < 30 else "down")
    with col_k5:
        render_kpi_card("R² Modèle", f"{r2:.3f}" if r2 else "0.000", "📐")
    
    st.markdown("---")
    st.markdown('<div class="section-title">📡 CAPTEURS PROCÉDÉ <span class="badge">TEMPS RÉEL</span></div>', unsafe_allow_html=True)
    cols = st.columns(3)
    for idx, cfg in enumerate(CARDS_CONFIG):
        with cols[idx % 3]:
            key = cfg["key"]
            if key in df.columns:
                val = latest[key]
                pv = prev[key]
                render_sensor_card(val, pv, cfg)
                if key in df.columns:
                    st.plotly_chart(sparkline(df[key].tail(40)), use_container_width=True,
                                   config={"displayModeBar": False}, key=f"spark_{key}")
    
    st.markdown("---")
    # Gauge et trend (inchangés)
    st.markdown('<div class="section-title">🎯 INDICATEUR PN RM <span class="badge">CRITIQUE</span></div>', unsafe_allow_html=True)
    col_gauge, col_trend = st.columns([1.2, 1.8])
    with col_gauge:
        with st.container():
            st.markdown('<div class="card-premium">', unsafe_allow_html=True)
            gauge_fig = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=pn_rm,
                number={"suffix": "", "font": {"size": 48, "color": "#F5FFF9", "family": "Orbitron"}},
                delta={"reference": TARGET_OPTIMAL, "increasing": {"color": "#00FF7F"},
                       "decreasing": {"color": "#FFB627"}},
                title={"text": "PN RM", "font": {"size": 14, "color": "rgba(255,255,255,0.4)"}},
                gauge={
                    "axis": {"range": [0.8, 1.8], "tickcolor": "rgba(255,255,255,0.2)", "tickwidth": 1},
                    "bar": {"color": "#00FF7F", "thickness": 0.25},
                    "bgcolor": "rgba(0,0,0,0.4)",
                    "borderwidth": 0,
                    "steps": [
                        {"range": [0.8, 1.2], "color": "rgba(255,59,59,0.15)"},
                        {"range": [1.2, 1.5], "color": "rgba(0,255,127,0.10)"},
                        {"range": [1.5, 1.8], "color": "rgba(255,182,39,0.15)"},
                    ],
                    "threshold": {
                        "line": {"color": "#39FF14", "width": 4},
                        "thickness": 0.8,
                        "value": TARGET_OPTIMAL
                    },
                }
            ))
            gauge_fig.update_layout(
                height=300,
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#F5FFF9"),
                margin=dict(l=20, r=20, t=40, b=10),
            )
            st.plotly_chart(gauge_fig, use_container_width=True, config={"displayModeBar": False})
            st.markdown('</div>', unsafe_allow_html=True)
    with col_trend:
        with st.container():
            st.markdown('<div class="card-premium">', unsafe_allow_html=True)
            st.markdown('<div style="font-size:0.6rem;color:rgba(255,255,255,0.3);letter-spacing:2px;text-transform:uppercase;margin-bottom:0.3rem;">Évolution temps réel</div>', unsafe_allow_html=True)
            trend_fig = go.Figure()
            x_vals = df.index if isinstance(df.index, pd.DatetimeIndex) else list(range(len(df)))
            trend_fig.add_trace(go.Scatter(
                x=x_vals, y=df[TARGET], mode="lines", name="PN RM",
                line=dict(color="#00FF7F", width=3),
                fill="tozeroy", fillcolor="rgba(0,255,127,0.06)"
            ))
            trend_fig.add_hrect(
                y0=TARGET_OPTIMAL - TARGET_TOLERANCE,
                y1=TARGET_OPTIMAL + TARGET_TOLERANCE,
                fillcolor="rgba(0,255,127,0.08)",
                line_width=0,
                annotation_text="Zone optimale",
                annotation_font_color="rgba(255,255,255,0.3)",
                annotation_font_size=10,
            )
            trend_fig.update_layout(
                template="plotly_dark",
                height=280,
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="rgba(255,255,255,0.6)", size=10),
                margin=dict(l=10, r=10, t=10, b=10),
                hovermode="x unified",
                xaxis=dict(showgrid=False, gridcolor="rgba(255,255,255,0.03)"),
                yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.03)"),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            )
            st.plotly_chart(trend_fig, use_container_width=True, config={"displayModeBar": False})
            st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown('<div class="section-title">📋 SYNTHÈSE & AIDE À LA DÉCISION <span class="badge">IA</span></div>', unsafe_allow_html=True)
    col_synth1, col_synth2, col_synth3 = st.columns(3)
    risk_class = "badge-ok" if risk_score < 30 else "badge-warning" if risk_score < 65 else "badge-danger"
    status_class = f"badge-{etat_class}"
    with col_synth1:
        st.markdown(f"""
        <div class="card-premium">
            <div style="display:flex;justify-content:space-between;align-items:center;">
                <span style="font-size:0.6rem;color:rgba(255,255,255,0.3);letter-spacing:2px;text-transform:uppercase;">Statut</span>
                <span class="badge-status {status_class}">{etat_label}</span>
            </div>
            <div style="margin-top:0.3rem;display:flex;justify-content:space-between;font-size:0.8rem;">
                <span style="color:rgba(255,255,255,0.4);">PN RM</span>
                <span style="font-weight:600;font-family:'Orbitron',sans-serif;">{pn_rm:.3f}</span>
            </div>
            <div style="display:flex;justify-content:space-between;font-size:0.8rem;">
                <span style="color:rgba(255,255,255,0.4);">Cible</span>
                <span style="font-weight:600;font-family:'Orbitron',sans-serif;color:#00FF7F;">{TARGET_OPTIMAL}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
    with col_synth2:
        st.markdown(f"""
        <div class="card-premium">
            <div style="display:flex;justify-content:space-between;align-items:center;">
                <span style="font-size:0.6rem;color:rgba(255,255,255,0.3);letter-spacing:2px;text-transform:uppercase;">Prédiction</span>
                <span style="font-size:0.7rem;color:rgba(255,255,255,0.3);">IA</span>
            </div>
            <div style="margin-top:0.3rem;display:flex;justify-content:space-between;font-size:0.8rem;">
                <span style="color:rgba(255,255,255,0.4);">Valeur prédite</span>
                <span style="font-weight:600;font-family:'Orbitron',sans-serif;">{pred:.3f}</span>
            </div>
            <div style="display:flex;justify-content:space-between;font-size:0.8rem;">
                <span style="color:rgba(255,255,255,0.4);">Écart</span>
                <span style="font-weight:600;font-family:'Orbitron',sans-serif;color:{'#00FF7F' if abs(ecart) < 0.02 else '#FFB627'};">{ecart:+.3f}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
    with col_synth3:
        st.markdown(f"""
        <div class="card-premium">
            <div style="display:flex;justify-content:space-between;align-items:center;">
                <span style="font-size:0.6rem;color:rgba(255,255,255,0.3);letter-spacing:2px;text-transform:uppercase;">Risque</span>
                <span class="badge-status {risk_class}">{risk_score:.0f}/100</span>
            </div>
            <div style="margin-top:0.3rem;display:flex;justify-content:space-between;font-size:0.8rem;">
                <span style="color:rgba(255,255,255,0.4);">Confiance</span>
                <span style="font-weight:600;">{confiance:.0%}</span>
            </div>
            <div style="display:flex;justify-content:space-between;font-size:0.8rem;">
                <span style="color:rgba(255,255,255,0.4);">Alarmes</span>
                <span style="font-weight:600;color:{'#FF3B3B' if len(generate_recommendations(latest)) > 0 else '#00FF7F'};">{len(generate_recommendations(latest))}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
    st.caption(f"🕒 Dernière synchronisation : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

def render_bi_dashboard(df):
    render_page_title("📊 TABLEAU DE BORD DÉCISIONNEL", "Intelligence décisionnelle · Analyse avancée")
    latest = df.iloc[-1]
    model, r2 = train_random_forest(df)
    st.session_state["model_r2"] = r2
    stability = compute_process_stability(df)
    st.markdown('<div class="section-title">🎯 INDICATEURS CLÉS</div>', unsafe_allow_html=True)
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        render_kpi_card("PN RM Actuel", f"{latest[TARGET]:.3f}", "🎯",
                       delta=f"{latest[TARGET] - TARGET_OPTIMAL:+.3f}")
    with col2:
        target_hit = (abs(df[TARGET] - TARGET_OPTIMAL) <= TARGET_TOLERANCE).mean() * 100
        render_kpi_card("Conformité", f"{target_hit:.1f}%", "✅")
    with col3:
        render_kpi_card("Stabilité", f"{stability.get('global_stability_score', 0):.0f}%", "📊")
    with col4:
        pred = model.predict(latest[FEATURES].values.reshape(1, -1))[0]
        render_kpi_card("Prédiction PN RM", f"{pred:.3f}", "🤖",
                       delta=f"{pred - TARGET_OPTIMAL:+.3f}")
    st.markdown("---")
    # Matrice de corrélation
    st.markdown('<div class="section-title">🔥 MATRICE DE CORRÉLATION</div>', unsafe_allow_html=True)
    corr = df[FEATURES + [TARGET]].corr()
    fig_corr = px.imshow(
        corr, text_auto=".2f", aspect="auto", template="plotly_dark",
        color_continuous_scale="Greens"
    )
    fig_corr.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="rgba(255,255,255,0.6)"),
        height=450,
    )
    st.plotly_chart(fig_corr, use_container_width=True)
    # Distributions
    # ... (similaire)
    # Recommandations
    # ...

def render_world_map(df):
    render_page_title("🌍 CARTE MONDIALE LOGISTIQUE", "Ammoniac · Acide phosphorique · Engrais phosphatés")
    # (contenu inchangé)