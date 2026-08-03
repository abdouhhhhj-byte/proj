# ======================================================================
# Page Analytics (3D, Monitoring, Analysis)
# ======================================================================

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np

from config import FEATURES, EXTRA_MONITORED, TARGET
from utils.helpers import render_page_title, compute_statistics, compute_correlation
from utils.visualization import GALLERY

def section_gallery3d(df):
    render_page_title("🌐 VISUALISATIONS 3D", "Scatter · Surface · Mesh · Cone · Isosurface · Terrain · Radar · Heatmap · Bubble · Network")
    st.write("")
    with st.container():
        st.markdown('<div class="card-premium">', unsafe_allow_html=True)
        choice = st.selectbox("Sélectionnez une visualisation 3D", list(GALLERY.keys()))
        fig = GALLERY[choice](df)
        st.plotly_chart(fig, use_container_width=True, config={"displaylogo": False})
        st.markdown('</div>', unsafe_allow_html=True)
    st.caption("Toutes les vues sont interactives : rotation 360°, zoom, survol et sélection des points.")

def section_monitoring(df):
    render_page_title("📈 MONITORING CAPTEURS", "Analyse temps réel des variables du procédé")
    st.write("")
    available_vars = [c for c in FEATURES + EXTRA_MONITORED + [TARGET] if c in df.columns]
    vars_to_plot = st.multiselect("Variables à afficher", available_vars,
                                   default=[c for c in available_vars if c in [FEATURES[0], FEATURES[3], TARGET]])
    if vars_to_plot:
        with st.container():
            st.markdown('<div class="card-premium">', unsafe_allow_html=True)
            fig = go.Figure()
            palette = ["#00FF7F", "#39FF14", "#00B4D8", "#FFB627", "#8FD6AB", "#F5FFF9"]
            x_vals = df.index if isinstance(df.index, pd.DatetimeIndex) else list(range(len(df)))
            for i, var in enumerate(vars_to_plot):
                fig.add_trace(go.Scatter(
                    x=x_vals, y=df[var], mode="lines", name=var,
                    line=dict(width=2.5, color=palette[i % len(palette)])
                ))
            fig.update_layout(
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="rgba(255,255,255,0.6)", size=11),
                hovermode="x unified",
                height=480,
                xaxis=dict(showgrid=False, gridcolor="rgba(255,255,255,0.03)"),
                yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.03)"),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            )
            st.plotly_chart(fig, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
    with st.expander("📊 Statistiques descriptives"):
        stats_cols = [c for c in FEATURES + [TARGET] if c in df.columns]
        st.dataframe(df[stats_cols].describe().style.background_gradient(cmap="Greens"),
                     use_container_width=True)

def section_analysis(df):
    render_page_title("🔍 ANALYSE EXPLORATOIRE", "Distributions, corrélations et dispersion des données")
    st.write("")
    tab1, tab2, tab3 = st.tabs(["📊 Distributions", "🔥 Corrélations", "📦 Boxplots"])
    with tab1:
        cols = [c for c in FEATURES + [TARGET] if c in df.columns]
        n_cols = len(cols)
        n_rows = (n_cols + 1) // 2
        fig = make_subplots(rows=n_rows, cols=2, subplot_titles=cols)
        for i, col in enumerate(cols):
            r, c = i // 2 + 1, i % 2 + 1
            fig.add_trace(go.Histogram(x=df[col], nbinsx=30, marker_color="#00FF7F", opacity=0.7), row=r, col=c)
        fig.update_layout(
            template="plotly_dark",
            height=400 * n_rows,
            showlegend=False,
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="rgba(255,255,255,0.6)"),
        )
        st.plotly_chart(fig, use_container_width=True)
    with tab2:
        corr = compute_correlation(df)
        fig = px.imshow(
            corr, text_auto=".2f", aspect="auto", template="plotly_dark",
            color_continuous_scale="Greens"
        )
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="rgba(255,255,255,0.6)"),
            height=500,
        )
        st.plotly_chart(fig, use_container_width=True)
    with tab3:
        fig = go.Figure()
        for col in [c for c in FEATURES + [TARGET] if c in df.columns]:
            fig.add_trace(go.Box(y=df[col], name=col, boxmean="sd", marker_color="#00FF7F"))
        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="rgba(255,255,255,0.6)"),
            height=500,
        )
        st.plotly_chart(fig, use_container_width=True)