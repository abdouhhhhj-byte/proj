# ======================================================================
# Prédiction avec les modèles
# ======================================================================

import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

from config import FEATURES, TARGET_OPTIMAL

def predict_pn_rm(model, input_values):
    """Renvoie la prédiction PN RM pour un dictionnaire de valeurs"""
    X = pd.DataFrame([input_values])[FEATURES]
    return model.predict(X)[0]

def predict_quality(clf, input_values):
    """Renvoie la classe et les probabilités"""
    X = pd.DataFrame([input_values])[FEATURES]
    class_ = clf.predict(X)[0]
    proba = clf.predict_proba(X)[0]
    classes = clf.classes_
    proba_dict = dict(zip(classes, proba))
    return class_, proba_dict

def render_prediction_result(pred_rm, class_, proba_dict):
    """Affiche les résultats de la prédiction avec métriques et jauges"""
    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="card-premium">', unsafe_allow_html=True)
        st.metric("PN RM prédit", f"{pred_rm:.3f}", delta=f"{pred_rm - TARGET_OPTIMAL:+.3f}")
        st.metric("Intervalle de confiance 95%", f"[{pred_rm-0.04:.3f} – {pred_rm+0.04:.3f}]")
        st.markdown('</div>', unsafe_allow_html=True)
    with col2:
        if class_ is not None:
            color = "#00B4D8" if class_ == "DAP" else "#39FF14"
            emoji = "🧪" if class_ == "DAP" else "🌱"
            st.markdown(f"""
            <div class="card-premium" style="border-left:4px solid {color};">
                <div style="font-size:0.6rem;color:rgba(255,255,255,0.3);letter-spacing:2px;text-transform:uppercase;">
                    Qualité prédite
                </div>
                <div style="font-size:2.2rem;font-weight:700;color:{color};">
                    {emoji} {class_}
                </div>
            </div>
            """, unsafe_allow_html=True)
    if class_ is not None and proba_dict:
        st.markdown("---")
        st.markdown("### 📊 Détail des probabilités")
        col_probs = st.columns(len(proba_dict))
        for i, (cl, prob) in enumerate(proba_dict.items()):
            with col_probs[i]:
                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=prob*100,
                    title={"text": cl, "font": {"size": 14, "color": "rgba(255,255,255,0.4)"}},
                    gauge={
                        "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "rgba(255,255,255,0.2)"},
                        "bar": {"color": "#00B4D8" if cl=="DAP" else "#39FF14"},
                        "steps": [
                            {"range": [0, 50], "color": "rgba(255,255,255,0.05)"},
                            {"range": [50, 100], "color": "rgba(0,255,127,0.05)"},
                        ],
                        "threshold": {
                            "line": {"color": "#39FF14", "width": 4},
                            "thickness": 0.8,
                            "value": 70
                        }
                    }
                ))
                fig.update_layout(height=200, paper_bgcolor="rgba(0,0,0,0)", font=dict(color="#F5FFF9"))
                st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        st.markdown("#### Répartition des probabilités")
        prob_df = pd.DataFrame({
            "Classe": list(proba_dict.keys()),
            "Probabilité": list(proba_dict.values())
        })
        fig_bar = px.bar(prob_df, x="Classe", y="Probabilité", color="Classe",
                         color_discrete_map={"DAP": "#00B4D8", "MAP": "#39FF14"},
                         template="plotly_dark", text_auto=".1%")
        fig_bar.update_layout(height=300, paper_bgcolor="rgba(0,0,0,0)",
                               font=dict(color="rgba(255,255,255,0.6)"),
                               showlegend=False)
        st.plotly_chart(fig_bar, use_container_width=True)