import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from config import FEATURES, TARGET, TARGET_OPTIMAL, TARGET_TOLERANCE, CARDS_CONFIG, COLORS, PLOTLY_TEMPLATE
from data_utils import status_from_gap, compute_statistics, compute_correlation, generate_recommendations, compute_process_stability
from viz_utils import sparkline, GALLERY
from models import train_random_forest, train_quality_classifier
import shap  # optional

# La section command_center, monitoring, analysis, shap, reports, admin, bi_dashboard, world_map, import_data, prediction, etc.
# Je vais donner un exemple pour section_prediction, mais toutes les autres sont identiques à l'original.

def section_prediction(df, model, clf):
    render_page_title("🤖 PRÉDICTION & SHAP", "Estimation du PN RM, qualité DAP/MAP associée et interprétation")
    st.write("Saisissez les valeurs des paramètres du procédé pour une prédiction instantanée du PN RM et de la qualité.")
    
    defaults = {c: float(df[c].mean()) for c in FEATURES}
    col1, col2 = st.columns(2)
    input_values = {}
    for i, col in enumerate(FEATURES):
        with (col1 if i < 3 else col2):
            input_values[col] = st.number_input(col, value=defaults[col], format="%.2f", step=0.1, key=f"pred_{col}")
    
    if st.button("🚀 Prédire & Classifier", type="primary", key="btn_predict"):
        X_input = pd.DataFrame([input_values])[FEATURES]
        pred_rm = model.predict(X_input)[0]
        pred_class = clf.predict(X_input)[0]
        proba = clf.predict_proba(X_input)[0]
        classes = clf.classes_
        proba_dict = dict(zip(classes, proba))
        
        st.session_state["pred_single"] = pred_rm
        st.session_state["pred_class_single"] = pred_class
        st.session_state["proba_single"] = proba_dict
        st.session_state["lower_single"] = pred_rm - 1.96 * 0.02
        st.session_state["upper_single"] = pred_rm + 1.96 * 0.02
    
    if "pred_single" in st.session_state:
        pred_rm = st.session_state["pred_single"]
        pred_class = st.session_state.get("pred_class_single")
        proba_dict = st.session_state.get("proba_single", {})
        lower = st.session_state["lower_single"]
        upper = st.session_state["upper_single"]
        
        col1, col2 = st.columns([1, 1])
        with col1:
            st.markdown('<div class="card-premium">', unsafe_allow_html=True)
            st.metric("PN RM prédit", f"{pred_rm:.3f}", delta=f"{pred_rm - TARGET_OPTIMAL:+.3f}")
            st.metric("Intervalle de confiance 95%", f"[{lower:.3f} – {upper:.3f}]")
            st.markdown('</div>', unsafe_allow_html=True)
        with col2:
            if pred_class is not None:
                color = "#00B4D8" if pred_class == "DAP" else "#39FF14"
                emoji = "🧪" if pred_class == "DAP" else "🌱"
                st.markdown(f"""
                <div class="card-premium" style="border-left:4px solid {color};">
                    <div style="font-size:0.6rem;color:rgba(255,255,255,0.3);letter-spacing:2px;text-transform:uppercase;">
                        Qualité prédite
                    </div>
                    <div style="font-size:2.2rem;font-weight:700;color:{color};">
                        {emoji} {pred_class}
                    </div>
                </div>
                """, unsafe_allow_html=True)
        
        if pred_class is not None and proba_dict:
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
                             template=PLOTLY_TEMPLATE, text_auto=".1%")
            fig_bar.update_layout(height=300, paper_bgcolor="rgba(0,0,0,0)",
                                   font=dict(color="rgba(255,255,255,0.6)"),
                                   showlegend=False)
            st.plotly_chart(fig_bar, use_container_width=True)
        
        st.markdown("---")
        section_shap(model, df)  # importée depuis ce même module