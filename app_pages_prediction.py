# ======================================================================
# Page Prédiction & SHAP
# ======================================================================

import streamlit as st
import pandas as pd

from config import FEATURES, TARGET_OPTIMAL
from utils.helpers import render_page_title
from models.prediction import predict_pn_rm, predict_quality, render_prediction_result
from models.training import train_random_forest
from utils.visualization import GALLERY  # pour SHAP ? non, SHAP est dans models

# Pour SHAP, on importe si disponible
try:
    import shap
    import matplotlib.pyplot as plt
    import numpy as np
except ImportError:
    shap = None

def section_shap(model, df):
    render_page_title("🧠 INTERPRÉTABILITÉ SHAP", "Explication des décisions du modèle prédictif")
    if shap is None:
        st.error("Le module SHAP n'est pas installé sur cet environnement.")
        return
    try:
        X_sample = df[FEATURES].sample(min(100, len(df)), random_state=42)
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_sample)
        if isinstance(shap_values, list):
            shap_values = shap_values[0]
        st.markdown("#### Importance moyenne des variables (|SHAP|)")
        mean_abs = np.abs(shap_values).mean(axis=0)
        fig_bar = px.bar(x=X_sample.columns, y=mean_abs, template="plotly_dark",
                         color=mean_abs, color_continuous_scale="Greens")
        fig_bar.update_layout(paper_bgcolor="rgba(0,0,0,0)", font=dict(color="rgba(255,255,255,0.6)"), showlegend=False)
        st.plotly_chart(fig_bar, use_container_width=True)
        # Beeswarm 3D
        # ...
    except Exception as e:
        st.error(f"Erreur lors du calcul SHAP : {e}")

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
        pred_rm = predict_pn_rm(model, input_values)
        class_, proba_dict = predict_quality(clf, input_values)
        st.session_state["pred_single"] = pred_rm
        st.session_state["pred_class_single"] = class_
        st.session_state["proba_single"] = proba_dict
        st.session_state["lower_single"] = pred_rm - 1.96 * 0.02
        st.session_state["upper_single"] = pred_rm + 1.96 * 0.02
    
    if "pred_single" in st.session_state:
        pred_rm = st.session_state["pred_single"]
        class_ = st.session_state.get("pred_class_single")
        proba_dict = st.session_state.get("proba_single", {})
        render_prediction_result(pred_rm, class_, proba_dict)
        
        st.markdown("---")
        section_shap(model, df)