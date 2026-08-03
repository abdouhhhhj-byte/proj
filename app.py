import streamlit as st
import pandas as pd
import numpy as np
import os
import base64
from datetime import datetime
from config import COLORS, LOGO_LOCAL_PATH, LOGO_FALLBACK_URL
from auth import init_db, authenticate, list_users, update_user_role, count_users, create_user, validate_signup, email_exists, ROLES, ROLE_ICONS
from data_utils import generate_synthetic_data, ensure_columns
from models import train_random_forest, train_quality_classifier
from sections import (
    section_command_center, section_gallery3d, section_monitoring,
    section_analysis, section_shap, section_reports, section_admin_users,
    render_bi_dashboard, render_world_map, render_import_data, section_prediction
)
import random

# Chargement du logo
def load_logo_data_uri(path, fallback_url):
    try:
        with open(path, "rb") as f:
            raw = f.read()
        ext = os.path.splitext(path)[1].lstrip(".").lower()
        mime = "webp" if ext == "webp" else ("png" if ext == "png" else ext or "png")
        b64 = base64.b64encode(raw).decode("utf-8")
        return f"data:image/{mime};base64,{b64}"
    except Exception:
        return fallback_url

LOGO_URL = load_logo_data_uri(LOGO_LOCAL_PATH, LOGO_FALLBACK_URL)

st.set_page_config(
    page_title="OCP Digital Twin — Manufacturing Intelligence",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Inject CSS (à partir de la fonction originale)
def inject_css():
    # Copier le contenu de la fonction inject_css du code original
    pass

def render_auth_screen():
    # Copier le contenu de render_auth_screen et ses sous-fonctions (login, signup)
    pass

def render_authenticated_app(user):
    inject_css()
    # Initialisation des états de session
    if "data_version" not in st.session_state:
        st.session_state.data_version = 0
    if "df" not in st.session_state:
        st.session_state.df = None
    if "model_r2" not in st.session_state:
        st.session_state.model_r2 = None

    # En-tête et barre latérale identiques à l'original
    # ... (copier le code)

    df = st.session_state.df
    if df is None:
        st.warning("Veuillez charger des données...")
        # Afficher l'onglet Data Import uniquement
        # ...
        return

    with st.spinner("Entraînement des modèles..."):
        model, r2 = train_random_forest(df)
        st.session_state.model_r2 = r2
        clf = train_quality_classifier(df)

    # Définition des onglets (TAB_DEFS et permissions)
    TAB_DEFS = [
        ("page1", "📁 Data Import"),
        ("page2", "🟢 Command Center & Global Insights"),
        ("page3", "🌐 3D Analytics & Monitoring"),
        ("page4", "🤖 Prédiction & SHAP"),
        ("page5", "📤 Reports & Export"),
        ("page6", "👤 User Management"),
    ]
    ROLE_PERMISSIONS = {
        "Administrateur": {"page1", "page2", "page3", "page4", "page5", "page6"},
        "Ingénieur procédé": {"page1", "page2", "page3", "page4", "page5"},
        "Opérateur": {"page1", "page2", "page3", "page5"},
        "Visiteur": {"page2", "page3"},
    }
    allowed = ROLE_PERMISSIONS.get(user["role"], set())
    visible_tabs = [(key, label) for key, label in TAB_DEFS if key in allowed]
    tabs = st.tabs([label for _, label in visible_tabs])

    for (key, _), tab in zip(visible_tabs, tabs):
        with tab:
            if key == "page1":
                render_import_data()
            elif key == "page2":
                st.markdown("---")
                section_command_center(df, model)
                st.markdown("---")
                render_world_map(df)
                st.markdown("---")
                render_bi_dashboard(df)
            elif key == "page3":
                section_gallery3d(df)
                st.markdown("---")
                section_analysis(df)
                st.markdown("---")
                section_monitoring(df)
            elif key == "page4":
                section_prediction(df, model, clf)
            elif key == "page5":
                section_reports(df)
            elif key == "page6":
                section_admin_users()

    # Footer

def main():
    init_db()
    if "auth_user" not in st.session_state:
        inject_css()
        render_auth_screen()
        return
    render_authenticated_app(st.session_state["auth_user"])

if __name__ == "__main__":
    main()