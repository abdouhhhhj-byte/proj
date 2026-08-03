# ======================================================================
# Application principale
# ======================================================================

import streamlit as st
from datetime import datetime

from config import COLORS, LOGO_URL, TAB_DEFS, ROLE_PERMISSIONS
from auth.authentication import init_db, render_auth_screen
from utils.helpers import inject_css
from models.training import train_random_forest, train_quality_classifier
from app_pages.dashboard import section_command_center, render_bi_dashboard, render_world_map
from app_pages.prediction import section_prediction
from app_pages.analytics import section_gallery3d, section_monitoring, section_analysis
from app_pages.settings import section_admin_users
from utils.export import section_reports  # à créer

# Initialisation de la base de données
init_db()

st.set_page_config(
    page_title="OCP Digital Twin — Manufacturing Intelligence",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Injection du CSS global
inject_css()

def render_authenticated_app(user):
    # Header
    st.markdown(f"""
    <div class="header-premium">
        <div class="header-left">
            <div class="logo-3d-scene"><img src="{LOGO_URL}" class="logo-header" alt="OCP"></div>
            <div>
                <div class="header-title">OCP STRATEGIC INTELLIGENCE</div>
                <div class="header-subtitle">Manufacturing Intelligence · Digital Twin</div>
            </div>
        </div>
        <div class="header-right">
            <div class="header-status">
                <span class="status-dot"></span>
                Système opérationnel
            </div>
            <div class="header-time">
                ⏱ {datetime.now().strftime('%H:%M:%S')}
            </div>
            <div class="header-user">
                👤 {user['nom']} · {user['role']}
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Sidebar
    with st.sidebar:
        st.markdown(f"""
        <img src="{LOGO_URL}" class="logo-sidebar-3d" alt="OCP">
        <div class="sidebar-title">Manufacturing Intelligence</div>
        <div class="sidebar-user">
            <div class="sidebar-user-name">{user['nom']}</div>
            <div class="sidebar-user-role">{user['role']}</div>
        </div>
        <hr class="sidebar-divider">
        """, unsafe_allow_html=True)
        if st.session_state.df is not None:
            st.success(f"📊 Données chargées : {len(st.session_state.df)} lignes")
        else:
            st.info("Aucune donnée chargée. Allez dans l'onglet Data Import.")
        st.markdown("---")
        st.caption("v20.0 · Modularisé")

    # Chargement des données
    if "df" not in st.session_state or st.session_state.df is None:
        st.warning("Veuillez charger des données (fichier ou génération) dans l'onglet **Data Import**.")
        # On affiche un onglet Data Import simple (à implémenter)
        render_import_data()
        return

    df = st.session_state.df
    with st.spinner("Entraînement des modèles..."):
        model, r2 = train_random_forest(df)
        st.session_state["model_r2"] = r2
        clf = train_quality_classifier(df)

    # Onglets
    allowed = ROLE_PERMISSIONS.get(user["role"], set())
    visible_tabs = [(key, label) for key, label in TAB_DEFS if key in allowed]
    tabs = st.tabs([label for _, label in visible_tabs])

    for (key, _), tab in zip(visible_tabs, tabs):
        with tab:
            if key == "dashboard":
                st.markdown("---")
                section_command_center(df, model)
                st.markdown("---")
                render_world_map(df)
                st.markdown("---")
                render_bi_dashboard(df)
            elif key == "prediction":
                section_prediction(df, model, clf)
            elif key == "analytics":
                section_gallery3d(df)
                st.markdown("---")
                section_analysis(df)
                st.markdown("---")
                section_monitoring(df)
            elif key == "settings":
                section_admin_users()

    # Footer
    st.markdown(f"""
    <div class="footer" style="display:flex;justify-content:space-between;align-items:center;padding:1rem 0.5rem;color:rgba(255,255,255,0.2);font-size:0.65rem;border-top:1px solid rgba(0,255,127,0.06);margin-top:2rem;letter-spacing:1px;">
        <span>⚡ OCP Strategic Intelligence · Manufacturing Digital Twin · v20.0</span>
        <span>{datetime.now().strftime('%Y-%m-%d %H:%M')}</span>
    </div>
    """, unsafe_allow_html=True)

def render_import_data():
    # Fonction d'import rapide si aucune donnée chargée (identique à celle du code original)
    # Pour simplifier, on reprend le code de render_import_data original.
    pass

def main():
    if "auth_user" not in st.session_state:
        render_auth_screen(COLORS, LOGO_URL)
        return
    render_authenticated_app(st.session_state["auth_user"])

if __name__ == "__main__":
    main()