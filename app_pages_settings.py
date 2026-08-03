# ======================================================================
# Page Settings (User Management)
# ======================================================================

import streamlit as st
from config import ROLES
from auth.authentication import list_users, update_user_role
from utils.helpers import render_page_title

def section_admin_users():
    render_page_title("👤 GESTION DES UTILISATEURS", "Réservé aux administrateurs")
    users = list_users()
    if not users:
        st.info("Aucun utilisateur enregistré.")
        return
    st.markdown('<div class="card-premium">', unsafe_allow_html=True)
    for u in users:
        c1, c2, c3, c4 = st.columns([2.2, 1.6, 1.4, 1.2])
        c1.markdown(f"**{u['nom']}**  \n{u['email']}")
        c2.markdown(f"{u['department']}  \nMatricule: {u['matricule']}")
        new_role = c3.selectbox(
            "Rôle", ROLES, index=ROLES.index(u["role"]) if u["role"] in ROLES else 0,
            key=f"role_{u['id']}", label_visibility="collapsed",
        )
        if new_role != u["role"]:
            if c4.button("💾 Enregistrer", key=f"save_role_{u['id']}"):
                update_user_role(u["id"], new_role)
                st.success(f"Rôle de {u['nom']} mis à jour → {new_role}")
                st.rerun()
        st.markdown("<hr style='border-color:rgba(0,255,127,0.06); margin:0.4rem 0;'>", unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)