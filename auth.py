# ======================================================================
# Authentification (SQLite)
# ======================================================================

import sqlite3
import hashlib
import secrets
import re
from contextlib import contextmanager
from datetime import datetime
import streamlit as st

from config import DB_PATH, ROLES, MASTER_PASSWORD

EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")

@contextmanager
def get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()

def init_db():
    with get_conn() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                nom TEXT NOT NULL,
                email TEXT NOT NULL UNIQUE,
                password_hash TEXT NOT NULL,
                salt TEXT NOT NULL,
                role TEXT NOT NULL DEFAULT 'Visiteur',
                department TEXT,
                matricule TEXT,
                date_creation TEXT NOT NULL
            )
        """)

def _hash_password(password: str, salt: str = None):
    if salt is None:
        salt = secrets.token_hex(16)
    digest = hashlib.pbkdf2_hmac(
        "sha256", password.encode("utf-8"), salt.encode("utf-8"), 200_000
    )
    return digest.hex(), salt

def _verify_password(password: str, password_hash: str, salt: str) -> bool:
    check, _ = _hash_password(password, salt)
    return secrets.compare_digest(check, password_hash)

def validate_signup(nom, email, department, matricule, accept_terms):
    errors = []
    if not nom or len(nom.strip()) < 2:
        errors.append("Le nom complet est requis (2 caractères minimum).")
    if not email or not EMAIL_RE.match(email):
        errors.append("Adresse email invalide.")
    if not department:
        errors.append("Le département est requis.")
    if not matricule or len(matricule.strip()) < 2:
        errors.append("Le matricule est requis.")
    if not accept_terms:
        errors.append("Vous devez accepter les conditions d'utilisation.")
    return errors

def email_exists(email: str) -> bool:
    with get_conn() as conn:
        row = conn.execute("SELECT 1 FROM users WHERE email = ?", (email.lower().strip(),)).fetchone()
        return row is not None

def create_user(nom, email, department, matricule, role="Visiteur"):
    with get_conn() as conn:
        conn.execute(
            """INSERT INTO users (nom, email, password_hash, salt, role, department, matricule, date_creation)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                nom.strip(),
                email.lower().strip(),
                "",
                "",
                role,
                department,
                matricule.strip(),
                datetime.now().isoformat(timespec="seconds"),
            ),
        )

def authenticate(email: str, password: str):
    if not email or not EMAIL_RE.match(email.strip()):
        return None
    if password != MASTER_PASSWORD:
        return None
    email = email.lower().strip()
    with get_conn() as conn:
        row = conn.execute("SELECT * FROM users WHERE email = ?", (email,)).fetchone()
    if row is None:
        role = "Administrateur" if count_users() == 0 else "Visiteur"
        nom = email.split("@")[0].replace(".", " ").replace("_", " ").title()
        with get_conn() as conn:
            conn.execute(
                """INSERT INTO users (nom, email, password_hash, salt, role, department, matricule, date_creation)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (nom, email, "", "", role, "Non renseigné", "", datetime.now().isoformat(timespec="seconds")),
            )
            row = conn.execute("SELECT * FROM users WHERE email = ?", (email,)).fetchone()
    return {
        "id": row["id"],
        "nom": row["nom"],
        "email": row["email"],
        "role": row["role"],
        "department": row["department"],
        "matricule": row["matricule"],
        "date_creation": row["date_creation"],
    }

def count_users() -> int:
    with get_conn() as conn:
        row = conn.execute("SELECT COUNT(*) AS n FROM users").fetchone()
        return row["n"]

def get_user_by_id(user_id: int):
    with get_conn() as conn:
        row = conn.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
    if row is None:
        return None
    return {
        "id": row["id"], "nom": row["nom"], "email": row["email"], "role": row["role"],
        "department": row["department"], "matricule": row["matricule"], "date_creation": row["date_creation"],
    }

def list_users():
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT id, nom, email, role, department, matricule, date_creation FROM users ORDER BY date_creation DESC"
        ).fetchall()
    return [dict(r) for r in rows]

def update_user_role(user_id: int, new_role: str):
    with get_conn() as conn:
        conn.execute("UPDATE users SET role = ? WHERE id = ?", (new_role, user_id))

ROLE_ICONS = {
    "Administrateur": "🛡️",
    "Ingénieur procédé": "⚙️",
    "Opérateur": "🧑‍🏭",
    "Visiteur": "👁️",
}

def render_login_form():
    st.caption("💡 Tout email valide est accepté — le mot de passe est celui de la plateforme.")
    with st.form("login_form", clear_on_submit=False):
        email = st.text_input("Email", placeholder="prenom.nom@ocpgroup.ma")
        password = st.text_input("Mot de passe", type="password", placeholder="••••••••")
        remember = st.checkbox("Se souvenir de moi")
        submitted = st.form_submit_button("🔓 Se connecter", use_container_width=True, type="primary")

    if submitted:
        if not email or not password:
            st.error("Veuillez renseigner votre email et votre mot de passe.")
            return
        if not EMAIL_RE.match(email.strip()):
            st.error("Adresse email invalide.")
            return
        user = authenticate(email, password)
        if user is None:
            st.error("Mot de passe incorrect.")
            return
        st.session_state["auth_user"] = user
        st.session_state["remember_me"] = remember
        st.success(f"Bienvenue, {user['nom']} ({user['role']}) — connexion en cours...")
        st.rerun()

    st.markdown('<div class="auth-divider"><span>Pas encore de compte ?</span></div>', unsafe_allow_html=True)
    if st.button("📝 Créer un compte", use_container_width=True, key="go_signup"):
        st.session_state["auth_view"] = "signup"
        st.rerun()

def render_signup_form():
    st.caption("💡 Le mot de passe de connexion est unique pour toute la plateforme — inutile de le définir ici.")
    with st.form("signup_form", clear_on_submit=False):
        nom = st.text_input("Nom complet")
        email = st.text_input("Email professionnel", placeholder="prenom.nom@ocpgroup.ma")
        department = st.selectbox(
            "Département",
            ["Production", "Maintenance", "Qualité", "R&D / Digital Twin", "HSE", "Logistique", "Direction"],
        )
        matricule = st.text_input("Matricule")
        accept = st.checkbox("J'accepte les conditions d'utilisation")
        submitted = st.form_submit_button("✅ Créer mon compte", use_container_width=True, type="primary")

    if submitted:
        errors = validate_signup(nom, email, department, matricule, accept)
        if email_exists(email):
            errors.append("Un compte existe déjà avec cet email.")
        if errors:
            for e in errors:
                st.error(e)
            return
        role = "Administrateur" if count_users() == 0 else "Visiteur"
        create_user(nom, email, department, matricule, role=role)
        st.success(f"Compte créé avec succès ({role}). Connectez-vous avec le mot de passe de la plateforme.")
        st.session_state["auth_view"] = "login"
        st.rerun()

    st.markdown('<div class="auth-divider"><span>Déjà inscrit ?</span></div>', unsafe_allow_html=True)
    if st.button("🔓 Retour à la connexion", use_container_width=True, key="go_login"):
        st.session_state["auth_view"] = "login"
        st.rerun()

def render_auth_screen(colors, logo_url):
    from utils.helpers import inject_auth_css, render_header  # import local pour éviter boucle
    inject_auth_css(colors, logo_url)
    if "auth_view" not in st.session_state:
        st.session_state["auth_view"] = "login"
    st.markdown('<div class="auth-wrapper">', unsafe_allow_html=True)
    render_header(colors, logo_url)
    if st.session_state["auth_view"] == "login":
        render_login_form()
    else:
        render_signup_form()
    st.markdown('</div>', unsafe_allow_html=True)