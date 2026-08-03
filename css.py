# ======================================================================
# OCP STRATEGIC INTELLIGENCE — Digital Twin Manufacturing
# Version 19.0 — Prédiction PN RM + SHAP + Classification DAP/MAP
# ======================================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import os
import random
import base64
from io import BytesIO
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings("ignore")
import hashlib
import re
import sqlite3
import secrets
from contextlib import contextmanager
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import r2_score

# Optional imports
try:
    import shap
except ImportError:
    shap = None

try:
    import matplotlib.pyplot as plt
    plt.switch_backend("Agg")
except ImportError:
    plt = None

try:
    from scipy.optimize import minimize
except ImportError:
    minimize = None

try:
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfgen import canvas
    from reportlab.lib.units import inch
except ImportError:
    A4 = None

# ======================================================================
# CONFIGURATION
# ======================================================================

FEATURES = [
    "INPUT_NH3_FLOW",
    "INPUT_PHOS_ACID_FLOW_54",
    "SLURRY_TEMPERATURE",
    "PUMP_SLURRY_FLOW_AP01",
    "WASHING_LIQUID_FLOW",
]
EXTRA_MONITORED = ["PUMP_AMPERAGE"]
TARGET = "PN RM"
TARGET_OPTIMAL = 1.35
TARGET_TOLERANCE = 0.02

RF_PARAMS = {"n_estimators": 200, "max_depth": 12, "random_state": 42, "n_jobs": -1}

CARDS_CONFIG = [
    {"key": "INPUT_NH3_FLOW", "label": "NH3 Flow", "unit": "t/h", "icon": "💨"},
    {"key": "INPUT_PHOS_ACID_FLOW_54", "label": "Acide 54%", "unit": "m³/h", "icon": "🧪"},
    {"key": "SLURRY_TEMPERATURE", "label": "Température", "unit": "°C", "icon": "🌡️"},
    {"key": "PUMP_SLURRY_FLOW_AP01", "label": "Pompe Slurry", "unit": "m³/h", "icon": "🔧"},
    {"key": "PUMP_AMPERAGE", "label": "Ampérage", "unit": "A", "icon": "⚡"},
    {"key": "WASHING_LIQUID_FLOW", "label": "Eau de lavage", "unit": "m³/h", "icon": "💧"},
]

# ======================================================================
# AUTHENTIFICATION
# ======================================================================

DB_PATH = "users.db"
ROLES = ["Administrateur", "Ingénieur procédé", "Opérateur", "Visiteur"]
EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
MASTER_PASSWORD = "azertocp"

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

# ======================================================================
# CSS ET LOGO
# ======================================================================

COLORS = {
    "primary": "#00FF7F",
    "secondary": "#0A4A3A",
    "accent": "#39FF14",
    "dark": "#0A0A0A",
    "surface": "#1A1A2E",
    "surface_light": "#2A2A3E",
    "white": "#F5FFF9",
    "gray": "#8A8A9A",
    "gray_light": "#B0B0C0",
    "alarm": "#FF3B3B",
    "warning": "#FFB627",
    "success": "#00FF7F",
    "info": "#00B4D8",
    "border": "rgba(0, 255, 127, 0.20)",
    "border_light": "rgba(0, 255, 127, 0.10)",
    "glass": "rgba(0, 0, 0, 0.40)",
    "glass_light": "rgba(0, 255, 127, 0.04)",
    "gradient_dark": "rgba(10, 10, 10, 0.95)",
    "gradient_green": "rgba(0, 255, 127, 0.08)",
    "emerald": "#00FF7F",
    "neon": "#39FF14",
    "dark_green": "#013220",
    "carbon": "#0A0A0A",
    "glass_bg": "rgba(0, 255, 127, 0.05)",
    "glass_bg_2": "rgba(57, 255, 20, 0.04)",
    "border_soft": "rgba(0, 255, 127, 0.18)",
}

PLOTLY_TEMPLATE = "plotly_dark"

LOGO_LOCAL_PATH = r"C:\Users\ABDEL\Downloads\pfe\ocp-logo-png_seeklogo-222172.webp"
LOGO_FALLBACK_URL = "https://upload.wikimedia.org/wikipedia/commons/7/7e/Ocp-group.png"

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

# ======================================================================
# INJECTION CSS COMPLÈTE
# ======================================================================

def inject_css():
    stars_html = ""
    for i in range(150):
        left = random.randint(0, 100)
        size = random.uniform(2, 6)
        duration = random.uniform(8, 20)
        delay = random.uniform(0, 20)
        opacity = random.uniform(0.3, 0.9)
        stars_html += (
            f'<div class="star" style="'
            f'left: {left}%; '
            f'width: {size}px; '
            f'height: {size}px; '
            f'animation-duration: {duration}s; '
            f'animation-delay: {delay}s; '
            f'opacity: {opacity};'
            f'"></div>'
        )

    st.markdown(f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=Orbitron:wght@400;600;700;800&display=swap');

    * {{
        margin: 0;
        padding: 0;
        box-sizing: border-box;
    }}

    html, body, .stApp {{
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        background: #0A0A0A;
        color: #F5FFF9;
    }}

    .stApp {{
        background: 
            radial-gradient(ellipse at 20% 0%, rgba(0,255,127,0.05) 0%, transparent 60%),
            radial-gradient(ellipse at 80% 100%, rgba(57,255,20,0.04) 0%, transparent 60%),
            linear-gradient(180deg, #0A0A0A 0%, #0D1A0D 40%, #0A0A0A 100%);
        background-attachment: fixed;
    }}

    #MainMenu {{visibility:hidden;}}
    footer {{visibility:hidden;}}
    header {{visibility:hidden;}}

    /* Logo 3D en arrière-plan */
    .logo-3d-container {{
        position: fixed;
        top: 0;
        left: 0;
        width: 100vw;
        height: 100vh;
        display: flex;
        align-items: center;
        justify-content: center;
        z-index: 0;
        pointer-events: none;
    }}

    .logo-3d {{
        width: 70vmin;
        height: 70vmin;
        max-width: 600px;
        max-height: 600px;
        animation: rotateLogo 22s linear infinite;
        transform-style: preserve-3d;
        filter: drop-shadow(0 0 60px rgba(0,255,127,0.08));
        opacity: 0.08;
    }}

    .logo-3d img {{
        width: 100%;
        height: 100%;
        object-fit: contain;
        animation: pulseGlow 5s ease-in-out infinite;
    }}

    @keyframes rotateLogo {{
        0% {{ transform: rotateY(0deg) scale(1) rotateX(5deg); }}
        50% {{ transform: rotateY(180deg) scale(1.04) rotateX(5deg); }}
        100% {{ transform: rotateY(360deg) scale(1) rotateX(5deg); }}
    }}

    @keyframes pulseGlow {{
        0%, 100% {{ filter: drop-shadow(0 0 20px rgba(0,255,127,0.04)); }}
        50% {{ filter: drop-shadow(0 0 60px rgba(0,255,127,0.12)); }}
    }}

    /* Étoiles filantes */
    .stars-container {{
        position: fixed;
        top: 0;
        left: 0;
        width: 100vw;
        height: 100vh;
        z-index: 1;
        pointer-events: none;
        overflow: hidden;
    }}

    .star {{
        position: absolute;
        background: #00FF7F;
        border-radius: 50%;
        animation: fallingStar linear infinite;
        box-shadow: 0 0 6px #00FF7F, 0 0 12px rgba(0,255,127,0.3);
    }}

    .star::after {{
        content: '';
        position: absolute;
        top: 100%;
        left: 50%;
        width: 1px;
        height: 15px;
        background: linear-gradient(to bottom, rgba(0,255,127,0.25), transparent);
        transform: translateX(-50%);
    }}

    @keyframes fallingStar {{
        0% {{
            transform: translateY(-20px) rotate(0deg) scale(1);
            opacity: 0;
        }}
        10% {{
            opacity: 1;
        }}
        90% {{
            opacity: 1;
        }}
        100% {{
            transform: translateY(calc(100vh + 50px)) rotate(720deg) scale(0.3);
            opacity: 0;
        }}
    }}

    section[data-testid="stMain"], section[data-testid="stSidebar"] {{
        position: relative;
        z-index: 10;
    }}

    .stApp > div {{
        position: relative;
        z-index: 10;
    }}

    .main-content {{
        position: relative;
        z-index: 10;
    }}

    /* Animations */
    @keyframes fadeInUp {{
        from {{ opacity: 0; transform: translateY(20px); }}
        to {{ opacity: 1; transform: translateY(0); }}
    }}
    @keyframes pulseGlowCard {{
        0%, 100% {{ box-shadow: 0 0 20px rgba(0,255,127,0.15); }}
        50% {{ box-shadow: 0 0 40px rgba(0,255,127,0.30); }}
    }}
    @keyframes slideIn {{
        from {{ opacity: 0; transform: translateX(-10px); }}
        to {{ opacity: 1; transform: translateX(0); }}
    }}
    @keyframes statusPulse {{
        0%, 100% {{ opacity: 1; }}
        50% {{ opacity: 0.6; }}
    }}

    /* Header */
    .header-premium {{
        background: linear-gradient(135deg, rgba(10,10,10,0.95), rgba(13,26,13,0.95));
        border-bottom: 1px solid rgba(0,255,127,0.15);
        padding: 0.8rem 2rem;
        display: flex;
        align-items: center;
        justify-content: space-between;
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        position: sticky;
        top: 0;
        z-index: 1000;
        animation: slideIn 0.5s ease;
    }}
    .header-left {{
        display: flex;
        align-items: center;
        gap: 1.2rem;
    }}
    .header-logo {{
        height: 42px;
        width: auto;
        filter: drop-shadow(0 0 20px rgba(0,255,127,0.4));
        transition: transform 0.3s ease;
    }}
    .header-logo:hover {{
        transform: scale(1.05) rotate(-2deg);
    }}
    .header-title {{
        font-family: 'Orbitron', sans-serif;
        font-size: 1.1rem;
        font-weight: 700;
        letter-spacing: 4px;
        background: linear-gradient(90deg, #00FF7F, #39FF14);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-shadow: 0 0 40px rgba(0,255,127,0.2);
    }}
    .header-subtitle {{
        font-size: 0.6rem;
        letter-spacing: 4px;
        text-transform: uppercase;
        color: rgba(255,255,255,0.4);
        font-weight: 400;
        margin-top: 2px;
    }}
    .header-right {{
        display: flex;
        align-items: center;
        gap: 1.5rem;
    }}
    .header-status {{
        display: flex;
        align-items: center;
        gap: 0.6rem;
        font-size: 0.7rem;
        color: rgba(255,255,255,0.6);
        letter-spacing: 1px;
    }}
    .status-dot {{
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background: #00FF7F;
        box-shadow: 0 0 15px rgba(0,255,127,0.6);
        animation: statusPulse 2s ease-in-out infinite;
    }}
    .header-time {{
        font-family: 'Orbitron', sans-serif;
        font-size: 0.75rem;
        color: rgba(255,255,255,0.7);
        letter-spacing: 2px;
        background: rgba(0,255,127,0.05);
        padding: 0.3rem 0.8rem;
        border-radius: 8px;
        border: 1px solid rgba(0,255,127,0.10);
    }}
    .header-user {{
        display: flex;
        align-items: center;
        gap: 0.6rem;
        font-size: 0.75rem;
        color: rgba(255,255,255,0.7);
        background: rgba(0,255,127,0.05);
        padding: 0.3rem 0.8rem;
        border-radius: 8px;
        border: 1px solid rgba(0,255,127,0.10);
    }}

    /* Sidebar */
    section[data-testid="stSidebar"] {{
        background: linear-gradient(180deg, rgba(10,10,10,0.98), rgba(13,26,13,0.98)) !important;
        border-right: 1px solid rgba(0,255,127,0.10) !important;
        padding-top: 1rem !important;
        z-index: 100 !important;
        position: relative !important;
    }}
    section[data-testid="stSidebar"] > div {{
        padding: 0 0.8rem;
    }}
    .sidebar-logo {{
        width: 100%;
        max-width: 160px;
        margin: 0 auto 1.2rem auto;
        display: block;
        filter: drop-shadow(0 0 25px rgba(0,255,127,0.3));
        transition: transform 0.4s ease;
    }}
    .sidebar-logo:hover {{
        transform: scale(1.05) rotate(-2deg);
    }}
    .sidebar-title {{
        font-family: 'Orbitron', sans-serif;
        font-size: 0.7rem;
        letter-spacing: 3px;
        color: rgba(255,255,255,0.4);
        text-transform: uppercase;
        text-align: center;
        margin-bottom: 0.8rem;
        border-bottom: 1px solid rgba(0,255,127,0.08);
        padding-bottom: 0.6rem;
    }}
    .sidebar-user {{
        background: rgba(0,255,127,0.04);
        border: 1px solid rgba(0,255,127,0.08);
        border-radius: 10px;
        padding: 0.6rem 0.8rem;
        margin-bottom: 0.8rem;
        text-align: center;
    }}
    .sidebar-user-name {{
        font-weight: 600;
        font-size: 0.85rem;
        color: #F5FFF9;
    }}
    .sidebar-user-role {{
        font-size: 0.65rem;
        color: rgba(255,255,255,0.4);
        letter-spacing: 1px;
        text-transform: uppercase;
    }}
    .sidebar-divider {{
        border: none;
        border-top: 1px solid rgba(0,255,127,0.06);
        margin: 0.6rem 0;
    }}

    /* Cartes */
    .card-premium {{
        background: linear-gradient(145deg, rgba(255,255,255,0.03), rgba(0,255,127,0.02));
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        border: 1px solid rgba(0,255,127,0.08);
        border-radius: 16px;
        padding: 1.2rem 1.4rem;
        box-shadow: 0 8px 32px rgba(0,0,0,0.4), inset 0 1px 0 rgba(255,255,255,0.03);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        animation: fadeInUp 0.6s ease both;
        position: relative;
        overflow: hidden;
        height: 100%;
        z-index: 10;
    }}
    .card-premium:hover {{
        transform: translateY(-2px);
        border-color: rgba(0,255,127,0.20);
        box-shadow: 0 12px 48px rgba(0,0,0,0.5), 0 0 30px rgba(0,255,127,0.05);
    }}
    .card-premium::before {{
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 2px;
        background: linear-gradient(90deg, transparent, rgba(0,255,127,0.3), transparent);
        opacity: 0;
        transition: opacity 0.3s ease;
    }}
    .card-premium:hover::before {{
        opacity: 1;
    }}

    /* KPI Cards */
    .kpi-card {{
        background: linear-gradient(145deg, rgba(0,255,127,0.04), rgba(0,10,0,0.6));
        border: 1px solid rgba(0,255,127,0.08);
        border-radius: 14px;
        padding: 1rem 1.2rem;
        text-align: center;
        transition: all 0.3s ease;
        animation: fadeInUp 0.5s ease both;
        height: 100%;
        min-height: 110px;
        display: flex;
        flex-direction: column;
        justify-content: center;
        z-index: 10;
        position: relative;
    }}
    .kpi-card:hover {{
        transform: translateY(-3px);
        border-color: rgba(0,255,127,0.20);
        box-shadow: 0 8px 30px rgba(0,0,0,0.4);
    }}
    .kpi-icon {{
        font-size: 1.5rem;
        margin-bottom: 0.3rem;
        filter: drop-shadow(0 0 10px rgba(0,255,127,0.2));
    }}
    .kpi-label {{
        font-size: 0.6rem;
        text-transform: uppercase;
        letter-spacing: 2px;
        color: rgba(255,255,255,0.4);
        font-weight: 600;
    }}
    .kpi-value {{
        font-family: 'Orbitron', sans-serif;
        font-size: 1.6rem;
        font-weight: 700;
        color: #F5FFF9;
        text-shadow: 0 0 30px rgba(0,255,127,0.1);
        line-height: 1.2;
    }}
    .kpi-delta {{
        font-size: 0.7rem;
        font-weight: 600;
        margin-top: 0.2rem;
    }}
    .kpi-delta-up {{ color: #00FF7F; }}
    .kpi-delta-down {{ color: #FFB627; }}
    .kpi-delta-neutral {{ color: rgba(255,255,255,0.3); }}

    /* Sections */
    .section-title {{
        font-family: 'Orbitron', sans-serif;
        font-size: 0.85rem;
        font-weight: 600;
        letter-spacing: 3px;
        color: #00FF7F;
        text-shadow: 0 0 20px rgba(0,255,127,0.15);
        margin: 1.5rem 0 0.8rem 0;
        padding-bottom: 0.4rem;
        border-bottom: 1px solid rgba(0,255,127,0.08);
        display: flex;
        align-items: center;
        gap: 0.6rem;
        z-index: 10;
        position: relative;
    }}
    .section-title .badge {{
        font-size: 0.5rem;
        background: rgba(0,255,127,0.10);
        padding: 0.1rem 0.6rem;
        border-radius: 12px;
        color: rgba(255,255,255,0.3);
        letter-spacing: 1px;
    }}
    .section-subtitle {{
        font-size: 0.7rem;
        color: rgba(255,255,255,0.3);
        letter-spacing: 2px;
        text-transform: uppercase;
        margin-top: -0.3rem;
        margin-bottom: 0.8rem;
        z-index: 10;
        position: relative;
    }}

    /* Badges */
    .badge-status {{
        display: inline-flex;
        align-items: center;
        gap: 0.4rem;
        padding: 0.2rem 0.8rem;
        border-radius: 20px;
        font-size: 0.6rem;
        font-weight: 600;
        letter-spacing: 1px;
        text-transform: uppercase;
    }}
    .badge-ok {{
        background: rgba(0,255,127,0.12);
        color: #00FF7F;
        border: 1px solid rgba(0,255,127,0.15);
    }}
    .badge-warning {{
        background: rgba(255,182,39,0.12);
        color: #FFB627;
        border: 1px solid rgba(255,182,39,0.15);
    }}
    .badge-danger {{
        background: rgba(255,59,59,0.12);
        color: #FF3B3B;
        border: 1px solid rgba(255,59,59,0.15);
        animation: statusPulse 1.5s ease-in-out infinite;
    }}

    /* Boutons */
    .stButton > button {{
        background: linear-gradient(135deg, rgba(0,255,127,0.08), rgba(0,50,25,0.3)) !important;
        color: #00FF7F !important;
        border: 1px solid rgba(0,255,127,0.15) !important;
        border-radius: 10px !important;
        font-weight: 600 !important;
        font-size: 0.75rem !important;
        letter-spacing: 1px !important;
        padding: 0.4rem 1.2rem !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2) !important;
        height: auto !important;
        z-index: 10 !important;
        position: relative !important;
    }}
    .stButton > button:hover {{
        transform: translateY(-2px) !important;
        border-color: rgba(0,255,127,0.4) !important;
        box-shadow: 0 8px 30px rgba(0,255,127,0.15) !important;
        background: linear-gradient(135deg, rgba(0,255,127,0.15), rgba(0,50,25,0.4)) !important;
    }}
    .stButton > button:active {{
        transform: scale(0.98) !important;
    }}

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {{
        gap: 6px;
        background: rgba(255,255,255,0.02);
        border-radius: 12px;
        padding: 4px;
        border: 1px solid rgba(0,255,127,0.06);
        flex-wrap: wrap;
        z-index: 10;
        position: relative;
    }}
    .stTabs [data-baseweb="tab"] {{
        border-radius: 8px;
        color: rgba(255,255,255,0.4);
        font-weight: 500;
        font-size: 0.7rem;
        letter-spacing: 0.5px;
        padding: 0.3rem 0.8rem !important;
        transition: all 0.3s ease;
    }}
    .stTabs [data-baseweb="tab"]:hover {{
        color: rgba(255,255,255,0.7);
        background: rgba(0,255,127,0.03);
    }}
    .stTabs [aria-selected="true"] {{
        background: linear-gradient(135deg, rgba(0,255,127,0.10), rgba(0,50,25,0.2)) !important;
        color: #00FF7F !important;
        border: 1px solid rgba(0,255,127,0.12) !important;
        box-shadow: 0 4px 20px rgba(0,255,127,0.05) !important;
    }}

    /* Metrics */
    div[data-testid="stMetric"] {{
        background: rgba(255,255,255,0.02);
        border-radius: 10px;
        padding: 0.4rem 0.8rem;
        border: 1px solid rgba(0,255,127,0.04);
        z-index: 10;
        position: relative;
    }}
    div[data-testid="stMetric"] label {{
        font-size: 0.6rem !important;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        color: rgba(255,255,255,0.3) !important;
    }}
    div[data-testid="stMetric"] div[data-testid="stMetricValue"] {{
        font-family: 'Orbitron', sans-serif !important;
        font-size: 1.2rem !important;
        color: #F5FFF9 !important;
    }}
    div[data-testid="stMetric"] div[data-testid="stMetricDelta"] {{
        font-size: 0.65rem !important;
    }}

    /* Expander */
    .streamlit-expanderHeader {{
        background: rgba(255,255,255,0.02) !important;
        border: 1px solid rgba(0,255,127,0.06) !important;
        border-radius: 10px !important;
        color: rgba(255,255,255,0.6) !important;
        font-weight: 500 !important;
        font-size: 0.75rem !important;
        letter-spacing: 1px !important;
        z-index: 10;
        position: relative;
    }}
    .streamlit-expanderContent {{
        background: rgba(255,255,255,0.01) !important;
        border-radius: 0 0 10px 10px !important;
        border: 1px solid rgba(0,255,127,0.04) !important;
        border-top: none !important;
        padding: 0.8rem !important;
        z-index: 10;
        position: relative;
    }}

    /* Selectbox */
    .stSelectbox > div > div {{
        background: rgba(255,255,255,0.02) !important;
        border: 1px solid rgba(0,255,127,0.08) !important;
        border-radius: 8px !important;
        color: #F5FFF9 !important;
        z-index: 10;
        position: relative;
    }}
    .stSelectbox label {{
        font-size: 0.65rem !important;
        color: rgba(255,255,255,0.3) !important;
        letter-spacing: 1px !important;
        text-transform: uppercase !important;
    }}

    /* MultiSelect */
    .stMultiSelect > div > div {{
        background: rgba(255,255,255,0.02) !important;
        border: 1px solid rgba(0,255,127,0.08) !important;
        border-radius: 8px !important;
        color: #F5FFF9 !important;
        z-index: 10;
        position: relative;
    }}

    /* Inputs */
    .stTextInput > div > div > input,
    .stNumberInput > div > div > input {{
        background: rgba(255,255,255,0.02) !important;
        border: 1px solid rgba(0,255,127,0.08) !important;
        border-radius: 8px !important;
        color: #F5FFF9 !important;
        font-size: 0.85rem !important;
        padding: 0.4rem 0.8rem !important;
        z-index: 10;
        position: relative;
    }}
    .stTextInput > div > div > input:focus,
    .stNumberInput > div > div > input:focus {{
        border-color: rgba(0,255,127,0.3) !important;
        box-shadow: 0 0 25px rgba(0,255,127,0.05) !important;
    }}
    .stTextInput label,
    .stNumberInput label {{
        font-size: 0.65rem !important;
        color: rgba(255,255,255,0.3) !important;
        letter-spacing: 1px !important;
        text-transform: uppercase !important;
    }}

    /* File Uploader */
    .stFileUploader > div {{
        background: rgba(255,255,255,0.02) !important;
        border: 2px dashed rgba(0,255,127,0.08) !important;
        border-radius: 12px !important;
        padding: 1.5rem !important;
        transition: all 0.3s ease !important;
        z-index: 10;
        position: relative;
    }}
    .stFileUploader > div:hover {{
        border-color: rgba(0,255,127,0.2) !important;
        background: rgba(0,255,127,0.02) !important;
    }}

    /* DataFrame */
    .stDataFrame {{
        background: rgba(255,255,255,0.02) !important;
        border-radius: 10px !important;
        border: 1px solid rgba(0,255,127,0.04) !important;
        overflow: hidden !important;
        z-index: 10;
        position: relative;
    }}
    .stDataFrame table {{
        font-size: 0.7rem !important;
    }}
    .stDataFrame thead tr th {{
        background: rgba(0,255,127,0.04) !important;
        color: rgba(255,255,255,0.5) !important;
        font-weight: 600 !important;
        letter-spacing: 0.5px !important;
        padding: 0.4rem 0.6rem !important;
    }}
    .stDataFrame tbody tr td {{
        padding: 0.3rem 0.6rem !important;
        border-bottom: 1px solid rgba(255,255,255,0.03) !important;
    }}
    .stDataFrame tbody tr:hover {{
        background: rgba(0,255,127,0.03) !important;
    }}

    /* Responsive */
    @media (max-width: 768px) {{
        .header-premium {{ flex-direction: column; gap: 0.5rem; padding: 0.5rem 1rem; }}
        .header-left {{ flex-direction: column; text-align: center; gap: 0.3rem; }}
        .header-right {{ flex-wrap: wrap; justify-content: center; gap: 0.5rem; }}
        .header-title {{ font-size: 0.9rem; }}
        .kpi-card {{ min-height: 80px; padding: 0.6rem 0.8rem; }}
        .kpi-value {{ font-size: 1.2rem; }}
        .card-premium {{ padding: 0.8rem 1rem; }}
        .section-title {{ font-size: 0.7rem; }}
    }}

    @media (min-width: 1920px) {{
        .kpi-value {{ font-size: 2rem; }}
        .card-premium {{ padding: 1.5rem 2rem; }}
        .header-title {{ font-size: 1.4rem; }}
    }}

    /* Scrollbar */
    ::-webkit-scrollbar {{ width: 6px; height: 6px; }}
    ::-webkit-scrollbar-track {{ background: rgba(255,255,255,0.02); }}
    ::-webkit-scrollbar-thumb {{ background: rgba(0,255,127,0.15); border-radius: 10px; }}
    ::-webkit-scrollbar-thumb:hover {{ background: rgba(0,255,127,0.25); }}

    /* Plotly */
    .js-plotly-plot .plotly .main-svg {{
        background: transparent !important;
    }}
    .js-plotly-plot .plotly .main-svg .bg {{
        fill: transparent !important;
    }}

    hr {{
        border: none;
        border-top: 1px solid rgba(0,255,127,0.06);
        margin: 0.8rem 0;
    }}

    .grid-gap {{
        gap: 1rem;
    }}
    .mt-1 {{ margin-top: 0.5rem; }}
    .mt-2 {{ margin-top: 1rem; }}
    .mt-3 {{ margin-top: 1.5rem; }}
    .mb-1 {{ margin-bottom: 0.5rem; }}
    .mb-2 {{ margin-bottom: 1rem; }}

    /* Logo 3D Scene */
    .logo-3d-scene {{
        display: inline-block;
        perspective: 800px;
        transition: transform 0.4s ease;
        z-index: 10;
        position: relative;
    }}
    .logo-3d-scene:hover {{
        transform: scale(1.05);
    }}
    .logo-header {{
        height: 56px;
        width: auto;
        max-width: 180px;
        object-fit: contain;
        filter: drop-shadow(0 0 20px rgba(0,255,127,0.5));
        transform-style: preserve-3d;
        animation: logoFloat 4s ease-in-out infinite, logoGlow 3s ease-in-out infinite;
        transition: all 0.6s cubic-bezier(0.4, 0, 0.2, 1);
        background: transparent;
        cursor: pointer;
    }}
    .logo-header:hover {{
        transform: rotateY(180deg) rotateX(10deg) scale(1.08);
        filter: drop-shadow(0 0 50px rgba(57,255,20,0.9)) brightness(1.15);
    }}
    @keyframes logoFloat {{
        0%, 100% {{ transform: translateY(0px) rotateY(0deg); }}
        50% {{ transform: translateY(-8px) rotateY(5deg); }}
    }}
    @keyframes logoGlow {{
        0%   {{ filter: drop-shadow(0 0 15px rgba(0,255,127,0.4)); }}
        50%  {{ filter: drop-shadow(0 0 35px rgba(57,255,20,0.8)); }}
        100% {{ filter: drop-shadow(0 0 15px rgba(0,255,127,0.4)); }}
    }}
    .logo-sidebar-3d {{
        width: 100%;
        max-width: 180px;
        height: auto;
        object-fit: contain;
        filter: drop-shadow(0 0 20px rgba(0,255,127,0.4));
        transform-style: preserve-3d;
        animation: logoFloat 5s ease-in-out infinite, logoGlow 4s ease-in-out infinite;
        transition: all 0.6s cubic-bezier(0.4, 0, 0.2, 1);
        background: transparent;
        cursor: pointer;
        margin: 0 auto;
        display: block;
    }}
    .logo-sidebar-3d:hover {{
        transform: rotateY(180deg) rotateX(8deg) scale(1.05);
        filter: drop-shadow(0 0 50px rgba(57,255,20,0.7)) brightness(1.1);
    }}

    .js-plotly-plot {{
        position: relative !important;
        z-index: 10 !important;
    }}

    .stMarkdown, .stDataFrame, .stExpander, .stTabs, .stSelectbox, .stMultiSelect, .stSlider {{
        position: relative !important;
        z-index: 10 !important;
    }}
    </style>

    <div class="logo-3d-container">
        <div class="logo-3d">
            <img src="{LOGO_URL}" alt="OCP Logo">
        </div>
    </div>
    <div class="stars-container">
        {stars_html}
    </div>
    """, unsafe_allow_html=True)

# ======================================================================
# FONCTIONS UTILITAIRES
# ======================================================================

def generate_synthetic_data(n=240, seed=None):
    if seed is None:
        seed = int(datetime.now().timestamp() * 1000) % 2**32
    np.random.seed(seed)
    timestamps = [datetime.now() - timedelta(minutes=i) for i in range(n, 0, -1)]
    data = {}
    trend = np.linspace(0, 5, n)
    for col in FEATURES:
        if "FLOW" in col:
            base = 100 if ("PHOS" in col or "SLURRY" in col) else 50 if "NH3" in col else 80
            data[col] = base + trend + np.cumsum(np.random.normal(0, 1.5, n))
        elif "TEMP" in col:
            data[col] = 85 + 0.02 * trend + np.cumsum(np.random.normal(0, 0.4, n))
        else:
            data[col] = 80 + 0.01 * trend + np.cumsum(np.random.normal(0, 1.2, n))
    data["PUMP_AMPERAGE"] = 45 + 0.01 * trend + np.cumsum(np.random.normal(0, 0.6, n))
    df = pd.DataFrame(data, index=timestamps)
    coeffs = np.array([0.002, -0.001, 0.003, -0.0015, -0.002])
    df[TARGET] = 1.35 + np.dot(df[FEATURES].values, coeffs) + np.random.normal(0, 0.015, n)
    df[TARGET] = df[TARGET].clip(0.8, 1.8)
    df.index.name = "Timestamp"
    return df

def ensure_columns(df):
    df_copy = df.copy()
    for col in FEATURES:
        if col not in df_copy.columns:
            df_copy[col] = np.random.normal(50, 10, len(df_copy))
    if "PUMP_AMPERAGE" not in df_copy.columns:
        df_copy["PUMP_AMPERAGE"] = 45 + np.random.normal(0, 5, len(df_copy))
    if TARGET not in df_copy.columns:
        df_copy[TARGET] = TARGET_OPTIMAL + np.random.normal(0, 0.05, len(df_copy))
    return df_copy

def compute_statistics(df):
    return df.describe()

def compute_correlation(df):
    cols = [c for c in FEATURES + [TARGET] if c in df.columns]
    return df[cols].corr()

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

def render_page_title(title, subtitle=None):
    sub_html = f'<div class="section-subtitle">{subtitle}</div>' if subtitle else ""
    st.markdown(f"""
    <div style="margin-bottom: 1rem;">
        <div class="section-title">{title}</div>
        {sub_html}
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

# ======================================================================
# MODÈLES AVEC CACHING
# ======================================================================

@st.cache_resource
def train_random_forest(df):
    X, y = df[FEATURES], df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(**RF_PARAMS)
    model.fit(X_train, y_train)
    r2 = r2_score(y_test, model.predict(X_test))
    return model, r2

@st.cache_resource
def train_quality_classifier(df):
    def quality(rm):
        if rm < 1.10:
            return "MAP"
        elif rm >= 1.11:
            return "DAP"
        else:
            return np.nan
    df_clean = df.copy()
    for col in FEATURES + [TARGET]:
        df_clean[col] = pd.to_numeric(df_clean[col], errors="coerce")
    df_clean["QUALITY"] = df_clean[TARGET].apply(quality)
    df_clean = df_clean.dropna(subset=["QUALITY"])
    if len(df_clean) < 20 or len(df_clean["QUALITY"].unique()) < 2:
        np.random.seed(42)
        n_synth = 200
        rm_map = np.random.uniform(0.8, 1.09, n_synth//2)
        rm_dap = np.random.uniform(1.11, 1.8, n_synth//2)
        rm_synth = np.concatenate([rm_map, rm_dap])
        quality_synth = ["MAP"]*(n_synth//2) + ["DAP"]*(n_synth//2)
        X_synth = pd.DataFrame({
            "INPUT_NH3_FLOW": 50 + 10*np.random.randn(n_synth),
            "INPUT_PHOS_ACID_FLOW_54": 100 + 15*np.random.randn(n_synth),
            "SLURRY_TEMPERATURE": 85 + 5*np.random.randn(n_synth),
            "PUMP_SLURRY_FLOW_AP01": 80 + 10*np.random.randn(n_synth),
            "WASHING_LIQUID_FLOW": 70 + 8*np.random.randn(n_synth),
        })
        X_synth["INPUT_NH3_FLOW"] += (rm_synth - 1.35) * 30
        X_synth["SLURRY_TEMPERATURE"] -= (rm_synth - 1.35) * 10
        df_synth = X_synth.copy()
        df_synth[TARGET] = rm_synth
        df_synth["QUALITY"] = quality_synth
        df_synth = df_synth.sample(frac=1, random_state=42).reset_index(drop=True)
        df_clean = pd.concat([df_clean, df_synth], ignore_index=True)
    X = df_clean[FEATURES]
    y = df_clean["QUALITY"]
    clf = RandomForestClassifier(
        n_estimators=500,
        max_depth=10,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced"
    )
    clf.fit(X, y)
    return clf

# ======================================================================
# VISUALISATIONS 3D (inchangées)
# ======================================================================

def gallery_scatter3d(df):
    fig = go.Figure(go.Scatter3d(
        x=df["INPUT_NH3_FLOW"], y=df["SLURRY_TEMPERATURE"], z=df[TARGET],
        mode="markers",
        marker=dict(size=4, color=df[TARGET], colorscale="Greens", opacity=0.85,
                    colorbar=dict(title="PN RM")),
    ))
    fig.update_layout(template=PLOTLY_TEMPLATE, height=560, paper_bgcolor="rgba(0,0,0,0)",
                       scene=dict(xaxis_title="NH3 Flow", yaxis_title="Température", zaxis_title="PN RM"))
    return fig

def gallery_surface3d(df):
    x = np.linspace(df["INPUT_NH3_FLOW"].min(), df["INPUT_NH3_FLOW"].max(), 40)
    y = np.linspace(df["SLURRY_TEMPERATURE"].min(), df["SLURRY_TEMPERATURE"].max(), 40)
    X, Y = np.meshgrid(x, y)
    Z = TARGET_OPTIMAL + 0.002 * (X - X.mean()) - 0.0015 * (Y - Y.mean()) + \
        0.00002 * (X - X.mean()) * (Y - Y.mean())
    fig = go.Figure(go.Surface(x=X, y=Y, z=Z, colorscale="Greens"))
    fig.update_layout(template=PLOTLY_TEMPLATE, height=560, paper_bgcolor="rgba(0,0,0,0)",
                       scene=dict(xaxis_title="NH3 Flow", yaxis_title="Température", zaxis_title="PN RM estimé"))
    return fig

def gallery_mesh3d():
    np.random.seed(1)
    x, y, z = np.random.randn(60), np.random.randn(60), np.random.randn(60)
    fig = go.Figure(go.Mesh3d(x=x, y=y, z=z, alphahull=5, opacity=0.55, color="#39FF14"))
    fig.update_layout(template=PLOTLY_TEMPLATE, height=560, paper_bgcolor="rgba(0,0,0,0)",
                       scene=dict(xaxis_title="X", yaxis_title="Y", zaxis_title="Z"))
    return fig

def gallery_cone(df):
    latest = df.iloc[-1]
    n = 6
    x, y, z = np.meshgrid(np.linspace(-3, 3, n), np.linspace(-3, 3, n), [0, 2, 4])
    x, y, z = x.flatten(), y.flatten(), z.flatten()
    u = np.full_like(x, latest["INPUT_NH3_FLOW"] / 100)
    v = np.full_like(y, latest["PUMP_SLURRY_FLOW_AP01"] / 150)
    w = np.full_like(z, 0.4)
    fig = go.Figure(go.Cone(x=x, y=y, z=z, u=u, v=v, w=w, colorscale="Greens",
                             sizemode="scaled", sizeref=1.4, showscale=False))
    fig.update_layout(template=PLOTLY_TEMPLATE, height=560, paper_bgcolor="rgba(0,0,0,0)",
                       scene=dict(xaxis_title="X", yaxis_title="Y", zaxis_title="Sens du flux"))
    return fig

def gallery_isosurface():
    X, Y, Z = np.mgrid[-4:4:30j, -4:4:30j, -4:4:30j]
    values = X * X + Y * Y + Z * Z
    fig = go.Figure(go.Isosurface(
        x=X.flatten(), y=Y.flatten(), z=Z.flatten(), value=values.flatten(),
        isomin=6, isomax=12, colorscale="Greens", surface_count=2, opacity=0.6,
        caps=dict(x_show=False, y_show=False, z_show=False),
    ))
    fig.update_layout(template=PLOTLY_TEMPLATE, height=560, paper_bgcolor="rgba(0,0,0,0)")
    return fig

def gallery_terrain3d(df):
    n = 50
    x = np.linspace(0, 10, n)
    y = np.linspace(0, 10, n)
    X, Y = np.meshgrid(x, y)
    np.random.seed(7)
    Z = (np.sin(X) * np.cos(Y) * 2 + np.random.normal(0, 0.15, (n, n))) + \
        (df[TARGET].std() * 3)
    fig = go.Figure(go.Surface(x=X, y=Y, z=Z, colorscale="Earth"))
    fig.update_layout(template=PLOTLY_TEMPLATE, height=560, paper_bgcolor="rgba(0,0,0,0)",
                       scene=dict(zaxis_title="Élévation (proxy variabilité procédé)"))
    return fig

def gallery_radar3d(df):
    latest = df.iloc[-1]
    cats = FEATURES
    norm = np.array([(latest[c] - df[c].min()) / (df[c].max() - df[c].min() + 1e-9) for c in cats])
    angles = np.linspace(0, 2 * np.pi, len(cats), endpoint=False)
    x = norm * np.cos(angles)
    y = norm * np.sin(angles)
    z = norm
    fig = go.Figure()
    fig.add_trace(go.Scatter3d(x=list(x) + [x[0]], y=list(y) + [y[0]], z=list(z) + [z[0]],
                                mode="lines+markers",
                                line=dict(color="#39FF14", width=6),
                                marker=dict(size=5, color="#00FF7F")))
    for xi, yi, zi in zip(x, y, z):
        fig.add_trace(go.Scatter3d(x=[0, xi], y=[0, yi], z=[0, zi], mode="lines",
                                    line=dict(color="rgba(0,255,127,0.25)", width=2), showlegend=False))
    fig.update_layout(template=PLOTLY_TEMPLATE, height=560, paper_bgcolor="rgba(0,0,0,0)", showlegend=False)
    return fig

def gallery_heatmap3d(df):
    corr = compute_correlation(df).values
    x = list(compute_correlation(df).columns)
    X, Y = np.meshgrid(np.arange(len(x)), np.arange(len(x)))
    fig = go.Figure(go.Surface(x=X, y=Y, z=corr, colorscale="RdYlGn", cmid=0))
    fig.update_layout(template=PLOTLY_TEMPLATE, height=560, paper_bgcolor="rgba(0,0,0,0)",
                       scene=dict(
                           xaxis=dict(tickvals=list(range(len(x))), ticktext=x),
                           yaxis=dict(tickvals=list(range(len(x))), ticktext=x),
                           zaxis_title="Corrélation"))
    return fig

def gallery_bubble3d(df):
    sample = df.sample(min(80, len(df)), random_state=1)
    fig = go.Figure(go.Scatter3d(
        x=sample["INPUT_NH3_FLOW"], y=sample["WASHING_LIQUID_FLOW"], z=sample[TARGET],
        mode="markers",
        marker=dict(size=(sample["SLURRY_TEMPERATURE"] / sample["SLURRY_TEMPERATURE"].max()) * 22 + 4,
                    color=sample[TARGET], colorscale="Greens", opacity=0.75,
                    colorbar=dict(title="PN RM")),
    ))
    fig.update_layout(template=PLOTLY_TEMPLATE, height=560, paper_bgcolor="rgba(0,0,0,0)",
                       scene=dict(xaxis_title="NH3 Flow", yaxis_title="Eau de lavage", zaxis_title="PN RM"))
    return fig

def gallery_network3d():
    np.random.seed(3)
    nodes = ["NH3 Tank", "Acid Tank", "Reactor", "Pump A", "Pump B", "Filter", "Output"]
    pos = np.random.randn(len(nodes), 3) * 2
    edges = [(0, 2), (1, 2), (2, 3), (2, 4), (3, 5), (4, 5), (5, 6)]
    fig = go.Figure()
    for a, b in edges:
        fig.add_trace(go.Scatter3d(x=[pos[a, 0], pos[b, 0]], y=[pos[a, 1], pos[b, 1]], z=[pos[a, 2], pos[b, 2]],
                                    mode="lines", line=dict(color="rgba(0,255,127,0.45)", width=4),
                                    showlegend=False))
    fig.add_trace(go.Scatter3d(x=pos[:, 0], y=pos[:, 1], z=pos[:, 2], mode="markers+text",
                                text=nodes, textposition="top center",
                                marker=dict(size=9, color="#39FF14", line=dict(color="white", width=1)),
                                textfont=dict(color="#F5FFF9")))
    fig.update_layout(template=PLOTLY_TEMPLATE, height=560, paper_bgcolor="rgba(0,0,0,0)", showlegend=False)
    return fig

GALLERY = {
    "Scatter 3D — Corrélation NH3/Température/PN RM": gallery_scatter3d,
    "Surface 3D — Surface de réponse du procédé": gallery_surface3d,
    "Mesh 3D — Nuage volumique": lambda df: gallery_mesh3d(),
    "Cone Plot — Sens & intensité des flux": gallery_cone,
    "Isosurface — Champ scalaire 3D": lambda df: gallery_isosurface(),
    "Terrain 3D — Cartographie de variabilité": gallery_terrain3d,
    "Radar 3D — Signature multivariée instantanée": gallery_radar3d,
    "Heatmap 3D — Corrélations en relief": gallery_heatmap3d,
    "Bubble Chart 3D — Flux / Lavage / PN RM": gallery_bubble3d,
    "Network Graph 3D — Topologie du procédé": lambda df: gallery_network3d(),
}

# ======================================================================
# AUTHENTIFICATION AFFICHAGE
# ======================================================================

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

def render_user_badge(user, colors):
    icon = ROLE_ICONS.get(user["role"], "👤")
    st.markdown(f"""
    <span style="display:flex; align-items:center; gap:6px; color:#8fd6ab; font-size:0.8rem;">
        {icon} <b style="color:{colors['white']};">{user['nom']}</b> · {user['role']}
    </span>
    """, unsafe_allow_html=True)

def logout():
    for key in ["auth_user", "auth_view", "remember_me"]:
        st.session_state.pop(key, None)
    st.rerun()

# ======================================================================
# ONGLETS : DÉFINITION ET PERMISSIONS
# ======================================================================

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

# ======================================================================
# SECTIONS DE L'APPLICATION
# ======================================================================

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
                template=PLOTLY_TEMPLATE,
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
                template=PLOTLY_TEMPLATE,
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
        st.dataframe(compute_statistics(df[stats_cols]).style.background_gradient(cmap="Greens"),
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
            template=PLOTLY_TEMPLATE,
            height=400 * n_rows,
            showlegend=False,
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="rgba(255,255,255,0.6)"),
        )
        st.plotly_chart(fig, use_container_width=True)
    with tab2:
        corr = compute_correlation(df)
        fig = px.imshow(
            corr, text_auto=".2f", aspect="auto", template=PLOTLY_TEMPLATE,
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
            template=PLOTLY_TEMPLATE,
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="rgba(255,255,255,0.6)"),
            height=500,
        )
        st.plotly_chart(fig, use_container_width=True)

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
        fig_bar = px.bar(x=X_sample.columns, y=mean_abs, template=PLOTLY_TEMPLATE,
                         color=mean_abs, color_continuous_scale="Greens")
        fig_bar.update_layout(paper_bgcolor="rgba(0,0,0,0)", font=dict(color="rgba(255,255,255,0.6)"), showlegend=False)
        st.plotly_chart(fig_bar, use_container_width=True)
        st.markdown("#### Beeswarm Plot 3D Interactif")
        data = []
        for i, feature in enumerate(X_sample.columns):
            for j in range(len(X_sample)):
                shap_val = shap_values[j, i]
                feat_val = X_sample.iloc[j][feature]
                data.append({
                    'Feature': feature,
                    'SHAP Value': shap_val,
                    'Feature Value': feat_val,
                    'Observation': j
                })
        df_beeswarm = pd.DataFrame(data)
        features_unique = X_sample.columns.tolist()
        feature_map = {f: i for i, f in enumerate(features_unique)}
        df_beeswarm['Feature Index'] = df_beeswarm['Feature'].map(feature_map)
        fig_beeswarm = px.scatter_3d(df_beeswarm,
                                     x='SHAP Value',
                                     y='Feature Index',
                                     z='Feature Value',
                                     color='Feature Value',
                                     hover_data=['Feature', 'SHAP Value', 'Feature Value'],
                                     color_continuous_scale='RdBu_r',
                                     opacity=0.7,
                                     title='Beeswarm 3D - SHAP Values')
        fig_beeswarm.update_layout(
            scene=dict(
                xaxis_title='SHAP Value',
                yaxis=dict(tickvals=list(feature_map.values()), ticktext=list(feature_map.keys()), title='Feature'),
                zaxis_title='Feature Value',
            ),
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="rgba(255,255,255,0.6)"),
            height=600,
        )
        st.plotly_chart(fig_beeswarm, use_container_width=True)
    except Exception as e:
        st.error(f"Erreur lors du calcul SHAP : {e}")

def section_reports(df):
    render_page_title("📤 EXPORT & RAPPORTS", "Téléchargement des données et rapports")
    c1, c2, c3 = st.columns(3)
    with c1:
        csv = df.to_csv(index=True).encode("utf-8")
        st.download_button("⬇️ CSV", csv, "rapport.csv", "text/csv", use_container_width=True, key="dl_csv")
    with c2:
        df_export = df.copy()
        if isinstance(df_export.index, pd.DatetimeIndex) and df_export.index.tz is not None:
            df_export.index = df_export.index.tz_localize(None)
        output = BytesIO()
        with pd.ExcelWriter(output, engine="openpyxl") as writer:
            df_export.to_excel(writer, sheet_name="Données")
        st.download_button(
            "⬇️ Excel",
            output.getvalue(),
            "rapport.xlsx",
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
            key="dl_xlsx"
        )
    with c3:
        if st.button("📄 Générer PDF", use_container_width=True, key="btn_pdf"):
            if A4 is None:
                st.error("ReportLab non installé.")
            else:
                buffer = BytesIO()
                c = canvas.Canvas(buffer, pagesize=A4)
                c.setFont("Helvetica-Bold", 16)
                c.drawString(1 * inch, 10 * inch, "Rapport PN RM — Digital Twin")
                c.setFont("Helvetica", 12)
                c.drawString(1 * inch, 9.5 * inch, f"Date : {datetime.now().strftime('%Y-%m-%d %H:%M')}")
                c.drawString(1 * inch, 9 * inch, f"Observations : {len(df)}")
                c.drawString(1 * inch, 8.5 * inch, f"PN RM moyen : {df[TARGET].mean():.3f}")
                c.save()
                buffer.seek(0)
                st.download_button("⬇️ PDF", buffer, "rapport.pdf", "application/pdf", use_container_width=True, key="dl_pdf")
    st.markdown("---")
    st.dataframe(df.tail(20), use_container_width=True)

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
    st.markdown('<div class="section-title">🔥 MATRICE DE CORRÉLATION</div>', unsafe_allow_html=True)
    with st.container():
        st.markdown('<div class="card-premium">', unsafe_allow_html=True)
        corr = compute_correlation(df)
        fig_corr = px.imshow(
            corr, text_auto=".2f", aspect="auto", template=PLOTLY_TEMPLATE,
            color_continuous_scale="Greens"
        )
        fig_corr.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="rgba(255,255,255,0.6)"),
            height=450,
        )
        st.plotly_chart(fig_corr, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    st.markdown("---")
    st.markdown('<div class="section-title">📊 DISTRIBUTIONS STATISTIQUES</div>', unsafe_allow_html=True)
    vars_dist = st.multiselect("Sélectionnez les variables", FEATURES + [TARGET], default=[FEATURES[0], TARGET])
    if vars_dist:
        with st.container():
            st.markdown('<div class="card-premium">', unsafe_allow_html=True)
            fig_dist = make_subplots(rows=1, cols=len(vars_dist), subplot_titles=vars_dist)
            for i, var in enumerate(vars_dist):
                fig_dist.add_trace(go.Histogram(x=df[var], nbinsx=30, marker_color="#00FF7F", 
                                               opacity=0.7, name=var), row=1, col=i+1)
            fig_dist.update_layout(
                template=PLOTLY_TEMPLATE,
                height=350,
                showlegend=False,
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="rgba(255,255,255,0.6)"),
            )
            st.plotly_chart(fig_dist, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
    st.markdown("---")
    st.markdown('<div class="section-title">🤖 INTELLIGENCE DÉCISIONNELLE</div>', unsafe_allow_html=True)
    col_pred, col_rec = st.columns([1, 1])
    with col_pred:
        st.markdown('<div class="card-premium">', unsafe_allow_html=True)
        st.markdown('<div style="font-size:0.6rem;color:rgba(255,255,255,0.3);letter-spacing:2px;text-transform:uppercase;margin-bottom:0.5rem;">Prédiction PN RM</div>', unsafe_allow_html=True)
        pred = model.predict(latest[FEATURES].values.reshape(1, -1))[0]
        st.metric("Valeur prédite", f"{pred:.3f}", delta=f"{pred - TARGET_OPTIMAL:+.3f}")
        st.metric("Intervalle 95%", f"[{pred-0.04:.3f} - {pred+0.04:.3f}]")
        st.markdown('</div>', unsafe_allow_html=True)
    with col_rec:
        st.markdown('<div class="card-premium">', unsafe_allow_html=True)
        st.markdown('<div style="font-size:0.6rem;color:rgba(255,255,255,0.3);letter-spacing:2px;text-transform:uppercase;margin-bottom:0.5rem;">Recommandations</div>', unsafe_allow_html=True)
        recos = generate_recommendations(latest)
        if recos:
            for r in recos[:2]:
                st.markdown(f'<div style="font-size:0.8rem;padding:0.3rem 0;border-bottom:1px solid rgba(255,255,255,0.03);">{r}</div>', unsafe_allow_html=True)
        else:
            st.success("✅ Procédé stable. Aucune recommandation.")
        st.markdown('</div>', unsafe_allow_html=True)
    st.markdown("---")
    st.markdown('<div class="section-title">📋 SYNTHÈSE DES PERFORMANCES</div>', unsafe_allow_html=True)
    cols_perf = st.columns(4)
    perf_score = 100 - (abs(latest[TARGET] - TARGET_OPTIMAL) * 50)
    perf_score = max(0, min(100, perf_score))
    efficiency = (df[TARGET].mean() / TARGET_OPTIMAL) * 100
    stab_score = stability.get('global_stability_score', 70)
    pred_score = max(0, 100 - abs(pred - latest[TARGET]) * 200)
    with cols_perf[0]:
        render_kpi_card("Performance", f"{perf_score:.0f}%", "⭐")
    with cols_perf[1]:
        render_kpi_card("Efficacité", f"{efficiency:.1f}%", "⚡")
    with cols_perf[2]:
        render_kpi_card("Stabilité", f"{stab_score:.0f}%", "📊")
    with cols_perf[3]:
        render_kpi_card("Qualité Prédiction", f"{min(100, pred_score):.0f}%", "🎯")

def render_world_map(df):
    render_page_title("🌍 CARTE MONDIALE LOGISTIQUE", "Ammoniac · Acide phosphorique · Engrais phosphatés")
    st.write("")
    countries_data = {
        "Exportateurs NH₃": {
            "Russia": 18.5, "Qatar": 12.3, "Saudi Arabia": 10.7, "Indonesia": 8.2, 
            "United States": 7.8, "Trinidad": 6.5, "Nigeria": 5.9, "Algeria": 5.1
        },
        "Importateurs NH₃": {
            "China": 12.8, "India": 9.5, "United States": 7.2, "Turkey": 6.3, 
            "Morocco": 5.8, "Spain": 4.9, "South Korea": 4.2, "Brazil": 3.8
        },
        "Exportateurs Acide Phosphorique": {
            "Morocco": 7.2, "China": 5.8, "United States": 4.3, "Russia": 3.9,
            "Saudi Arabia": 3.5, "Tunisia": 2.8, "Jordan": 2.5, "South Africa": 2.1
        },
        "Importateurs Acide Phosphorique": {
            "India": 3.5, "Brazil": 2.8, "Indonesia": 2.2, "Vietnam": 1.9,
            "Turkey": 1.7, "Thailand": 1.5, "Bangladesh": 1.2, "Pakistan": 1.0
        },
        "Producteurs Engrais Phosphatés": {
            "Morocco": 12.5, "China": 9.8, "United States": 7.2, "Russia": 5.5,
            "Saudi Arabia": 4.8, "Tunisia": 3.2, "Jordan": 2.8, "Israel": 2.2,
            "Brazil": 1.8, "India": 1.5
        }
    }
    country_coords = {
        "Morocco": (31.7917, -7.0926), "China": (35.8617, 104.1954), 
        "United States": (37.0902, -95.7129), "India": (20.5937, 78.9629),
        "Russia": (61.5240, 105.3188), "Brazil": (-14.2350, -51.9253),
        "Saudi Arabia": (23.8859, 45.0792), "Indonesia": (-0.7893, 113.9213),
        "Qatar": (25.3548, 51.1839), "Turkey": (38.9637, 35.2433),
        "Spain": (40.4637, -3.7492), "South Korea": (35.9078, 127.7669),
        "Tunisia": (33.8869, 9.5375), "Jordan": (30.5852, 36.2384),
        "Israel": (31.0461, 34.8516), "South Africa": (-30.5595, 22.9375),
        "Nigeria": (9.0820, 8.6753), "Algeria": (28.0339, 1.6596),
        "Trinidad": (10.6918, -61.2225), "Vietnam": (14.0583, 108.2772),
        "Thailand": (15.8700, 100.9925), "Bangladesh": (23.6850, 90.3563),
        "Pakistan": (30.3753, 69.3451)
    }
    map_option = st.selectbox("Sélectionnez la carte à afficher", list(countries_data.keys()))
    st.markdown("---")
    data_dict = countries_data[map_option]
    map_data = []
    for country, value in data_dict.items():
        if country in country_coords:
            lat, lon = country_coords[country]
            map_data.append({"Country": country, "Value": value, "Lat": lat, "Lon": lon})
    df_map = pd.DataFrame(map_data)
    morocco_value = data_dict.get("Morocco", 0)
    col_left, col_right = st.columns([3, 1])
    with col_left:
        if not df_map.empty:
            with st.container():
                st.markdown('<div class="card-premium">', unsafe_allow_html=True)
                fig = go.Figure()
                max_val = df_map["Value"].max()
                sizes = (df_map["Value"] / max_val) * 50 + 20
                fig.add_trace(go.Scattergeo(
                    lon=df_map["Lon"],
                    lat=df_map["Lat"],
                    text=df_map.apply(lambda row: f"{row['Country']}<br>Valeur: {row['Value']:.1f} Mt", axis=1),
                    mode="markers",
                    marker=dict(
                        size=sizes,
                        color=df_map["Value"],
                        colorscale="Greens",
                        showscale=True,
                        colorbar=dict(title="Valeur (Mt)"),
                        line=dict(width=1, color="white")
                    ),
                    hovertemplate="<b>%{text}</b><br>%{marker.color:.1f} Mt<extra></extra>"
                ))
                fig.update_layout(
                    geo=dict(
                        showland=True,
                        landcolor="#1a2a1a",
                        oceancolor="#0a1a0a",
                        coastlinecolor="#00FF7F",
                        countrycolor="#00FF7F",
                        showcountries=True,
                        projection_type="equirectangular",
                    ),
                    template=PLOTLY_TEMPLATE,
                    height=480,
                    paper_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="rgba(255,255,255,0.6)"),
                    title=f"{map_option} — Données 2024",
                    title_font=dict(size=14, color="rgba(255,255,255,0.6)"),
                )
                st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": True})
                st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.info("Aucune donnée disponible pour cette carte.")
    with col_right:
        st.markdown("""
        <div class="card-premium">
            <div style="font-size:0.6rem;color:rgba(255,255,255,0.3);letter-spacing:2px;text-transform:uppercase;margin-bottom:0.5rem;">Comparaison</div>
        """, unsafe_allow_html=True)
        st.markdown(f"""
        <div style="font-size:0.9rem;">
            <div style="display:flex; justify-content:space-between; padding:0.3rem 0;">
                <span style="color:rgba(255,255,255,0.4);">Maroc</span>
                <span style="font-weight:700; color:#00FF7F;">{morocco_value:.1f} Mt</span>
            </div>
            <div style="display:flex; justify-content:space-between; padding:0.3rem 0; border-top:1px solid rgba(255,255,255,0.05);">
                <span style="color:rgba(255,255,255,0.4);">Total</span>
                <span style="font-weight:700;">{sum(data_dict.values()):.1f} Mt</span>
            </div>
            <div style="display:flex; justify-content:space-between; padding:0.3rem 0; border-top:1px solid rgba(255,255,255,0.05);">
                <span style="color:rgba(255,255,255,0.4);">Pays</span>
                <span style="font-weight:700;">{len(data_dict)}</span>
            </div>
            <div style="display:flex; justify-content:space-between; padding:0.3rem 0; border-top:1px solid rgba(255,255,255,0.05);">
                <span style="color:rgba(255,255,255,0.4);">Part Maroc</span>
                <span style="font-weight:700; color:#00FF7F;">{morocco_value/sum(data_dict.values())*100:.1f}%</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("---")
        top3 = sorted(data_dict.items(), key=lambda x: x[1], reverse=True)[:3]
        st.markdown("#### 🏆 Top 3")
        for i, (country, value) in enumerate(top3, 1):
            medal = "🥇" if i == 1 else "🥈" if i == 2 else "🥉"
            st.markdown(f"{medal} **{country}** : {value:.1f} Mt")
        st.markdown('</div>', unsafe_allow_html=True)
    st.markdown("---")
    st.markdown('<div class="section-title">📈 COMPARAISON ANNUELLE AVEC LE MAROC</div>', unsafe_allow_html=True)
    with st.container():
        st.markdown('<div class="card-premium">', unsafe_allow_html=True)
        years = list(range(2019, 2025))
        morocco_data = [5.8, 6.1, 6.5, 7.2, 7.8, 8.2]
        competitor_data = [4.2, 4.5, 4.8, 5.1, 5.5, 5.8]
        fig_comp = go.Figure()
        fig_comp.add_trace(go.Scatter(x=years, y=morocco_data, mode="lines+markers",
                                      name="Maroc", line=dict(color="#00FF7F", width=3),
                                      marker=dict(size=10)))
        fig_comp.add_trace(go.Scatter(x=years, y=competitor_data, mode="lines+markers",
                                      name="Concurrent (moyenne)", line=dict(color="#FFB627", width=2, dash="dash"),
                                      marker=dict(size=8)))
        fig_comp.update_layout(
            template=PLOTLY_TEMPLATE,
            height=320,
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="rgba(255,255,255,0.6)"),
            title="Évolution des exportations d'engrais phosphatés",
            title_font=dict(size=14, color="rgba(255,255,255,0.6)"),
            xaxis_title="Année",
            yaxis_title="Exportations (Millions de tonnes)",
            hovermode="x unified",
            xaxis=dict(showgrid=False, gridcolor="rgba(255,255,255,0.03)"),
            yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.03)"),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        st.plotly_chart(fig_comp, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

# ======================================================================
# IMPORT DES DONNÉES
# ======================================================================

def render_import_data():
    render_page_title("📁 IMPORTATION INTERACTIVE", "Prévisualisez, associez, filtrez et validez vos données")
    
    if "raw_df" not in st.session_state:
        st.session_state.raw_df = None
    if "mapping" not in st.session_state:
        st.session_state.mapping = {}
    if "time_column" not in st.session_state:
        st.session_state.time_column = None
    if "applied_import" not in st.session_state:
        st.session_state.applied_import = False

    uploaded = st.file_uploader("📂 Déposez un fichier CSV", type=["csv"], key="import_uploader")
    if uploaded is not None:
        try:
            df_raw = pd.read_csv(uploaded)
            st.session_state.raw_df = df_raw
            st.session_state.applied_import = False
            st.success(f"✅ {len(df_raw)} lignes chargées en mémoire")
        except Exception as e:
            st.error(f"Erreur de lecture : {e}")
            st.session_state.raw_df = None

    if st.button("🔄 Générer des données synthétiques", use_container_width=True):
        df_synth = generate_synthetic_data(seed=int(datetime.now().timestamp()))
        if isinstance(df_synth.index, pd.DatetimeIndex):
            df_synth = df_synth.reset_index()
            st.session_state.time_column = "Timestamp"
        st.session_state.raw_df = df_synth
        st.session_state.applied_import = False
        st.success(f"✅ {len(df_synth)} lignes synthétiques générées")
        st.rerun()

    if st.session_state.raw_df is not None:
        raw = st.session_state.raw_df
        st.markdown("---")
        st.markdown("### 🧩 Associer les colonnes")
        
        time_options = ["(aucune)"] + list(raw.columns)
        time_candidates = [col for col in raw.columns if any(kw in col.lower() for kw in ["time", "date", "timestamp", "heure", "jour", "mois", "année"])]
        default_time = time_candidates[0] if time_candidates else "(aucune)"
        time_col = st.selectbox("Colonne de date/heure (optionnelle)", time_options, 
                                index=time_options.index(default_time) if default_time in time_options else 0)
        st.session_state.time_column = None if time_col == "(aucune)" else time_col
        
        cols_mapping = st.columns(2)
        mapping = {}
        all_columns = ["(ignorer)"] + list(raw.columns)
        
        def suggest_column(target, columns):
            target_clean = re.sub(r'[^a-zA-Z0-9]', '', target).lower()
            for col in columns:
                col_clean = re.sub(r'[^a-zA-Z0-9]', '', col).lower()
                if target_clean in col_clean or col_clean in target_clean:
                    return col
            return "(ignorer)"
        
        for i, feat in enumerate(FEATURES + [TARGET]):
            with cols_mapping[i % 2]:
                suggestion = suggest_column(feat, raw.columns)
                default_index = 0
                if suggestion != "(ignorer)":
                    try:
                        default_index = all_columns.index(suggestion)
                    except ValueError:
                        default_index = 0
                mapping[feat] = st.selectbox(
                    f"Colonne pour **{feat}**",
                    options=all_columns,
                    index=default_index,
                    key=f"map_{feat}"
                )
        st.session_state.mapping = mapping
        
        st.markdown("---")
        st.markdown("### 🎚️ Filtrage pré-import")
        col_filt1, col_filt2 = st.columns(2)
        with col_filt1:
            min_rows = st.slider("Nombre minimum de lignes", 10, len(raw), min(50, len(raw)))
        with col_filt2:
            outlier_threshold = st.slider("Seuil d'écart-type pour exclure (σ)", 1.0, 5.0, 3.0, step=0.5)
        
        preview_data = {}
        for feat, col in mapping.items():
            if col != "(ignorer)" and col in raw.columns:
                preview_data[feat] = raw[col]
        if preview_data:
            preview_df = pd.DataFrame(preview_data)
            z_scores = np.abs(stats.zscore(preview_df[FEATURES], nan_policy='omit'))
            preview_clean = preview_df[(z_scores < outlier_threshold).all(axis=1)]
            st.markdown("#### 👁️ Aperçu (10 premières lignes après filtrage)")
            st.dataframe(preview_clean.head(10), use_container_width=True)
            st.caption(f"Lignes après filtrage : {len(preview_clean)} sur {len(preview_df)} (seuil {outlier_threshold}σ)")
        else:
            st.warning("Aucune colonne mappée pour l'aperçu.")
            preview_clean = pd.DataFrame()
        
        if st.button("🚀 Appliquer au tableau de bord", type="primary", use_container_width=True):
            if not preview_data:
                st.error("Aucune colonne n'a été associée. Veuillez mapper au moins une variable.")
                return
            
            final_data = {}
            for feat, col in mapping.items():
                if col != "(ignorer)" and col in raw.columns:
                    final_data[feat] = raw[col]
            df_applied = pd.DataFrame(final_data)
            df_applied = df_applied.dropna()
            if len(df_applied) > 0:
                z_scores = np.abs(stats.zscore(df_applied[FEATURES], nan_policy='omit'))
                df_clean = df_applied[(z_scores < outlier_threshold).all(axis=1)]
            else:
                df_clean = df_applied
            if len(df_clean) >= min_rows:
                if st.session_state.time_column and st.session_state.time_column in raw.columns:
                    try:
                        time_series = pd.to_datetime(raw[st.session_state.time_column])
                        df_clean.index = time_series.loc[df_clean.index]
                    except Exception as e:
                        st.warning(f"Impossible de convertir la colonne de temps en datetime : {e}. Index séquentiel utilisé.")
                        df_clean.index = pd.date_range(start=datetime(2024,1,1), periods=len(df_clean), freq='h')
                else:
                    start = datetime(2024, 1, 1)
                    df_clean.index = pd.date_range(start=start, periods=len(df_clean), freq='h')
                
                if "PUMP_AMPERAGE" not in df_clean.columns:
                    df_clean["PUMP_AMPERAGE"] = 45 + np.random.normal(0, 5, len(df_clean))
                for col in FEATURES:
                    if col not in df_clean.columns:
                        df_clean[col] = np.random.normal(50, 10, len(df_clean))
                if TARGET not in df_clean.columns:
                    df_clean[TARGET] = TARGET_OPTIMAL + np.random.normal(0, 0.05, len(df_clean))
                
                st.session_state.df = df_clean
                st.session_state.data_version = st.session_state.get("data_version", 0) + 1
                st.session_state.applied_import = True
                st.success(f"✅ {len(df_clean)} lignes appliquées avec succès !")
                st.rerun()
            else:
                st.error(f"❌ Après filtrage, il reste {len(df_clean)} lignes (seuil = {min_rows}).")
    else:
        st.info("Aucune donnée chargée. Utilisez l'upload ou générez des données synthétiques.")

# ======================================================================
# SECTION PRÉDICTION (PN RM + CLASSIFICATION DAP/MAP + SHAP)
# ======================================================================

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
        section_shap(model, df)

# ======================================================================
# APPLICATION PRINCIPALE
# ======================================================================

def render_authenticated_app(user):
    inject_css()
    if "data_version" not in st.session_state:
        st.session_state.data_version = 0
    if "df" not in st.session_state:
        st.session_state.df = None
    if "model_r2" not in st.session_state:
        st.session_state.model_r2 = None

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
        st.caption("v19.0 · Prédiction + SHAP + Classification DAP/MAP")

    df = st.session_state.df
    if df is None:
        st.warning("Veuillez charger des données (fichier ou génération) dans l'onglet **Data Import**.")
        allowed = ROLE_PERMISSIONS.get(user["role"], set())
        visible_tabs = [(key, label) for key, label in TAB_DEFS if key in allowed]
        tabs = st.tabs([label for _, label in visible_tabs])
        for (key, _), tab in zip(visible_tabs, tabs):
            with tab:
                if key == "page1":
                    render_import_data()
                else:
                    st.warning("⚠️ Veuillez d'abord charger des données via l'onglet **Data Import**.")
        return

    with st.spinner("Entraînement des modèles..."):
        model, r2 = train_random_forest(df)
        st.session_state.model_r2 = r2
        clf = train_quality_classifier(df)

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

    st.markdown(f"""
    <div class="footer" style="display:flex;justify-content:space-between;align-items:center;padding:1rem 0.5rem;color:rgba(255,255,255,0.2);font-size:0.65rem;border-top:1px solid rgba(0,255,127,0.06);margin-top:2rem;letter-spacing:1px;">
        <span>⚡ OCP Strategic Intelligence · Manufacturing Digital Twin · v19.0</span>
        <span>{datetime.now().strftime('%Y-%m-%d %H:%M')}</span>
    </div>
    """, unsafe_allow_html=True)

def main():
    init_db()
    if "auth_user" not in st.session_state:
        inject_css()
        render_auth_screen(COLORS, LOGO_URL)
        return
    render_authenticated_app(st.session_state["auth_user"])

if __name__ == "__main__":
    main()