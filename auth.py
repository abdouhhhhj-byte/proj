import sqlite3
import hashlib
import secrets
import re
from contextlib import contextmanager
from datetime import datetime
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