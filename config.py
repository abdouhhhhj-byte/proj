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

# Chemins et autres constantes
DB_PATH = "users.db"
ROLES = ["Administrateur", "Ingénieur procédé", "Opérateur", "Visiteur"]
MASTER_PASSWORD = "azertocp"
LOGO_LOCAL_PATH = r"C:\Users\ABDEL\Downloads\pfe\ocp-logo-png_seeklogo-222172.webp"
LOGO_FALLBACK_URL = "https://upload.wikimedia.org/wikipedia/commons/7/7e/Ocp-group.png"