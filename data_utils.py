import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import re
from config import FEATURES, TARGET, TARGET_OPTIMAL

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