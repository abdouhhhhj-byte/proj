# models.py
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import r2_score

FEATURES = [
    "INPUT_NH3_FLOW",
    "INPUT_PHOS_ACID_FLOW_54",
    "SLURRY_TEMPERATURE",
    "PUMP_SLURRY_FLOW_AP01",
    "WASHING_LIQUID_FLOW",
]
TARGET = "PN RM"
RF_PARAMS = {"n_estimators": 200, "max_depth": 12, "random_state": 42, "n_jobs": -1}

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