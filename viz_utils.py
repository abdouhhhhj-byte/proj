# ======================================================================
# Visualisations 3D et graphiques
# ======================================================================

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np
import pandas as pd

from config import FEATURES, TARGET, TARGET_OPTIMAL, PLOTLY_TEMPLATE

# ------------------------------------------------------------
# Fonctions 3D (extraites de la galerie)
# ------------------------------------------------------------

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
    corr = df[FEATURES + [TARGET]].corr().values
    x = list(df[FEATURES + [TARGET]].columns)
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