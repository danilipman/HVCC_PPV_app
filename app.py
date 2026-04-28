import streamlit as st
import os
import numpy as np
import pandas as pd
from dash import Dash, dcc, html, Input, Output
import plotly.express as px
import plotly.graph_objects as go
import plotly
import json
from ellipse_util import *
from astropy.table import Table


# -------------------------
# CONFIG
# -------------------------
st.set_page_config(layout="wide")
st.title("ACES High Velocity Clouds PPV")

# -------------------------
# DATA LOADING (CACHED)
# -------------------------
@st.cache_data
def load_data():

    # --- scouse fits ---
    scouse_fits = pd.read_csv(
        './DATA/final_cmz_scouse_hnco_fits.csv',
        usecols=[0,1,2,3,5,7],
        names=['n', 'l', 'b', 'amp', 'velocity', 'FWHM'],
        sep=r"\s+"
    )

    # --- ratio stats ---
    ratio_stats_tab = Table.read('./DATA/RatioMap_Stats_AllEVFs.csv')

    evf_name = ratio_stats_tab['ID_lb_Name']
    HNCO_SiO21_Median = ratio_stats_tab['HNCO_7m12mTP_SiO21 Median']
    HNCO_CS21_Median = ratio_stats_tab['HNCO_7m12mTP_CS21 Median']
    HNCO_HC3N_Median = ratio_stats_tab['HNCO_7m12mTP_HC3N Median']

    # --- extract EVF IDs ---
    ratio_evf_ID = []
    for evf in evf_name:
        matchcoord = evf.split('_',2)[0].split('ID',1)[1]
        ratio_evf_ID.append(int(matchcoord))

    # --- stvec ---
    with open('./DATA/stvec.json', 'r') as f:
        loaded_dict = json.load(f)

    # Handle both normal JSON and double-encoded JSON
    if isinstance(loaded_dict, str):
        loaded_dict = json.loads(loaded_dict)

    stvec = {k: np.array(v) for k, v in loaded_dict.items()}

    return (
        scouse_fits,
        HNCO_SiO21_Median,
        HNCO_CS21_Median,
        HNCO_HC3N_Median,
        ratio_evf_ID,
        stvec
    )


# Load data
(
    scouse_fits,
    HNCO_SiO21_Median,
    HNCO_CS21_Median,
    HNCO_HC3N_Median,
    ratio_evf_ID,
    stvec
) = load_data()


# -------------------------
# UI CONTROLS
# -------------------------
color = st.selectbox(
    "Line Ratio:",
    options=['HNCO/SiO', 'HNCO/CS', 'HNCO/HC3N']
)

color_range = st.slider(
    "Color Range",
    0.0, 2.5,
    (0.0, 2.5),
    step=0.01
)

# -------------------------
# SELECT DATA
# -------------------------
if color == 'HNCO/SiO':
    lineratio_toplot = HNCO_SiO21_Median
elif color == 'HNCO/CS':
    lineratio_toplot = HNCO_CS21_Median
elif color == 'HNCO/HC3N':
    lineratio_toplot = HNCO_HC3N_Median


# -------------------------
# BUILD FIGURE
# -------------------------
trace_scouse = go.Scatter3d(
    x=scouse_fits.l,
    y=scouse_fits.b,
    z=scouse_fits.velocity,
    mode='markers',
    name='MOPRA HNCO (Henshaw+2016)',
    marker=dict(
        color=np.log10(scouse_fits.amp),
        size=1,
        colorscale='Blues',
        opacity=0.5,
        symbol='square'
    )
)

trace_sgra = go.Scatter3d(
    x=[-0.056], y=[-0.046], z=[0],
    mode='markers',
    name='Sgr A*',
    marker=dict(color='black', size=2, symbol='x')
)

# NOTE: Rings must already exist in your environment
trace_ellipse = go.Scatter3d(
    x=np.degrees(Rings.l),
    y=np.degrees(Rings.b),
    z=Rings.vr,
    mode='lines',
    name='ellipse model (Lipman+2026)',
    line=dict(color='grey')
)

# -------------------------
# FIRST HVCC
# -------------------------
first_key = list(stvec.keys())[0]
k = stvec[first_key][0]

l1, b1, v1 = k[:,0], k[:,1], k[:,2]

hvcc_c = float(
    lineratio_toplot[
        int(np.where(int(first_key) == np.array(ratio_evf_ID))[0][0])
    ]
)

trace_hvcc_0 = go.Scatter3d(
    x=l1, y=b1, z=v1,
    mode='markers',
    name='HVCC data',
    legendgroup='HVCC',
    showlegend=True,
    marker=dict(
        color=np.full(len(l1), hvcc_c),
        size=1,
        colorscale="magma_r",
        cmin=color_range[0],
        cmid=(color_range[0] + color_range[1]) / 2,
        cmax=color_range[1],
        opacity=0.5,
        symbol='circle'
    )
)

data = [trace_scouse, trace_ellipse, trace_sgra, trace_hvcc_0]

# -------------------------
# LOOP HVCCs
# -------------------------
for reg in stvec:
    for k in stvec[reg]:

        if (int(reg) in ratio_evf_ID) and (np.shape(k)[-1] == 3):

            plotc = float(
                lineratio_toplot[
                    int(np.where(int(reg) == np.array(ratio_evf_ID))[0][0])
                ]
            )

            l1, b1, v1 = k[:,0], k[:,1], k[:,2]

            trace_hvcc = go.Scatter3d(
                x=l1, y=b1, z=v1,
                mode='markers',
                name='HVCC data',
                legendgroup='HVCC',
                showlegend=False,
                marker=dict(
                    color=np.full(len(l1), plotc),
                    size=1,
                    colorscale="magma_r",
                    cmin=color_range[0],
                    cmid=(color_range[0] + color_range[1]) / 2,
                    cmax=color_range[1],
                    opacity=0.5,
                    symbol='circle',
                    showscale=True,
                    colorbar=dict(
                        orientation='h',
                        y=0,
                        title=color + ' Median'
                    )
                ),
                hovertemplate=(
                    f"<b>ID: {reg}</b><br>"
                    "l: %{x:.2f}<br>"
                    "b: %{y:.2f}<br>"
                    "vr: %{z:.2f}<br>"
                    f"ratio: {plotc:.2f}<br>"
                )
            )

            data.append(trace_hvcc)


# -------------------------
# LAYOUT
# -------------------------
fig = go.Figure(data)

fig.update_scenes(xaxis_autorange="reversed")

fig.update_layout(
    width=1300,
    height=800,
    paper_bgcolor='white',
    margin=dict(r=20, l=50, b=10, t=0),
    legend=dict(
        orientation="v",
        y=0.5,
        x=1.0,
        bgcolor="white",
        bordercolor="black",
        borderwidth=2
    ),
    scene=dict(
        aspectmode='manual',
        aspectratio=dict(x=1.5, y=0.8, z=0.8),
        xaxis_title='GLON [deg]',
        yaxis_title='GLAT [deg]',
        zaxis_title='Radial Velocity [km/s]',
        bgcolor='white'
    )
)

fig.update_layout(scene=dict(camera=dict(eye=dict(x=0.1, y=-2, z=0.4))))

# -------------------------
# DISPLAY
# -------------------------
st.plotly_chart(fig, use_container_width=True)

# -------------------------
# DOWNLOADS
# -------------------------
st.download_button(
    "Download PNG",
    data=fig.to_image(format="png", scale=6),
    file_name="PPV_view.png"
)

st.download_button(
    "Download HTML",
    data=fig.to_html(),
    file_name="PPV_view.html"
)
