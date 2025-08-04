import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
import requests
import io
import time
import os
from datetime import datetime

st.set_page_config(
    page_title="Women's Cricket Shot Intelligence",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load from GitHub CSV
@st.cache
def load_data():
    url = "https://raw.githubusercontent.com/Krithika0808/Hundred/main/Hundred.csv"
    response = requests.get(url)
    if response.status_code == 200:
        df = pd.read_csv(io.StringIO(response.content.decode('utf-8')))
        return df
    else:
        st.error("Failed to load data from GitHub.")
        return pd.DataFrame()

df = load_data()

if df.empty:
    st.warning("No data to show.")
    st.stop()

# Sidebar filters
st.sidebar.header("Filters")
selected_player = st.sidebar.selectbox("Select a player", sorted(df['batsman'].dropna().unique()))
bowling_types = sorted(df['bowlingType'].dropna().unique())
selected_bowling_types = st.sidebar.multiselect("Bowling Types", bowling_types, default=bowling_types)
phases = sorted(df['phase'].dropna().unique())
selected_phases = st.sidebar.multiselect("Phase", phases, default=phases)

# Filter the data
filtered_df = df[
    (df['batter'] == selected_player) &
    (df['bowlingType'].isin(selected_bowling_types)) &
    (df['phase'].isin(selected_phases))
]

# Create Zone Map
zone_counts = filtered_df['fieldingPosition'].value_counts().reset_index()
zone_counts.columns = ['Zone', 'Count']
zone_chart = px.bar(zone_counts, x='Zone', y='Count', title='Zone-wise Shot Frequency')

# Timing vs Control Heatmap
if 'battingConnectionId' in filtered_df.columns and 'fieldingPosition' in filtered_df.columns:
    heatmap_data = filtered_df.groupby(['battingConnectionId', 'fieldingPosition']).size().reset_index(name='Count')
    heatmap = px.density_heatmap(
        heatmap_data,
        x='fieldingPosition',
        y='battingConnectionId',
        z='Count',
        color_continuous_scale='Viridis',
        title='Timing vs Control Heatmap'
    )
else:
    heatmap = None

# Shot Type by Zone
if 'shotType' in filtered_df.columns and 'fieldingPosition' in filtered_df.columns:
    shot_zone_data = filtered_df.groupby(['fieldingPosition', 'shotType']).size().reset_index(name='Count')
    shot_zone_chart = px.bar(
        shot_zone_data,
        x='fieldingPosition',
        y='Count',
        color='shotType',
        barmode='group',
        title='Shot Type Distribution by Zone'
    )
else:
    shot_zone_chart = None

# Display visualizations
st.title("Women's Cricket Shot Intelligence Dashboard")
st.markdown(f"### Shot Intelligence for **{selected_player}**")

col1, col2 = st.columns(2)
with col1:
    st.plotly_chart(zone_chart, use_container_width=True)
with col2:
    if heatmap:
        st.plotly_chart(heatmap, use_container_width=True)
    else:
        st.info("Timing vs Control data not available.")

st.markdown("### Shot Type by Zone")
if shot_zone_chart:
    st.plotly_chart(shot_zone_chart, use_container_width=True)
else:
    st.info("Shot Type data not available.")

