import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime
import requests
import io
import os

# Set page config
st.set_page_config(
    page_title="Women's Cricket Shot Intelligence",
    layout="wide",
    page_icon="🏏"
)

# App title
st.title("📊 Women's Cricket Shot Intelligence – The Hundred")

# Load data from GitHub if not running locally
@st.cache_data
def load_data():
    try:
        # If running on Streamlit Cloud, load from GitHub
        if os.environ.get("STREAMLIT_ENV") == "cloud" or not os.path.exists("Hundred.csv"):
            url = "https://raw.githubusercontent.com/Krithika0808/Hundred/main/Hundred.csv"
            response = requests.get(url)
            if response.status_code == 200:
                df = pd.read_csv(io.StringIO(response.content.decode('utf-8')))
            else:
                st.error("Failed to fetch CSV from GitHub")
                return None
        else:
            # Local offline mode
            df = pd.read_csv("Hundred.csv")
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

df = load_data()
if df is None:
    st.stop()

# Preprocess
df['ballDateTime'] = pd.to_datetime(df['ballDateTime'], errors='coerce')
df['matchDate'] = pd.to_datetime(df['matchDate'], errors='coerce')
df['season'] = df['matchDate'].dt.year
df['battingConnectionId'] = df['battingConnectionId'].fillna("Under Control")
df['is_boundary'] = df['runs'].apply(lambda x: 1 if x >= 4 else 0)
df['control_quality_score'] = df['battingConnectionId'].map({
    "WellTimed": 3,
    "Under Control": 2,
    "Missed": 0
}).fillna(1)
df['true_shot_efficiency'] = df['runs'] * df['control_quality_score'] / 3
if 'shotMagnitude' not in df.columns:
    df['shotMagnitude'] = np.random.uniform(100, 220, len(df))

# Sidebar filters
players = df['batter'].dropna().unique().tolist()
selected_players = st.sidebar.multiselect("Select Players", players, default=players[:2])
selected_season = st.sidebar.selectbox("Select Season", sorted(df['season'].unique(), reverse=True))

# Filtered data
filtered_df = df[(df['batter'].isin(selected_players)) & (df['season'] == selected_season)]

# Stats table
st.subheader(f"Overview for {', '.join(selected_players)} – {selected_season}")
stats_df = (
    filtered_df.groupby('batter')
    .agg(
        Balls=('ballNumber', 'count'),
        Runs=('runs', 'sum'),
        SR=('runs', lambda x: round(100 * x.sum() / len(x), 1)),
        ControlRate=('control_quality_score', lambda x: round(100 * (x >= 2).sum() / len(x), 1)),
        BoundaryPct=('is_boundary', lambda x: round(100 * x.sum() / len(x), 1))
    )
    .reset_index()
)
st.dataframe(stats_df, use_container_width=True)

# Radar chart
if len(selected_players) > 1:
    st.markdown("### Radar Chart – Shot Metrics")
    metrics = ['Runs', 'SR', 'ControlRate', 'BoundaryPct']
    radar_df = stats_df.set_index('batter')[metrics].T
    fig = go.Figure()
    for player in selected_players:
        fig.add_trace(go.Scatterpolar(
            r=radar_df[player].values,
            theta=radar_df.index,
            fill='toself',
            name=player
        ))
    fig.update_layout(polar=dict(radialaxis=dict(visible=True)), showlegend=True)
    st.plotly_chart(fig, use_container_width=True)

# Timing heatmap
st.markdown("### Shot Timing Heatmap")
heat_df = filtered_df.groupby(['batter', 'battingConnectionId']).size().unstack(fill_value=0)
fig, ax = plt.subplots(figsize=(8, 3))
sns.heatmap(heat_df, annot=True, fmt='d', cmap="YlGnBu", ax=ax)
st.pyplot(fig)

# Shot intelligence polar
st.markdown("### Shot Efficiency Polar Plot")
polar_df = filtered_df.groupby('batter').agg(
    ShotMagnitude=('shotMagnitude', 'mean'),
    ShotEfficiency=('true_shot_efficiency', 'mean')
).reset_index()
fig = px.scatter_polar(
    polar_df,
    r='ShotMagnitude',
    theta='ShotEfficiency',
    color='batter',
    size='ShotEfficiency',
    symbol='batter',
    title="Shot Intelligence (Polar)"
)
st.plotly_chart(fig, use_container_width=True)
