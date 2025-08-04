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
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title="Women's Cricket Shot Intelligence",
    layout="wide",
    page_icon="🏏"
)

# App Title
st.title("📊 Women's Cricket Shot Intelligence – The Hundred")

@st.cache_data
def load_data(from_github=True):
    if from_github:
        url = "https://raw.githubusercontent.com/Krithika0808/Hundred/main/Hundred.csv"
        response = requests.get(url)
        if response.status_code == 200:
            df = pd.read_csv(io.StringIO(response.content.decode('utf-8')))
        else:
            st.error("Failed to fetch CSV from GitHub")
            return None
    else:
        df = pd.read_csv("Hundred.csv")
    return df

@st.cache_data
def preprocess_data(df):
    df['ballDateTime'] = pd.to_datetime(df['ballDateTime'], errors='coerce')
    df['is_boundary'] = df['is_boundary'].astype(bool)
    df['matchDate'] = pd.to_datetime(df['matchDate'], errors='coerce')
    df['season'] = df['matchDate'].dt.year
    return df

@st.cache_data
def calculate_shot_intelligence_metrics(df):
    df = df.copy()
    
    # Sample logic – replace with real values if available
    df['battingConnectionId'] = df['battingConnectionId'].fillna("Under Control")
    df['control_quality_score'] = df['battingConnectionId'].map({
        "WellTimed": 3,
        "Under Control": 2,
        "Missed": 0
    }).fillna(1)
    
    df['true_shot_efficiency'] = df['runs'] * df['control_quality_score'] / 3

    # If shotMagnitude not available, simulate for plotting
    if 'shotMagnitude' not in df.columns:
        df['shotMagnitude'] = np.random.uniform(100, 220, len(df))

    return df

# Sidebar controls
with st.sidebar:
    st.header("Filters")
    use_github = st.checkbox("Load data from GitHub", value=True)
    df = load_data(use_github)
    
    if df is not None:
        df = preprocess_data(df)
        df = calculate_shot_intelligence_metrics(df)

        unique_players = df['batter'].dropna().unique().tolist()
        selected_players = st.multiselect("Select Players", unique_players, default=unique_players[:2])
        selected_season = st.selectbox("Select Season", sorted(df['season'].unique(), reverse=True))
    else:
        st.stop()

# Main Analysis Section
if df is not None and selected_players:
    filtered_df = df[
        (df['batter'].isin(selected_players)) &
        (df['season'] == selected_season)
    ]

    st.subheader(f"Overview for {', '.join(selected_players)} – Season {selected_season}")

    col1, col2 = st.columns(2)

    with col1:
        player_stats = (
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
        st.dataframe(player_stats, use_container_width=True)

    with col2:
        st.markdown("### Radar Chart – Shot Metrics")
        if len(selected_players) < 2:
            st.warning("Select at least two players to compare")
        else:
            metrics = ['Runs', 'SR', 'ControlRate', 'BoundaryPct']
            radar_df = player_stats.set_index('batter')[metrics].T
            fig = go.Figure()
            for player in selected_players[:4]:
                fig.add_trace(go.Scatterpolar(
                    r=radar_df[player].values,
                    theta=radar_df.index,
                    fill='toself',
                    name=player
                ))
            fig.update_layout(polar=dict(radialaxis=dict(visible=True)), showlegend=True)
            st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Shot Timing Heatmap")
    heat_df = filtered_df.groupby(['batter', 'battingConnectionId']).size().unstack(fill_value=0)
    fig, ax = plt.subplots(figsize=(8, 3))
    sns.heatmap(heat_df, annot=True, fmt='d', cmap="YlGnBu", ax=ax)
    st.pyplot(fig)

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

else:
    st.warning("No players selected or data not loaded properly.")
