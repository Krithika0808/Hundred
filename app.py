import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt
import requests
import time
import os
import io
from datetime import datetime

# Constants
GITHUB_SOURCE = "https://raw.githubusercontent.com/Krithika0808/Hundred/main/Hundred.csv"
LOCAL_CACHE = "hundred_cache.pkl"
DATA_RETRY_DELAY = 5
REQUIRED_COLUMNS = {
    'batsman', 'runs', 'totalBallNumber', 'shotAngle',
    'battingShotTypeId', 'battingConnectionId', 'matchDate'
}

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data(ttl=3600)
def load_data():
    """Loads data from GitHub or local fallback"""
    for attempt in range(3):
        try:
            response = requests.get(GITHUB_SOURCE, timeout=10)
            if response.status_code == 200:
                df = pd.read_csv(io.StringIO(response.content.decode('utf-8')))
                if validate_dataframe(df):
                    return preprocess_data(df)
        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {e}")
            time.sleep(DATA_RETRY_DELAY)
    
    if os.path.exists(LOCAL_CACHE):
        return pd.read_pickle(LOCAL_CACHE)
    
    return create_sample_data()

def validate_dataframe(df):
    return (
        not df.empty and
        REQUIRED_COLUMNS.issubset(df.columns)
    )

def preprocess_data(df):
    df = df.dropna(subset=['batsman', 'runs', 'totalBallNumber'])
    df['runs'] = df['runs'].astype(int)
    df['totalBallNumber'] = df['totalBallNumber'].astype(int)
    df['shotAngle'] = df['shotAngle'].apply(lambda x: x % 360 if pd.notnull(x) else 0).astype(float)
    df['is_boundary'] = df['runs'].isin([4, 6])
    if 'shotMagnitude' not in df.columns:
        df['shotMagnitude'] = np.random.uniform(100, 200, len(df))
    df.to_pickle(LOCAL_CACHE)
    return df

def create_sample_data():
    np.random.seed(42)
    players = ['Smriti Mandhana', 'Harmanpreet Kaur', 'Beth Mooney',
               'Alyssa Healy', 'Meg Lanning', 'Elyse Villani', 'Deepti Jones']
    shot_types = ['Drive', 'Pull', 'Cut', 'Sweep', 'Flick', 'Hook',
                  'Reverse Sweep', 'Defensive', 'Cover Drive']
    connection_types = ['Middled', 'WellTimed', 'Undercontrol',
                        'MisTimed', 'Missed', 'HitBody', 'TopEdge']

    data = {
        'batsman': np.random.choice(players, 1000),
        'battingShotTypeId': np.random.choice(shot_types, 1000),
        'battingConnectionId': np.random.choice(connection_types, 1000, p=[0.3, 0.25, 0.2, 0.1, 0.05, 0.05, 0.05]),
        'runs': np.random.choice([0, 1, 2, 3, 4, 6], 1000),
        'totalBallNumber': np.random.randint(1, 101, 1000),
        'shotAngle': np.random.uniform(0, 360, 1000),
        'shotMagnitude': np.random.uniform(100, 200, 1000),
        'matchDate': pd.date_range(start='2023-01-01', periods=1000, freq='D')
    }
    df = pd.DataFrame(data)
    df['is_boundary'] = df['runs'].isin([4, 6])
    return df

def calculate_shot_intelligence_metrics(df):
    control_scores = {
        'Middled': 3, 'WellTimed': 3, 'Undercontrol': 3,
        'MisTimed': 2, 'Missed': 1, 'HitBody': 0.5,
        'TopEdge': 2, 'BatPad': 1, 'BottomEdge': 2,
        'Gloved': 2, 'HitHelmet': 0, 'HitPad': 0,
        'InsideEdge': 2, 'LeadingEdge': 2, 'Left': 3
    }

    df['control_quality_score'] = df['battingConnectionId'].map(control_scores).fillna(1.5)

    angle_bins = [0, 45, 90, 135, 180, 225, 270, 315, 360]
    angle_labels = ['Long Off', 'Cover', 'Point', 'Third Man',
                    'Fine Leg', 'Square Leg', 'Mid Wicket', 'Long On']
    df['angle_zone'] = pd.cut(df['shotAngle'], bins=angle_bins, labels=angle_labels, include_lowest=True)

    df['control_score'] = (
        df['control_quality_score'] * 33.33 +
        df['runs'] * 5 +
        df['is_boundary'].astype(int) * 20 +
        (df['angle_zone'].isin(['Cover', 'Mid Wicket', 'Long On', 'Long Off']).astype(int) * 10)
    ).clip(0, 100)

    df['true_shot_efficiency'] = df['runs'] * df['control_quality_score'] / 3
    df['true_risk_reward'] = np.where(df['control_score'] >= 50, df['runs'] * 1.3, df['runs'] * 0.7)

    return df

def create_shot_angle_heatmap(df, player_name):
    player_data = df[df['batsman'] == player_name]
    if player_data.empty:
        return go.Figure()
    
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=player_data['shotMagnitude'],
        theta=player_data['shotAngle'],
        mode='markers',
        marker=dict(
            size=player_data['runs'] * 3 + 8,
            color=player_data['control_score'],
            colorscale='RdYlGn',
            showscale=True,
            colorbar=dict(title="Control Score")
        ),
        text=player_data.apply(
            lambda row: f"Shot: {row.battingShotTypeId}<br>Runs: {row.runs}<br>Score: {row.control_score:.1f}", axis=1),
        hoverinfo='text'
    ))
    fig.update_layout(
        title=f"{player_name} – Shot Map",
        polar=dict(
            radialaxis=dict(range=[0, 200], visible=True),
            angularaxis=dict(
                tickvals=[0, 45, 90, 135, 180, 225, 270, 315],
                ticktext=['Long Off', 'Cover', 'Point', 'Third Man',
                          'Fine Leg', 'Square Leg', 'Mid Wicket', 'Long On']
            )
        ),
        height=600
    )
    return fig

def main():
    st.set_page_config(
        page_title="Women's Cricket Shot Intelligence",
        page_icon="🏏",
        layout="wide"
    )

    st.markdown('<h1 class="main-header">🏏 Women\'s Cricket Shot Intelligence Matrix</h1>', unsafe_allow_html=True)

    with st.spinner("Loading data..."):
        try:
            df = load_data()
            df = calculate_shot_intelligence_metrics(df)
        except Exception as e:
            st.error(f"Data load failed: {e}")
            df = create_sample_data()
            df = calculate_shot_intelligence_metrics(df)

    st.sidebar.header("🎯 Player Selection")
    players = df['batsman'].unique()
    selected_players = st.sidebar.multiselect("Choose Players", options=players, default=players[:2])

    tab1, tab2, tab3 = st.tabs(["📍 Shot Map", "📈 Shot Types", "📊 Radar"])

    with tab1:
        st.subheader("360° Shot Map")
        for player in selected_players:
            st.plotly_chart(create_shot_angle_heatmap(df, player), use_container_width=True)

    with tab2:
        st.subheader("Shot Control vs Aggression")
        shot_agg = df.groupby('battingShotTypeId').agg(
            control_score=('control_score', 'mean'),
            runs=('runs', 'mean'),
            is_boundary=('is_boundary', 'mean')
        ).reset_index()

        fig = px.scatter(
            shot_agg, x='control_score', y='runs',
            size='is_boundary', color='control_score',
            hover_name='battingShotTypeId',
            labels={'control_score': 'Control Score', 'runs': 'Avg Runs', 'is_boundary': 'Boundary %'},
            title="Shot Type Performance"
        )
        st.plotly_chart(fig)

    with tab3:
        st.subheader("Radar Comparison")
        if len(selected_players) < 2:
            st.warning("Please select at least 2 players")
        else:
            radar_data = df[df['batsman'].isin(selected_players)].groupby('batsman').agg({
                'control_score': 'mean',
                'is_boundary': 'mean',
                'runs': 'mean',
                'true_shot_efficiency': 'mean',
                'true_risk_reward': 'mean'
            }).rename(columns={
                'control_score': 'Control',
                'is_boundary': 'Boundary %',
                'runs': 'Runs/Shot',
                'true_shot_efficiency': 'Efficiency',
                'true_risk_reward': 'Risk-Reward'
            }).T

            fig = go.Figure()
            for i, player in enumerate(selected_players):
                if player in radar_data.columns:
                    fig.add_trace(go.Scatterpolar(
                        r=radar_data[player].values,
                        theta=radar_data.index,
                        fill='toself',
                        name=player
                    ))
            fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 100])))
            st.plotly_chart(fig)

if __name__ == "__main__":
    main()
