import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
import requests
import time
import os
from datetime import datetime

# Configuration
GITHUB_SOURCE = "https://raw.githubusercontent.com/Krithika0808/Hundred/main/Hundred.csv"
LOCAL_CACHE = "hundred_cache.pkl"
DATA_RETRY_DELAY = 5  # Seconds between retries
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
    .metric-card {
        background: linear-gradient(135deg, #f0f2f6 0%, #e8f4fd 100%);
        padding: 1.5rem;
        border-radius: 0.75rem;
        border-left: 4px solid #1f77b4;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 0.5rem 0;
    }
    .insight-box {
        background: linear-gradient(135deg, #e8f4fd 0%, #f0f8ff 100%);
        padding: 1.5rem;
        border-radius: 0.75rem;
        border: 2px solid #1f77b4;
        margin: 1rem 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .player-card {
        background: linear-gradient(135deg, #fff 0%, #f8f9fa 100%);
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #dee2e6;
        margin: 0.5rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    .stMetric > div[data-testid="metric-container"] {
        background-color: rgba(255,255,255,0.05);
        border: 1px solid rgba(49,51,63,0.2);
        padding: 0.5rem;
        border-radius: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data(ttl=3600)
def load_data():
    """Robust data loading with GitHub priority and validation"""
    max_retries = 3
    for attempt in range(max_retries):
        try:
            # 1. GitHub Priority Load
            if os.getenv("GITHUB_LOAD", "true").lower() == "true":
                print(f"Attempt {attempt+1}: Loading from GitHub")
                response = requests.get(
                    GITHUB_SOURCE,
                    headers={'User-Agent': 'WomenCricketApp/1.0'},
                    timeout=15
                )
                if response.status_code == 200:
                    df = pd.read_csv(response.content)
                    if validate_dataframe(df):
                        return preprocess_data(df)
            
            # 2. Local Cache Fallback
            if os.path.exists(LOCAL_CACHE):
                print("Loading from local cache")
                return pd.read_pickle(LOCAL_CACHE)
            
            # 3. Sample Data Fallback
            print("Generating sample data")
            return create_sample_data()
            
        except Exception as e:
            print(f"Attempt {attempt+1} failed: {str(e)}")
            time.sleep(DATA_RETRY_DELAY)
    
    raise Exception("All data loading attempts failed")

def validate_dataframe(df):
    """Comprehensive dataframe validation"""
    return (
        not df.empty and
        REQUIRED_COLUMNS.issubset(df.columns) and
        df['runs'].dtype in (np.int64, np.float64) and
        len(df['batsman'].unique()) >= 5
    )

def preprocess_data(df):
    """Optimized data preprocessing pipeline"""
    # Type conversions
    df = df.astype({
        'runs': 'int32',
        'totalBallNumber': 'int32',
        'shotAngle': 'float32',
        'is_boundary': 'bool'
    })
    
    # Data cleaning
    df = df.dropna(subset=['batsman', 'runs', 'totalBallNumber'])
    df = df[df['runs'] >= 0]
    
    # Angle normalization
    df['shotAngle'] = df['shotAngle'].apply(lambda x: x % 360 if pd.notnull(x) else 0)
    
    # Save to local cache
    df.to_pickle(LOCAL_CACHE)
    
    return df

def create_sample_data():
    """High-fidelity sample data generator"""
    np.random.seed(42)
    
    players = ['Smriti Mandhana', 'Harmanpreet Kaur', 'Beth Mooney', 
              'Alyssa Healy', 'Meg Lanning', 'Elyse Villani', 'Deepti Jones']
    
    shot_types = ['Drive', 'Pull', 'Cut', 'Sweep', 'Flick', 'Hook', 
                 'Reverse Sweep', 'Defensive', 'Cover Drive']
    
    connection_types = ['Middled', 'WellTimed', 'Undercontrol', 
                       'MisTimed', 'Missed', 'HitBody', 'TopEdge']
    
    data = {
        'batsman': np.random.choice(players, 1500),
        'battingShotTypeId': np.random.choice(shot_types, 1500),
        'battingConnectionId': np.random.choice(connection_types, 1500, p=[0.3, 0.25, 0.2, 0.1, 0.05, 0.05, 0.05]),
        'runs': np.random.choice([0, 1, 2, 3, 4, 6], 1500, p=[0.3, 0.25, 0.2, 0.1, 0.1, 0.05]),
        'totalBallNumber': np.random.randint(1, 101, 1500),
        'shotAngle': np.random.uniform(0, 360, 1500),
        'is_boundary': np.random.choice([True, False], 1500, p=[0.15, 0.85]),
        'matchDate': pd.date_range(start='2023-01-01', periods=1500, freq='D').date
    }
    
    df = pd.DataFrame(data)
    
    # Add realistic patterns
    df.loc[df['runs'] == 4, 'is_boundary'] = True
    df.loc[df['runs'] == 6, 'is_boundary'] = True
    
    # Create match phases
    max_balls = df['totalBallNumber'].max()
    df['match_phase'] = pd.cut(
        df['totalBallNumber'],
        bins=[0, 25, 75, max_balls+1],
        labels=['Powerplay', 'Middle', 'Death']
    )
    
    return df

def calculate_shot_intelligence_metrics(df):
    """Advanced metrics calculation with validation"""
    if not validate_dataframe(df):
        raise ValueError("Invalid dataframe structure for metrics calculation")
    
    df = df.copy()
    
    # Control Quality Score
    control_scores = {
        'Middled': 3, 'WellTimed': 3, 'Undercontrol': 3,
        'MisTimed': 2, 'Missed': 1, 'HitBody': 0.5,
        'TopEdge': 2, 'BatPad': 1, 'BottomEdge': 2,
        'Gloved': 2, 'HitHelmet': 0, 'HitPad': 0,
        'InsideEdge': 2, 'LeadingEdge': 2, 'Left': 3
    }
    
    df['control_quality_score'] = df['battingConnectionId'].map(control_scores).fillna(1.5)
    
    # Angle zones
    angle_bins = [0, 45, 90, 135, 180, 225, 270, 315, 360]
    angle_labels = ['Long Off', 'Cover', 'Point', 'Third Man', 
                   'Fine Leg', 'Square Leg', 'Mid Wicket', 'Long On']
    df['angle_zone'] = pd.cut(df['shotAngle'], bins=angle_bins, labels=angle_labels)
    
    # Control Score Calculation
    df['control_score'] = (
        df['control_quality_score'] * 33.33 +
        df['runs'] * 5 +
        df['is_boundary'].astype(int) * 20 +
        (df['angle_zone'].isin(['Cover', 'Mid Wicket', 'Long On', 'Long Off']).astype(int) * 10)
    ).clip(0, 100)
    
    # Risk-Reward Metrics
    df['true_risk_reward'] = np.where(
        df['control_score'] >= 50,
        df['runs'] * 1.3,
        df['runs'] * 0.7
    )
    
    return df

def create_shot_angle_heatmap(df, player_name):
    """Advanced 360° shot visualization"""
    player_data = df[df['batsman'] == player_name]
    if player_data.empty:
        return go.Figure()
    
    # Color mapping based on control score
    colorscale = [
        [0, '#ff0000'], [0.3, '#ff7f00'], [0.7, '#ffff00'], 
        [1.0, '#00ff00']
    ]
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=player_data['shotMagnitude'],
        theta=player_data['shotAngle'],
        mode='markers',
        marker=dict(
            size=player_data['runs'] * 3 + 8,
            color=player_data['control_score'],
            colorscale=colorscale,
            showscale=True,
            colorbar=dict(title="Control Score (0-100)")
        ),
        text=player_data.apply(
            lambda row: f"Shot: {row.battingShotTypeId}<br>" +
                       f"Runs: {row.runs}<br>" +
                       f" Control: {row.control_score:.1f}",
            axis=1
        ),
        hoverinfo='text'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(range=[0, 200], visible=True),
            angularaxis=dict(
                tickvals=[0, 45, 90, 135, 180, 225, 270, 315],
                ticktext=['Long Off', 'Cover', 'Point', 'Third Man', 
                         'Fine Leg', 'Square Leg', 'Mid Wicket', 'Long On']
            )
        ),
        title=f"{player_name} - Shot Intelligence Matrix",
        height=600
    )
    
    return fig

def main():
    st.set_page_config(
        page_title="Women's Cricket Analytics",
        page_icon="🏏",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.markdown('<h1 class="main-header">🏏 Women\'s Cricket Shot Intelligence Matrix</h1>', unsafe_allow_html=True)
    
    # Data loading with progress
    with st.spinner("🔄 Fetching and preparing data..."):
        try:
            df = load_data()
            df = calculate_shot_intelligence_metrics(df)
            
            if df.empty:
                st.error("⚠️ No valid data available after all fallbacks")
                return
                
        except Exception as e:
            st.error(f"❌ Critical error: {str(e)}")
            st.info("Showing sample data instead...")
            df = create_sample_data()
            df = calculate_shot_intelligence_metrics(df)
    
    # Data Quality Dashboard
    with st.expander("🔍 Data Quality Dashboard"):
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Balls", len(df))
        with col2:
            st.metric("Unique Batsmen", df['batsman'].nunique())
        with col3:
            st.metric("Boundary Shots", df['is_boundary'].sum())
        with col4:
            st.metric("Control Score Avg", f"{df['control_score'].mean():.1f}/100")
        
        st.write("📊 Data Summary:")
        st.dataframe(df.describe().T.style.background_gradient(cmap='RdYlGn_r'))
    
    # Player Selection
    st.sidebar.header("🎯 Player Analysis")
    available_players = df['batsman'].unique()
    selected_players = st.sidebar.multiselect(
        "Select Players",
        options=available_players,
        default=available_players[:2] if len(available_players) >= 2 else []
    )
    
    # Main Analysis Tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "🎯 Shot Placement", 
        "⚡ Control vs Aggression", 
        "📊 Match Phase", 
        "📈 Player Comparison"
    ])
    
    with tab1:
        st.subheader("360° Shot Placement Analysis")
        col1, col2 = st.columns([2, 1])
        
        with col1:
            if selected_players:
                for player in selected_players:
                    fig = create_shot_angle_heatmap(df, player)
                    st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("##### 🧠 Control Metrics Guide")
            st.markdown("""
            - **Green**: >80 Control Score
            - **Yellow**: 50-80 Control Score
            - **Red**: <50 Control Score
            - Marker size = Runs scored (1-6)
            """)
    
    with tab2:
        st.subheader("Control vs Aggression Matrix")
        shot_analysis = df.groupby('battingShotTypeId').agg({
            'control_score': 'mean',
            'runs': 'mean',
            'is_boundary': 'mean'
        }).reset_index()
        
        fig = px.scatter(
            shot_analysis,
            x='control_score',
            y='runs',
            size='is_boundary',
            color='control_score',
            title="Shot Type Performance Matrix",
            labels={
                'control_score': 'Control Score (0-100)',
                'runs': 'Average Runs',
                'is_boundary': 'Boundary Frequency'
            },
            color_continuous_scale='RdYlGn'
        )
        fig.add_hline(y=shot_analysis['runs'].mean(), line_dash="dash", line_color="gray")
        fig.add_vline(x=shot_analysis['control_score'].mean(), line_dash="dash", line_color="gray")
        st.plotly_chart(fig)
    
    with tab3:
        st.subheader("Match Phase Performance")
        phase_analysis = df.groupby('match_phase').agg({
            'runs': 'mean',
            'control_score': 'mean',
            'is_boundary': 'mean'
        }).reset_index()
        
        fig = px.bar(
            phase_analysis,
            x='match_phase',
            y=['runs', 'control_score'],
            title="Phase-Wise Performance",
            barmode='group',
            labels={'value': 'Score', 'variable': 'Metric'}
        )
        st.plotly_chart(fig)
    
    with tab4:
        st.subheader("Player Comparison Radar")
        if len(selected_players) >= 2:
            metrics = {
                'Control Score': df.groupby('batsman')['control_score'].mean(),
                'Boundary %': df.groupby('batsman')['is_boundary'].mean() * 100,
                'Runs/Shot': df.groupby('batsman')['runs'].mean(),
                'Shot Efficiency': df.groupby('batsman')['true_shot_efficiency'].mean(),
                'Match Impact': df.groupby('batsman')['true_risk_reward'].mean()
            }
            
            radar_df = pd.DataFrame(metrics).T
            fig = go.Figure()
            
            for i, player in enumerate(selected_players[:4]):
                fig.add_trace(go.Scatterpolar(
                    r=radar_df[player].values,
                    theta=radar_df.columns,
                    fill='toself',
                    name=player,
                    line_color=px.colors.qualitative.Plotly[i]
                ))
            
            fig.update_layout(
                polar=dict(radialaxis=dict(range=[0, 100])),
                title="Multi-Dimensional Player Comparison"
            )
            st.plotly_chart(fig)

if __name__ == "__main__":
    main()
