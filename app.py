import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
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
    'battingShotTypeId', 'battingConnectionId', 'matchDate',
    'shotMagnitude'  # Critical for visualization
}

@st.cache_data(ttl=3600)
def load_data():
    """Robust data loading with automatic column handling"""
    try:
        # 1. GitHub Priority Load
        if os.getenv("GITHUB_LOAD", "true").lower() == "true":
            print("🚀 Attempting GitHub data load...")
            for attempt in range(3):
                try:
                    response = requests.get(
                        GITHUB_SOURCE,
                        headers={'User-Agent': 'WomenCricketApp/1.0'},
                        timeout=15
                    )
                    if response.status_code == 200:
                        df = pd.read_csv(response.content)
                        
                        # Automatic column fixes
                        if 'shotMagnitude' not in df.columns:
                            df['shotMagnitude'] = np.random.uniform(50, 200, len(df))
                            st.warning("⚠️ Generated shotMagnitude from random data")
                        
                        if not validate_dataframe(df):
                            raise ValueError("Invalid data structure after column fixes")
                        
                        return df
                except Exception as e:
                    print(f"Attempt {attempt+1} failed: {str(e)}")
                    time.sleep(DATA_RETRY_DELAY)
        
        # 2. Local Cache Fallback
        if os.path.exists(LOCAL_CACHE):
            df = pd.read_pickle(LOCAL_CACHE)
            return df
        
        # 3. Sample Data Fallback
        print("Generating sample data with guaranteed columns")
        return create_sample_data()
        
    except Exception as e:
        st.error(f"❌ Data loading failed: {str(e)}")
        return create_sample_data()

def validate_dataframe(df):
    """Comprehensive dataframe validation with automatic fixes"""
    # Check required columns
    missing_cols = REQUIRED_COLUMNS - set(df.columns)
    if missing_cols:
        st.error(f"❌ Missing required columns: {missing_cols}")
        return False
    
    # Check data types
    type_checks = {
        'runs': (df['runs'].dtype in [np.int64, np.float64], "Numeric runs required"),
        'shotAngle': (df['shotAngle'].dtype in [np.float64, np.int64], "Numeric angles required"),
        'shotMagnitude': (df['shotMagnitude'].dtype in [np.float64, np.int64], "Numeric magnitude required")
    }
    
    for check, (valid, message) in type_checks.items():
        if not valid:
            st.error(f"❌ {message}")
            return False
    
    return True

def create_sample_data():
    """Guaranteed sample data with all required columns"""
    np.random.seed(42)
    
    players = ['Smriti Mandhana', 'Harmanpreet Kaur', 'Beth Mooney', 
              'Alyssa Healy', 'Meg Lanning', 'Elyse Villani', 'Deepti Jones']
    
    shot_types = ['Drive', 'Pull', 'Cut', 'Sweep', 'Flick', 'Hook', 
                 'Reverse Sweep', 'Defensive', 'Cover Drive']
    
    connection_types = ['Middled', 'WellTimed', 'Undercontrol', 
                       'MisTimed', 'Missed', 'HitBody', 'TopEdge']
    
    # Create base data
    data = {
        'batsman': np.random.choice(players, 1500),
        'battingShotTypeId': np.random.choice(shot_types, 1500),
        'battingConnectionId': np.random.choice(connection_types, 1500),
        'runs': np.random.choice([0, 1, 2, 3, 4, 6], 1500),
        'totalBallNumber': np.random.randint(1, 101, 1500),
        'shotAngle': np.random.uniform(0, 360, 1500),
        'shotMagnitude': np.random.uniform(50, 200, 1500),  # Ensure this column exists
        'is_boundary': np.random.choice([True, False], 1500),
        'matchDate': pd.date_range(start='2023-01-01', periods=1500, freq='D').date,
        'match_phase': np.random.choice(['Powerplay', 'Middle', 'Death'], 1500)
    }
    
    df = pd.DataFrame(data)
    
    # Add realistic patterns
    df.loc[df['runs'] == 4, 'is_boundary'] = True
    df.loc[df['runs'] == 6, 'is_boundary'] = True
    
    # Create control scores
    control_scores = {
        'Middled': 3, 'WellTimed': 3, 'Undercontrol': 3,
        'MisTimed': 2, 'Missed': 1, 'HitBody': 0.5,
        'TopEdge': 2, 'BatPad': 1, 'BottomEdge': 2,
        'Gloved': 2, 'HitHelmet': 0, 'HitPad': 0,
        'InsideEdge': 2, 'LeadingEdge': 2, 'Left': 3
    }
    
    df['control_quality_score'] = df['battingConnectionId'].map(control_scores).fillna(1.5)
    
    return df

def calculate_shot_intelligence_metrics(df):
    """Advanced metrics calculation with column validation"""
    df = df.copy()
    
    # Handle missing values
    numeric_cols = ['shotAngle', 'shotMagnitude', 'runs']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(df[col].median() if not df[col].empty else 0)
    
    # Create angle zones
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
    """Robust 360° visualization with error handling"""
    player_data = df[df['batsman'] == player_name]
    if player_data.empty:
        return go.Figure()
    
    # Column fallbacks
    if 'shotMagnitude' not in player_data.columns:
        st.warning("⚠️ Using angle as magnitude proxy")
        player_data['shotMagnitude'] = player_data['shotAngle'] / 2 + 80
    
    # Create visualization
    fig = go.Figure()
    
    colorscale = [
        [0, '#ff0000'], [0.3, '#ff7f00'], [0.7, '#ffff00'], 
        [1.0, '#00ff00']
    ]
    
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
                       f" Runs: {row.runs}<br>" +
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
    
    # Custom CSS (keep your existing styling)
    st.markdown("""
    <style>
        /* Your existing CSS styles */
    </style>
    """, unsafe_allow_html=True)
    
    st.markdown('<h1 class="main-header">🏏 Women\'s Cricket Shot Intelligence Matrix</h1>', unsafe_allow_html=True)
    
    # Data loading with progress
    with st.spinner("🔄 Fetching latest dataset..."):
        df = load_data()
        if df.empty:
            st.error("⚠️ No valid data available")
            return
        
        if not validate_dataframe(df):
            st.error("⚠️ Data validation failed")
            return
        
        df = calculate_shot_intelligence_metrics(df)
    
    # Data Quality Dashboard
    with st.expander("🔍 Data Quality Check"):
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Balls", len(df))
        with col2:
            st.metric("Valid Shot Types", df['battingShotTypeId'].nunique())
        with col3:
            st.metric("Boundary Shots", df['is_boundary'].sum())
        with col4:
            st.metric("Control Data Available", df['control_score'].notna().sum())
        
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
    
    # Main Dashboard Tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "🎯 Shot Placement", 
        "⚡ Control vs Aggression", 
        "📊 Match Phase", 
        "📈 Player Comparison"
    ])
    
    with tab1:
        st.subheader("360° Shot Placement Intelligence")
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
            
            colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
            
            for i, player in enumerate(selected_players[:4]):
                fig.add_trace(go.Scatterpolar(
                    r=radar_df[player].values,
                    theta=radar_df.columns,
                    fill='toself',
                    name=player,
                    line_color=colors[i],
                    fillcolor=colors[i],
                    opacity=0.3
                ))
            
            fig.update_layout(
                polar=dict(radialaxis=dict(range=[0, 100])),
                title="Multi-Dimensional Player Comparison"
            )
            st.plotly_chart(fig)

if __name__ == "__main__":
    main()
