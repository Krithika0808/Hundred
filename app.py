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
DATA_CACHE_TTL = 3600  # 1 hour cache
REQUIRED_COLUMNS = {
    'batsman', 'runs', 'totalBallNumber', 'shotAngle',
    'battingShotTypeId', 'battingConnectionId', 'matchDate',
    'shotMagnitude'  # Critical for visualization
}

# Custom CSS for production
st.markdown("""
<style>
    /* Your existing CSS styles */
    .github-loader {
        color: #1f77b4;
        font-style: italic;
        margin-top: 1rem;
        opacity: 0.7;
    }
    /* Hide file upload elements */
    .stFileUploader > div > div > div > div {
        display: none !important;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data(ttl=DATA_CACHE_TTL)
def load_data():
    """GitHub-first data loading with production-grade error handling"""
    try:
        # 1. GitHub Priority Load
        print("🚀 Attempting GitHub data load...")
        response = requests.get(
            GITHUB_SOURCE,
            headers={
                'User-Agent': 'WomenCricketApp/1.0',
                'Accept': 'application/vnd.github.v3.raw'
            },
            timeout=10
        )
        if response.status_code == 200:
            print("✅ GitHub response received")
            df = pd.read_csv(response.content)
            
            # Validate and preprocess
            if validate_dataframe(df):
                print("✅ GitHub data validated")
                return preprocess_data(df)
            else:
                print("❌ GitHub data validation failed")
                raise ValueError("Invalid GitHub data structure")
                
    except Exception as e:
        print(f"❌ GitHub error: {str(e)}")
        st.warning("🔄 Fallback to sample data...")
        
    # 2. Sample Data Fallback
    print("Generating sample data...")
    return create_sample_data()

def validate_dataframe(df):
    """Comprehensive dataframe validation"""
    if df.empty:
        return False
    
    missing_cols = REQUIRED_COLUMNS - set(df.columns)
    if missing_cols:
        print(f"⚠️ Missing columns: {missing_cols}")
        return False
    
    # Type validation
    if df['runs'].dtype not in [np.int64, np.float64]:
        print("⚠️ Invalid runs column type")
        return False
    
    if df['shotAngle'].dtype not in [np.float64, np.int64]:
        print("⚠️ Invalid shotAngle type")
        return False
    
    return True

def preprocess_data(df):
    """Optimized data processing pipeline"""
    df = df.copy()
    
    # Type conversions
    df = df.astype({
        'runs': 'int32',
        'totalBallNumber': 'int32',
        'shotAngle': 'float32',
        'shotMagnitude': 'float32'
    })
    
    # Data cleaning
    df = df.dropna(subset=['batsman', 'runs', 'totalBallNumber'])
    df = df[df['runs'] >= 0]
    
    # Angle normalization
    df['shotAngle'] = df['shotAngle'].apply(lambda x: x % 360 if pd.notnull(x) else 0)
    
    # Control score calculation
    control_scores = {
        'Middled': 3, 'WellTimed': 3, 'Undercontrol': 3,
        'MisTimed': 2, 'Missed': 1, 'HitBody': 0.5,
        'TopEdge': 2, 'BatPad':1, 'BottomEdge':2,    
        'Gloved': 2, 'HitHelmet': 0, 'HitPad': 0,
        'InsideEdge': 2, 'LeadingEdge': 2, 'Left': 3
    }
    
    df['control_quality_score'] = df['battingConnectionId'].map(control_scores).fillna(1.5)
    
    # Control Score Calculation
    df['control_score'] = (
        df['control_quality_score'] * 33.33 +
        df['runs'] * 5 +
        df['is_boundary'].astype(int) * 20 +
        (df['angle_zone'].isin(['Cover', 'Mid Wicket', 'Long On', 'Long Off']).astype(int) * 10
    ).clip(0, 100)
    
    return df

def create_sample_data():
    """High-fidelity sample data generator"""
    np.random.seed(42)
    
    players = ['Smriti Mandhana', 'Harmanpreet Kaur', 'Beth Mooney', 
              'Alyssa Healy', 'Meg Lanning', 'Elyse Villani', 'Deepti Jones']
    
    shot_types = ['Drive', 'Pull', 'Cut', 'Sweep', 'Flick', 'Hook', 
                 'Reverse Sweep', 'Defensive', 'Cover Drive']
    
    data = {
        'batsman': np.random.choice(players, 1500),
        'battingShotTypeId': np.random.choice(shot_types, 1500),
        'battingConnectionId': np.random.choice(['Middled', 'WellTimed', 'Undercontrol', 'MisTimed', 'Missed'], 1500),
        'runs': np.random.choice([0, 1, 2, 3, 4, 6], 1500),
        'totalBallNumber': np.random.randint(1, 101, 1500),
        'shotAngle': np.random.uniform(0, 360, 1500),
        'shotMagnitude': np.random.uniform(50, 200, 1500),
        'is_boundary': np.random.choice([True, False], 1500),
        'matchDate': pd.date_range(start='2023-01-01', periods=1500, freq='D').date
    }
    
    df = pd.DataFrame(data)
    
    # Add realistic patterns
    df.loc[df['runs'] == 4, 'is_boundary'] = True
    df.loc[df['runs'] == 6, 'is_boundary'] = True
    
    return df

def create_shot_angle_heatmap(df, player_name):
    """Robust 360° visualization with error handling"""
    player_data = df[df['batsman'] == player_name]
    if player_data.empty:
        return go.Figure()
    
    # Column fallbacks
    if 'shotMagnitude' not in player_data.columns:
        player_data['shotMagnitude'] = player_data['shotAngle'] / 2 + 80
        st.warning("⚠️ Using angle as magnitude proxy")
    
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
                tickmode='array',
                tickvals=[0, 45, 90, 135, 180, 225, 270, 315],
                ticktext=['Long Off', 'Cover', 'Point', 'Third Man', 
                         'Fine Leg', 'Square Leg', 'Mid Wicket', 'Long On']
            )
        ),
        title=f"{player_name} - Shot Intelligence Matrix",
        height=600
    )
    
    return fig

# Keep your existing visualization functions here
# (control vs aggression, match phase, player comparison, etc.)

def main():
    st.set_page_config(
        page_title="Women's Cricket Analytics",
        page_icon="🏏",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.markdown('<h1 class="main-header">🏏 Women's Cricket Shot Intelligence Matrix</h1>', unsafe_allow_html=True)
    
    # Load data with enhanced diagnostics
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
            missing_angles = df['shotAngle'].isna().sum()
            st.metric("Missing Shot Angles", missing_angles)
        with col2:
            valid_shots = df['battingShotTypeId'].notna().sum() if 'battingShotTypeId' in df.columns else 0
            st.metric("Valid Shot Types", valid_shots)
        with col3:
            boundary_shots = df['is_boundary'].sum()
            st.metric("Boundary Shots Detected", boundary_shots)
        with col4:
            control_data = df['battingConnectionId'].notna().sum()
            st.metric("Control Data Available", control_data)
        
        st.write("📊 Data Summary:")
        st.dataframe(df.describe().T.style.background_gradient(cmap='RdYlGn_r'))
    
    # Player Selection
    st.sidebar.header("🎯 Player Analysis")
    available_players = df['batsman'].unique()
    if len(available_players) == 0:
        st.error("❌ No batsman data found in the dataset")
        return
    
    selected_players = st.sidebar.multiselect(
        "Select Players",
        options=available_players,
        default=available_players[:2] if len(available_players) >= 2 else []
    
    # Main Dashboard Tabs
    tab1, tab2, tab3 = st.tabs([
        "🎯 Shot Placement", 
        "⚡ Control vs Aggression", 
        "📊 Match Phase Analysis"
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
        st.plotly.chart(fig)

if __name__ == "__main__":
    main()
