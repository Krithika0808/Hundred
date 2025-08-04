import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime
import os
import requests
import timeit

# Security Configuration
GITHUB_DATA_URL = "https://raw.githubusercontent.com/Krithika0808/Hundred/main/Hundred.csv"
DATA_CACHE_TTL = 3600  # 1 hour cache
MAX_FILE_SIZE = 200 * 1024 * 1024  # 200MB limit

# Custom CSS
st.markdown("""
<style>
    .file-path {display: none !important;}
    .github-loader {
        color: #1f77b4;
        font-style: italic;
        margin-top: 1rem;
        opacity: 0.7;
    }
    .stFileUploader > div > div > div > div {
        visibility: hidden !important;
    }
    .stTextInput {display: none !important;}
    a[href^="https://github.com"] {
        color: #1f77b4 !important;
        text-decoration: none !important;
    }
</style>
""", unsafe_allow_html=True)

# Data Loading
@st.cache_data(ttl=DATA_CACHE_TTL)
def load_data(uploaded_file=None):
    """Secure data loading: Upload > GitHub > Sample"""
    try:
        if uploaded_file:
            if uploaded_file.size > MAX_FILE_SIZE:
                raise ValueError("File exceeds 200MB limit")
            return pd.read_csv(uploaded_file)
        
        @st.cache_data(ttl=86400)  # 24-hour cache
        def load_github_data():
            response = requests.get(
                GITHUB_DATA_URL,
                headers={
                    'User-Agent': 'WomenCricketApp/1.0',
                    'Accept': 'application/vnd.github.v3.raw'
                },
                timeout=10
            )
            return pd.read_csv(response.content)
            
        return load_github_data()
        
    except Exception as e:
        st.warning("🔄 Fallback to sample data (GitHub unavailable)")
        return create_sample_data()

# Keep your existing data processing functions here
# (calculate_shot_intelligence_metrics, create_shot_angle_heatmap, etc.)

# Modified Main Function
def main():
    st.set_page_config(
        page_title="Women's Cricket Analytics",
        page_icon="🏏",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Security Notice
    st.markdown("""
    <div style='text-align: center; margin: 1rem 0;'>
        <h1>🏏 Women's Cricket Intelligence</h1>
        <p style='color: #666;'>Powered by GitHub-hosted dataset</p>
    </div>
    """, unsafe_allow_html=True)
    
    # File upload interface
    st.sidebar.header("📤 Secure Data Loading")
    uploaded_file = st.sidebar.file_uploader(
        "Upload Custom CSV (200MB max)", 
        type=['csv'],
        help="Upload your own dataset or use GitHub version"
    )
    
    # Load data
    with st.spinner("🔄 Fetching latest dataset..."):
        df = load_data(uploaded_file)
        
        if df.empty:
            st.error("⚠ No valid data available")
            return
    
    df = calculate_shot_intelligence_metrics(df)
    
    # Data Quality Check
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
    
    # Sidebar filters (keep your existing filters)
    st.sidebar.header("🎯 Analysis Filters")
    available_players = df['batsman'].unique() if 'batsman' in df.columns else []
    if len(available_players) == 0:
        st.error("❌ No batsman data found in the dataset")
        return
    
    selected_players = st.sidebar.multiselect(
        "Select Players",
        options=available_players,
        default=list(available_players)[:min(5, len(available_players))]
    )
    # Shot types filter
    available_shots = df['battingShotTypeId'].unique() if 'battingShotTypeId' in df.columns else []
    if len(available_shots) > 0:
        selected_shots = st.sidebar.multiselect(
            "Select Shot Types",
            options=available_shots,
            default=list(available_shots)
        )
    else:
        selected_shots = []
    
    # Ball range filter
    if 'totalBallNumber' in df.columns and df['totalBallNumber'].notna().any():
        max_balls = int(df['totalBallNumber'].max())
        min_balls = int(df['totalBallNumber'].min())
        ball_range = st.sidebar.slider(
            "Ball Range",
            min_value=min_balls,
            max_value=max_balls,
            value=(min_balls, max_balls)
        )
    else:
        ball_range = (1, 100)
    
    # Apply filters
    filtered_df = df[df['batsman'].isin(selected_players)]
    
    if selected_shots and 'battingShotTypeId' in df.columns:
        filtered_df = filtered_df[filtered_df['battingShotTypeId'].isin(selected_shots)]
    
    if 'totalBallNumber' in df.columns:
        filtered_df = filtered_df[filtered_df['totalBallNumber'].between(ball_range[0], ball_range[1])]
    
    if filtered_df.empty:
        st.warning("⚠️ No data matches your current filters")
        return
    
    # Dashboard Tabs (keep your existing tabs)
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "🎯 Shot Placement", 
        "⚡ Control vs Aggression", 
        "📊 Match Phase Analysis",
        "🏆 Player Intelligence",
        "📈 Player Comparison",
        "🔍 Advanced Analytics"
    ])
    
    with tab1:
        st.subheader("360° Shot Placement Intelligence")
        col1, col2 = st.columns([2, 1])
        
        with col1:
            selected_player = st.selectbox("Select Player", selected_players)
            if selected_player:
                fig = create_shot_angle_heatmap(filtered_df, selected_player)
                st.plotly_chart(fig, use_container_width=True)
        with col2:
            st.markdown("##### 🧠 Interpretation Guide")
            st.markdown("""
                - **Green**: Control Score 80-100
                - **Orange**: Control Score 50-79
                - **Red**: Control Score 0-49
                - Marker size = Runs scored (1-6)
                - Hover for detailed stats
            """)
    
    # Keep your other tabs exactly as they are
    # (tab2 to tab6 code remains unchanged)
    
    # Example of modified tab2 content
    with tab2:
        st.subheader("Control vs Aggression Matrix")
        fig = create_control_vs_aggression_chart(filtered_df)
        st.plotly_chart(fig, use_container_width=True)
    
if __name__ == "__main__":
    main()

