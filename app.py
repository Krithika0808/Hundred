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

# Configuration
GITHUB_SOURCE = "https://raw.githubusercontent.com/Krithika0808/Hundred/main/Hundred.csv"
LOCAL_FALL = "Hundred.csv"  # Local fallback file
DATA_CACHE_TTL = 3600  # 1 hour cache

@st.cache_data(ttl=DATA_CACHE_TTL)
def load_data():
    """GitHub-first data loading strategy with automatic fallbacks"""
    try:
        # 1. Try GitHub first
        @st.cache_data(ttl=86400)  # 24-hour GitHub cache
        def load_github_data():
            response = requests.get(
                GITHUB_SOURCE,
                headers={
                    'User-Agent': 'WomenCricketApp/1.0',
                    'Accept': 'application/vnd.github.v3.raw'
                },
                timeout=10
            )
            return pd.read_csv(response.content)
            
        df = load_github_data()
        st.success("✅ Data loaded from GitHub repository")
        return df
        
    except Exception as e:
        st.warning(f"🔄 Fallback to local file ({str(e)})")
        
        # 2. Try local file in same directory
        if os.path.exists(LOCAL_SOURCE):
            df = pd.read_csv(LOCAL_SOURCE)
            st.success(f"✅ Fallback to local file: {LOCAL_SOURCE}")
            return df
        
        # 3. Fallback to sample data
        st.info("📊 Using sample data for demonstration")
        return create_sample_data()

# Keep your existing data processing functions here
# (calculate_shot_intelligence_metrics, create_shot_angle_heatmap, etc.)

def main():
    st.set_page_config(
        page_title="Women's Cricket Analytics",
        page_icon="🏏",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Remove file upload components from UI
    st.markdown("""
    <style>
        /* Hide file upload elements */
        .stFileUploader > div > div > div > div {
            display: none !important;
        }
    </style>
    """, unsafe_allow_html=True)
    
    # Load data automatically
    with st.spinner("🔄 Fetching latest dataset..."):
        df = load_data()
        
        if df.empty:
            st.error("⚠️ No valid data available")
            return
    
    # Process data
    df = calculate_shot_intelligence_metrics(df)
    
    # Data quality check
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
    
    # Main dashboard tabs (keep your existing tabs structure)
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "🎯 Shot Placement", 
        "⚡ Control vs Aggression", 
        "📊 Match Phase Analysis",
        "🏆 Player Intelligence",
        "📈 Player Comparison",
        "🔍 Advanced Analytics"
    ])
    
    # Keep your existing tab implementations...
    # (Your tab1 to tab6 code remains unchanged)

# Keep your existing helper functions here
# (calculate_shot_intelligence_metrics, create_sample_data, etc.)
