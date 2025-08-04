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

# Configuration
DATA_SOURCES = {
    "local": "Hundred.csv",  # Local file path
    "github": "https://raw.githubusercontent.com/Krithika0808/Hundred/main/Hundred.csv"
}
DATA_CACHE_TTL = 3600  # 1 hour cache
MAX_FILE_SIZE = 200 * 1024 * 1024  # 200MB limit

# Custom CSS
st.markdown("""
<style>
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
    """Secure data loading: Local > GitHub > Sample"""
    try:
        # 1. Try local file first
        if os.path.exists(DATA_SOURCES["local"]):
            return pd.read_csv(DATA_SOURCES["local"])
            
        # 2. Try GitHub if local file not found
        @st.cache_data(ttl=86400)  # 24-hour cache
        def load_github_data():
            response = requests.get(
                DATA_SOURCES["github"],
                headers={
                    'User-Agent': 'WomenCricketApp/1.0',
                    'Accept': 'application/vnd.github.v3.raw'
                },
                timeout=10
            )
            return pd.read_csv(response.content)
            
        return load_github_data()
        
    except Exception as e:
        st.warning(f"🔄 Fallback to sample data ({str(e)})")
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
        <p style='color: #666;'>Loading from local file first, GitHub as backup</p>
    </div>
    """, unsafe_allow_html=True)
    
    # File upload interface
    st.sidebar.header("📤 Data Loading Options")
    uploaded_file = st.sidebar.file_uploader(
        "Upload Custom CSV (200MB max)", 
        type=['csv'],
        help="Upload your own dataset or use local/GitHub version"
    )
    
    # Load data
    with st.spinner("🔄 Loading dataset..."):
        df = load_data(uploaded_file)
        
        if df.empty:
            st.error("⚠️ No valid data available")
            return
    
    # Rest of your existing code remains unchanged...
    # (filters, tabs, visualizations, etc.)

if __name__ == "__main__":
    main()
