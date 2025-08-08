import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, date
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Hundred Women's Cricket Analysis",
    page_icon="🏏",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #FF6B35;
        text-align: center;
        margin-bottom: 2rem;
    }
    .kpi-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 0.5rem;
        text-align: center;
        margin: 0.5rem 0;
    }
    .tab-header {
        font-size: 1.5rem;
        font-weight: bold;
        color: #2E4057;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    """Load cricket data from GitHub repository"""
    try:
        # GitHub raw URL for the CSV file
        url = "https://raw.githubusercontent.com/Krithika0808/Hundred/main/Hundred.csv"
        df = pd.read_csv(url)
        
        # Debug: Show original data info
        st.write(f"**Debug Info:** Loaded {len(df)} rows, {len(df.columns)} columns")
        
        # Data cleaning and preprocessing
        # Convert date columns with better error handling
        if 'matchDate' in df.columns:
            df['matchDate'] = pd.to_datetime(df['matchDate'], errors='coerce')
        if 'ballDateTime' in df.columns:
            df['ballDateTime'] = pd.to_datetime(df['ballDateTime'], errors='coerce')
        
        # Convert numeric columns with validation
        numeric_cols = ['runs', 'shotAngle', 'shotMagnitude', 'runrate', 'totalBallNumber', 
                       'ballNumber', 'overNumber', 'totalInningRuns', 'totalInningWickets']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                # Debug: Check for conversion issues
                null_count = df[col].isna().sum()
                if null_count > 0:
                    st.warning(f"**Debug:** {null_count} values in '{col}' couldn't be converted to numeric")
        
        # Convert boolean columns with validation
        bool_cols = ['isWicket', 'isBoundry', 'isWide', 'isNoBall', 'isPowerPlay']
        for col in bool_cols:
            if col in df.columns:
                try:
                    df[col] = df[col].astype(bool)
                except Exception as e:
                    st.warning(f"**Debug:** Error converting '{col}' to boolean: {e}")
                    df[col] = False  # Default to False
        
        # Fill missing values more safely
        if 'batsman' in df.columns:
            df['batsman'] = df['batsman'].fillna('Unknown')
        if 'bowler' in df.columns:
            df['bowler'] = df['bowler'].fillna('Unknown')
        if 'runs' in df.columns:
            df['runs'] = df['runs'].fillna(0)
        
        # Debug: Show cleaned data info
        st.write(f"**Debug:** After cleaning: {len(df)} rows with valid data")
        
        return df
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        st.error(f"**Debug:** Full error details: {type(e).__name__}: {e}")
        return pd.DataFrame()

def create_match_phase(row):
    """Determine match phase based on ball number"""
    try:
        if pd.isna(row.get('totalBallNumber')):
            return 'Unknown'
        
        ball_num = row['totalBallNumber']
        if ball_num <= 25:  # First 25 balls (Powerplay in T20)
            return 'Powerplay'
        elif ball_num <= 75:  # Middle overs
            return 'Middle'
        else:  # Death overs
            return 'Death'
    except Exception as e:
        st.warning(f"**Debug:** Error in create_match_phase: {e}")
        return 'Unknown'

def safe_groupby_operation(df, group_col, agg_dict, operation_name="operation"):
    """Safely perform groupby operations with error handling"""
    try:
        if group_col not in df.columns:
            st.error(f"**Debug:** Column '{group_col}' not found for {operation_name}")
            return pd.DataFrame()
        
        # Remove rows where group column is null
        df_clean = df[df[group_col].notna()]
        
        if df_clean.empty:
            st.warning(f"**Debug:** No valid data for {operation_name}")
            return pd.DataFrame()
        
        result = df_clean.groupby(group_col).agg(agg_dict)
        return result
    except Exception as e:
        st.error(f"**Debug:** Error in {operation_name}: {e}")
        return pd.DataFrame()

def main():
    """Main application function"""
    
    # Add debug mode toggle
    debug_mode = st.sidebar.checkbox("Enable Debug Mode", value=False)
    
    # Load data
    with st.spinner("Loading Hundred Women's Cricket data..."):
        df = load_data()
    
    if df.empty:
        st.error("Failed to load data. Please check the data source.")
        return
    
    # Show debug info if enabled
    if debug_mode:
        st.sidebar.write("**Debug: Data Info**")
        st.sidebar.write(f"Shape: {df.shape}")
        st.sidebar.write("Columns:", list(df.columns))
        
        with st.expander("View Raw Data Sample"):
            st.write(df.head())
    
    # Add match phase column safely
    try:
        df['match_phase'] = df.apply(create_match_phase, axis=1)
    except Exception as e:
        st.error(f"**Debug:** Error creating match_phase: {e}")
        df['match_phase'] = 'Unknown'
    
    # Main title
    st.markdown("<h1 class='main-header'>🏏 Hundred Women's Cricket Analysis Dashboard</h1>", 
                unsafe_allow_html=True)
    
    # Sidebar filters with better error handling
    st.sidebar.header("🔧 Filters")
    
    # Date filter with validation
    df_filtered = df.copy()
    
    if 'matchDate' in df.columns and not df['matchDate'].isna().all():
        try:
            min_date = df['matchDate'].min().date()
            max_date = df['matchDate'].max().date()
            
            date_range = st.sidebar.date_input(
                "Select Date Range",
                value=(min_date, max_date),
                min_value=min_date,
                max_value=max_date
            )
            
            # Apply date filter
            if len(date_range) == 2:
                start_date, end_date = date_range
                df_filtered = df_filtered[
                    (df_filtered['matchDate'].dt.date >= start_date) & 
                    (df_filtered['matchDate'].dt.date <= end_date)
                ]
        except Exception as e:
            st.sidebar.warning(f"Date filter error: {e}")
    
    # Team filter with validation
    if 'battingTeam' in df_filtered.columns:
        try:
            teams = sorted([team for team in df_filtered['battingTeam'].unique() 
                          if pd.notna(team) and team != ''])
            
            if teams:
                selected_teams = st.sidebar.multiselect(
                    "Select Teams",
                    options=teams,
                    default=teams[:5] if len(teams) > 5 else teams  # Limit default selection
                )
                
                # Apply team filter
                if selected_teams:
                    df_filtered = df_filtered[df_filtered['battingTeam'].isin(selected_teams)]
            else:
                st.sidebar.warning("No valid teams found in data")
        except Exception as e:
            st.sidebar.error(f"Team filter error: {e}")
    
    st.sidebar.markdown(f"**Data Points:** {len(df_filtered):,}")
    
    # Create tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🏠 Home", 
        "🏏 Batting Analysis", 
        "🎳 Bowling Analysis", 
        "🎯 Shot Analysis", 
        "📊 Match Explorer"
    ])
    
    # HOME PAGE
    with tab1:
        st.markdown("<div class='tab-header'>Tournament Overview</div>", unsafe_allow_html=True)
        
        # KPI Metrics with error handling
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            try:
                if 'fixtureId' in df_filtered.columns:
                    total_matches = df_filtered['fixtureId'].nunique()
                else:
                    total_matches = 0
                st.metric("Total Matches", f"{total_matches:,}")
            except Exception as e:
                st.metric("Total Matches", "Error")
                if debug_mode:
                    st.error(f"Match count error: {e}")
        
        with col2:
            try:
                total_runs = int(df_filtered['runs'].sum()) if 'runs' in df_filtered.columns else 0
                st.metric("Total Runs", f"{total_runs:,}")
            except Exception as e:
                st.metric("Total Runs", "Error")
                if debug_mode:
                    st.error(f"Total runs error: {e}")
        
        with col3:
            try:
                total_wickets = int(df_filtered['isWicket'].sum()) if 'isWicket' in df_filtered.columns else 0
                st.metric("Total Wickets", f"{total_wickets:,}")
            except Exception as e:
                st.metric("Total Wickets", "Error")
                if debug_mode:
                    st.error(f"Total wickets error: {e}")
        
        with col4:
            try:
                avg_run_rate = df_filtered['runrate'].mean() if 'runrate' in df_filtered.columns else 0
                st.metric("Average Run Rate", f"{avg_run_rate:.2f}")
            except Exception as e:
                st.metric("Average Run Rate", "Error")
                if debug_mode:
                    st.error(f"Run rate error: {e}")
        
        # Tournament insights with better error handling
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📈 Runs Trend by Date")
            try:
                if 'matchDate' in df_filtered.columns and not df_filtered['matchDate'].isna().all():
                    daily_runs = df_filtered.groupby(df_filtered['matchDate'].dt.date)['runs'].sum().reset_index()
                    if not daily_runs.empty:
                        fig_trend = px.line(
                            daily_runs, 
                            x='matchDate', 
                            y='runs',
                            title='Daily Runs Scored',
                            labels={'runs': 'Total Runs', 'matchDate': 'Date'}
                        )
                        st.plotly_chart(fig_trend, use_container_width=True)
                    else:
                        st.info("No data available for trend analysis")
                else:
                    st.info("Date information not available for trend analysis")
            except Exception as e:
                st.error("Error creating runs trend chart")
                if debug_mode:
                    st.error(f"Trend chart error: {e}")
        
        with col2:
            st.subheader("🏟️ Runs by Team")
            try:
                if 'battingTeam' in df_filtered.columns:
                    team_runs = df_filtered.groupby('battingTeam')['runs'].sum().sort_values(ascending=True)
                    if not team_runs.empty:
                        fig_team = px.bar(
                            x=team_runs.values,
                            y=team_runs.index,
                            orientation='h',
                            title='Total Runs by Team',
                            labels={'x': 'Total Runs', 'y': 'Team'}
                        )
                        st.plotly_chart(fig_team, use_container_width=True)
                    else:
                        st.info("No team data available")
                else:
                    st.info("Team information not available")
            except Exception as e:
                st.error("Error creating team runs chart")
                if debug_mode:
                    st.error(f"Team chart error: {e}")
    
    # BATTING ANALYSIS
    with tab2:
        st.markdown("<div class='tab-header'>🏏 Batting Performance Analysis</div>", unsafe_allow_html=True)
        
        try:
            # Check if required columns exist
            required_cols = ['batsman', 'runs', 'isBoundry']
            missing_cols = [col for col in required_cols if col not in df_filtered.columns]
            
            if missing_cols:
                st.error(f"Missing required columns for batting analysis: {missing_cols}")
            else:
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("🏆 Top Run Scorers")
                    
                    # Calculate batting statistics safely
                    batting_stats = safe_groupby_operation(
                        df_filtered, 
                        'batsman', 
                        {
                            'runs': ['sum', 'count', 'mean'],
                            'isBoundry': 'sum'
                        },
                        "batting statistics"
                    )
                    
                    if not batting_stats.empty:
                        batting_stats.columns = ['Total_Runs', 'Balls_Faced', 'Average', 'Boundaries']
                        batting_stats['Strike_Rate'] = (batting_stats['Total_Runs'] / batting_stats['Balls_Faced'] * 100).round(2)
                        batting_stats = batting_stats.sort_values('Total_Runs', ascending=False).head(10)
                        
                        st.dataframe(batting_stats, use_container_width=True)
                    else:
                        st.info("No batting statistics available")
                
                with col2:
                    st.subheader("📊 Strike Rate vs Average")
                    if 'batting_stats' in locals() and len(batting_stats) > 0:
                        try:
                            fig_batting = px.scatter(
                                batting_stats.reset_index(),
                                x='Average',
                                y='Strike_Rate',
                                size='Total_Runs',
                                hover_name='batsman',
                                title='Batting Performance Matrix',
                                labels={'Average': 'Batting Average', 'Strike_Rate': 'Strike Rate'}
                            )
                            st.plotly_chart(fig_batting, use_container_width=True)
                        except Exception as e:
                            st.error("Error creating batting scatter plot")
                            if debug_mode:
                                st.error(f"Scatter plot error: {e}")
                    else:
                        st.info("No data available for performance matrix")
        
        except Exception as e:
            st.error("Error in batting analysis section")
            if debug_mode:
                st.error(f"Batting analysis error: {e}")
    
    # BOWLING ANALYSIS (simplified for brevity)
    with tab3:
        st.markdown("<div class='tab-header'>🎳 Bowling Performance Analysis</div>", unsafe_allow_html=True)
        
        try:
            if all(col in df_filtered.columns for col in ['bowler', 'isWicket', 'runs']):
                bowling_stats = safe_groupby_operation(
                    df_filtered,
                    'bowler',
                    {
                        'isWicket': 'sum',
                        'runs': 'sum',
                        'batsman': 'count'
                    },
                    "bowling statistics"
                )
                
                if not bowling_stats.empty:
                    bowling_stats.columns = ['Wickets', 'Runs_Conceded', 'Balls_Bowled']
                    bowling_stats = bowling_stats[bowling_stats['Balls_Bowled'] >= 10]
                    bowling_stats['Economy_Rate'] = (bowling_stats['Runs_Conceded'] / bowling_stats['Balls_Bowled'] * 6).round(2)
                    
                    st.subheader("🏆 Top Wicket Takers")
                    top_bowlers = bowling_stats.sort_values('Wickets', ascending=False).head(10)
                    st.dataframe(top_bowlers, use_container_width=True)
                else:
                    st.info("No bowling statistics available")
            else:
                st.error("Missing required columns for bowling analysis")
        except Exception as e:
            st.error("Error in bowling analysis")
            if debug_mode:
                st.error(f"Bowling analysis error: {e}")
    
    # SHOT ANALYSIS (simplified)
    with tab4:
        st.markdown("<div class='tab-header'>🎯 Shot Analysis</div>", unsafe_allow_html=True)
        
        try:
            if 'batsman' in df_filtered.columns:
                available_batsmen = [b for b in df_filtered['batsman'].unique() 
                                   if pd.notna(b) and b != 'Unknown' and b != '']
                
                if available_batsmen:
                    selected_batsman = st.selectbox(
                        "Select Batsman for Shot Analysis",
                        options=sorted(available_batsmen)
                    )
                    
                    player_data = df_filtered[df_filtered['batsman'] == selected_batsman]
                    
                    if not player_data.empty:
                        st.write(f"**Selected:** {selected_batsman}")
                        st.write(f"**Total balls faced:** {len(player_data)}")
                        st.write(f"**Total runs scored:** {player_data['runs'].sum()}")
                    else:
                        st.info("No data available for selected batsman")
                else:
                    st.info("No batsmen available for analysis")
            else:
                st.error("Batsman column not found in data")
        except Exception as e:
            st.error("Error in shot analysis")
            if debug_mode:
                st.error(f"Shot analysis error: {e}")
    
    # MATCH EXPLORER (simplified)
    with tab5:
        st.markdown("<div class='tab-header'>📊 Match Explorer</div>", unsafe_allow_html=True)
        
        try:
            if 'fixtureId' in df_filtered.columns:
                available_matches = sorted(df_filtered['fixtureId'].unique())
                
                if available_matches:
                    selected_match = st.selectbox(
                        "Select Match",
                        options=available_matches
                    )
                    
                    match_data = df_filtered[df_filtered['fixtureId'] == selected_match]
                    
                    if not match_data.empty:
                        st.subheader(f"📋 Match Summary - Fixture {selected_match}")
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Total Balls", len(match_data))
                        with col2:
                            st.metric("Total Runs", int(match_data['runs'].sum()))
                        with col3:
                            if 'isWicket' in match_data.columns:
                                st.metric("Total Wickets", int(match_data['isWicket'].sum()))
                            else:
                                st.metric("Total Wickets", "N/A")
                    else:
                        st.info("No data available for selected match")
                else:
                    st.info("No matches available")
            else:
                st.error("Match ID column not found in data")
        except Exception as e:
            st.error("Error in match explorer")
            if debug_mode:
                st.error(f"Match explorer error: {e}")
    
    # Footer
    st.markdown("---")
    st.markdown("**🏏 Hundred Women's Cricket Analysis Dashboard** | Data Source: The Hundred Tournament")

if __name__ == "__main__":
    main()
