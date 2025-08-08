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
        
        # Data cleaning and preprocessing
        # Convert date columns
        df['matchDate'] = pd.to_datetime(df['matchDate'], errors='coerce')
        df['ballDateTime'] = pd.to_datetime(df['ballDateTime'], errors='coerce')
        
        # Convert numeric columns
        numeric_cols = ['runs', 'shotAngle', 'shotMagnitude', 'runrate', 'totalBallNumber', 
                       'ballNumber', 'overNumber', 'totalInningRuns', 'totalInningWickets']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Convert boolean columns
        bool_cols = ['isWicket', 'isBoundry', 'isWide', 'isNoBall', 'isPowerPlay']
        for col in bool_cols:
            if col in df.columns:
                df[col] = df[col].astype(bool)
        
        # Fill missing values
        df['batsman'] = df['batsman'].fillna('Unknown')
        df['bowler'] = df['bowler'].fillna('Unknown')
        df['runs'] = df['runs'].fillna(0)
        
        return df
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return pd.DataFrame()

def create_match_phase(row):
    """Determine match phase based on ball number"""
    if pd.isna(row['totalBallNumber']):
        return 'Unknown'
    
    ball_num = row['totalBallNumber']
    if ball_num <= 25:  # First 25 balls (Powerplay in T20)
        return 'Powerplay'
    elif ball_num <= 75:  # Middle overs
        return 'Middle'
    else:  # Death overs
        return 'Death'

def main():
    """Main application function"""
    
    # Load data
    with st.spinner("Loading Hundred Women's Cricket data..."):
        df = load_data()
    
    if df.empty:
        st.error("Failed to load data. Please check the data source.")
        return
    
    # Add match phase column
    df['match_phase'] = df.apply(create_match_phase, axis=1)
    
    # Main title
    st.markdown("<h1 class='main-header'>🏏 Hundred Women's Cricket Analysis Dashboard</h1>", 
                unsafe_allow_html=True)
    
    # Sidebar filters
    st.sidebar.header("🔧 Filters")
    
    # Date filter
    if not df['matchDate'].isna().all():
        date_range = st.sidebar.date_input(
            "Select Date Range",
            value=(df['matchDate'].min().date(), df['matchDate'].max().date()),
            min_value=df['matchDate'].min().date(),
            max_value=df['matchDate'].max().date()
        )
        
        # Apply date filter
        if len(date_range) == 2:
            start_date, end_date = date_range
            df_filtered = df[
                (df['matchDate'].dt.date >= start_date) & 
                (df['matchDate'].dt.date <= end_date)
            ]
        else:
            df_filtered = df.copy()
    else:
        df_filtered = df.copy()
    
    # Team filter
    teams = sorted([team for team in df_filtered['battingTeam'].unique() if pd.notna(team)])
    selected_teams = st.sidebar.multiselect(
        "Select Teams",
        options=teams,
        default=teams
    )
    
    # Apply team filter
    if selected_teams:
        df_filtered = df_filtered[df_filtered['battingTeam'].isin(selected_teams)]
    
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
        
        # KPI Metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            total_matches = df_filtered['fixtureId'].nunique()
            st.metric("Total Matches", f"{total_matches:,}")
        
        with col2:
            total_runs = int(df_filtered['runs'].sum())
            st.metric("Total Runs", f"{total_runs:,}")
        
        with col3:
            total_wickets = int(df_filtered['isWicket'].sum())
            st.metric("Total Wickets", f"{total_wickets:,}")
        
        with col4:
            avg_run_rate = df_filtered['runrate'].mean()
            st.metric("Average Run Rate", f"{avg_run_rate:.2f}")
        
        # Tournament insights
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📈 Runs Trend by Date")
            if not df_filtered['matchDate'].isna().all():
                daily_runs = df_filtered.groupby(df_filtered['matchDate'].dt.date)['runs'].sum().reset_index()
                fig_trend = px.line(
                    daily_runs, 
                    x='matchDate', 
                    y='runs',
                    title='Daily Runs Scored',
                    labels={'runs': 'Total Runs', 'matchDate': 'Date'}
                )
                st.plotly_chart(fig_trend, use_container_width=True)
            else:
                st.info("Date information not available for trend analysis")
        
        with col2:
            st.subheader("🏟️ Runs by Team")
            team_runs = df_filtered.groupby('battingTeam')['runs'].sum().sort_values(ascending=True)
            fig_team = px.bar(
                x=team_runs.values,
                y=team_runs.index,
                orientation='h',
                title='Total Runs by Team',
                labels={'x': 'Total Runs', 'y': 'Team'}
            )
            st.plotly_chart(fig_team, use_container_width=True)
    
    # BATTING ANALYSIS
    with tab2:
        st.markdown("<div class='tab-header'>🏏 Batting Performance Analysis</div>", unsafe_allow_html=True)
        
        # Top run-scorers
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🏆 Top Run Scorers")
            batting_stats = df_filtered.groupby('batsman').agg({
                'runs': ['sum', 'count', 'mean'],
                'isBoundry': 'sum'
            }).round(2)
            
            batting_stats.columns = ['Total_Runs', 'Balls_Faced', 'Average', 'Boundaries']
            batting_stats['Strike_Rate'] = (batting_stats['Total_Runs'] / batting_stats['Balls_Faced'] * 100).round(2)
            batting_stats = batting_stats.sort_values('Total_Runs', ascending=False).head(10)
            
            st.dataframe(batting_stats, use_container_width=True)
        
        with col2:
            st.subheader("📊 Strike Rate vs Average")
            if len(batting_stats) > 0:
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
        
        # Runs by match phase
        st.subheader("⏱️ Runs by Match Phase")
        phase_runs = df_filtered.groupby(['batsman', 'match_phase'])['runs'].sum().reset_index()
        
        # Select top batsmen for phase analysis
        top_batsmen = df_filtered.groupby('batsman')['runs'].sum().nlargest(8).index.tolist()
        phase_runs_filtered = phase_runs[phase_runs['batsman'].isin(top_batsmen)]
        
        if not phase_runs_filtered.empty:
            fig_phase = px.bar(
                phase_runs_filtered,
                x='batsman',
                y='runs',
                color='match_phase',
                title='Runs Distribution by Match Phase (Top 8 Batsmen)',
                labels={'runs': 'Total Runs', 'batsman': 'Batsman'}
            )
            fig_phase.update_layout(xaxis_tickangle=45)
            st.plotly_chart(fig_phase, use_container_width=True)
        
        # Boundaries analysis
        st.subheader("🎯 Boundary Analysis")
        boundary_stats = df_filtered[df_filtered['isBoundry'] == True].groupby('batsman').size().sort_values(ascending=False).head(10)
        
        if not boundary_stats.empty:
            fig_boundaries = px.bar(
                x=boundary_stats.index,
                y=boundary_stats.values,
                title='Top 10 Boundary Hitters',
                labels={'x': 'Batsman', 'y': 'Total Boundaries'}
            )
            fig_boundaries.update_layout(xaxis_tickangle=45)
            st.plotly_chart(fig_boundaries, use_container_width=True)
    
    # BOWLING ANALYSIS
    with tab3:
        st.markdown("<div class='tab-header'>🎳 Bowling Performance Analysis</div>", unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🏆 Top Wicket Takers")
            # Calculate bowling statistics
            bowling_stats = df_filtered.groupby('bowler').agg({
                'isWicket': 'sum',
                'runs': 'sum',
                'batsman': 'count'  # balls bowled
            }).round(2)
            
            bowling_stats.columns = ['Wickets', 'Runs_Conceded', 'Balls_Bowled']
            bowling_stats = bowling_stats[bowling_stats['Balls_Bowled'] >= 10]  # Minimum 10 balls
            
            # Calculate economy rate (runs per 6 balls)
            bowling_stats['Economy_Rate'] = (bowling_stats['Runs_Conceded'] / bowling_stats['Balls_Bowled'] * 6).round(2)
            
            # Calculate bowling average (runs per wicket)
            bowling_stats['Bowling_Average'] = (bowling_stats['Runs_Conceded'] / bowling_stats['Wickets']).round(2)
            bowling_stats['Bowling_Average'] = bowling_stats['Bowling_Average'].replace([np.inf], 999)
            
            top_bowlers = bowling_stats.sort_values('Wickets', ascending=False).head(10)
            st.dataframe(top_bowlers, use_container_width=True)
        
        with col2:
            st.subheader("💹 Economy Rate vs Wickets")
            if len(top_bowlers) > 0:
                fig_bowling = px.scatter(
                    top_bowlers.reset_index(),
                    x='Economy_Rate',
                    y='Wickets',
                    size='Balls_Bowled',
                    hover_name='bowler',
                    title='Bowling Performance Matrix',
                    labels={'Economy_Rate': 'Economy Rate', 'Wickets': 'Total Wickets'}
                )
                st.plotly_chart(fig_bowling, use_container_width=True)
        
        # Dot ball analysis
        st.subheader("🎯 Dot Ball Analysis")
        dot_balls = df_filtered.groupby('bowler').agg({
            'runs': lambda x: (x == 0).sum(),  # Count dot balls
            'batsman': 'count'  # Total balls
        })
        dot_balls.columns = ['Dot_Balls', 'Total_Balls']
        dot_balls = dot_balls[dot_balls['Total_Balls'] >= 10]
        dot_balls['Dot_Ball_Percentage'] = (dot_balls['Dot_Balls'] / dot_balls['Total_Balls'] * 100).round(1)
        
        top_dot_ball_bowlers = dot_balls.sort_values('Dot_Ball_Percentage', ascending=False).head(10)
        
        if not top_dot_ball_bowlers.empty:
            fig_dots = px.bar(
                top_dot_ball_bowlers.reset_index(),
                x='bowler',
                y='Dot_Ball_Percentage',
                title='Top 10 Bowlers - Dot Ball Percentage',
                labels={'Dot_Ball_Percentage': 'Dot Ball %', 'bowler': 'Bowler'}
            )
            fig_dots.update_layout(xaxis_tickangle=45)
            st.plotly_chart(fig_dots, use_container_width=True)
    
    # SHOT ANALYSIS
    with tab4:
        st.markdown("<div class='tab-header'>🎯 Shot Analysis</div>", unsafe_allow_html=True)
        
        # Player selection for shot analysis
        available_batsmen = [b for b in df_filtered['batsman'].unique() if pd.notna(b) and b != 'Unknown']
        selected_batsman = st.selectbox(
            "Select Batsman for Shot Analysis",
            options=sorted(available_batsmen)
        )
        
        player_data = df_filtered[
            (df_filtered['batsman'] == selected_batsman) & 
            (df_filtered['shotAngle'].notna()) & 
            (df_filtered['shotMagnitude'].notna())
        ]
        
        if not player_data.empty:
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.subheader(f"🌐 Wagon Wheel - {selected_batsman}")
                
                # Create wagon wheel using polar plot
                fig_wagon = go.Figure()
                
                # Color by runs scored
                colors = {0: 'gray', 1: 'blue', 2: 'green', 3: 'orange', 4: 'red', 6: 'purple'}
                
                for runs, color in colors.items():
                    data = player_data[player_data['runs'] == runs]
                    if not data.empty:
                        fig_wagon.add_trace(go.Scatterpolar(
                            r=data['shotMagnitude'],
                            theta=data['shotAngle'],
                            mode='markers',
                            name=f'{runs} runs',
                            marker=dict(color=color, size=8),
                            hovertemplate=f'<b>{runs} runs</b><br>Angle: %{{theta}}°<br>Distance: %{{r}}<extra></extra>'
                        ))
                
                fig_wagon.update_layout(
                    title=f'Shot Distribution - {selected_batsman}',
                    polar=dict(
                        radialaxis=dict(visible=True, range=[0, player_data['shotMagnitude'].max() * 1.1]),
                        angularaxis=dict(direction="clockwise")
                    ),
                    height=500
                )
                st.plotly_chart(fig_wagon, use_container_width=True)
            
            with col2:
                st.subheader("📊 Shot Statistics")
                
                # Player shot stats
                total_shots = len(player_data)
                boundaries = len(player_data[player_data['runs'] >= 4])
                avg_shot_distance = player_data['shotMagnitude'].mean()
                
                st.metric("Total Shots", total_shots)
                st.metric("Boundaries", boundaries)
                st.metric("Avg Shot Distance", f"{avg_shot_distance:.1f}")
                
                # Shot type analysis
                if 'battingShotTypeId' in player_data.columns:
                    st.subheader("🎯 Shot Types")
                    shot_types = player_data.groupby('battingShotTypeId').agg({
                        'runs': ['count', 'mean', 'sum']
                    }).round(2)
                    shot_types.columns = ['Frequency', 'Avg_Runs', 'Total_Runs']
                    shot_types = shot_types.sort_values('Total_Runs', ascending=False)
                    st.dataframe(shot_types, use_container_width=True)
        else:
            st.info(f"No shot data available for {selected_batsman}")
    
    # MATCH EXPLORER
    with tab5:
        st.markdown("<div class='tab-header'>📊 Match Explorer</div>", unsafe_allow_html=True)
        
        # Match selection
        available_matches = df_filtered['fixtureId'].unique()
        selected_match = st.selectbox(
            "Select Match",
            options=sorted(available_matches)
        )
        
        match_data = df_filtered[df_filtered['fixtureId'] == selected_match]
        
        if not match_data.empty:
            # Match summary
            st.subheader(f"📋 Match Summary - Fixture {selected_match}")
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Balls", len(match_data))
            with col2:
                st.metric("Total Runs", int(match_data['runs'].sum()))
            with col3:
                st.metric("Total Wickets", int(match_data['isWicket'].sum()))
            with col4:
                avg_rr = match_data['runrate'].mean()
                st.metric("Average Run Rate", f"{avg_rr:.2f}")
            
            # Ball by ball data
            st.subheader("⚾ Ball-by-Ball Data")
            
            # Select relevant columns for ball-by-ball
            ball_by_ball = match_data[[
                'ballNumber', 'batsman', 'bowler', 'runs', 'isWicket', 
                'commentary', 'runrate'
            ]].copy()
            
            ball_by_ball = ball_by_ball.sort_values('ballNumber')
            st.dataframe(ball_by_ball, use_container_width=True, height=300)
            
            # Cumulative runs chart
            st.subheader("📈 Cumulative Runs Progress")
            
            match_data_sorted = match_data.sort_values('ballNumber')
            match_data_sorted['cumulative_runs'] = match_data_sorted['runs'].cumsum()
            
            fig_cumulative = px.line(
                match_data_sorted,
                x='ballNumber',
                y='cumulative_runs',
                title=f'Cumulative Runs - Match {selected_match}',
                labels={'ballNumber': 'Ball Number', 'cumulative_runs': 'Cumulative Runs'}
            )
            st.plotly_chart(fig_cumulative, use_container_width=True)
        else:
            st.info("No data available for selected match")
    
    # Footer
    st.markdown("---")
    st.markdown("**🏏 Hundred Women's Cricket Analysis Dashboard** | Data Source: The Hundred Tournament")

if __name__ == "__main__":
    main()
