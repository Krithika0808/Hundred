import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
from io import StringIO
from datetime import datetime
import numpy as np

# --- Configuration and Data Loading ---

# Set Streamlit page configuration
st.set_page_config(
    page_title="Hundred Women's Cricket Analysis",
    page_icon="🏏",
    layout="wide",
    initial_sidebar_state="expanded"
)

# URL for the raw CSV data on GitHub
GITHUB_CSV_URL = "https://raw.githubusercontent.com/Krithika0808/Hundred/main/Hundred.csv"

@st.cache_data
def load_data_from_github(url):
    """
    Loads cricket data from a public GitHub repository.
    Uses requests to fetch the raw CSV content and pandas to read it.
    """
    try:
        response = requests.get(url)
        response.raise_for_status()  # Check for HTTP errors
        
        csv_content = StringIO(response.text)
        df = pd.read_csv(csv_content)
        
        # --- Data Cleaning and Preprocessing ---
        
        # Check for essential columns to prevent KeyErrors later
        required_cols = ['matchDate', 'runs', 'isWicket', 'overNumber', 'totalBallNumber', 
                         'runsConceded', 'shotAngle', 'shotMagnitude', 'runsScored', 
                         'extras', 'runsWide', 'runsByes', 'runsLegByes', 'batsman', 
                         'bowler', 'battingTeam', 'bowlingTeam', 'battingShotTypeId', 'isBoundary']
        
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            st.error(f"The following required columns are missing from the dataset: {', '.join(missing_cols)}")
            return pd.DataFrame()

        # Convert date column to datetime objects
        df['matchDate'] = pd.to_datetime(df['matchDate'], errors='coerce')
        
        # Ensure key numeric columns have the correct type
        numeric_cols = ['runs', 'isWicket', 'overNumber', 'totalBallNumber', 'runsConceded',
                        'shotAngle', 'shotMagnitude', 'runsScored', 'extras', 'runsWide', 'runsByes', 'runsLegByes']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        
        # Fill missing string data with 'Unknown'
        string_cols = ['batsman', 'bowler', 'battingTeam', 'bowlingTeam', 'battingShotTypeId']
        for col in string_cols:
            if col in df.columns:
                df[col] = df[col].fillna('Unknown')
        
        # Create a `match_phase` column based on ball number
        df['match_phase'] = df['totalBallNumber'].apply(lambda x: 
            'Powerplay' if x <= 25 else ('Middle Overs' if x <= 75 else 'Death Overs')
        )
        
        return df

    except requests.exceptions.RequestException as e:
        st.error(f"Error loading data from GitHub: {e}")
        return pd.DataFrame()

def main():
    """
    Main function to run the Streamlit application.
    """
    st.markdown("<h1 style='text-align: center;'>Hundred Women's Tournament 🏏</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #6c757d;'>Comprehensive Data Analysis</p>", unsafe_allow_html=True)

    # Load data
    df = load_data_from_github(GITHUB_CSV_URL)

    if df.empty:
        st.warning("Could not load data. Please check the data source.")
        return

    # --- Sidebar Filters ---
    st.sidebar.header("Filters")
    
    # Date filter
    min_date = df['matchDate'].min().date()
    max_date = df['matchDate'].max().date()
    date_range = st.sidebar.date_input(
        "Select Date Range",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date
    )
    
    # Check if date_range is a tuple with two elements
    if isinstance(date_range, tuple) and len(date_range) == 2:
        start_date = pd.to_datetime(date_range[0])
        end_date = pd.to_datetime(date_range[1])
        filtered_df = df[(df['matchDate'] >= start_date) & (df['matchDate'] <= end_date)]
    else:
        # Handle cases where only one date is selected
        filtered_df = df
        
    # Team filter
    all_teams = sorted(filtered_df['battingTeam'].unique())
    selected_teams = st.sidebar.multiselect(
        "Select Teams",
        options=all_teams,
        default=all_teams
    )
    
    if selected_teams:
        filtered_df = filtered_df[filtered_df['battingTeam'].isin(selected_teams) | filtered_df['bowlingTeam'].isin(selected_teams)]

    if filtered_df.empty:
        st.warning("No data available for the selected filters.")
        return

    # --- Tabs for different analysis sections ---
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Home", 
        "Batting Analysis", 
        "Bowling Analysis", 
        "Shot Analysis", 
        "Match Explorer"
    ])

    with tab1:
        st.header("Home Page: Tournament Summary")
        
        # Calculate summary KPIs
        total_matches = filtered_df['fixtureId'].nunique()
        total_runs = filtered_df['runsScored'].sum()
        total_wickets = filtered_df['isWicket'].sum()
        total_balls = filtered_df['totalBallNumber'].max()
        
        if total_balls > 0:
            avg_run_rate = (total_runs / total_balls) * 6
        else:
            avg_run_rate = 0
            
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Matches", total_matches)
        with col2:
            st.metric("Total Runs", int(total_runs))
        with col3:
            st.metric("Total Wickets", int(total_wickets))
        with col4:
            st.metric("Average Run Rate", f"{avg_run_rate:.2f}")

    with tab2:
        st.header("Batting Analysis")
        
        # Top run-scorers table
        player_stats = filtered_df.groupby('batsman').agg(
            Runs=('runsScored', 'sum'),
            Balls=('runsScored', 'count'),
            Boundaries=('isBoundary', 'sum')
        ).reset_index()
        
        # Calculate strike rate and average
        player_stats['Strike Rate'] = (player_stats['Runs'] / player_stats['Balls']) * 100
        
        # To calculate average, we need total outs which is tricky with this dataset,
        # so we'll use a simple proxy or note the limitation
        wickets_per_player = filtered_df[filtered_df['isWicket'] == 1].groupby('batsman').size()
        player_stats = player_stats.set_index('batsman')
        player_stats['Wickets'] = wickets_per_player
        player_stats['Wickets'] = player_stats['Wickets'].fillna(0)
        player_stats['Average'] = player_stats['Runs'] / player_stats['Wickets']
        player_stats['Average'] = player_stats['Average'].replace([np.inf, -np.inf], np.nan).fillna(0)
        player_stats = player_stats.reset_index()
        
        st.subheader("Top Run-Scorers")
        st.dataframe(player_stats.sort_values('Runs', ascending=False), hide_index=True)

        # Runs by phase bar chart
        runs_by_phase = filtered_df.groupby(['batsman', 'match_phase'])['runsScored'].sum().reset_index()
        runs_by_phase_fig = px.bar(
            runs_by_phase,
            x='batsman',
            y='runsScored',
            color='match_phase',
            title="Runs by Match Phase per Player",
            labels={'runsScored': 'Total Runs', 'batsman': 'Batsman'},
            barmode='group'
        )
        st.plotly_chart(runs_by_phase_fig, use_container_width=True)
        
        # Boundaries per player bar chart
        boundaries_df = player_stats.sort_values('Boundaries', ascending=False)
        boundaries_fig = px.bar(
            boundaries_df,
            x='batsman',
            y='Boundaries',
            title="Boundaries per Player",
            labels={'Boundaries': 'Total Boundaries', 'batsman': 'Batsman'}
        )
        st.plotly_chart(boundaries_fig, use_container_width=True)

    with tab3:
        st.header("Bowling Analysis")
        
        # Top wicket-takers table
        bowler_stats = filtered_df.groupby('bowler').agg(
            Wickets=('isWicket', 'sum'),
            Balls=('isWicket', 'count'),
            RunsConceded=('runsConceded', 'sum')
        ).reset_index()
        
        # Calculate bowling average and economy rate
        bowler_stats['Bowling Average'] = bowler_stats['RunsConceded'] / bowler_stats['Wickets']
        bowler_stats['Bowling Average'] = bowler_stats['Bowling Average'].replace([np.inf, -np.inf], np.nan).fillna(0)
        bowler_stats['Economy Rate'] = (bowler_stats['RunsConceded'] / bowler_stats['Balls']) * 6
        
        st.subheader("Top Wicket-Takers")
        st.dataframe(bowler_stats.sort_values('Wickets', ascending=False), hide_index=True)
        
        # Dot ball percentage
        dot_balls = filtered_df[(filtered_df['runsScored'] == 0) & (filtered_df['extras'] == 0)].groupby('bowler').size().reset_index(name='Dot Balls')
        bowler_stats = bowler_stats.merge(dot_balls, on='bowler', how='left').fillna(0)
        bowler_stats['Dot Ball Percentage'] = (bowler_stats['Dot Balls'] / bowler_stats['Balls']) * 100
        
        st.subheader("Dot Ball Percentage")
        dot_ball_fig = px.bar(
            bowler_stats.sort_values('Dot Ball Percentage', ascending=False),
            x='bowler',
            y='Dot Ball Percentage',
            title="Dot Ball Percentage per Bowler",
            labels={'bowler': 'Bowler', 'Dot Ball Percentage': 'Dot Ball %'}
        )
        st.plotly_chart(dot_ball_fig, use_container_width=True)
        
        # Scatter plot: Economy rate vs. Wickets
        economy_wicket_fig = px.scatter(
            bowler_stats,
            x='Economy Rate',
            y='Wickets',
            hover_name='bowler',
            title="Economy Rate vs. Wickets",
            labels={'Economy Rate': 'Economy Rate', 'Wickets': 'Total Wickets'}
        )
        st.plotly_chart(economy_wicket_fig, use_container_width=True)

    with tab4:
        st.header("Shot Analysis")
        
        # Player filter for this tab
        shot_players = sorted(filtered_df['batsman'].unique())
        selected_shot_player = st.selectbox("Select Player for Shot Analysis", options=shot_players)
        
        if selected_shot_player:
            player_shots_df = filtered_df[filtered_df['batsman'] == selected_shot_player].copy()
            
            # Wagon wheel plot
            st.subheader(f"Wagon Wheel for {selected_shot_player}")
            wagon_wheel_fig = px.scatter_polar(
                player_shots_df,
                r='shotMagnitude',
                theta='shotAngle',
                color='battingShotTypeId',
                size='runsScored',
                hover_data=['battingShotTypeId', 'runsScored'],
                title=f"Shot Placement and Runs for {selected_shot_player}"
            )
            wagon_wheel_fig.update_layout(polar_angularaxis=dict(direction="clockwise", tickvals=np.arange(0, 361, 45)))
            st.plotly_chart(wagon_wheel_fig, use_container_width=True)
            
            # Shot types with average runs table
            shot_type_runs = player_shots_df.groupby('battingShotTypeId').agg(
                TotalRuns=('runsScored', 'sum'),
                TotalShots=('battingShotTypeId', 'count'),
                AverageRuns=('runsScored', 'mean')
            ).reset_index().sort_values('TotalRuns', ascending=False)
            
            st.subheader("Shot Types and Average Runs")
            st.dataframe(shot_type_runs.round(2), hide_index=True)

    with tab5:
        st.header("Match Explorer")
        
        all_matches = sorted(filtered_df['fixtureId'].unique())
        selected_match_id = st.selectbox("Select a Match", options=all_matches)
        
        if selected_match_id:
            match_df = filtered_df[filtered_df['fixtureId'] == selected_match_id].copy()
            
            st.subheader("Ball-by-Ball Commentary")
            st.dataframe(
                match_df[['ballNumber', 'battingTeam', 'bowlingTeam', 'batsman', 'bowler', 'commentary', 'runsScored']],
                hide_index=True
            )
            
            # Cumulative runs over balls graph
            st.subheader("Cumulative Runs Over Balls")
            match_df['cumulative_runs_batting'] = match_df.groupby(['fixtureId', 'battingTeam'])['runsScored'].cumsum()
            match_df['cumulative_runs_bowling'] = match_df.groupby(['fixtureId', 'bowlingTeam'])['runsConceded'].cumsum()
            
            # Get the two teams in the match
            teams_in_match = match_df['battingTeam'].unique()
            
            fig = make_subplots(specs=[[{"secondary_y": True}]])
            
            for team in teams_in_match:
                team_data = match_df[match_df['battingTeam'] == team]
                fig.add_trace(go.Scatter(
                    x=team_data['totalBallNumber'], 
                    y=team_data['cumulative_runs_batting'],
                    mode='lines+markers',
                    name=f'{team} Runs'
                ))
            
            fig.update_layout(
                title_text="Cumulative Runs Over Balls",
                xaxis_title="Ball Number",
                yaxis_title="Total Runs"
            )
            st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()
