import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
from io import StringIO
from datetime import datetime
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from collections import defaultdict

# --- Set Page Configuration ---
st.set_page_config(
    page_title="Women's Cricket Shot Intelligence Matrix",
    page_icon="🏏",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom CSS for better styling ---
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
    .insight-card {
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        padding: 1.5rem;
        border-radius: 0.75rem;
        border: 1px solid #dee2e6;
        margin: 1rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    .insight-title {
        font-weight: bold;
        color: #495057;
        margin-bottom: 0.5rem;
    }
    .insight-content {
        color: #6c757d;
        font-size: 0.95rem;
    }
    .bowling-recommendation {
        background-color: #f8f9fa;
        padding: 0.5rem;
        border-radius: 0.375rem;
        margin-top: 0.5rem;
        border-left: 3px solid #dc3545;
    }
    .recommendation-item {
        margin-bottom: 0.25rem;
        font-size: 0.9rem;
    }
    .connection-note {
        font-style: italic;
        color: #6c757d;
        font-size: 0.85rem;
        margin-top: 0.25rem;
    }
    .advanced-metric {
        background: linear-gradient(135deg, #e3f2fd 0%, #bbdefb 100%);
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #2196f3;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)


# --- Data Loading and Preprocessing Functions ---

@st.cache_data
def load_data_from_github(github_url=None):
    """Load cricket data from GitHub repository"""
    try:
        if github_url is None:
            github_url = "https://raw.githubusercontent.com/Krithika0808/Hundred/main/Hundred.csv"
        
        response = requests.get(github_url)
        response.raise_for_status() 
        
        csv_content = StringIO(response.text)
        df = pd.read_csv(csv_content)
        
        if df.empty:
            return df
            
        numeric_columns = ['shotAngle', 'shotMagnitude', 'runs', 'totalBallNumber']
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        bool_columns = ['isWicket', 'isBoundary', 'isAirControlled', 'isWide', 'isNoBall']
        for col in bool_columns:
            if col in df.columns:
                df[col] = df[col].astype(bool)
        
        string_columns = ['batsman', 'bowler', 'battingShotTypeId', 'battingConnectionId', 'commentary']
        for col in string_columns:
            if col in df.columns:
                df[col] = df[col].astype(str).str.strip()
        
        essential_columns = [col for col in ['batsman', 'runs', 'totalBallNumber'] if col in df.columns]
        if essential_columns:
            df = df.dropna(subset=essential_columns)
        
        if 'matchDate' in df.columns:
            df['matchDate'] = pd.to_datetime(df['matchDate'], errors='coerce')
            df['season'] = df['matchDate'].dt.year
        
        return df
        
    except requests.exceptions.RequestException as e:
        st.error(f"❌ Error fetching data from GitHub: {str(e)}")
        st.info("Creating sample data for demonstration...")
        return create_sample_data()
    except Exception as e:
        st.error(f"❌ Error processing data: {str(e)}")
        st.info("Creating sample data for demonstration...")
        return create_sample_data()

def create_sample_data():
    """Create sample cricket data for demonstration"""
    np.random.seed(42)
    
    players = ['Smriti Mandhana', 'Harmanpreet Kaur', 'Beth Mooney', 'Alyssa Healy', 'Meg Lanning']
    bowlers = ['Jess Jonassen', 'Sophie Ecclestone', 'Ashleigh Gardner', 'Shabnim Ismail', 'Marizanne Kapp']
    shot_types = ['Drive', 'Pull', 'Cut', 'Sweep', 'Flick', 'Hook', 'Reverse Sweep', 'Loft']
    connection_types = ['Middled', 'WellTimed', 'Undercontrol', 'MisTimed', 'Missed', 'HitBody']
    
    length_types = ['Yorker', 'Full', 'Good Length', 'Short', 'Bouncer']
    line_types = ['Off Stump', 'Middle Stump', 'Leg Stump', 'Wide Outside Off', 'DownLeg']
    bowling_types = ['Fast', 'Medium', 'Spin', 'Swing', 'Seam']
    bowling_from = ['Over the Wicket', 'Around the Wicket']
    bowling_hands = ['Right Arm', 'Left Arm']
    
    n_rows = 1000
    
    data = {
        'batsman': np.random.choice(players, n_rows),
        'bowler': np.random.choice(bowlers, n_rows),
        'battingShotTypeId': np.random.choice(shot_types, n_rows),
        'battingConnectionId': np.random.choice(connection_types, n_rows, p=[0.3, 0.25, 0.2, 0.15, 0.05, 0.05]),
        'runs': np.random.choice([0, 1, 2, 3, 4, 6], n_rows, p=[0.3, 0.25, 0.2, 0.1, 0.1, 0.05]),
        'totalBallNumber': np.random.randint(1, 101, n_rows),
        'shotAngle': np.random.uniform(0, 360, n_rows),
        'shotMagnitude': np.random.uniform(50, 200, n_rows),
        'isAirControlled': np.random.choice([True, False], n_rows, p=[0.3, 0.7]),
        'isBoundary': np.random.choice([True, False], n_rows, p=[0.15, 0.85]),
        'isWicket': np.random.choice([True, False], n_rows, p=[0.05, 0.95]),
        'commentary': ['Good shot!', 'Excellent timing!', 'Mistimed!', 'Great connection!'] * (n_rows // 4),
        'fixtureId': np.random.randint(1, 21, n_rows),
        'battingTeam': np.random.choice(['Team A', 'Team B', 'Team C', 'Team D'], n_rows),
        'matchDate': pd.to_datetime(pd.to_datetime('2023-01-01') + pd.to_timedelta(np.random.randint(0, 365, n_rows), unit='d')),
        'lengthTypeId': np.random.choice(length_types, n_rows),
        'lineTypeId': np.random.choice(line_types, n_rows),
        'bowlingTypeId': np.random.choice(bowling_types, n_rows),
        'bowlingFromId': np.random.choice(bowling_from, n_rows),
        'bowlingHandId': np.random.choice(bowling_hands, n_rows),
        'match_situation': np.random.choice(['Powerplay', 'Middle Overs', 'Death Overs'], n_rows)
    }
    
    df = pd.DataFrame(data)
    
    df.loc[df['runs'] == 4, 'isBoundary'] = True
    df.loc[df['runs'] == 6, 'isBoundary'] = True
    df.loc[df['runs'] < 4, 'isBoundary'] = False
    
    def generate_commentary(row):
        if row['runs'] == 6:
            return f"SIX! {row['battingShotTypeId']} shot for maximum!"
        elif row['runs'] == 4:
            return f"FOUR! Beautiful {row['battingShotTypeId']} to the boundary!"
        elif row['battingConnectionId'] == 'Middled':
            return f"Perfect {row['battingShotTypeId']} shot!"
        elif row['battingConnectionId'] == 'MisTimed':
            return f"Mistimed {row['battingShotTypeId']}"
        else:
            return f"{row['battingShotTypeId']} shot for {row['runs']} runs"
    
    df['commentary'] = df.apply(generate_commentary, axis=1)
    
    return df

def false_shot_score(connection_id):
    """Calculate false shot score based on batting connection"""
    false_shot_mapping = {
        'Middled': 0, 'WellTimed': 0, 'Undercontrol': 0, 'Left': 0,
        'MisTimed': 3, 'Mis-timed': 3, 'BottomEdge': 4, 'TopEdge': 4,
        'BatPad': 4, 'InsideEdge': 3, 'LeadingEdge': 4, 'OutsideEdge': 3,
        'Gloved': 3, 'ThickEdge': 3, 'Missed': 5, 'PlayAndMiss': 5,
        'PlayAndMissLegSide': 5, 'HitBody': 2, 'HitHelmet': 2, 'HitPad': 2,
        'Padded': 2, 'Spliced': 4, 'Unknown': 2
    }
    return false_shot_mapping.get(str(connection_id).strip(), 2)

def calculate_shot_intelligence_metrics(df):
    """Calculate advanced shot intelligence metrics"""
    
    if df.empty:
        return df
    
    df = df.copy()
    
    df['shotAngle'] = df['shotAngle'].fillna(0).clip(0, 360)
    df['shotMagnitude'] = df['shotMagnitude'].fillna(100)
    df['isAirControlled'] = df['isAirControlled'].fillna(False)
    df['battingConnectionId'] = df['battingConnectionId'].fillna('Unknown')
    df['commentary'] = df['commentary'].fillna('')
    df['false_shot_score'] = df['battingConnectionId'].apply(false_shot_score)
    
    def detect_boundary_from_data(row):
        runs = row.get('runs', 0)
        commentary = str(row['commentary']).lower()
        is_boundary_col = row.get('isBoundary', False)
        
        if runs == 4 or runs == 6:
            return True
        if is_boundary_col:
            return True
        boundary_keywords = ['four', 'boundary', 'six', 'maximum']
        for keyword in boundary_keywords:
            if keyword in commentary:
                return True
        return False
    
    df['is_boundary'] = df.apply(detect_boundary_from_data, axis=1)
    
    control_scores = {
        'Middled': 3, 'WellTimed': 3, 'Undercontrol': 3, 'Left': 3,
        'MisTimed': 2, 'Mis-timed': 2, 'BottomEdge': 2, 'TopEdge': 2,
        'BatPad': 1, 'InsideEdge': 2, 'LeadingEdge': 2, 'OutsideEdge': 2,
        'Gloved': 2, 'ThickEdge': 2, 'Missed': 0, 'PlayAndMiss': 0,
        'PlayAndMissLegSide': 0, 'HitBody': 0.5, 'HitHelmet': 0, 'HitPad': 0,
        'Padded': 0, 'Spliced': 1
    }
    df['control_quality_score'] = df['battingConnectionId'].map(control_scores).fillna(1.5)
    
    angle_bins = [0, 45, 90, 135, 180, 225, 270, 315, 360]
    angle_labels = ['Long Off', 'Cover', 'Point', 'Third Man', 'Fine Leg', 'Square Leg', 'Mid Wicket', 'Long On']
    df['angle_zone'] = pd.cut(df['shotAngle'], bins=angle_bins, labels=angle_labels, include_lowest=True)
    
    # Calculate base control score
    df['control_score'] = df['control_quality_score'] * 33.33
    df['control_score'] += df['runs'] * 5
    df['control_score'] += df['is_boundary'] * 20
    good_placements = ['Cover', 'Mid Wicket', 'Long On', 'Long Off']
    df['control_score'] += df['angle_zone'].isin(good_placements) * 10
    df['control_score'] = df['control_score'].clip(0, 100)
    df['is_controlled_shot'] = (df['control_score'] >= 50).astype(int)
    
    if 'totalBallNumber' in df.columns:
        max_ball = int(df['totalBallNumber'].max())
        bins = [0, 25, 75, max_ball + 1]
        labels = ['Powerplay (1-25)', 'Middle (26-75)', f'Death (76-{max_ball})']
        df['match_phase'] = pd.cut(df['totalBallNumber'], bins=bins, labels=labels, include_lowest=True)
    
    return df

def calculate_xruns(df):
    """Calculates Expected Runs (xRuns) for each shot based on historical averages."""
    if df.empty or 'battingShotTypeId' not in df.columns or 'angle_zone' not in df.columns:
        df['xRuns'] = 0.0
        return df
    
    # Create a lookup table for average runs per shot type and angle zone
    xruns_lookup = df.groupby(['battingShotTypeId', 'angle_zone'])['runs'].mean().reset_index()
    xruns_lookup.rename(columns={'runs': 'xRuns_avg'}, inplace=True)
    
    df = pd.merge(df, xruns_lookup, on=['battingShotTypeId', 'angle_zone'], how='left')
    df['xRuns'] = df['xRuns_avg'].fillna(df['runs'].mean()) # Fill NaNs with overall average
    df.drop(columns='xRuns_avg', inplace=True)
    return df

def predict_wicket_probability(df):
    """Simple rule-based model for wicket probability index."""
    if df.empty:
        df['wicket_prob_index'] = 0.0
        return df
    
    df['wicket_prob_index'] = df['false_shot_score'] * 10 
    
    # Add multipliers for specific conditions
    df.loc[df['isWicket'] == True, 'wicket_prob_index'] = 100
    df.loc[(df['false_shot_score'] >= 4) & (df['match_phase'] == 'Death (76-100)'), 'wicket_prob_index'] *= 1.5
    df['wicket_prob_index'] = df['wicket_prob_index'].clip(0, 100)
    
    return df

def cluster_shots(df, n_clusters=4):
    """Uses KMeans to cluster shots based on angle and magnitude."""
    if df.empty or 'shotAngle' not in df.columns or 'shotMagnitude' not in df.columns:
        df['shot_cluster'] = 'No Cluster'
        return df
    
    cluster_df = df[['shotAngle', 'shotMagnitude']].dropna()
    if len(cluster_df) < n_clusters:
        df['shot_cluster'] = 'Cluster 1'
        return df
    
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(cluster_df)
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_df['shot_cluster'] = kmeans.fit_predict(scaled_data)
    
    # Map cluster labels back to the original dataframe
    df = df.copy()
    df['shot_cluster'] = 'No Cluster'
    df.loc[cluster_df.index, 'shot_cluster'] = cluster_df['shot_cluster'].astype(str)
    return df

def get_player_insights(player_data):
    """Generate player insights with new advanced metrics."""
    insights = {}
    
    if 'battingShotTypeId' in player_data.columns:
        shot_counts = player_data['battingShotTypeId'].value_counts()
        if not shot_counts.empty:
            favorite_shot = shot_counts.index[0]
            favorite_count = shot_counts.iloc[0]
            total_shots = len(player_data)
            favorite_percentage = (favorite_count / total_shots) * 100
            insights['favorite_shot'] = f"{favorite_shot} ({favorite_percentage:.1f}% of shots)"
    
    if 'angle_zone' in player_data.columns and 'false_shot_score' in player_data.columns:
        risky_shots = player_data[player_data['false_shot_score'] >= 4]
        if not risky_shots.empty:
            risky_zone = risky_shots['angle_zone'].value_counts().index[0]
            insights['risky_zone'] = risky_zone
    
    if 'control_score' in player_data.columns and 'runs' in player_data.columns and 'xRuns' in player_data.columns:
        avg_control = player_data['control_score'].mean()
        avg_xruns = player_data['xRuns'].mean()
        avg_actual_runs = player_data['runs'].mean()
        
        if avg_actual_runs > avg_xruns * 1.1:
            insights['efficiency_insight'] = f"Performing **above expected efficiency** (actual runs: {avg_actual_runs:.2f}, xRuns: {avg_xruns:.2f})."
        elif avg_actual_runs < avg_xruns * 0.9:
            insights['efficiency_insight'] = f"Performing **below expected efficiency** (actual runs: {avg_actual_runs:.2f}, xRuns: {avg_xruns:.2f})."
        else:
            insights['efficiency_insight'] = f"Performing **as expected** (actual runs: {avg_actual_runs:.2f}, xRuns: {avg_xruns:.2f})."
    
    bowling_columns = ['lengthTypeId', 'lineTypeId', 'bowlingTypeId', 'bowlingFromId', 'bowlingHandId']
    available_bowling_cols = [col for col in bowling_columns if col in player_data.columns]
    
    if available_bowling_cols and 'control_score' in player_data.columns and 'risky_zone' in insights:
        risky_zone_shots = player_data[player_data['angle_zone'] == insights['risky_zone']]
        
        if not risky_zone_shots.empty:
            bowling_recommendations = []
            for col in available_bowling_cols:
                attr_control = risky_zone_shots.groupby(col)['control_score'].mean()
                if not attr_control.empty and len(attr_control) >= 2:
                    weakest_attr = attr_control.idxmin()
                    weakest_score = attr_control.min()
                    if weakest_score < 60:
                        col_names = {'lengthTypeId': 'Length', 'lineTypeId': 'Line', 'bowlingTypeId': 'Type', 'bowlingFromId': 'From', 'bowlingHandId': 'Hand'}
                        display_name = col_names.get(col, col)
                        bowling_recommendations.append((display_name, weakest_attr, weakest_score))
            
            if bowling_recommendations:
                bowling_recommendations.sort(key=lambda x: x[2])
                top_weaknesses = bowling_recommendations[:3]
                recommendations = [f"{name}: {value} ({score:.1f}/100)" for name, value, score in top_weaknesses]
                insights['bowl_to'] = recommendations
                insights['bowl_to_connection'] = insights['risky_zone']
    
    return insights

def create_player_form_chart(df, player_name):
    """Creates a line chart for a player's form based on a rolling control score."""
    player_data = df[df['batsman'] == player_name].sort_values('matchDate')
    if player_data.empty or len(player_data['matchDate'].unique()) < 2:
        return go.Figure()
        
    player_data['rolling_control'] = player_data['control_score'].rolling(window=10, min_periods=1).mean()
    
    fig = px.line(player_data, x='matchDate', y='rolling_control',
                  title=f"{player_name}'s Form: Rolling Average Control Score (10-ball window)",
                  labels={'rolling_control': 'Rolling Avg. Control Score', 'matchDate': 'Match Date'},
                  line_shape='spline')
    fig.update_layout(yaxis_range=[0, 100])
    return fig

def create_xruns_comparison_chart(df, player_name):
    """Creates a bar chart comparing actual runs to xRuns."""
    player_data = df[df['batsman'] == player_name]
    if player_data.empty:
        return go.Figure()
        
    runs_summary = player_data.groupby('battingShotTypeId').agg(
        avg_runs=('runs', 'mean'),
        avg_xruns=('xRuns', 'mean')
    ).reset_index()
    
    fig = go.Figure()
    fig.add_trace(go.Bar(x=runs_summary['battingShotTypeId'], y=runs_summary['avg_runs'], name='Actual Runs'))
    fig.add_trace(go.Bar(x=runs_summary['battingShotTypeId'], y=runs_summary['avg_xruns'], name='Expected Runs (xRuns)'))
    
    fig.update_layout(barmode='group', title=f"{player_name} - Actual Runs vs. Expected Runs by Shot Type")
    return fig

def create_shot_cluster_chart(df, player_name):
    """Creates a polar chart visualizing shot clusters."""
    player_data = df[df['batsman'] == player_name]
    if player_data.empty or 'shot_cluster' not in player_data.columns:
        return go.Figure()
    
    fig = px.scatter_polar(
        player_data,
        r='shotMagnitude',
        theta='shotAngle',
        color='shot_cluster',
        title=f"{player_name} - Shot Clusters by Type",
        hover_data=['battingShotTypeId', 'runs', 'control_score'],
        color_discrete_sequence=px.colors.qualitative.Pastel
    )
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, player_data['shotMagnitude'].max() * 1.1]),
            angularaxis=dict(direction="clockwise")
        ),
        showlegend=True
    )
    return fig

# --- Main App Structure ---
if 'data_df' not in st.session_state:
    st.session_state.data_df = load_data_from_github()

if st.session_state.data_df.empty:
    st.warning("No data available to display. Please check the data source or try again later.")
else:
    df = st.session_state.data_df.copy()
    
    df = calculate_shot_intelligence_metrics(df)
    df = calculate_xruns(df)
    df = predict_wicket_probability(df)
    df = cluster_shots(df)
    
    all_players = sorted(df['batsman'].unique())
    all_bowlers = sorted(df['bowler'].unique())
    all_seasons = sorted(df['season'].unique())
    
    st.markdown("<h1 class='main-header'>Women's Cricket Shot Intelligence Matrix 🏏</h1>", unsafe_allow_html=True)
    st.sidebar.header("Player and Match Selection")
    
    selected_players = st.sidebar.multiselect(
        "Select Player(s)",
        options=all_players,
        default=all_players[:1]
    )
    
    if len(selected_players) == 0:
        st.warning("Please select at least one player.")
        st.stop()
    
    selected_season = st.sidebar.multiselect("Filter by Season", options=all_seasons, default=all_seasons)
    
    filtered_df = df[
        df['batsman'].isin(selected_players) &
        df['season'].isin(selected_season)
    ]

    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
        "Dashboard",
        "Player Form",
        "Shot Placement",
        "Shot Matrix",
        "Match Phase Analysis",
        "Player Comparison",
        "Advanced Analytics",
        "Bowler Pressure"
    ])
    
    with tab1:
        st.subheader("📊 Player Dashboard")
        if len(selected_players) == 1:
            player_name = selected_players[0]
            player_data = filtered_df[filtered_df['batsman'] == player_name]
            
            if not player_data.empty:
                col1, col2, col3, col4 = st.columns(4)
                with col1: st.metric("Total Runs", f"{player_data['runs'].sum()}")
                with col2: st.metric("Avg Control Score", f"{player_data['control_score'].mean():.1f}/100")
                with col3: st.metric("Boundary Rate", f"{(player_data['is_boundary'].sum() / len(player_data)) * 100:.1f}%")
                with col4: st.metric("Avg False Shot Score", f"{player_data['false_shot_score'].mean():.2f}")

                st.markdown('<div class="insight-box">', unsafe_allow_html=True)
                st.markdown("### 🧠 Analyst Insights")
                insights = get_player_insights(player_data)
                
                if 'favorite_shot' in insights: st.write(f"**Favorite Shot:** {insights['favorite_shot']}")
                if 'risky_zone' in insights: st.write(f"**Potential Weakness Zone:** {insights['risky_zone']}")
                if 'efficiency_insight' in insights: st.write(f"**Efficiency Insight:** {insights['efficiency_insight']}")
                if 'bowl_to' in insights:
                    st.markdown("---")
                    st.markdown("#### 🎯 Bowling Strategy Recommendation")
                    st.markdown('<div class="bowling-recommendation">', unsafe_allow_html=True)
                    st.write(f"**To target {player_name}'s weakness:**")
                    if 'bowl_to_connection' in insights: st.write(f"Focus on bowling to the **{insights['bowl_to_connection']}** zone with these attributes:")
                    for rec in insights['bowl_to']: st.write(f"- {rec}")
                    st.markdown('</div>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
    
    with tab2:
        st.subheader("📈 Player Form Analysis")
        if len(selected_players) == 1:
            player_name = selected_players[0]
            st.markdown(f"##### {player_name} - Control Score Trend")
            form_fig = create_player_form_chart(filtered_df, player_name)
            st.plotly_chart(form_fig, use_container_width=True)
        else:
            st.info("Please select only one player to view their form trend.")

    with tab3:
        st.subheader(f"🌐 Shot Placement Intelligence for {', '.join(selected_players)}")
        for player in selected_players:
            st.markdown(f"##### {player}")
            shot_fig = create_shot_angle_heatmap(df, player)
            st.plotly_chart(shot_fig, use_container_width=True)
    
    with tab4:
        st.subheader("📊 Shot Intelligence Matrix")
        control_aggression_fig = create_control_vs_aggression_chart(filtered_df)
        st.plotly_chart(control_aggression_fig, use_container_width=True)
        
    with tab5:
        st.subheader("⏳ Performance Across Match Phases")
        match_phase_fig = create_match_phase_analysis(filtered_df)
        st.plotly_chart(match_phase_fig, use_container_width=True)

    with tab6:
        st.subheader("⚖️ Player Comparison")
        if len(selected_players) >= 2:
            radar_fig = create_player_comparison_radar(df, selected_players)
            st.plotly_chart(radar_fig, use_container_width=True)
        else:
            st.info("Please select at least two players for comparison.")
    
    with tab7:
        st.subheader("🔍 Advanced Cricket Analytics")
        adv_tab1, adv_tab2, adv_tab3, adv_tab4 = st.tabs([
            "💰 xRuns vs Actual Runs",
            "⚡ Pressure Performance", 
            "🎯 Shot Clusters",
            "📈 Wicket Probability"
        ])
        
        with adv_tab1:
            st.markdown("##### 💰 xRuns (Expected Runs) Analysis")
            if len(selected_players) == 1:
                player_name = selected_players[0]
                xruns_fig = create_xruns_comparison_chart(filtered_df, player_name)
                st.plotly_chart(xruns_fig, use_container_width=True)
            else:
                st.info("Please select only one player for xRuns analysis.")
        
        with adv_tab2:
            st.markdown("##### ⚡ Performance Under Pressure")
            pressure_fig = create_pressure_performance_analysis(filtered_df)
            st.plotly_chart(pressure_fig, use_container_width=True)
        
        with adv_tab3:
            st.markdown("##### 🎯 Shot Clusters (AI-powered analysis)")
            if len(selected_players) == 1:
                player_name = selected_players[0]
                cluster_fig = create_shot_cluster_chart(filtered_df, player_name)
                st.plotly_chart(cluster_fig, use_container_width=True)
            else:
                st.info("Please select only one player for shot cluster analysis.")
        
        with adv_tab4:
            st.markdown("##### 📈 Wicket Probability Index")
            risk_fig = create_wicket_probability_heatmap(filtered_df)
            st.plotly_chart(risk_fig, use_container_width=True)
    
    with tab8:
        st.subheader("🎯 Bowler Pressure Analysis")
        pressure_tab1, pressure_tab2 = st.tabs([
            "📊 Bowler Pressure Index",
            "🔍 False Shot Patterns"
        ])
        with pressure_tab1:
            st.markdown("##### 📈 Bowler Pressure Index")
            bowler_pressure_df = filtered_df.groupby('bowler').agg(
                total_false_shots=('false_shot_score', 'sum'),
                balls_bowled=('runs', 'count'),
                runs_conceded=('runs', 'sum'),
                avg_control_against=('control_score', 'mean')
            ).reset_index()
            bowler_pressure_df = bowler_pressure_df[bowler_pressure_df['balls_bowled'] >= 20]
            if not bowler_pressure_df.empty:
                bowler_pressure_df['Pressure Index'] = bowler_pressure_df['total_false_shots'] / bowler_pressure_df['balls_bowled']
                bowler_pressure_df['Economy'] = bowler_pressure_df['runs_conceded'] / bowler_pressure_df['balls_bowled']
                st.dataframe(bowler_pressure_df.sort_values('Pressure Index', ascending=False), use_container_width=True)
            else:
                st.info("Not enough bowling data to show pressure index.")
        with pressure_tab2:
            st.markdown("##### 🔍 False Shot Patterns by Bowler")
            if 'bowler' in filtered_df.columns:
                available_bowlers = filtered_df['bowler'].unique()
                selected_bowler = st.selectbox("Select a Bowler", options=sorted(available_bowlers))
                if selected_bowler:
                    bowler_data = filtered_df[filtered_df['bowler'] == selected_bowler]
                    if not bowler_data.empty and len(bowler_data) >= 10:
                        shot_vulnerability = bowler_data.groupby('battingShotTypeId').agg(
                            avg_false_shot_score=('false_shot_score', 'mean'),
                            count=('false_shot_score', 'count')
                        ).reset_index()
                        shot_vulnerability = shot_vulnerability[shot_vulnerability['count'] >= 3]
                        st.dataframe(shot_vulnerability.sort_values('avg_false_shot_score', ascending=False), use_container_width=True)
                    else:
                        st.info("Not enough data for this bowler.")
