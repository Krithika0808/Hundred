import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
from io import StringIO
from datetime import datetime

# Set page config
st.set_page_config(
    page_title="Women's Cricket Shot Intelligence Matrix",
    page_icon="🏏",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
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
    .analytical-grid {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 2rem;
        padding: 1rem;
    }
    .analytics-card {
        background: rgba(255,255,255,0.95);
        padding: 1.5rem;
        border-radius: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

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
        
        # Data cleaning
        numeric_cols = ['shotAngle', 'shotMagnitude', 'runs', 'totalBallNumber']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        bool_cols = ['isWicket', 'isBoundary', 'isAirControlled', 'isWide', 'isNoBall']
        for col in bool_cols:
            if col in df.columns:
                df[col] = df[col].astype(bool)
        string_cols = ['batsman', 'bowler', 'battingShotTypeId', 'battingConnectionId', 'commentary']
        for col in string_cols:
            if col in df.columns:
                df[col] = df[col].astype(str).str.strip()
        essential_cols = [col for col in ['batsman', 'runs', 'totalBallNumber'] if col in df.columns]
        if essential_cols:
            df = df.dropna(subset=essential_cols)
        if 'matchDate' in df.columns:
            df['matchDate'] = pd.to_datetime(df['matchDate'], errors='coerce')
            df['season'] = df['matchDate'].dt.year
        return df
    except Exception as e:
        st.error(f"❌ Data loading error: {str(e)}")
        return create_sample_data()

def create_sample_data():
    """Create sample cricket data for demonstration"""
    np.random.seed(42)
    players = ['Smriti Mandhana', 'Harmanpreet Kaur', 'Beth Mooney', 'Alyssa Healy', 'Meg Lanning']
    shot_types = ['Drive', 'Pull', 'Cut', 'Sweep', 'Flick', 'Hook', 'Reverse Sweep', 'Loft']
    connection_types = ['Middled', 'WellTimed', 'Undercontrol', 'MisTimed', 'Missed', 'HitBody']
    length_types = ['Yorker', 'Full', 'Good Length', 'Short', 'Bouncer']
    line_types = ['Off Stump', 'Middle Stump', 'Leg Stump', 'Wide Outside Off', 'Down Leg']
    bowling_types = ['Fast', 'Medium', 'Spin', 'Swing', 'Seam']
    bowling_from = ['Over the Wicket', 'Around the Wicket']
    bowling_hands = ['Right Arm', 'Left Arm']
    fielding_positions = ['Cover', 'Mid Wicket', 'Point', 'Third Man', 'Fine Leg', 'Square Leg', 'Long On', 'Long Off']
    
    n_rows = 1000
    data = {
        'batsman': np.random.choice(players, n_rows),
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
        'fieldingPosition': np.random.choice(fielding_positions, n_rows)
    }
    df = pd.DataFrame(data)
    
    # Data adjustments
    df.loc[df['runs'] == 4, 'isBoundary'] = True
    df.loc[df['runs'] == 6, 'isBoundary'] = True
    df.loc[df['runs'] < 4, 'isBoundary'] = False
    df['commentary'] = df.apply(generate_commentary, axis=1)
    return df

def calculate_shot_intelligence_metrics(df):
    """Calculate advanced shot intelligence metrics"""
    if df.empty:
        return df
    
    df = df.copy()
    df['shotAngle'] = df['shotAngle'].fillna(0)
    df['shotMagnitude'] = df['shotMagnitude'].fillna(100)
    df['isAirControlled'] = df['isAirControlled'].fillna(False)
    df['battingConnectionId'] = df['battingConnectionId'].fillna('Unknown')
    df['commentary'] = df['commentary'].fillna('')
    
    # Boundary detection
    df['is_boundary'] = df.apply(detect_boundary_from_data, axis=1)
    
    # Control categories
    df['true_control_category'] = df.apply(determine_true_control, axis=1)
    
    # Control score calculation
    control_scores = {
        'Middled': 3, 'WellTimed': 3, 'Undercontrol': 3,
        'MisTimed': 2, 'Missed': 0, 'HitBody': 0.5,
        'TopEdge': 2, 'BatPad':1, 'BottomEdge':2,
        'Gloved': 2, 'HitHelmet': 0, 'InsideEdge': 2,
        'LeadingEdge': 2, 'Left': 3, 'mis-timed': 2,
        'OutsideEdge': 2, 'Padded': 0, 'PlayAndMiss': 0
    }
    df['control_quality_score'] = df['battingConnectionId'].map(control_scores).fillna(1.5)
    
    # Angle zones
    angle_bins = [0, 45, 90, 135, 180, 225, 270, 315, 360]
    angle_labels = ['Long Off', 'Cover', 'Point', 'Third Man', 'Fine Leg', 'Square Leg', 'Mid Wicket', 'Long On']
    df['angle_zone'] = pd.cut(df['shotAngle'], bins=angle_bins, labels=angle_labels, include_lowest=True)
    
    # Hybrid control score
    df['control_score'] = df['control_quality_score'] * 33.33
    df['control_score'] += df['runs'] * 5
    df['control_score'] += df['is_boundary'] * 20
    df['control_score'] += df['angle_zone'].isin(['Cover', 'Mid Wicket', 'Long On', 'Long Off']) * 10
    df['control_score'] = df['control_score'].clip(0, 100)
    
    # Control flags
    df['is_controlled_shot'] = (df['control_score'] >= 50).astype(int)
    
    # Additional metrics
    df['execution_intelligence'] = df['control_quality_score'] * (df['runs'] + 1) / 2
    df['true_risk_reward'] = np.where(df['is_controlled_shot'], df['runs'] * 1.3, df['runs'] * 0.7)
    df['true_shot_efficiency'] = np.where(
        df['shotMagnitude'] > 0,
        (df['runs'] * df['control_quality_score']) / (df['shotMagnitude'] / 100),
        df['runs'] * df['control_quality_score']
    )
    df['shot_category'] = np.where(
        df['isAirControlled'], 
        'Aerial_' + df['true_control_category'],
        'Ground_' + df['true_control_category']
    )
    
    # Match phases
    if 'totalBallNumber' in df.columns:
        max_ball = df['totalBallNumber'].max()
        if max_ball <= 25:
            df['match_phase'] = pd.cut(df['totalBallNumber'], bins=[0, 10, 25], labels=['Powerplay', 'Death'])
        elif max_ball <= 50:
            df['match_phase'] = pd.cut(df['totalBallNumber'], bins=[0, 25, 50], labels=['Powerplay', 'Middle'])
        else:
            df['match_phase'] = pd.cut(df['totalBallNumber'], bins=[0, 25, 75, max_ball], 
                         labels=['Powerplay', 'Middle', 'Death'])
    return df

def get_player_insights(player_data):
    """Generate player-specific insights"""
    insights = {}
    
    # Favorite shot
    if 'battingShotTypeId' in player_data.columns:
        shot_counts = player_data['battingShotTypeId'].value_counts()
        if not shot_counts.empty:
            insights['favorite_shot'] = f"{shot_counts.index[0]} ({(shot_counts.iloc[0]/len(player_data)*100):.1f}%)"
    
    # Dismissal patterns
    if 'angle_zone' in player_data.columns and 'true_control_category' in player_data.columns:
        poor_control = player_data[player_data['true_control_category'].isin(['Less Control', 'Poor Control'])]
        if not poor_control.empty:
            dismissal_zones = poor_control['angle_zone'].mode()[0]
            insights['dismissal_zone'] = f"{dismissal_zone} ({(len(poor_control[poor_control['angle_zone'] == dismissal_zone])/len(poor_control)*100):.1f}%)"
    
    # Bowling weaknesses
    bowling_cols = ['lengthTypeId', 'lineTypeId', 'bowlingTypeId']
    if any(col in player_data.columns for col in bowling_cols):
        weaknesses = {}
        for col in bowling_cols:
            if col in player_data.columns:
                group_means = player_data.groupby(col)['control_score'].mean()
                min_score = group_means.min()
                if min_score < 60:
                    weaknesses[col] = f"{group_means.idxmin()}: {min_score:.1f}/100"
        if weaknesses:
            insights['bowling_weaknesses'] = weaknesses
    
    # Strength area
    if 'angle_zone' in player_data.columns:
        strong_zone = player_data.groupby('angle_zone')['control_score'].mean().idxmax()
        insights['strength_zone'] = f"{strong_zone} ({player_data[player_data['angle_zone'] == strong_zone]['control_score'].mean():.1f}/100)"
    
    return insights

def create_shot_angle_heatmap(df, player_name):
    """Create polar plot of shot distribution"""
    player_data = df[df['batsman'] == player_name]
    if player_data.empty:
        return go.Figure()
    
    fig = go.Figure()
    hover_text = []
    for _, row in player_data.iterrows():
        hover_text.append(f"Shot: {row['battingShotTypeId']}<br>Runs: {row['runs']}<br>Control: {row['control_score']:.1f}/100")
    
    fig.add_trace(go.Scatterpolar(
        r=player_data['shotMagnitude'],
        theta=player_data['shotAngle'],
        mode='markers',
        marker=dict(
            size=player_data['runs'] * 4,
            color=player_data['control_score'],
            colorscale='RdYlGn',
            cmin=0, cmax=100
        ),
        text=hover_text,
        hovertemplate='%{text}'
    ))
    fig.update_layout(
        polar=dict(
            radialaxis=dict(range=[0, player_data['shotMagnitude'].max()*1.1]),
            angularaxis=dict(tickvals=[0, 45, 90, 135, 180, 225, 270, 315])
        ),
        title=f"{player_name} Shot Map",
        height=600
    )
    return fig

def create_control_matrix(df):
    """Create control vs aggression matrix"""
    analysis = df.groupby('battingShotTypeId').agg({
        'is_controlled_shot': 'mean',
        'runs': 'mean',
        'control_score': 'mean',
        'shot_count': 'size'
    }).reset_index()
    analysis = analysis[analysis['shot_count'] >= 5]
    
    fig = px.scatter(
        analysis,
        x='is_controlled_shot',
        y='runs',
        size='shot_count',
        color='control_score',
        text='battingShotTypeId',
        title='Control vs Aggression Matrix',
        labels={
            'is_controlled_shot': 'Control Rate',
            'runs': 'Avg Runs',
            'control_score': 'Control Quality'
        },
        color_continuous_scale='RdYlGn'
    )
    fig.add_hline(y=analysis['runs'].mean(), line_dash="dash", line_color="gray")
    fig.add_vline(x=analysis['is_controlled_shot'].mean(), line_dash="dash", line_color="gray")
    return fig

def create_phase_analysis(df):
    """Match phase performance analysis"""
    analysis = df.groupby(['match_phase', 'battingShotTypeId']).agg({
        'runs': 'mean',
        'is_controlled_shot': 'mean'
    }).reset_index()
    return px.bar(
        analysis,
        x='battingShotTypeId',
        y='runs',
        color='match_phase',
        barmode='group',
        title='Phase-wise Shot Performance'
    )

def main():
    """Main application logic"""
    df = load_data_from_github()
    if df.empty:
        st.error("No data available")
        return
    
    df = calculate_shot_intelligence_metrics(df)
    
    # Sidebar
    st.sidebar.header("Filters")
    selected_players = st.sidebar.multiselect(
        "Choose Batters",
        df['batsman'].unique(),
        default=df['batsman'].unique()[:2]
    )
    
    # Main tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "🎯 Shot Patterns",
        "⚡ Control Analysis",
        "📊 Phase Performance",
        "🔍 Player Insights"
    ])
    
    # Shot Patterns
    with tab1:
        st.subheader("360° Shot Map")
        col1, col2 = st.columns([2, 1])
        with col1:
            if selected_players:
                player = st.selectbox("Select Player", selected_players)
                fig = create_shot_angle_heatmap(df, player)
                st.plotly_chart(fig)
        with col2:
            st.markdown("""
            **Legend:**
            - Green: Control >80
            - Orange: Control 50-79
            - Red: Control <50
            - Size: Runs scored
            """)
    
    # Control Analysis
    with tab2:
        st.subheader("Control vs Aggression")
        fig = create_control_matrix(df)
        st.plotly_chart(fig)
    
    # Phase Analysis
    with tab3:
        st.subheader("Match Phase Performance")
        fig = create_phase_analysis(df)
        st.plotly_chart(fig)
    
    # Player Insights
    with tab4:
        st.subheader("Player Comparisons")
        if len(selected_players) >= 2:
            radar_data = []
            for player in selected_players[:4]:
                pdata = df[df['batsman'] == player].iloc[0]
                radar_data.append({
                    'Player': player,
                    'Control Rate': pdata['is_controlled_shot']*100,
                    'Avg Runs': pdata['runs'],
                    'Boundary %': data['is_boundary']*100,
                    'Control Score': data['control_score']
                })
            
            radar_df = pd.DataFrame(radar_data)
            fig = go.Figure()
            for i, player in enumerate(selected_players[:4]):
                values = radar_df[radar_df['Player'] == player].values[0][1:]
                fig.add_trace(go.Scatterpolar(
                    r=values.tolist() + [values[0]],
                    theta=['Control', 'Runs', 'Boundaries', 'Control Score', 'Control'],
                    name=player,
                    marker_color=px.colors.qualitative.Plotly[i]
                ))
            fig.update_layout(polar=dict(radialaxis=dict(range=[0, 100])))
            st.plotly_chart(fig)
        else:
            st.warning("Select at least 2 players for comparison")

if __name__ == "__main__":
    main()
