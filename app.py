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

# Custom CSS
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
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data_from_github(github_url=None):
    """Load cricket data from GitHub repository with error handling"""
    try:
        if github_url is None:
            github_url = "https://raw.githubusercontent.com/Krithika0808/Hundred/main/Hundred.csv"
        
        response = requests.get(github_url)
        response.raise_for_status()
        
        csv_content = StringIO(response.text)
        df = pd.read_csv(csv_content)
        
        # Essential columns check
        required_columns = ['batsman', 'runs', 'totalBallNumber', 'shotAngle', 'commentary']
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing critical columns: {missing_cols}")
        
        # Data cleaning pipeline
        df = df.pipe        for col in required_columns:
            if col in df.columns:
                if col in ['runs', 'totalBallNumber', 'shotAngle']:
                    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
                elif col == 'commentary':
                    df[col] = df[col].astype(str).str.strip().fillna('')
                else:
                    df[col] = df[col].astype(str).str.strip()
        
        # Generate boundary column
        df['is_boundary'] = df.apply(
            lambda x: 
                1 if x['runs'] in [4,6] else 
                (1 if x['isBoundary'] else 
                 any(keyword in str(x['commentary']).lower() 
                 for keyword in ['four', 'six', 'boundary'])
            else 0
        )
        
        return df
    
    except Exception as e:
        st.error(f"❌ Data Load Error: {str(e)}")
        return create_sample_data()

def create_sample_data():
    """Generate comprehensive sample data with all required columns"""
    np.random.seed(42)
    
    players = ['Smriti Mandhana', 'Harmanpreet Kaur', 'Beth Mooney', 
             'Alyssa Healy', 'Meg Lanning']
    shot_types = ['Drive', 'Pull', 'Cut', 'Sweep', 'Flick', 'Hook', 
                'Reverse Sweep', 'Loft', 'Defended', 'Cover Drive']
    connection_types = ['Middled', 'WellTimed', 'Undercontrol', 
                   'MisTimed', 'Edge', 'Body', 'Top Edge']
    
    # Bowling parameters
    length_types = ['Yorker', 'Full', 'Good Length', 'Short', 'Bouncer']
    line_types = ['Off', 'Middle', 'Leg', 'Wide Off', 'Down Leg']
    bowling_types = ['Fast', 'Medium', 'Spin', 'Swing', 'Seam']
    fielding_zones = ['Cover', 'Point', 'Mid Wicket', 'Third Man', 
                'Fine Leg', 'Square Leg', 'Long On', 'Long Off']
    
    n_rows = 2000
    
    data = {
        'batsman': np.random.choice(players, n_rows),
        'battingShotTypeId': np.random.choice(shot_types, n_rows),
        'battingConnectionId': np.random.choice(connection_types, n_rows, 
                           p=[0.3, 0.25, 0.2, 0.15, 0.08, 0.07]),
        'runs': np.random.choice([0,1,2,3,4,6], n_rows, 
                        p=[0.3, 0.25, 0.2, 0.1, 0.1, 0.05]),
        'totalBallNumber': np.random.randint(1, 101, n_rows),
        'shotAngle': np.random.uniform(0, 360, n_rows),
        'shotMagnitude': np.random.uniform(50, 200, n_rows),
        'isAirControlled': np.random.choice([True, False], n_rows, p=[0.3, 0.7]),
        'commentary': np.random.choice(
            ['Good shot!', 'Perfect timing!', 'Mistimed!', 'Great connection!'] * 3 
            + ['Boundary!', 'Six!', 'Edge!', 'Body hit!'] * 2, 
            n_rows
        ),
        'fixtureId': np.random.randint(1, 21, n_rows),
        'battingTeam': np.random.choice(['Team A', 'Team B', 'Team C', 'Team D'], n_rows),
        'matchDate': pd.to_datetime(
            pd.to_datetime('2023-01-01') + 
            pd.to_timedelta(np.random.randint(0, 365, n_rows), unit='d')
        ),
        'lengthTypeId': np.random.choice(length_types, n_rows),
        'lineTypeId': np.random.choice(line_types, n_rows),
        'bowlingTypeId': np.random.choice(bowling_types, n_rows),
        'fieldingPosition': np.random.choice(fielding_zones, n_rows)
    }
    
    df = pd.DataFrame(data)
    
    # Boundary logic
    df.loc[df['runs'].isin([4,6]), 'is_boundary'] = 1
    df.loc[df['runs'] < 4, 'is_boundary'] = 0
    df['is_boundary'] = df['is_boundary'].astype(bool)
    
    # Commentary enhancement
    df['commentary'] = df.apply(
        lambda x: f"SIX! {x['battingShotTypeId']}" if x['runs'] == 6 
        else f"FOUR! {x['battingShotTypeId']}" if x['runs'] == 4 
        else f"{x['battingShotTypeId']} for {x['runs']} runs", 
        axis=1
    )
    
    return df

def calculate_shot_intelligence_metrics(df):
    """Comprehensive metrics calculation with column validation"""
    required_cols = ['batsman', 'runs', 'totalBallNumber', 'shotAngle', 
                 'battingConnectionId', 'is_boundary', 'commentary']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    df = df.copy()
    
    # Control quality scoring
    control_map = {
        'Middled': 3, 'WellTimed': 3, 'Undercontrol': 3,
        'MisTimed': 2, 'Edge': 2, 'Body': 1,
        'Top Edge': 2, 'PlayAndMiss': 0
    }
    df['control_quality'] = df['battingConnectionId'].map(control_map).fillna(1.5)
    
    # Angle zones
    df['angle_zone'] = pd.cut(df['shotAngle'], 
        bins=[0,45,90,135,180,225,270,315,360],
        labels=['Long Off', 'Cover', 'Point', 'Third Man', 
               'Fine Leg', 'Square Leg', 'Mid Wicket', 'Long On'],
        include_lowest=True
    )
    
    # Hybrid control score
    df['control_score'] = (
        df['control_quality'] * 33.33 +  # Base control
        df['runs'] * 5 +             # Runs multiplier
        df['is_boundary'] * 20 +      # Boundary bonus
        (df['angle_zone'].isin(['Cover', 'Mid Wicket', 'Long On', 'Long Off']) * 10)  # Strategic zones
    ).clip(0, 100)
    
    # Control flags
    df['is_controlled'] = df['control_score'] >= 50
    
    # Shot efficiency
    df['shot_efficiency'] = np.where(
        df['shotMagnitude'] > 0,
        (df['runs'] * df['control_quality']) / (df['shotMagnitude'] / 100),
        df['runs'] * df['control_quality']
    )
    
    # Match phases
    max_balls = df['totalBallNumber'].max()
    if max_balls <= 25:
        phases = ['Powerplay']
    elif max_balls <= 50:
        phases = ['Powerplay', 'Middle']
    else:
        phases = ['Powerplay', 'Middle', 'Death']
    
    df['match_phase'] = pd.cut(
        df['totalBallNumber'],
        bins=[0, 25, 50, max_balls],
        labels=phases,
        include_lowest=True
    )
    
    return df

def get_player_insights(player_data):
    """Generate player insights with comprehensive analysis"""
    insights = {}
    
    # Favorite shot analysis
    if 'battingShotTypeId' in player_data.columns:
        shot_counts = player_data['battingShotTypeId'].value_counts()
        if not shot_counts.empty:
            insights['favorite_shot'] = f"{shot_counts.index[0]} ({(shot_counts[0]/len(player_data)*100):.1f}%)"
    
    # Dismissal analysis
    if 'fieldingPosition' in player_data.columns and 'control_score' in player_data.columns:
        dismissals = player_data[player_data['isWicket'] == True]
        if not dismissals.empty:
            dismissal_zone = dismissals['fieldingPosition'].mode()[0]
            dismissal_rate = (len(dismissals) / len(player_data)) * 100
            insights['dismissal_zone'] = f"{dismissal_zone} ({dismissal_rate:.1f}%)"
    
    # Bowling weaknesses
    bowling_cols = ['lengthTypeId', 'lineTypeId', 'bowlingTypeId']
    weaknesses = {}
    for col in bowling_cols:
        if col in player_data.columns:
            group_means = player_data.groupby(col)['control_score'].mean()
            if not group_means.empty:
                min_score = group_means.min()
                if min_score < 60:
                    weaknesses[col] = f"{group_means.idxmin()}: {min_score:.1f}/100"
    if weaknesses:
        insights['bowling_weaknesses'] = weaknesses
    
    # Strength zones
    if 'angle_zone' in player_data.columns:
        strong_zone = player_data.groupby('angle_zone')['control_score'].mean().idxmax()
        insights['strength_zone'] = f"{strong_zone} ({player_data[player_data['angle_zone'] == strong_zone]['control_score'].mean():.1f}/100)"
    
    return insights

def create_shot_map(df, player_name):
    """Advanced 360° shot map visualization"""
    player_df = df[df['batsman'] == player_name].copy()
    if player_df.empty:
        return go.Figure()
    
    # Data validation
    required_cols = ['shotAngle', 'shotMagnitude', 'runs', 'control_score']
    missing_cols = [col for col in required_cols if col not in player_df.columns]
    if missing_cols:
        st.error(f"Missing map data for {player_name}: {missing_cols}")
        return go.Figure()
    
    # Create figure
    fig = go.Figure()
    hover_text = []
    
    for _, row in player_df.iterrows():
        hover_text.append(
            f"Shot: {row['battingShotTypeId']}<br>"
            f"Runs: {row['runs']}<br>"
            f"Control: {row['control_score']:.1f}/100<br>"
            f"Connection: {row['battingConnectionId']}"
        )
    
    fig.add_trace(go.Scatterpolar(
        r=player_df['shotMagnitude'],
        theta=player_df['shotAngle'],
        mode='markers',
        marker=dict(
            size=player_df['runs'] * 3 + 8,
            color=player_df['control_score'],
            colorscale='RdYlGn',
            cmin=0,
            cmax=100,
            line=dict(width=1, color='white')
        ),
        text=hover_text,
        hovertemplate='%{text}<extra></extra>',
        name='Shots'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, player_df['shotMagnitude'].max() * 1.1],
                title="Shot Distance (cm)"
            ),
            angularaxis=dict(
                tickvals=[0,45,90,135,180,225,270,315],
                ticktext=['Long Off', 'Cover', 'Point', 'Third Man', 
                       'Fine Leg', 'Square Leg', 'Mid Wicket', 'Long On'],
                direction='clockwise'
            )
        ),
        title=f"{player_name}'s Shot Map Analysis",
        height=600,
        margin=dict(l=0, r=0, b=0, t=40)
    )
    return fig

def create_control_matrix(df):
    """Control vs aggression matrix with quadrant analysis"""
    analysis = df.groupby('battingShotTypeId').agg({
        'is_controlled': 'mean',
        'runs': 'mean',
        'control_score': 'mean',
        'shot_count': 'size'
    }).reset_index()
    analysis = analysis[analysis['shot_count'] >= 10]
    
    fig = px.scatter(
        analysis,
        x='is_controlled',
        y='runs',
        size='shot_count',
        color='control_score',
        text='battingShotTypeId',
        title='Shot Control vs Aggression Matrix',
        labels={
            'is_controlled': 'Control Rate',
            'runs': 'Avg Runs/Shot',
            'control_score': 'Control Quality'
        },
        color_continuous_scale='RdYlGn',
        range_color=[0, 100]
    )
    
    # Add reference lines
    fig.add_hline(y=analysis['runs'].mean(), line_dash="dash", line_color="gray")
    fig.add_vline(x=analysis['is_controlled'].mean(), line_dash="dash", line_color="gray")
    
    # Quadrant labels
    fig.add_annotation(
        x=0.1, y=analysis['runs'].max(),
        text="High Risk/Low Reward",
        showarrow=False,
        font=dict(color="red"),
        align="left"
    )
    fig.add_annotation(
        x=0.9, y=analysis['runs'].max(),
        text="Smart Cricket/High Reward",
        showarrow=False,
        font=dict(color="green"),
        align="right"
    )
    fig.add_annotation(
        x=0.1, y=analysis['runs'].min(),
        text="Poor Execution/Low Reward",
        showarrow=False,
        font=dict(color="orange"),
        align="left"
    )
    fig.add_annotation(
        x=0.9, y=analysis['runs'].min(),
        text="Defensive/Controlled",
        showarrow=False,
        font=dict(color="blue"),
        align="right"
    )
    return fig

def create_phase_analysis(df):
    """Match phase analysis with interactive controls"""
    if 'match_phase' not in df.columns:
        return go.Figure()
    
    phase_data = df.groupby(['match_phase', 'battingShotTypeId']).agg({
        'runs': 'mean',
        'is_controlled': 'mean',
        'boundary': 'mean'
    }).reset_index()
    
    fig = px.bar(
        phase_data,
        x='battingShotTypeId',
        y='runs',
        color='match_phase',
        barmode='group',
        hover_data=['is_controlled', 'boundary'],
        title='Phase-wise Shot Performance',
        labels={'runs': 'Avg Runs', 'battingShotTypeId': 'Shot Type'}
    )
    fig.update_layout(
        xaxis_tickangle=-45,
        hovermode='x unified',
        legend=dict(orientation='bottom center')
    )
    return fig

def create_radar_comparison(players):
    """Player comparison radar chart"""
    if len(players) < 2:
        return go.Figure()
    
    radar_data = []
    for player in players:
        pdata = df[df['batsman'] == player].iloc[0]
        radar_data.append({
            'Player': player,
            'Control Rate': pdata['is_controlled'] * 100,
            'Avg Runs': pdata['runs'],
            'Boundary %': pdata['boundary'] * 100,
            'Control Score': pdata['control_score'],
            'Efficiency': (data['runs'] * data['control_score']) / (data['shotMagnitude'] / 100) if data['shotMagnitude'] > 0 else 0
        })
    
    radar_df = pd.DataFrame(radar_data)
    
    fig = go.Figure()
    categories = ['Control Rate', 'Avg Runs', 'Boundary %', 'Control Score', 'Efficiency']
    
    for i, player in enumerate(players):
        player_data = radar_df[radar_df['Player'] == player].iloc[0]
        values = [player_data[cat] for cat in categories] + [player_data[categories[0]]  # Close the polygon
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=categories + [categories[0]],
            fill='toself',
            name=player,
            line_color=px.colors.qualitative.Plotly[i],
            opacity=0.3
        ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(range=[0, 100], visible=True),
            angularaxis=dict(tickmode='array', visible=True)
        ),
        title="Player Comparison Radar Chart",
        height=500
    )
    return fig

def main():
    """Main application with proper error handling"""
    st.markdown('<h1 class="main-header">Women\'s Cricket Shot Intelligence Matrix</h1>', unsafe_allow_html=True)
    
    # Load and process data
    df = load_data_from_github()
    if df.empty:
        st.error("No data available. Check GitHub connection and data format.")
        return
    
    try:
        df = calculate_shot_intelligence_metrics(df)
    except Exception as e:
        st.error(f"Data processing failed: {str(e)}")
        return
    
    # Sidebar configuration
    st.sidebar.header("🎯 Analysis Controls")
    selected_players = st.sidebar.multiselect(
        "Select Players for Analysis",
        df['batsman'].unique(),
        default=df['batsman'].unique()[:2]
    )
    
    # Main dashboard
    tab1, tab2, tab3, tab4 = st.tabs([
        "🎯 Shot Maps",
        "⚡ Control Analysis",
        "📊 Phase Performance",
        "🔍 Player Comparisons"
    ])
    
    # Shot Maps Tab
    with tab1:
        st.subheader("360° Shot Placement Analysis")
        col1, col2 = st.columns([2, 1])
        with col1:
            if selected_players:
                player = st.selectbox("Choose Player", selected_players)
                fig = create_shot_map(df, player)
                st.plotly_chart(fig)
        with col2:
            st.markdown("**Legend:**")
            st.markdown("""
            🔵 Green: Control >80
            🟡 Orange: Control 50-79
            🔴 Red: Control <50
            📏 Size: Runs scored
            """)
    
    # Control Analysis Tab
    with tab2:
        st.subheader("Control vs Aggression Matrix")
        fig = create_control_matrix(df)
        st.plotly_chart(fig)
    
    # Phase Analysis Tab
    with tab3:
        st.subheader("Match Phase Performance")
        fig = create_phase_analysis(df)
        st.plotly_chart(fig)
    
    # Player Comparisons Tab
    with tab4:
        st.subheader("Player Radar Comparison")
        if len(selected_players) >= 2:
            fig = create_radar_comparison(selected_players)
            st.plotly_chart(fig)
        else:
            st.warning("Select at least 2 players for comparison")
    
    # Advanced Analytics (Right Panel)
    with st.expander("🔍 Advanced Analytics", expanded=False):
        st.subheader("Dismissal Pattern Analysis")
        if selected_players:
            selected_batter = st.selectbox("Choose Batter", selected_players)
            dismissals = df[(df['batsman'] == selected_batter) & (df['isWicket'] == True)]
            if not dismissals.empty:
                st.subheader("Dismissal Hotspots")
                zone_counts = dismissals['fieldingPosition'].value_counts().reset_index()
                fig = px.bar(zone_counts, x='fieldingPosition', y='dismissals')
                st.plotly_chart(fig)
    
                st.subheader("Bowling Strategy")
                heatmap_data = dismissals.groupby(['lineTypeId', 'lengthTypeId']).size().unstack(fill_value=0)
                fig = go.Figure(
                    data=go.Heatmap(
                        z=heatmap_data.values,
                        x=heatmap_data.columns,
                        y=heatmap_data.index,
                        colorscale='Reds'
                    )
                )
                st.plotly_chart(fig)
            else:
                st.info("No dismissal data for selected batter")
        else:
            st.warning("Select a player for dismissal analysis")

if __name__ == "__main__":
    main()
