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
    .stmetric > div[data-testid="metric-container"] {
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
    """Load cricket data from GitHub with comprehensive error handling"""
    try:
        if github_url is None:
            github_url = "https://raw.githubusercontent.com/Krithika0808/Hundred/main/Hundred.csv"
        response = requests.get(github_url)
        response.raise_for_status()
        csv_content = StringIO(response.text)
        df = pd.read_csv(csv_content)
        
        required_cols = {'batsman', 'runs', 'totalBallNumber', 'shotAngle', 'commentary'}
        missing_cols = required_cols - set(df.columns)
        if missing_cols:
            raise ValueError(f"Missing critical columns: {missing_cols}")
        
        df = df.astype({
            'runs': 'float',
            'totalBallNumber': 'int',
            'shotAngle': 'float',
            'isWicket': 'bool',
            'isBoundary': 'bool',
            'isAirControlled': 'bool'
        })
        
        df['is_boundary'] = df.apply(
            lambda x: 1 if x['runs'] in [4,6] else 
            (1 if x['isBoundary'] else 
             any(keyword in str(x['commentary']).lower() 
             for keyword in ['four', 'six', 'boundary'])
            else 0
        )
        return df
    except Exception as e:
        st.error(f"❌ Data Load Failed: {str(e)}")
        return create_sample_data()

def create_sample_data():
    """Generate realistic sample data with all required fields"""
    np.random.seed(42)
    players = ['Smriti Mandhana', 'Harmanpreet Kaur', 'Beth Mooney', 
             'Alyssa Healy', 'Meg Lanning']
    shot_types = ['Drive', 'Pull', 'Cut', 'Sweep', 'Flick', 'Hook', 'Reverse Sweep', 'Loft']
    connection_types = ['Middled', 'WellTimed', 'Undercontrol', 'MisTimed', 'Edge', 'Body']
    
    data = {
        'batsman': np.random.choice(players, 1000),
        'battingShotTypeId': np.random.choice(shot_types, 1000),
        'battingConnectionId': np.random.choice(connection_types, 1000, p=[0.3, 0.25, 0.2, 0.15, 0.1]),
        'runs': np.random.choice([0,1,2,3,4,6], 1000, p=[0.3, 0.25, 0.2, 0.1, 0.1, 0.05]),
        'totalBallNumber': np.random.randint(1, 101, 1000),
        'shotAngle': np.random.uniform(0, 360, 1000),
        'shotMagnitude': np.random.uniform(50, 200, 1000),
        'isAirControlled': np.random.choice([True, False], 1000, p=[0.3, 0.7]),
        'commentary': np.random.choice(
            ['Good shot!', 'Perfect timing!', 'Mistimed!', 'Great connection!'] * 250, 
            1000
        ),
        'fixtureId': np.random.randint(1, 21, 1000),
        'battingTeam': np.random.choice(['Team A', 'Team B', 'Team C', 'Team D'], 1000),
        'matchDate': pd.to_datetime(
            pd.to_datetime('2023-01-01') + 
            pd.to_timedelta(np.random.randint(0, 365, 1000), unit='d')
        ),
        'lengthTypeId': np.random.choice(['Yorker', 'Full', 'Good Length', 'Short', 'Bouncer'], 1000),
        'lineTypeId': np.random.choice(['Off', 'Middle', 'Leg', 'Wide', 'Down'], 1000),
        'bowlingTypeId': np.random.choice(['Fast', 'Medium', 'Spin', 'Swing', 'Seam'], 1000),
        'fieldingPosition': np.random.choice(
            ['Cover', 'Point', 'Mid Wicket', 'Third Man', 
            'Fine Leg', 'Square Leg', 'Long On', 'Long Off'], 
            1000
        )
    }
    
    df = pd.DataFrame(data)
    df.loc[df['runs'] == 4, 'isBoundary'] = True
    df.loc[df['runs'] == 6, 'isBoundary'] = True
    df.loc[df['runs'] < 4, 'isBoundary'] = False
    return df

def calculate_shot_intelligence_metrics(df):
    """Comprehensive metrics calculation with validation"""
    required_cols = {'batsman', 'runs', 'totalBallNumber', 'shotAngle', 
               'battingConnectionId', 'is_boundary'}
    missing_cols = required_cols - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing metrics columns: {missing_cols}")
    
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
        df['control_quality'] * 33.33 +
        df['runs'] * 5 +
        df['is_boundary'] * 20 +
        (df['angle_zone'].isin(['Cover', 'Mid Wicket', 'Long On', 'Long Off']) * 10)
    ).clip(0, 100)
    
    df['is_controlled'] = df['control_score'] >= 50
    
    # Efficiency metrics
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
    
    df['match_phase'] = pd.cut(df['totalBallNumber'],
        bins=[0,25,50,max_balls],
        labels=phases,
        include_lowest=True
    )
    
    return df

def get_player_insights(player_data):
    """Generate player insights with defensive patterns"""
    insights = {}
    
    # Favorite shot
    if 'battingShotTypeId' in player_data.columns:
        shot_counts = player_data['battingShotTypeId'].value_counts()
        if not shot_counts.empty:
            insights['favorite_shot'] = f"{shot_counts.index[0]} ({(shot_counts[0]/len(player_data)*100):.1f}%)"
    
    # Dismissal zones
    if 'fieldingPosition' in player_data.columns:
        dismissals = player_data[player_data['isWicket'] == True]
        if not dismissals.empty:
            zone_counts = dismissals['fieldingPosition'].value_counts()
            if not zone_counts.empty:
                insights['dismissal_zone'] = f"{zone_counts.index[0]} ({(zone_counts[0]/len(dismissals)*100):.1f}%)"
    
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
    
    return insights

def create_shot_map(df, player_name):
    """Advanced 360° shot visualization"""
    player_df = df[df['batsman'] == player_name].copy()
    if player_df.empty:
        return go.Figure()
    
    required_cols = ['shotAngle', 'shotMagnitude', 'runs', 'control_score']
    for col in required_cols:
        if col not in player_df.columns:
            player_df[col] = 0  # Fallback values
    
    hover_text = []
    for _, row in player_df.iterrows():
        hover_text.append(
            f"Shot: {row['battingShotTypeId']}<br>"
            f"Runs: {row['runs']}<br>"
            f"Control: {row['control_score']:.1f}/100"
        )
    
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=player_df['shotMagnitude'],
        theta=player_df['shotAngle'],
        mode='markers',
        marker=dict(
            size=player_df['runs'] * 3 + 8,
            color=player_df['control_score'],
            colorscale='RdYlGn',
            cmin=0,
            cmax=100
        ),
        text=hover_text,
        hovertemplate='%{text}<extra></extra>',
        name='Shots'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                range=[0, player_df['shotMagnitude'].max()*1.1],
                title="Shot Distance (cm)"
            ),
            angularaxis=dict(
                tickvals=[0,45,90,135,180,225,270,315],
                ticktext=['Long Off', 'Cover', 'Point', 'Third Man', 
                       'Fine Leg', 'Square Leg', 'Mid Wicket', 'Long On']
            )
        ),
        title=f"{player_name} Shot Map Analysis",
        height=600
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
        title='Control vs Aggression Matrix',
        labels={
            'is_controlled': 'Control Rate',
            'runs': 'Avg Runs/Shot',
            'control_score': 'Control Quality'
        },
        color_continuous_scale='RdYlGn',
        range_color=[0, 100]
    )
    fig.add_hline(y=analysis['runs'].mean(), line_dash="dash", line_color="gray")
    fig.add_vline(x=analysis['is_controlled'].mean(), line_dash="dash", line_color="gray")
    return fig

def create_phase_analysis(df):
    """Match phase performance analysis"""
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
        title='Phase-wise Shot Performance',
        labels={'runs': 'Avg Runs', 'battingShotTypeId': 'Shot Type'}
    )
    fig.update_layout(xaxis_tickangle=-45)
    return fig

def create_radar_comparison(players):
    """Multi-player radar comparison"""
    if len(players) < 2:
        return go.Figure()
    
    radar_data = []
    for player in players:
        pdata = df[df['batsman'] == player].iloc[0]
        radar_data.append({
            'Player': player,
            'Control Rate': pdata['is_controlled'] * 100,
            'Avg Runs': pdata['runs'],
            'Boundary %': data['boundary'] * 100,
            'Control Score': data['control_score']
        })
    
    radar_df = pd.DataFrame(radar_data)
    
    fig = go.Figure()
    categories = ['Control Rate', 'Avg Runs', 'Boundary %', 'Control Score']
    
    for i, player in enumerate(players):
        values = radar_df[radar_df['Player'] == player][categories].values[0].tolist() + [radar_df[radar_df['Player'] == player][categories].values[0][0]]
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=categories + [categories[0]],
            fill='toself',
            name=player,
            line_color=px.colors.qualitative.Plotly[i]
        ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(range=[0, 100],
            visible=True),
            angularaxis=dict(tickmode='array')
        ),
        title="Player Comparison Radar Chart",
        height=500
    )
    return fig

def create_dismissal_analysis(df, selected_batter):
    """Advanced dismissal pattern analysis"""
    dismissals = df[(df['batsman'] == selected_batter) & (df['isWicket'] == True)].copy()
    
    if dismissals.empty:
        return pd.DataFrame(), go.Figure(), go.Figure(), go.Figure()
    
    # Timing mapping
    timing_map = {
        'WellTimed': 'Well Timed',
        'Undercontrol': 'Controlled',
        'Missed': 'Missed',
        'Edge': 'Edged',
        'NotApplicable': 'Unknown'
    }
    dismissals['Timing'] = dismissals['battingConnectionId'].map(timing_map).fillna('Unknown')
    
    # Summary table
    summary = dismissals.groupby(
        ['fieldingPosition', 'lineTypeId', 'lengthTypeId', 'bowlingTypeId', 'Timing']
    ).size().reset_index(name='dismissals')
    
    # Fielding position chart
    fig1 = px.bar(
        dismissals['fieldingPosition'].value_counts().reset_index(),
        x='fieldingPosition',
        y='dismissals',
        title='Dismissals by Fielding Position'
    )
    
    # Timing pie chart
    fig2 = px.pie(
        dismissals,
        names='Timing',
        hole=0.4,
        title='Dismissal Timing Distribution'
    )
    
    # Line/length heatmap
    heatmap_data = dismissals.groupby(['lineTypeId', 'lengthTypeId']).size().unstack(fill_value=0)
    fig3 = go.Figure(
        data=go.Heatmap(
            z=heatmap_data.values,
            x=heatmap_data.columns,
            y=heatmap_data.index,
            colorscale='Reds'
        ),
        layout=go.Layout(
            xaxis_title='Length Type',
            yaxis_title='Line Type'
        )
    )
    
    return summary, fig1, fig2, fig3

def main():
    """Main application entry point"""
    st.markdown('<h1 class="main-header">Women\'s Cricket Shot Intelligence Matrix</h1>', unsafe_allow_html=True)
    
    # Load and process data
    df = load_data_from_github()
    if df.empty:
        st.error("⚠️ No data available")
        return
    
    df = calculate_shot_intelligence_metrics(df)
    
    # Sidebar
    st.sidebar.header("🎯 Analysis Controls")
    available_players = df['batsman'].unique()
    selected_players = st.sidebar.multiselect(
        "Select Players",
        available_players,
        default=available_players[:2]
    )
    
    # Main tabs
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "🎯 Shot Maps",
        "⚡ Control Matrix",
        "📊 Phase Analysis",
        "🏆 Player Cards",
        "📈 Radar Comparison",
        "🔍 Dismissal Analysis"
    ])
    
    # Shot Maps
    with tab1:
        st.subheader("360° Shot Placement")
        col1, col2 = st.columns([2, 1])
        with col1:
            if selected_players:
                player = st.selectbox("Select Player", selected_players)
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
    
    # Control Matrix
    with tab2:
        st.subheader("Control vs Aggression Analysis")
        fig = create_control_matrix(df)
        st.plotly_chart(fig)
    
    # Phase Analysis
    with tab3:
        st.subheader("Match Phase Performance")
        fig = create_phase_analysis(df)
        st.plotly_chart(fig)
    
    # Player Cards
    with tab4:
        st.subheader("Player Intelligence Cards")
        if selected_players:
            for player in selected_players:
                player_data = df[df['batsman'] == player]
                if not player_data.empty:
                    # Metrics card
                    st.markdown(f"#### {player}")
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Control Rate", f"{player_data['is_controlled'].mean()*100:.1f}%")
                    with col2:
                        st.metric("Avg Runs", f"{player_data['runs'].mean():.2f}")
                    with col3:
                        st.metric("Boundary %", f"{player_data['boundary'].mean()*100:.1f}%")
                    st.progress(player_data['control_score'].mean()/100)
                    st.caption("Control Score (0-100)")
                    
                    # Insights
                    insights = get_player_insights(player_data)
                    if insights:
                        st.markdown('<div class="insight-card">', unsafe_allow_html=True)
                        st.markdown("##### 🎯 Key Insights")
                        if 'favorite_shot' in insights:
                            st.markdown(f"🏏 **Favorite Shot:** {insights['favorite_shot']}")
                        if 'dismissal_zone' in insights:
                            st.markdown(f"⚠️ **Weak Zone:** {insights['dismissal_zone']}")
                        if 'bowling_weaknesses' in insights:
                            st.markdown("💡 **Bowling Tips:**")
                            for weakness in insights['bowling_weaknesses'].values():
                                st.markdown(f"• {weakness}")
                        st.markdown('</div>', unsafe_allow_html=True)
    
    # Radar Comparison
    with tab5:
        st.subheader("Player Radar Comparison")
        if len(selected_players) >= 2:
            fig = create_radar_comparison(selected_players)
            st.plotly_chart(fig)
        else:
            st.warning("Select ≥2 players for comparison")
    
    # Dismissal Analysis
    with tab6:
        st.header("🔍 Dismissal Pattern Analysis")
        if selected_players:
            selected_batter = st.selectbox("Select Batsman", selected_players)
            summary, fig1, fig2, fig3 = create_dismissal_analysis(df, selected_batter)
            
            if not summary.empty:
                # Summary Table
                st.subheader("Dismissal Hotspots")
                st.dataframe(summary.sort_values('dismissals', ascending=False), use_container_width=True)
                
                # Visualizations
                col1, col2 = st.columns(2)
                with col1:
                    st.plotly_chart(fig1, use_container_width=True)
                    st.plotly_chart(fig2, use_container_width=True)
                with col2:
                    st.plotly_chart(fig3, use_container_width=True)
                
                # Strategy
                st.subheader("🎯 Tactical Recommendation")
                if not summary.empty:
                    top_row = summary.iloc[0]
                    st.markdown(f"""
                    **Optimal Bowling Strategy:**
                    • **Type:** {top_row.get('bowlingTypeId', 'N/A')}
                    • **Line:** {top_row.get('lineTypeId', 'N/A')}
                    • **Length:** {top_row.get('lengthTypeId', 'N/A')}
                    • **Target Zone:** {top_row.get('fieldingPosition', 'N/A')}
                    """)
            else:
                st.info("No dismissal data for selected batsman")
        else:
            st.warning("Select a batsman from the sidebar")
    
if __name__ == "__main__":
    main()
