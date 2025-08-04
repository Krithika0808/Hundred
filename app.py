import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
from io import StringIO
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Set page config
st.set_page_config(
    page_title="Women's Cricket Shot Intelligence Matrix",
    page_icon="🏏",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.8rem;
        font-weight: bold;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    .metric-card {
        background: linear-gradient(135deg, #f0f2f6 0%, #e8f4fd 100%);
        padding: 1.5rem;
        border-radius: 1rem;
        border-left: 5px solid #667eea;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        margin: 0.5rem 0;
        transition: transform 0.2s ease;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(0,0,0,0.15);
    }
    .insight-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 1rem;
        margin: 1rem 0;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
    }
    .player-stats {
        background: rgba(255,255,255,0.95);
        padding: 1.5rem;
        border-radius: 0.75rem;
        border: 1px solid #e0e6ed;
        margin: 1rem 0;
        box-shadow: 0 2px 10px rgba(0,0,0,0.05);
    }
    .warning-box {
        background: linear-gradient(135deg, #ff9a9e 0%, #fecfef 100%);
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #ff6b6b;
        margin: 1rem 0;
        color: #721c24;
    }
    .success-box {
        background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%);
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #28a745;
        margin: 1rem 0;
        color: #155724;
    }
    .stSelectbox > div > div {
        background-color: rgba(255,255,255,0.9);
    }
    .analytics-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
        gap: 1.5rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data(ttl=3600)  # Cache for 1 hour
def load_data_from_github(github_url=None):
    """Load cricket data from GitHub repository with improved error handling"""
    try:
        if github_url is None:
            github_url = "https://raw.githubusercontent.com/Krithika0808/Hundred/main/Hundred.csv"
        
        with st.spinner("Loading cricket data..."):
            response = requests.get(github_url, timeout=30)
            response.raise_for_status()
            csv_content = StringIO(response.text)
            df = pd.read_csv(csv_content)
        
        if df.empty:
            st.warning("Dataset is empty. Using sample data.")
            return create_sample_data()
        
        # Enhanced data cleaning
        numeric_cols = ['shotAngle', 'shotMagnitude', 'runs', 'totalBallNumber']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        bool_cols = ['isWicket', 'isBoundary', 'isAirControlled', 'isWide', 'isNoBall']
        for col in bool_cols:
            if col in df.columns:
                df[col] = df[col].fillna(False).astype(bool)
        
        string_cols = ['batsman', 'bowler', 'battingShotTypeId', 'battingConnectionId', 'commentary']
        for col in string_cols:
            if col in df.columns:
                df[col] = df[col].astype(str).str.strip().replace('nan', 'Unknown')
        
        # Remove rows with missing essential data
        essential_cols = [col for col in ['batsman', 'runs', 'totalBallNumber'] if col in df.columns]
        if essential_cols:
            df = df.dropna(subset=essential_cols)
        
        # Enhanced date handling
        if 'matchDate' in df.columns:
            df['matchDate'] = pd.to_datetime(df['matchDate'], errors='coerce')
            df['season'] = df['matchDate'].dt.year
            df['month'] = df['matchDate'].dt.month
        
        st.success(f"✅ Successfully loaded {len(df)} records from dataset")
        return df
        
    except requests.exceptions.RequestException as e:
        st.error(f"❌ Network error loading data: {str(e)}")
        return create_sample_data()
    except Exception as e:
        st.error(f"❌ Data loading error: {str(e)}")
        return create_sample_data()

def create_sample_data():
    """Create enhanced sample cricket data for demonstration"""
    np.random.seed(42)
    st.info("🎯 Using sample data for demonstration purposes")
    
    # Enhanced player roster with real women's cricket stars
    players = [
        'Smriti Mandhana', 'Harmanpreet Kaur', 'Beth Mooney', 'Alyssa Healy', 
        'Meg Lanning', 'Ellyse Perry', 'Stafanie Taylor', 'Deandra Dottin',
        'Chamari Athapaththu', 'Lizelle Lee'
    ]
    
    shot_types = ['Drive', 'Pull', 'Cut', 'Sweep', 'Flick', 'Hook', 'Reverse Sweep', 'Loft', 'Glance', 'Dab']
    connection_types = ['Middled', 'WellTimed', 'Undercontrol', 'MisTimed', 'Missed', 'HitBody', 'TopEdge', 'BottomEdge']
    length_types = ['Yorker', 'Full', 'Good Length', 'Short', 'Bouncer']
    line_types = ['Off Stump', 'Middle Stump', 'Leg Stump', 'Wide Outside Off', 'Down Leg']
    bowling_types = ['Fast', 'Medium', 'Spin', 'Swing', 'Seam']
    bowling_from = ['Over the Wicket', 'Around the Wicket']
    bowling_hands = ['Right Arm', 'Left Arm']
    fielding_positions = ['Cover', 'Mid Wicket', 'Point', 'Third Man', 'Fine Leg', 'Square Leg', 'Long On', 'Long Off']
    teams = ['Mumbai Indians', 'Delhi Capitals', 'RCB', 'CSK', 'KKR', 'PBKS', 'RR', 'SRH']
    
    n_rows = 1500  # Increased sample size
    
    # Generate more realistic data with correlations
    data = {
        'batsman': np.random.choice(players, n_rows),
        'battingShotTypeId': np.random.choice(shot_types, n_rows),
        'battingConnectionId': np.random.choice(connection_types, n_rows, p=[0.25, 0.2, 0.2, 0.15, 0.08, 0.05, 0.04, 0.03]),
        'runs': np.random.choice([0, 1, 2, 3, 4, 6], n_rows, p=[0.25, 0.3, 0.2, 0.1, 0.1, 0.05]),
        'totalBallNumber': np.random.randint(1, 121, n_rows),
        'shotAngle': np.random.uniform(0, 360, n_rows),
        'shotMagnitude': np.random.uniform(50, 250, n_rows),
        'isAirControlled': np.random.choice([True, False], n_rows, p=[0.35, 0.65]),
        'isBoundary': np.random.choice([True, False], n_rows, p=[0.15, 0.85]),
        'isWicket': np.random.choice([True, False], n_rows, p=[0.06, 0.94]),
        'fixtureId': np.random.randint(1, 31, n_rows),
        'battingTeam': np.random.choice(teams, n_rows),
        'matchDate': pd.to_datetime('2023-01-01') + pd.to_timedelta(np.random.randint(0, 365, n_rows), unit='d'),
        'lengthTypeId': np.random.choice(length_types, n_rows),
        'lineTypeId': np.random.choice(line_types, n_rows),
        'bowlingTypeId': np.random.choice(bowling_types, n_rows),
        'bowlingFromId': np.random.choice(bowling_from, n_rows),
        'bowlingHandId': np.random.choice(bowling_hands, n_rows),
        'fieldingPosition': np.random.choice(fielding_positions, n_rows)
    }
    
    df = pd.DataFrame(data)
    
    # Create realistic correlations
    df.loc[df['runs'] >= 4, 'isBoundary'] = True
    df.loc[df['runs'] < 4, 'isBoundary'] = False
    df.loc[df['battingConnectionId'].isin(['Middled', 'WellTimed']), 'runs'] = np.random.choice([2, 3, 4, 6], size=(df['battingConnectionId'].isin(['Middled', 'WellTimed'])).sum(), p=[0.3, 0.2, 0.3, 0.2])
    
    # Generate contextual commentary
    df['commentary'] = df.apply(generate_enhanced_commentary, axis=1)
    df['season'] = df['matchDate'].dt.year
    df['month'] = df['matchDate'].dt.month
    
    return df

def generate_enhanced_commentary(row):
    """Generate contextual commentary based on shot outcome"""
    shot = row['battingShotTypeId']
    runs = row['runs']
    connection = row['battingConnectionId']
    
    if runs == 6:
        return f"Magnificent {shot}! That's sailed over the boundary!"
    elif runs == 4:
        return f"Beautiful {shot} races to the boundary!"
    elif connection in ['Middled', 'WellTimed']:
        return f"Well-timed {shot} brings {runs} runs"
    elif connection == 'MisTimed':
        return f"Mistimed {shot}, manages {runs}"
    elif connection == 'Missed':
        return "Swing and a miss!"
    else:
        return f"{shot} played, {runs} runs"

def calculate_shot_intelligence_metrics(df):
    """Calculate comprehensive shot intelligence metrics"""
    if df.empty:
        return df
    
    df = df.copy()
    
    # Fill missing values with sensible defaults
    df['shotAngle'] = df['shotAngle'].fillna(180)  # Straight shot default
    df['shotMagnitude'] = df['shotMagnitude'].fillna(100)
    df['isAirControlled'] = df['isAirControlled'].fillna(False)
    df['battingConnectionId'] = df['battingConnectionId'].fillna('Unknown')
    df['commentary'] = df['commentary'].fillna('')
    
    # Enhanced boundary detection
    df['is_boundary'] = df.apply(detect_boundary_from_data, axis=1)
    
    # Advanced control categories
    df['true_control_category'] = df.apply(determine_enhanced_control, axis=1)
    
    # Refined control scoring system
    control_scores = {
        'Middled': 4.0, 'WellTimed': 3.8, 'Undercontrol': 3.5,
        'MisTimed': 2.2, 'Missed': 0.0, 'HitBody': 0.5,
        'TopEdge': 1.8, 'BatPad': 1.0, 'BottomEdge': 2.0,
        'Gloved': 1.5, 'HitHelmet': 0.0, 'InsideEdge': 2.5,
        'LeadingEdge': 1.8, 'OutsideEdge': 2.0, 'Unknown': 2.0
    }
    df['control_quality_score'] = df['battingConnectionId'].map(control_scores).fillna(2.0)
    
    # Strategic angle zones with cricket field positions
    angle_bins = [0, 22.5, 67.5, 112.5, 157.5, 202.5, 247.5, 292.5, 337.5, 360]
    angle_labels = ['Long On', 'Mid Wicket', 'Square Leg', 'Fine Leg', 'Third Man', 'Point', 'Cover', 'Long Off']
    df['angle_zone'] = pd.cut(df['shotAngle'], bins=angle_bins, labels=angle_labels, include_lowest=True)
    
    # Advanced hybrid control score
    df['control_score'] = df['control_quality_score'] * 20  # Base control component
    df['control_score'] += df['runs'] * 8  # Run value component
    df['control_score'] += df['is_boundary'] * 15  # Boundary bonus
    df['control_score'] += df['angle_zone'].isin(['Cover', 'Mid Wicket', 'Long On', 'Long Off']) * 5  # Strategic zone bonus
    df['control_score'] -= df['isAirControlled'] * 3  # Risk penalty for aerial shots
    df['control_score'] = df['control_score'].clip(0, 100)
    
    # Intelligence metrics
    df['is_controlled_shot'] = (df['control_score'] >= 55).astype(int)
    df['execution_intelligence'] = (df['control_quality_score'] * (df['runs'] + 1) * 2) / 3
    df['risk_reward_ratio'] = np.where(
        df['is_controlled_shot'], 
        df['runs'] * 1.4, 
        df['runs'] * 0.6
    )
    df['shot_efficiency'] = np.where(
        df['shotMagnitude'] > 0,
        (df['runs'] * df['control_quality_score'] * 100) / df['shotMagnitude'],
        df['runs'] * df['control_quality_score']
    )
    
    # Shot categorization
    df['shot_category'] = np.where(
        df['isAirControlled'], 
        'Aerial_' + df['true_control_category'],
        'Ground_' + df['true_control_category']
    )
    
    # Match phase analysis
    if 'totalBallNumber' in df.columns:
        max_ball = df['totalBallNumber'].max()
        if max_ball <= 20:  # T20 format
            df['match_phase'] = pd.cut(df['totalBallNumber'], 
                                     bins=[0, 6, 15, max_ball], 
                                     labels=['Powerplay', 'Middle', 'Death'])
        elif max_ball <= 50:  # T10 or shorter format
            df['match_phase'] = pd.cut(df['totalBallNumber'], 
                                     bins=[0, 20, max_ball], 
                                     labels=['Powerplay', 'Death'])
        else:  # ODI or longer format
            df['match_phase'] = pd.cut(df['totalBallNumber'], 
                                     bins=[0, 10, 40, max_ball], 
                                     labels=['Powerplay', 'Middle', 'Death'])
    
    return df

def detect_boundary_from_data(row):
    """Enhanced boundary detection logic"""
    if row['runs'] >= 4:
        return True
    if row['isBoundary'] if pd.notna(row['isBoundary']) else False:
        return True
    if row['shotMagnitude'] > 180 and row['control_quality_score'] > 3:
        return True
    return False

def determine_enhanced_control(row):
    """Determine enhanced control category"""
    connection = row['battingConnectionId']
    if connection in ['Middled', 'WellTimed', 'Undercontrol']:
        return 'Excellent Control'
    elif connection in ['MisTimed', 'TopEdge', 'BottomEdge', 'InsideEdge', 'OutsideEdge']:
        return 'Moderate Control'
    elif connection in ['Missed', 'HitBody', 'HitHelmet']:
        return 'Poor Control'
    else:
        return 'Average Control'

def get_comprehensive_player_insights(player_data):
    """Generate comprehensive player-specific insights"""
    insights = {}
    
    if player_data.empty:
        return insights
    
    # Favorite shots analysis
    if 'battingShotTypeId' in player_data.columns:
        shot_analysis = player_data.groupby('battingShotTypeId').agg({
            'runs': ['count', 'mean', 'sum'],
            'control_score': 'mean',
            'is_boundary': 'sum'
        }).round(2)
        shot_analysis.columns = ['frequency', 'avg_runs', 'total_runs', 'avg_control', 'boundaries']
        shot_analysis = shot_analysis.sort_values('frequency', ascending=False)
        
        if not shot_analysis.empty:
            top_shot = shot_analysis.index[0]
            insights['signature_shot'] = {
                'shot': top_shot,
                'frequency': f"{shot_analysis.loc[top_shot, 'frequency']} balls ({(shot_analysis.loc[top_shot, 'frequency']/len(player_data)*100):.1f}%)",
                'effectiveness': f"{shot_analysis.loc[top_shot, 'avg_runs']:.1f} runs/ball",
                'control': f"{shot_analysis.loc[top_shot, 'avg_control']:.1f}/100"
            }
    
    # Strength and weakness zones
    if 'angle_zone' in player_data.columns and len(player_data['angle_zone'].dropna()) > 0:
        zone_analysis = player_data.groupby('angle_zone').agg({
            'runs': 'mean',
            'control_score': 'mean',
            'is_boundary': 'mean'
        }).round(2)
        
        if not zone_analysis.empty:
            best_zone = zone_analysis['runs'].idxmax()
            worst_zone = zone_analysis['runs'].idxmin()
            
            insights['strength_zone'] = {
                'zone': best_zone,
                'avg_runs': f"{zone_analysis.loc[best_zone, 'runs']:.1f}",
                'control': f"{zone_analysis.loc[best_zone, 'control_score']:.1f}/100"
            }
            
            insights['vulnerability_zone'] = {
                'zone': worst_zone,
                'avg_runs': f"{zone_analysis.loc[worst_zone, 'runs']:.1f}",
                'control': f"{zone_analysis.loc[worst_zone, 'control_score']:.1f}/100"
            }
    
    # Bowling matchup analysis
    bowling_cols = ['lengthTypeId', 'bowlingTypeId', 'lineTypeId']
    bowling_weaknesses = {}
    
    for col in bowling_cols:
        if col in player_data.columns and len(player_data[col].dropna()) > 0:
            bowling_analysis = player_data.groupby(col).agg({
                'runs': 'mean',
                'control_score': 'mean',
                'isWicket': 'mean'
            }).round(2)
            
            if not bowling_analysis.empty and len(bowling_analysis) > 1:
                weakness = bowling_analysis['runs'].idxmin()
                bowling_weaknesses[col.replace('Id', '')] = {
                    'type': weakness,
                    'avg_runs': f"{bowling_analysis.loc[weakness, 'runs']:.1f}",
                    'control': f"{bowling_analysis.loc[weakness, 'control_score']:.1f}/100"
                }
    
    if bowling_weaknesses:
        insights['bowling_vulnerabilities'] = bowling_weaknesses
    
    # Phase-wise performance
    if 'match_phase' in player_data.columns and len(player_data['match_phase'].dropna()) > 0:
        phase_analysis = player_data.groupby('match_phase').agg({
            'runs': 'mean',
            'control_score': 'mean',
            'is_boundary': 'mean'
        }).round(2)
        
        if not phase_analysis.empty:
            best_phase = phase_analysis['runs'].idxmax()
            insights['best_phase'] = {
                'phase': best_phase,
                'avg_runs': f"{phase_analysis.loc[best_phase, 'runs']:.1f}",
                'boundary_rate': f"{phase_analysis.loc[best_phase, 'is_boundary']*100:.1f}%"
            }
    
    return insights

def create_enhanced_shot_heatmap(df, player_name):
    """Create enhanced polar plot of shot distribution"""
    player_data = df[df['batsman'] == player_name].copy()
    
    if player_data.empty:
        fig = go.Figure()
        fig.add_annotation(
            text=f"No data available for {player_name}",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=16)
        )
        return fig
    
    # Create hover text with comprehensive information
    hover_text = []
    for _, row in player_data.iterrows():
        hover_text.append(
            f"<b>{row['battingShotTypeId']}</b><br>" +
            f"Runs: {row['runs']}<br>" +
            f"Control: {row['control_score']:.1f}/100<br>" +
            f"Connection: {row['battingConnectionId']}<br>" +
            f"Phase: {row.get('match_phase', 'Unknown')}<br>" +
            f"Zone: {row.get('angle_zone', 'Unknown')}"
        )
    
    fig = go.Figure()
    
    # Add shot scatter plot
    fig.add_trace(go.Scatterpolar(
        r=player_data['shotMagnitude'],
        theta=player_data['shotAngle'],
        mode='markers',
        marker=dict(
            size=player_data['runs'] * 3 + 5,  # Minimum visible size
            color=player_data['control_score'],
            colorscale='RdYlGn',
            cmin=0, cmax=100,
            opacity=0.8,
            line=dict(width=1, color='white')
        ),
        text=hover_text,
        hovertemplate='%{text}<extra></extra>',
        name='Shots'
    ))
    
    # Update layout with field positions
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                range=[0, player_data['shotMagnitude'].max()*1.1],
                title="Shot Distance"
            ),
            angularaxis=dict(
                tickvals=[0, 45, 90, 135, 180, 225, 270, 315],
                ticktext=['Long On', 'Mid Wicket', 'Square Leg', 'Fine Leg', 
                         'Third Man', 'Point', 'Cover', 'Long Off'],
                direction='clockwise'
            )
        ),
        title=dict(
            text=f"{player_name} - 360° Shot Map",
            x=0.5,
            font=dict(size=18)
        ),
        height=650,
        showlegend=True
    )
    
    return fig

def create_advanced_control_matrix(df):
    """Create enhanced control vs aggression matrix"""
    shot_analysis = df.groupby('battingShotTypeId').agg({
        'is_controlled_shot': 'mean',
        'runs': 'mean',
        'control_score': 'mean',
        'is_boundary': 'mean',
        'batsman': 'count'  # frequency
    }).round(3)
    shot_analysis.columns = ['control_rate', 'avg_runs', 'avg_control', 'boundary_rate', 'frequency']
    shot_analysis = shot_analysis[shot_analysis['frequency'] >= 5]  # Filter low-frequency shots
    shot_analysis = shot_analysis.reset_index()
    
    if shot_analysis.empty:
        return go.Figure().add_annotation(text="Insufficient data for analysis", 
                                        xref="paper", yref="paper", x=0.5, y=0.5)
    
    fig = px.scatter(
        shot_analysis,
        x='control_rate',
        y='avg_runs',
        size='frequency',
        color='avg_control',
        text='battingShotTypeId',
        title='Shot Control vs Aggression Matrix',
        labels={
            'control_rate': 'Control Rate (0-1)',
            'avg_runs': 'Average Runs per Shot',
            'avg_control': 'Average Control Score',
            'frequency': 'Shot Frequency'
        },
        color_continuous_scale='RdYlGn',
        size_max=50
    )
    
    # Add quadrant lines
    fig.add_hline(y=shot_analysis['avg_runs'].mean(), 
                  line_dash="dash", line_color="gray", 
                  annotation_text="Avg Runs")
    fig.add_vline(x=shot_analysis['control_rate'].mean(), 
                  line_dash="dash", line_color="gray", 
                  annotation_text="Avg Control")
    
    # Add quadrant labels
    fig.add_annotation(x=0.8, y=shot_analysis['avg_runs'].max()*0.9,
                      text="High Control<br>High Runs", showarrow=False,
                      bgcolor="lightgreen", opacity=0.7)
    fig.add_annotation(x=0.2, y=shot_analysis['avg_runs'].max()*0.9,
                      text="Low Control<br>High Risk", showarrow=False,
                      bgcolor="lightcoral", opacity=0.7)
    
    fig.update_traces(textposition='top center')
    fig.update_layout(height=600, showlegend=True)
    
    return fig

def create_phase_performance_analysis(df):
    """Enhanced match phase performance analysis"""
    if 'match_phase' not in df.columns or df['match_phase'].isna().all():
        return go.Figure().add_annotation(text="No phase data available", 
                                        xref="paper", yref="paper", x=0.5, y=0.5)
    
    phase_analysis = df.groupby(['match_phase', 'battingShotTypeId']).agg({
        'runs': ['mean', 'sum', 'count'],
        'control_score': 'mean',
        'is_boundary': 'mean'
    }).round(2)
    
    phase_analysis.columns = ['avg_runs', 'total_runs', 'frequency', 'avg_control', 'boundary_rate']
    phase_analysis = phase_analysis.reset_index()
    phase_analysis = phase_analysis[phase_analysis['frequency'] >= 3]  # Filter low-frequency combinations
    
    if phase_analysis.empty:
        return go.Figure().add_annotation(text="Insufficient phase data", 
                                        xref="paper", yref="paper", x=0.5, y=0.5)
    
    # Create subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Average Runs by Phase', 'Control Score by Phase',
                       'Boundary Rate by Phase', 'Shot Frequency by Phase'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    # Plot 1: Average runs
    for phase in phase_analysis['match_phase'].unique():
        phase_data = phase_analysis[phase_analysis['match_phase'] == phase]
        fig.add_trace(
            go.Bar(x=phase_data['battingShotTypeId'], y=phase_data['avg_runs'],
                  name=f'{phase} - Avg Runs', showlegend=True),
            row=1, col=1
        )
    
    # Plot 2: Control scores
    for phase in phase_analysis['match_phase'].unique():
        phase_data = phase_analysis[phase_analysis['match_phase'] == phase]
        fig.add_trace(
            go.Bar(x=phase_data['battingShotTypeId'], y=phase_data['avg_control'],
                  name=f'{phase} - Control', showlegend=False),
            row=1, col=2
        )
    
    # Plot 3: Boundary rates
    for phase in phase_analysis['match_phase'].unique():
        phase_data = phase_analysis[phase_analysis['match_phase'] == phase]
        fig.add_trace(
            go.Bar(x=phase_data['battingShotTypeId'], y=phase_data['boundary_rate']*100,
                  name=f'{phase} - Boundary%', showlegend=False),
            row=2, col=1
        )
    
    # Plot 4: Shot frequency
    for phase in phase_analysis['match_phase'].unique():
        phase_data = phase_analysis[phase_analysis['match_phase'] == phase]
        fig.add_trace(
            go.Bar(x=phase_data['battingShotTypeId'], y=phase_data['frequency'],
                  name=f'{phase} - Frequency', showlegend=False),
            row=2, col=2
        )
    
    fig.update_layout(height=800, title_text="Comprehensive Phase Analysis")
    return fig

def create_player_comparison_radar(df, selected_players):
    """Create comprehensive player comparison radar chart"""
    if len(selected_players) < 2:
        return go.Figure().add_annotation(text="Select at least 2 players for comparison", 
                                        xref="paper", yref="paper", x=0.5, y=0.5)
    
    radar_metrics = []
    categories = ['Strike Rate', 'Control Rate', 'Boundary Rate', 'Avg Runs', 'Risk Management', 'Consistency']
    
    for player in selected_players[:5]:  # Limit to 5 players for readability
        player_data = df[df['batsman'] == player]
        if player_data.empty:
            continue
            
        # Calculate metrics
        strike_rate = (player_data['runs'].sum() / len(player_data)) * 100
        control_rate = player_data['is_controlled_shot'].mean() * 100
        boundary_rate = player_data['is_boundary'].mean() * 100
        avg_runs = player_data['runs'].mean()
        risk_mgmt = 100 - (player_data['isWicket'].mean() * 100 * 10)  # Inverted wicket rate
        consistency = 100 - (player_data['runs'].std() * 10)  # Lower std dev = higher consistency
        
        radar_metrics.append({
            'Player': player,
            'Strike Rate': min(strike_rate, 100),
            'Control Rate': control_rate,
            'Boundary Rate': min(boundary_rate * 5, 100),  # Scale up for visibility
            'Avg Runs': min(avg_runs * 25, 100),  # Scale for 0-100 range
            'Risk Management': max(risk_mgmt, 0),
            'Consistency': max(consistency, 0)
        })
    
    if not radar_metrics:
        return go.Figure().add_annotation(text="No data available for selected players", 
                                        xref="paper", yref="paper", x=0.5, y=0.5)
    
    fig = go.Figure()
    
    colors = px.colors.qualitative.Set3
    for i, player_stats in enumerate(radar_metrics):
        player = player_stats['Player']
        values = [player_stats[cat] for cat in categories]
        
        fig.add_trace(go.Scatterpolar(
            r=values + [values[0]],  # Close the polygon
            theta=categories + [categories[0]],
            fill='toself',
            name=player,
            line_color=colors[i % len(colors)],
            fillcolor=colors[i % len(colors)],
            opacity=0.6
        ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100]
            )
        ),
        showlegend=True,
        title="Player Performance Comparison",
        height=600
    )
    
    return fig

def create_bowling_vulnerability_heatmap(df, player_name):
    """Create bowling vulnerability heatmap for a player"""
    player_data = df[df['batsman'] == player_name]
    
    if player_data.empty or 'lengthTypeId' not in player_data.columns or 'lineTypeId' not in player_data.columns:
        return go.Figure().add_annotation(text=f"No bowling data available for {player_name}", 
                                        xref="paper", yref="paper", x=0.5, y=0.5)
    
    # Create length vs line heatmap
    vulnerability_matrix = player_data.groupby(['lengthTypeId', 'lineTypeId']).agg({
        'runs': 'mean',
        'control_score': 'mean',
        'isWicket': 'mean'
    }).round(2)
    
    if vulnerability_matrix.empty:
        return go.Figure().add_annotation(text="Insufficient bowling data", 
                                        xref="paper", yref="paper", x=0.5, y=0.5)
    
    # Pivot for heatmap
    runs_matrix = vulnerability_matrix['runs'].unstack(fill_value=0)
    control_matrix = vulnerability_matrix['control_score'].unstack(fill_value=0)
    
    # Create subplot with two heatmaps
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Average Runs Conceded', 'Control Score'),
        specs=[[{"type": "heatmap"}, {"type": "heatmap"}]]
    )
    
    # Runs heatmap
    fig.add_trace(
        go.Heatmap(
            z=runs_matrix.values,
            x=runs_matrix.columns,
            y=runs_matrix.index,
            colorscale='Reds',
            name='Runs'
        ),
        row=1, col=1
    )
    
    # Control heatmap  
    fig.add_trace(
        go.Heatmap(
            z=control_matrix.values,
            x=control_matrix.columns,
            y=control_matrix.index,
            colorscale='RdYlGn',
            name='Control'
        ),
        row=1, col=2
    )
    
    fig.update_layout(
        title=f"{player_name} - Bowling Vulnerability Analysis",
        height=500
    )
    
    return fig

def main():
    """Enhanced main application logic"""
    # Header
    st.markdown('<h1 class="main-header">🏏 Women\'s Cricket Shot Intelligence Matrix</h1>', 
                unsafe_allow_html=True)
    
    # Load and process data
    df = load_data_from_github()
    if df.empty:
        st.error("❌ Unable to load data. Please check your connection and try again.")
        return
    
    df = calculate_shot_intelligence_metrics(df)
    
    # Data overview
    with st.expander("📊 Dataset Overview", expanded=False):
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Balls", len(df))
        with col2:
            st.metric("Players", df['batsman'].nunique())
        with col3:
            st.metric("Matches", df['fixtureId'].nunique() if 'fixtureId' in df.columns else 0)
        with col4:
            st.metric("Avg Control Score", f"{df['control_score'].mean():.1f}")
    
    # Enhanced Sidebar
    st.sidebar.header("🎯 Analysis Filters")
    
    # Player selection
    available_players = sorted(df['batsman'].unique())
    selected_players = st.sidebar.multiselect(
        "Select Players",
        available_players,
        default=available_players[:3] if len(available_players) >= 3 else available_players,
        help="Choose players for analysis"
    )
    
    # Phase filter
    if 'match_phase' in df.columns:
        phases = df['match_phase'].dropna().unique()
        selected_phases = st.sidebar.multiselect(
            "Match Phases",
            phases,
            default=list(phases),
            help="Filter by match phases"
        )
        df = df[df['match_phase'].isin(selected_phases)] if selected_phases else df
    
    # Shot type filter
    shot_types = df['battingShotTypeId'].unique()
    selected_shots = st.sidebar.multiselect(
        "Shot Types",
        shot_types,
        default=list(shot_types),
        help="Filter by shot types"
    )
    df = df[df['battingShotTypeId'].isin(selected_shots)] if selected_shots else df
    
    # Control threshold
    control_threshold = st.sidebar.slider(
        "Minimum Control Score",
        0, 100, 0,
        help="Filter shots by minimum control score"
    )
    df = df[df['control_score'] >= control_threshold] if control_threshold > 0 else df
    
    # Main content tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🎯 Shot Patterns", "⚡ Control Analysis", "📊 Phase Performance", 
        "👥 Player Comparison", "🎪 Bowling Analysis"
    ])
    
    # Tab 1: Shot Patterns
    with tab1:
        st.subheader("360° Shot Pattern Analysis")
        
        if not selected_players:
            st.warning("⚠️ Please select at least one player from the sidebar")
        else:
            col1, col2 = st.columns([3, 1])
            
            with col1:
                selected_player = st.selectbox(
                    "Select Player for Shot Map",
                    selected_players,
                    help="Choose a player to visualize their shot patterns"
                )
                
                if selected_player:
                    fig = create_enhanced_shot_heatmap(df, selected_player)
                    st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                if selected_player:
                    player_data = df[df['batsman'] == selected_player]
                    
                    # Player stats
                    st.markdown('<div class="player-stats">', unsafe_allow_html=True)
                    st.markdown(f"**{selected_player} Statistics**")
                    
                    col_a, col_b = st.columns(2)
                    with col_a:
                        st.metric("Balls Faced", len(player_data))
                        st.metric("Strike Rate", f"{(player_data['runs'].sum()/len(player_data)*100):.1f}")
                    with col_b:
                        st.metric("Avg Control", f"{player_data['control_score'].mean():.1f}")
                        st.metric("Boundary %", f"{player_data['is_boundary'].mean()*100:.1f}%")
                    
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                    # Shot distribution
                    shot_dist = player_data['battingShotTypeId'].value_counts().head(5)
                    if not shot_dist.empty:
                        st.markdown("**Top 5 Shots:**")
                        for shot, count in shot_dist.items():
                            percentage = (count/len(player_data)*100)
                            st.markdown(f"• {shot}: {count} ({percentage:.1f}%)")
    
    # Tab 2: Control Analysis
    with tab2:
        st.subheader("Shot Control Intelligence Matrix")
        
        fig = create_advanced_control_matrix(df)
        st.plotly_chart(fig, use_container_width=True)
        
        # Control insights
        st.markdown('<div class="insight-box">', unsafe_allow_html=True)
        st.markdown("**📈 Control Analysis Insights**")
        
        high_control_shots = df.groupby('battingShotTypeId')['control_score'].mean().sort_values(ascending=False).head(3)
        st.markdown("**Highest Control Shots:**")
        for shot, score in high_control_shots.items():
            st.markdown(f"• {shot}: {score:.1f}/100")
        
        high_risk_shots = df.groupby('battingShotTypeId')['control_score'].mean().sort_values().head(3)
        st.markdown("**Highest Risk Shots:**")
        for shot, score in high_risk_shots.items():
            st.markdown(f"• {shot}: {score:.1f}/100")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Tab 3: Phase Performance
    with tab3:
        st.subheader("Match Phase Performance Analysis")
        
        fig = create_phase_performance_analysis(df)
        st.plotly_chart(fig, use_container_width=True)
        
        # Phase insights
        if 'match_phase' in df.columns and not df['match_phase'].isna().all():
            phase_stats = df.groupby('match_phase').agg({
                'runs': 'mean',
                'control_score': 'mean',
                'is_boundary': 'mean'
            }).round(2)
            
            st.markdown("**Phase-wise Performance Summary:**")
            for phase, stats in phase_stats.iterrows():
                with st.expander(f"{phase} Phase"):
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Avg Runs", f"{stats['runs']:.2f}")
                    with col2:
                        st.metric("Control Score", f"{stats['control_score']:.1f}")
                    with col3:
                        st.metric("Boundary Rate", f"{stats['is_boundary']*100:.1f}%")
    
    # Tab 4: Player Comparison
    with tab4:
        st.subheader("Multi-Player Performance Comparison")
        
        if len(selected_players) >= 2:
            # Radar chart comparison
            fig = create_player_comparison_radar(df, selected_players)
            st.plotly_chart(fig, use_container_width=True)
            
            # Detailed comparison table
            st.subheader("Detailed Statistics Comparison")
            comparison_stats = []
            
            for player in selected_players:
                player_data = df[df['batsman'] == player]
                if not player_data.empty:
                    stats = {
                        'Player': player,
                        'Balls Faced': len(player_data),
                        'Total Runs': player_data['runs'].sum(),
                        'Strike Rate': f"{(player_data['runs'].sum()/len(player_data)*100):.1f}",
                        'Avg Control': f"{player_data['control_score'].mean():.1f}",
                        'Boundary %': f"{player_data['is_boundary'].mean()*100:.1f}%",
                        'Controlled Shots %': f"{player_data['is_controlled_shot'].mean()*100:.1f}%",
                        'Favorite Shot': player_data['battingShotTypeId'].mode().iloc[0] if not player_data['battingShotTypeId'].mode().empty else 'N/A'
                    }
                    comparison_stats.append(stats)
            
            if comparison_stats:
                comparison_df = pd.DataFrame(comparison_stats)
                st.dataframe(comparison_df, use_container_width=True)
        else:
            st.warning("⚠️ Please select at least 2 players for comparison")
    
    # Tab 5: Bowling Analysis
    with tab5:
        st.subheader("Bowling Vulnerability Analysis")
        
        if selected_players:
            selected_player_bowling = st.selectbox(
                "Select Player for Bowling Analysis",
                selected_players,
                key="bowling_analysis"
            )
            
            if selected_player_bowling:
                fig = create_bowling_vulnerability_heatmap(df, selected_player_bowling)
                st.plotly_chart(fig, use_container_width=True)
                
                # Player insights
                insights = get_comprehensive_player_insights(df[df['batsman'] == selected_player_bowling])
                
                if insights:
                    st.markdown('<div class="insight-box">', unsafe_allow_html=True)
                    st.markdown(f"**🔍 {selected_player_bowling} - Tactical Insights**")
                    
                    if 'signature_shot' in insights:
                        st.markdown(f"**Signature Shot:** {insights['signature_shot']['shot']}")
                        st.markdown(f"• Frequency: {insights['signature_shot']['frequency']}")
                        st.markdown(f"• Effectiveness: {insights['signature_shot']['effectiveness']}")
                    
                    if 'strength_zone' in insights:
                        st.markdown(f"**Strength Zone:** {insights['strength_zone']['zone']}")
                        st.markdown(f"• Average: {insights['strength_zone']['avg_runs']} runs")
                    
                    if 'vulnerability_zone' in insights:
                        st.markdown(f"**Vulnerability:** {insights['vulnerability_zone']['zone']}")
                        st.markdown(f"• Average: {insights['vulnerability_zone']['avg_runs']} runs")
                    
                    if 'bowling_vulnerabilities' in insights:
                        st.markdown("**Bowling Vulnerabilities:**")
                        for bowling_type, vuln in insights['bowling_vulnerabilities'].items():
                            st.markdown(f"• {bowling_type}: {vuln['type']} ({vuln['avg_runs']} avg runs)")
                    
                    st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.warning("⚠️ Please select a player for bowling analysis")
    
    # Footer
    st.markdown("---")
    st.markdown("**🏏 Women's Cricket Shot Intelligence Matrix** - Advanced Analytics Dashboard")
    st.markdown("*Empowering tactical decisions through data-driven insights*")

if __name__ == "__main__":
    main()
