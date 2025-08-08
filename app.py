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
        'bowlingHandId': np.random.choice(bowling_hands, n_rows)
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
        'Middled': 0,
        'WellTimed': 0,
        'Undercontrol': 0,
        'Left': 0,
        'MisTimed': 3,
        'Mis-timed': 3,
        'BottomEdge': 4,
        'TopEdge': 4,
        'BatPad': 4,
        'InsideEdge': 3,
        'LeadingEdge': 4,
        'OutsideEdge': 3,
        'Gloved': 3,
        'ThickEdge': 3,
        'Missed': 5,
        'PlayAndMiss': 5,
        'PlayAndMissLegSide': 5,
        'HitBody': 2,
        'HitHelmet': 2,
        'HitPad': 2,
        'Padded': 2,
        'Spliced': 4
    }
    return false_shot_mapping.get(str(connection_id).strip(), 2)

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
    
    def detect_boundary_from_data(row):
        runs = row.get('runs', 0)
        commentary = str(row['commentary']).lower()
        is_boundary_col = row.get('isBoundary', False)
        
        if runs == 4 or runs == 6:
            return True
        
        if is_boundary_col:
            return True
        
        boundary_keywords = ['four', '4 runs', 'boundary', 'reaches the rope', 'to the fence', 
                             'six', '6 runs', 'maximum', 'over the rope', 'into the stands',
                             'hits it for four', 'hits it for six']
        
        for keyword in boundary_keywords:
            if keyword in commentary:
                return True
                
        return False
    
    df['is_boundary'] = df.apply(detect_boundary_from_data, axis=1)
    
    def determine_true_control(row):
        connection = str(row['battingConnectionId']).strip()
        
        if connection in ['Middled', 'WellTimed', 'Undercontrol', 'Left']:
            return 'High Control'
        
        elif connection in ['MisTimed', 'BottomEdge', 'TopEdge', 'BatPad', 'Mis-timed', 'InsideEdge', 'LeadingEdge', 'OutsideEdge', 'Gloved', 'ThickEdge', 'TopEdge']:
            return 'Less Control'
        
        else:
            return 'Poor Control'
    
    df['true_control_category'] = df.apply(determine_true_control, axis=1)
    
    control_scores = {
        'Middled': 3,
        'WellTimed': 3,
        'Undercontrol': 3,
        'MisTimed': 2,
        'Missed': 0,
        'HitBody': 0.5,
        'TopEdge': 2,
        'BatPad':1,
        'BottomEdge':2,
        'Gloved': 2,
        'HitHelmet': 0,
        'HitPad': 0,
        'InsideEdge': 2,
        'LeadingEdge': 2,
        'Left': 3,
        'mis-timed': 2,
        'OutsideEdge': 2,
        'Padded': 0,
        'PlayAndMiss': 0,
        'PlayAndMissLegSide': 0,
        'Spliced': 1,
        'ThickEdge': 2,
        'TopEdge': 2
    }
    
    df['control_quality_score'] = df['battingConnectionId'].map(control_scores).fillna(1.5)
    
    valid_angles = (df['shotAngle'] >= 0) & (df['shotAngle'] <= 360)
    df.loc[~valid_angles, 'shotAngle'] = 0
    
    angle_bins = [0, 45, 90, 135, 180, 225, 270, 315, 360]
    angle_labels = ['Long Off', 'Cover', 'Point', 'Third Man', 'Fine Leg', 'Square Leg', 'Mid Wicket', 'Long On']
    
    df['angle_zone'] = pd.cut(df['shotAngle'], bins=angle_bins, labels=angle_labels, include_lowest=True)
    
    df['control_score'] = df['control_quality_score'] * 33.33
    
    df['control_score'] += df['runs'] * 5
    df['control_score'] += df['is_boundary'] * 20
    
    good_placements = ['Cover', 'Mid Wicket', 'Long On', 'Long Off']
    df['control_score'] += df['angle_zone'].isin(good_placements) * 10
    
    if 'match_phase' in df.columns:
        phase_bonus = {
            'Powerplay (1-25)': 10,
            'Middle (26-75)': 5,
            'Death (76-100)': 0
        }
        df['control_score'] += df['match_phase'].map(phase_bonus).fillna(5)
    
    df['control_score'] = df['control_score'].clip(0, 100)
    
    df['is_controlled_shot'] = (df['control_score'] >= 50).astype(int)
    
    df['execution_intelligence'] = df['control_quality_score'] * (df['runs'] + 1) / 2
    
    df['true_risk_reward'] = np.where(
        df['is_controlled_shot'] == 1,
        df['runs'] * 1.3,
        df['runs'] * 0.7
    )
    
    df['true_shot_efficiency'] = np.where(
        df['shotMagnitude'] > 0,
        (df['runs'] * df['control_quality_score']) / (df['shotMagnitude'] / 100),
        df['runs'] * df['control_quality_score']
    )
    
    df['shot_category'] = np.where(
        df['isAirControlled'] == True,
        'Aerial_' + df['true_control_category'],
        'Ground_' + df['true_control_category']
    )
    
    if 'totalBallNumber' in df.columns:
        max_ball = int(df['totalBallNumber'].max())
        
        if max_ball <= 10:
            bins = [0, max_ball + 1]
            labels = ['Early (1-10)']
        elif max_ball <= 20:
            bins = [0, 10, max_ball + 1]
            labels = ['Early (1-10)', f'Mid-Powerplay (11-{max_ball})']
        elif max_ball <= 25:
            bins = [0, 10, 20, max_ball + 1]
            labels = ['Early (1-10)', 'Mid-Powerplay (11-20)', f'Late-Powerplay (21-{max_ball})']
        elif max_ball <= 50:
            bins = [0, 25, max_ball + 1]
            labels = ['Powerplay (1-25)', f'Middle (26-{max_ball})']
        elif max_ball <= 75:
            bins = [0, 25, max_ball + 1]
            labels = ['Powerplay (1-25)', f'Middle (26-{max_ball})']
        else:
            bins = [0, 25, 75, max_ball + 1]
            labels = ['Powerplay (1-25)', 'Middle (26-75)', f'Death (76-{max_ball})']
        
        df['match_phase'] = pd.cut(
            df['totalBallNumber'], 
            bins=bins,
            labels=labels,
            include_lowest=True
        )
    
    if 'totalBallNumber' in df.columns:
        df['is_death_phase'] = (df['totalBallNumber'] >= 76).astype(int)
        df['death_control_performance'] = df['is_death_phase'] * df['is_controlled_shot']
    
    if 'battingShotTypeId' in df.columns:
        shot_control_avg = df.groupby('battingShotTypeId')['control_quality_score'].mean()
        df['shot_type_control_avg'] = df['battingShotTypeId'].map(shot_control_avg)
        
        df['selection_wisdom'] = np.where(
            df['control_quality_score'] >= df['shot_type_control_avg'],
            'Smart Selection',
            'Risky Selection'
        )
    
    df['false_shot_score'] = df['battingConnectionId'].apply(false_shot_score)
    
    return df

def get_player_insights(player_data):
    """Generate player insights including dismissal patterns, favorite shots, and bowling recommendations"""
    insights = {}
    
    if 'battingShotTypeId' in player_data.columns:
        shot_counts = player_data['battingShotTypeId'].value_counts()
        if not shot_counts.empty:
            favorite_shot = shot_counts.index[0]
            favorite_count = shot_counts.iloc[0]
            total_shots = len(player_data)
            favorite_percentage = (favorite_count / total_shots) * 100
            insights['favorite_shot'] = f"{favorite_shot} ({favorite_percentage:.1f}% of shots)"
    
    dismissal_zone = None
    if 'angle_zone' in player_data.columns and 'true_control_category' in player_data.columns:
        poor_control_shots = player_data[player_data['true_control_category'].isin(['Poor Control', 'Less Control'])]
        if not poor_control_shots.empty:
            dismissal_zones = poor_control_shots['angle_zone'].value_counts()
            if not dismissal_zones.empty:
                dismissal_zone = dismissal_zones.index[0]
                dismissal_count = dismissal_zones.iloc[0]
                total_poor = len(poor_control_shots)
                dismissal_percentage = (dismissal_count / total_poor) * 100
                insights['dismissal_pattern'] = f"{dismissal_zone} ({dismissal_percentage:.1f}% of poor control shots)"
    
    bowling_columns = ['lengthTypeId', 'lineTypeId', 'bowlingTypeId', 'bowlingFromId', 'bowlingHandId']
    available_bowling_cols = [col for col in bowling_columns if col in player_data.columns]
    
    if available_bowling_cols and 'control_score' in player_data.columns and dismissal_zone:
        poor_control_shots = player_data[player_data['true_control_category'].isin(['Poor Control', 'Less Control'])]
        dismissal_area_shots = poor_control_shots[poor_control_shots['angle_zone'] == dismissal_zone]
        
        if not dismissal_area_shots.empty:
            bowling_recommendations = []
            
            for col in available_bowling_cols:
                if col in player_data.columns:
                    attr_control = dismissal_area_shots.groupby(col)['control_score'].mean()
                    
                    if not attr_control.empty and len(attr_control) >= 3:
                        weakest_attr = attr_control.idxmin()
                        weakest_score = attr_control.min()
                        
                        if weakest_score < 60:
                            col_names = {
                                'lengthTypeId': 'Length',
                                'lineTypeId': 'Line',
                                'bowlingTypeId': 'Type',
                                'bowlingFromId': 'From',
                                'bowlingHandId': 'Hand'
                            }
                            
                            display_name = col_names.get(col, col)
                            bowling_recommendations.append((display_name, weakest_attr, weakest_score))
            
            if bowling_recommendations:
                bowling_recommendations.sort(key=lambda x: x[2])
                
                top_weaknesses = bowling_recommendations[:3]
                
                recommendations = []
                for name, value, score in top_weaknesses:
                    recommendations.append(f"{name}: {value} ({score:.1f}/100)")
                
                insights['bowl_to'] = recommendations
                insights['bowl_to_connection'] = dismissal_zone
        else:
            bowling_weaknesses = []
            
            for col in available_bowling_cols:
                if col in player_data.columns:
                    attr_control = player_data.groupby(col)['control_score'].mean()
                    if not attr_control.empty:
                        weakest_attr = attr_control.idxmin()
                        weakest_score = attr_control.min()
                        
                        if weakest_score < 60:
                            col_names = {
                                'lengthTypeId': 'Length',
                                'lineTypeId': 'Line',
                                'bowlingTypeId': 'Type',
                                'bowlingFromId': 'From',
                                'bowlingHandId': 'Hand'
                            }
                            
                            display_name = col_names.get(col, col)
                            bowling_weaknesses.append((display_name, weakest_attr, weakest_score))
            
            if bowling_weaknesses:
                bowling_weaknesses.sort(key=lambda x: x[2])
                
                top_weaknesses = bowling_weaknesses[:3]
                
                recommendations = []
                for name, value, score in top_weaknesses:
                    recommendations.append(f"{name}: {value} ({score:.1f}/100)")
                
                insights['bowl_to'] = recommendations
    
    if 'angle_zone' in player_data.columns and 'control_score' in player_data.columns:
        zone_control = player_data.groupby('angle_zone')['control_score'].mean().sort_values(ascending=False)
        if not zone_control.empty:
            strongest_zone = zone_control.index[0]
            strongest_score = zone_control.iloc[0]
            insights['strength_area'] = f"{strongest_zone} (avg control: {strongest_score:.1f}/100)"
    
    if 'battingShotTypeId' in player_data.columns and 'runs' in player_data.columns:
        shot_runs = player_data.groupby('battingShotTypeId')['runs'].agg(['mean', 'count'])
        shot_runs = shot_runs[shot_runs['count'] >= 5]
        if not shot_runs.empty:
            most_effective = shot_runs['mean'].idxmax()
            avg_runs = shot_runs.loc[most_effective, 'mean']
            insights['most_effective'] = f"{most_effective} (avg {avg_runs:.2f} runs)"
    
    return insights

def create_shot_angle_heatmap(df, player_name):
    """Create 360-degree shot angle heatmap"""
    
    player_data = df[df['batsman'] == player_name]
    
    if player_data.empty:
        st.warning(f"No data available for {player_name}")
        return go.Figure()
    
    player_data = player_data[
        (player_data['shotAngle'].notna()) & 
        (player_data['shotMagnitude'].notna()) &
        (player_data['shotAngle'] >= 0) & 
        (player_data['shotAngle'] <= 360)
    ]
    
    if player_data.empty:
        st.warning(f"No valid shot angle data for {player_name}")
        return go.Figure()
    
    fig = go.Figure()
    
    colors = []
    for _, row in player_data.iterrows():
        if row['control_score'] >= 80:
            colors.append('green')
        elif row['control_score'] >= 50:
            colors.append('orange')
        else:
            colors.append('red')
    
    hover_text = []
    for idx, row in player_data.iterrows():
        shot_type = row.get('battingShotTypeId', 'Unknown')
        runs = row.get('runs', 0)
        connection = row.get('battingConnectionId', 'Unknown')
        control_score = row.get('control_score', 0)
        hover_text.append(f"Shot: {shot_type}<br>Runs: {runs}<br>Connection: {connection}<br>Control Score: {control_score:.1f}/100")
    
    fig.add_trace(go.Scatterpolar(
        r=player_data['shotMagnitude'],
        theta=player_data['shotAngle'],
        mode='markers',
        marker=dict(
            size=player_data['runs'] * 4 + 8,
            color=player_data['control_score'],
            colorscale='RdYlGn',
            showscale=True,
            colorbar=dict(title="Control Score"),
            line=dict(width=2, color='white'),
            cmin=0,
            cmax=100
        ),
        text=hover_text,
        hovertemplate='<b>%{text}</b><br>Angle: %{theta}°<br>Distance: %{r}<extra></extra>',
        name='Shots'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, player_data['shotMagnitude'].max() * 1.1] if not player_data.empty else [0, 200],
                title="Shot Distance"
            ),
            angularaxis=dict(
                tickmode='array',
                tickvals=[0, 45, 90, 135, 180, 225, 270, 315],
                ticktext=['Long Off', 'Cover', 'Point', 'Third Man', 
                          'Fine Leg', 'Square Leg', 'Mid Wicket', 'Long On'],
                direction='clockwise'
            )
        ),
        title=f"{player_name} - Shot Placement Intelligence<br><sub>Size = Runs Scored, Color = Control Score (0-100)</sub>",
        height=600,
        showlegend=False
    )
    
    return fig

def create_control_vs_aggression_chart(df):
    """Create TRUE control vs aggression analysis"""
    
    if df.empty or 'battingShotTypeId' not in df.columns:
        return go.Figure()
    
    shot_analysis = df.groupby(['battingShotTypeId']).agg({
        'is_controlled_shot': 'mean',
        'runs': 'mean',
        'is_boundary': 'mean',
        'control_score': 'mean'
    }).reset_index()
    
    shot_counts = df.groupby(['battingShotTypeId']).size().reset_index(name='shot_count')
    shot_analysis = shot_analysis.merge(shot_counts, on='battingShotTypeId')
    
    shot_analysis.columns = ['Shot Type', 'True Control Rate', 'Avg Runs', 'Boundary %', 'Avg Control Score', 'Shot Count']
    
    shot_analysis = shot_analysis[shot_analysis['Shot Count'] >= 3]
    
    if shot_analysis.empty:
        return go.Figure()
    
    fig = px.scatter(
        shot_analysis,
        x='True Control Rate',
        y='Avg Runs',
        size='Shot Count',
        color='Avg Control Score',
        text='Shot Type',
        title='TRUE Shot Intelligence: Control Mastery vs Aggression Matrix',
        labels={
            'True Control Rate': 'True Control Rate (Enhanced Hybrid Method)',
            'Avg Runs': 'Average Runs per Shot',
            'Shot Count': 'Number of Shots',
            'Avg Control Score': 'Control Score (0-100)'
        },
        color_continuous_scale='RdYlGn',
        range_color=[0, 100]
    )
    
    fig.add_hline(y=shot_analysis['Avg Runs'].mean(), line_dash="dash", line_color="gray", opacity=0.5)
    fig.add_vline(x=shot_analysis['True Control Rate'].mean(), line_dash="dash", line_color="gray", opacity=0.5)
    
    fig.add_annotation(x=0.1, y=shot_analysis['Avg Runs'].max()*0.9, text="HIGH RISK<br>LOW REWARD", 
                       bgcolor="rgba(255,0,0,0.1)", bordercolor="red")
    fig.add_annotation(x=0.9, y=shot_analysis['Avg Runs'].max()*0.9, text="SMART CRICKET<br>HIGH REWARD", 
                       bgcolor="rgba(0,255,0,0.1)", bordercolor="green")
    fig.add_annotation(x=0.1, y=shot_analysis['Avg Runs'].min()*1.5, text="POOR EXECUTION<br>LOW REWARD", 
                       bgcolor="rgba(255,100,0,0.1)", bordercolor="orange")
    fig.add_annotation(x=0.9, y=shot_analysis['Avg Runs'].min()*1.5, text="DEFENSIVE<br>CONTROLLED", 
                       bgcolor="rgba(0,100,255,0.1)", bordercolor="blue")
    
    fig.update_traces(textposition="top center")
    fig.update_layout(height=600)
    
    return fig

def create_match_phase_analysis(df):
    """Analyze shot selection across match phases"""
    
    if df.empty or 'totalBallNumber' not in df.columns or 'battingShotTypeId' not in df.columns or 'match_phase' not in df.columns:
        return go.Figure()
    
    phase_analysis = df.groupby(['match_phase', 'battingShotTypeId']).agg({
        'runs': 'mean',
        'is_controlled_shot': 'mean',
        'is_boundary': 'mean'
    }).reset_index()
    
    if phase_analysis.empty:
        return go.Figure()
    
    fig = px.bar(
        phase_analysis,
        x='battingShotTypeId',
        y='runs',
        color='match_phase',
        barmode='group',
        title='Shot Selection Intelligence Across Match Phases',
        labels={'runs': 'Average Runs', 'battingShotTypeId': 'Shot Type'},
        height=500
    )
    
    fig.update_layout(xaxis_tickangle=-45)
    
    return fig

def create_player_comparison_radar(df, selected_players):
    """Create radar chart comparing players across multiple dimensions"""
    
    if len(selected_players) < 2:
        return go.Figure()
    
    metrics_data = {}
    categories = ['Control Rate', 'Avg Runs/Shot', 'Boundary %', 'Control Quality', 'Shot Efficiency']
    
    for player in selected_players[:4]:
        player_data = df[df['batsman'] == player]
        if not player_data.empty:
            control_rate = player_data['is_controlled_shot'].mean() * 100
            avg_runs = player_data['runs'].mean() * 25
            boundary_pct = player_data['is_boundary'].mean() * 100
            control_quality = player_data['control_score'].mean()
            shot_efficiency = min(player_data['true_shot_efficiency'].mean() * 10, 100)
            
            metrics_data[player] = {
                'Control Rate': control_rate,
                'Avg Runs/Shot': avg_runs,
                'Boundary %': boundary_pct,
                'Control Quality': control_quality,
                'Shot Efficiency': shot_efficiency
            }
    
    if not metrics_data:
        return go.Figure()
    
    fig = go.Figure()
    
    for player, metrics in metrics_data.items():
        fig.add_trace(go.Scatterpolar(
            r=[metrics[c] for c in categories],
            theta=categories,
            fill='toself',
            name=player
        ))
        
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100]
            )),
        showlegend=True,
        title='Player Comparison Radar Chart'
    )
    
    return fig


def create_bowling_effectiveness_chart(df):
    """
    Creates a scatter plot to analyze bowler effectiveness.
    X-axis: Average Runs per Ball (Economy)
    Y-axis: Average False Shot Score per Ball (Pressure Index)
    Size: Number of Balls Bowled (Workload)
    Color: Average Control Score against the bowler (Vulnerability)
    """
    if df.empty or 'bowler' not in df.columns or 'false_shot_score' not in df.columns or 'runs' not in df.columns or 'totalBallNumber' not in df.columns:
        return None
    
    bowler_summary = df.groupby('bowler').agg(
        balls_bowled=('totalBallNumber', 'count'),
        runs_conceded=('runs', 'sum'),
        total_false_shot_score=('false_shot_score', 'sum'),
        avg_control_score_against=('control_score', 'mean')
    ).reset_index()
    
    # Filter for bowlers with a minimum number of balls to ensure statistical significance
    bowler_summary = bowler_summary[bowler_summary['balls_bowled'] >= 20]
    
    if bowler_summary.empty:
        return None
    
    bowler_summary['avg_runs_per_ball'] = bowler_summary['runs_conceded'] / bowler_summary['balls_bowled']
    bowler_summary['avg_false_shot_score'] = bowler_summary['total_false_shot_score'] / bowler_summary['balls_bowled']
    
    fig = px.scatter(
        bowler_summary,
        x='avg_runs_per_ball',
        y='avg_false_shot_score',
        size='balls_bowled',
        color='avg_control_score_against',
        text='bowler',
        title='Bowling Effectiveness Matrix: Pressure vs. Economy',
        labels={
            'avg_runs_per_ball': 'Average Runs per Ball (Economy)',
            'avg_false_shot_score': 'Average False Shot Score (Pressure Index)',
            'balls_bowled': 'Balls Bowled',
            'avg_control_score_against': 'Avg. Control Score Against'
        },
        color_continuous_scale='RdYlGn_r', # Reversed scale: Red for low control, Green for high
        range_color=[0, 100],
        hover_data={
            'balls_bowled': True,
            'runs_conceded': True,
            'avg_control_score_against': ':.1f'
        }
    )
    
    avg_economy = bowler_summary['avg_runs_per_ball'].mean()
    avg_pressure = bowler_summary['avg_false_shot_score'].mean()
    
    fig.add_hline(y=avg_pressure, line_dash="dash", line_color="gray", opacity=0.5)
    fig.add_vline(x=avg_economy, line_dash="dash", line_color="gray", opacity=0.5)
    
    fig.add_annotation(x=avg_economy * 1.5, y=avg_pressure * 1.5, text="High Pressure,<br>High Economy", showarrow=False)
    fig.add_annotation(x=avg_economy * 0.5, y=avg_pressure * 1.5, text="Elite Bowler<br>(High Pressure,<br>Low Economy)", showarrow=False)
    fig.add_annotation(x=avg_economy * 1.5, y=avg_pressure * 0.5, text="Passive,<br>High Economy", showarrow=False)
    fig.add_annotation(x=avg_economy * 0.5, y=avg_pressure * 0.5, text="Containment<br>Bowler", showarrow=False)
    
    fig.update_traces(textposition="top center")
    fig.update_layout(height=600)
    
    return fig

def create_pressure_performance_analysis(df):
    """
    Creates a grouped bar chart to analyze player performance across match phases.
    """
    if df.empty or 'batsman' not in df.columns or 'match_phase' not in df.columns:
        return None
    
    phase_performance = df.groupby(['batsman', 'match_phase']).agg(
        avg_control_score=('control_score', 'mean'),
        avg_runs=('runs', 'mean'),
        boundary_rate=('is_boundary', 'mean'),
        balls_faced=('runs', 'count')
    ).reset_index()
    
    if phase_performance.empty:
        return None
    
    fig = px.bar(
        phase_performance,
        x='match_phase',
        y='avg_control_score',
        color='batsman',
        barmode='group',
        text='balls_faced',
        title='Player Performance by Match Phase',
        labels={
            'avg_control_score': 'Average Control Score (0-100)',
            'match_phase': 'Match Phase',
            'balls_faced': 'Balls Faced'
        },
        hover_data=['avg_runs', 'boundary_rate'],
        height=600
    )
    
    fig.update_layout(xaxis_title_text='Match Phase', yaxis_title_text='Average Control Score')
    
    return fig

def create_wicket_probability_heatmap(df):
    """
    Creates a heatmap to visualize average false shot score by shot zone and connection type.
    """
    if df.empty or 'angle_zone' not in df.columns or 'battingConnectionId' not in df.columns or 'false_shot_score' not in df.columns:
        return None
    
    # Filter for non-controlled shots to focus on risk
    risk_df = df[df['false_shot_score'] > 0]
    if risk_df.empty:
        return None
        
    risk_summary = risk_df.groupby(['angle_zone', 'battingConnectionId'])['false_shot_score'].mean().unstack(fill_value=0)
    
    fig = px.imshow(
        risk_summary,
        color_continuous_scale='YlOrRd',
        title='Risk Assessment Heatmap (Average False Shot Score)',
        labels={
            'x': 'Connection Type',
            'y': 'Shot Zone',
            'color': 'Avg False Shot Score'
        },
        height=600,
        width=1000,
        text_auto=".1f"
    )
    
    return fig


# --- Main App Structure ---
if 'data_df' not in st.session_state:
    st.session_state.data_df = load_data_from_github()

if st.session_state.data_df.empty:
    st.warning("No data available to display. Please check the data source or try again later.")
else:
    df = st.session_state.data_df.copy()
    
    # Calculate all the new metrics
    df = calculate_shot_intelligence_metrics(df)
    
    all_players = sorted(df['batsman'].unique())
    all_bowlers = sorted(df['bowler'].unique())
    
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
    
    filtered_df = df[df['batsman'].isin(selected_players)]

    # Dynamic tabs
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "Dashboard",
        "Shot Placement",
        "Shot Matrix",
        "Match Phase Analysis",
        "Player Comparison",
        "Advanced Analytics",
        "Bowler Pressure Analysis"
    ])
    
    with tab1:
        st.subheader("📊 Player Dashboard")
        
        if len(selected_players) == 1:
            player_name = selected_players[0]
            player_data = filtered_df[filtered_df['batsman'] == player_name]
            
            if not player_data.empty:
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    total_runs = player_data['runs'].sum()
                    st.metric("Total Runs", f"{total_runs}")
                with col2:
                    avg_control = player_data['control_score'].mean()
                    st.metric("Avg Control Score", f"{avg_control:.1f}/100")
                with col3:
                    boundary_rate = (player_data['is_boundary'].sum() / len(player_data)) * 100
                    st.metric("Boundary Rate", f"{boundary_rate:.1f}%")
                with col4:
                    false_shot_rate = (player_data['false_shot_score'].sum() / len(player_data))
                    st.metric("Avg False Shot Score", f"{false_shot_rate:.2f}")

                st.markdown('<div class="insight-box">', unsafe_allow_html=True)
                st.markdown("### 🧠 Analyst Insights")
                insights = get_player_insights(player_data)
                
                if 'favorite_shot' in insights:
                    st.write(f"**Favorite Shot:** {insights['favorite_shot']}")
                if 'strength_area' in insights:
                    st.write(f"**Strength Area:** {insights['strength_area']}")
                if 'most_effective' in insights:
                    st.write(f"**Most Effective Shot:** {insights['most_effective']}")
                if 'dismissal_pattern' in insights:
                    st.write(f"**Potential Weakness Zone:** {insights['dismissal_pattern']}")
                if 'bowl_to' in insights:
                    st.markdown("---")
                    st.markdown("#### 🎯 Bowling Strategy Recommendation")
                    st.markdown('<div class="bowling-recommendation">', unsafe_allow_html=True)
                    st.write(f"**To target {player_name}'s weakness:**")
                    if 'bowl_to_connection' in insights:
                         st.write(f"Focus on bowling to the **{insights['bowl_to_connection']}** zone with these attributes:")
                    
                    for rec in insights['bowl_to']:
                        st.write(f"- {rec}")
                    st.markdown('</div>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)

    with tab2:
        st.subheader(f"🌐 Shot Placement Intelligence for {', '.join(selected_players)}")
        
        for player in selected_players:
            st.markdown(f"##### {player}")
            shot_fig = create_shot_angle_heatmap(df, player)
            st.plotly_chart(shot_fig, use_container_width=True)
    
    with tab3:
        st.subheader("📊 Shot Intelligence Matrix")
        
        control_aggression_fig = create_control_vs_aggression_chart(filtered_df)
        st.plotly_chart(control_aggression_fig, use_container_width=True)
        
    with tab4:
        st.subheader("⏳ Performance Across Match Phases")
        match_phase_fig = create_match_phase_analysis(filtered_df)
        st.plotly_chart(match_phase_fig, use_container_width=True)

    with tab5:
        st.subheader("⚖️ Player Comparison")
        if len(selected_players) >= 2:
            radar_fig = create_player_comparison_radar(df, selected_players)
            st.plotly_chart(radar_fig, use_container_width=True)
        else:
            st.info("Please select at least two players for comparison.")
            
    with tab6:
        st.subheader("🔍 Advanced Cricket Analytics")
        
        adv_tab1, adv_tab2, adv_tab3, adv_tab4 = st.tabs([
            "🎳 Bowling Effectiveness",
            "⚡ Pressure Performance", 
            "🎯 Risk Heatmap",
            "📊 False Shot Analysis"
        ])
        
        with adv_tab1:
            st.markdown("##### 🎳 Bowler Effectiveness Matrix")
            st.markdown("Analyzing bowlers based on their ability to induce false shots while maintaining economy")
            
            try:
                bowling_fig = create_bowling_effectiveness_chart(filtered_df)
                if bowling_fig is not None:
                    st.plotly_chart(bowling_fig, use_container_width=True)
                else:
                    st.info("📊 Not enough bowling data to create effectiveness chart (need at least 20 balls per bowler)")
                
                if 'bowler' in filtered_df.columns and 'false_shot_score' in filtered_df.columns:
                    st.markdown("##### 🏆 Top Pressure Bowlers (by Average False Shot Score)")
                    
                    bowler_summary = filtered_df.groupby('bowler').agg(
                        total_false_shots=('false_shot_score', 'sum'),
                        balls_bowled=('totalBallNumber', 'count'),
                        runs_conceded=('runs', 'sum'),
                        avg_control_against=('control_score', 'mean')
                    ).round(2).reset_index()
                    
                    bowler_summary = bowler_summary[bowler_summary['balls_bowled'] >= 20]
                    
                    if not bowler_summary.empty:
                        bowler_summary['Avg False Shot Score'] = bowler_summary['total_false_shots'] / bowler_summary['balls_bowled']
                        bowler_summary['Economy'] = bowler_summary['runs_conceded'] / bowler_summary['balls_bowled']
                        
                        bowler_summary = bowler_summary[['bowler', 'Avg False Shot Score', 'Economy', 'balls_bowled', 'avg_control_against']]
                        bowler_summary.columns = ['Bowler', 'Avg False Shot Score', 'Economy', 'Balls Bowled', 'Avg Control Against']
                        bowler_summary = bowler_summary.sort_values('Avg False Shot Score', ascending=False)
                        st.dataframe(bowler_summary.head(10), use_container_width=True)
                    else:
                        st.info("No bowlers with sufficient data (minimum 20 balls)")
                else:
                    st.info("Bowler data not available in current selection")
            
            except Exception as e:
                st.error(f"Error in bowling analysis: {e}")
        
        with adv_tab2:
            st.markdown("##### ⚡ Performance Under Pressure")
            st.markdown("Comparing player performance across different match phases based on control score")
            
            try:
                pressure_fig = create_pressure_performance_analysis(filtered_df)
                if pressure_fig is not None:
                    st.plotly_chart(pressure_fig, use_container_width=True)
                else:
                    st.info("📊 Not enough data to create pressure analysis")
                
                if 'match_phase' in filtered_df.columns and selected_players:
                    st.markdown("##### 📈 Pressure Statistics")
                    
                    pressure_stats = filtered_df.groupby(['batsman', 'match_phase']).agg(
                        avg_control_score=('control_score', 'mean'),
                        avg_runs_per_ball=('runs', lambda x: x.sum() / x.count()),
                        boundary_rate=('is_boundary', 'mean'),
                        balls_faced=('runs', 'count')
                    ).round(2).reset_index()
                    
                    if not pressure_stats.empty:
                        st.dataframe(pressure_stats, use_container_width=True)
                    else:
                        st.info("No pressure statistics available for selected players")
                else:
                    st.info("Please select players to view pressure statistics")
            
            except Exception as e:
                st.error(f"Error in pressure analysis: {e}")
        
        with adv_tab3:
            st.markdown("##### 🎯 Risk Assessment Heatmap")
            st.markdown("Visualizing risk levels across different shot zones and connection types")
            
            try:
                risk_fig = create_wicket_probability_heatmap(filtered_df)
                if risk_fig is not None:
                    st.plotly_chart(risk_fig, use_container_width=True)
                else:
                    st.info("📊 Not enough data to create risk heatmap")
                    
                if 'angle_zone' in filtered_df.columns and 'battingConnectionId' in filtered_df.columns:
                    st.markdown("##### 📊 Risk Summary Table")
                    risk_summary = filtered_df.groupby(['angle_zone', 'battingConnectionId']).agg({
                        'false_shot_score': ['mean', 'count'],
                        'runs': 'mean'
                    }).round(2)
                    risk_summary.columns = ['Avg Risk Score', 'Frequency', 'Avg Runs']
                    risk_summary = risk_summary.reset_index()
                    risk_summary = risk_summary[risk_summary['Frequency'] >= 2]
                    risk_summary = risk_summary.sort_values('Avg Risk Score', ascending=False)
                    st.dataframe(risk_summary.head(15), use_container_width=True)
            
            except Exception as e:
                st.error(f"Error in risk analysis: {e}")
        
        with adv_tab4:
            st.markdown("##### 📊 Comprehensive False Shot Analysis")
            st.markdown("Detailed breakdown of false shot patterns and their implications")
            
            try:
                if 'battingConnectionId' in filtered_df.columns:
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("##### 🔍 False Shot Score by Connection")
                        connection_scores = filtered_df.groupby('battingConnectionId').agg({
                            'false_shot_score': ['mean', 'count'],
                            'runs': 'mean',
                            'control_score': 'mean'
                        }).round(2)
                        
                        connection_scores.columns = ['Avg False Shot Score', 'Frequency', 'Avg Runs', 'Avg Control Score']
                        connection_scores = connection_scores.sort_values('Avg False Shot Score', ascending=False)
                        st.dataframe(connection_scores, use_container_width=True)
                    
                    with col2:
                        st.markdown("##### 🎯 Most Vulnerable Shot Types")
                        if 'battingShotTypeId' in filtered_df.columns:
                            shot_vulnerability = filtered_df.groupby('battingShotTypeId').agg({
                                'false_shot_score': 'mean',
                                'control_score': 'mean',
                                'runs': 'mean'
                            }).round(2)
                            
                            shot_vulnerability.columns = ['Avg False Shot Score', 'Avg Control Score', 'Avg Runs']
                            shot_vulnerability = shot_vulnerability.sort_values('Avg False Shot Score', ascending=False)
                            st.dataframe(shot_vulnerability, use_container_width=True)
                        else:
                            st.info("Shot type data not available")
                    
                    st.markdown("##### 📈 False Shot Interpretation Guide")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown("""
                        **False Shot Score Scale:**
                        - **0-1:** Excellent control and timing ✅
                        - **2-3:** Moderate risk, some mistiming ⚠️
                        - **4-5:** High risk, poor connection/miss ❌
                        """)
                    
                    with col2:
                        st.markdown("""
                        **Key Connection Types:**
                        - **Middled/WellTimed:** Perfect shots (0 score)
                        - **MisTimed/Edges:** Risky shots (3-4 score)
                        - **Missed/PlayAndMiss:** Dangerous (5 score)
                        """)
                    
                    st.markdown("##### 📊 Summary Statistics")
                    st.markdown('<div class="advanced-metric">', unsafe_allow_html=True)
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        avg_false_score = filtered_df['false_shot_score'].mean()
                        st.metric("Avg False Shot Score", f"{avg_false_score:.2f}")
                    
                    with col2:
                        high_risk_pct = (filtered_df['false_shot_score'] >= 4).mean() * 100
                        st.metric("High Risk Shots %", f"{high_risk_pct:.1f}%")
                    
                    with col3:
                        perfect_shots_pct = (filtered_df['false_shot_score'] == 0).mean() * 100
                        st.metric("Perfect Shots %", f"{perfect_shots_pct:.1f}%")
                    
                    with col4:
                        avg_control = filtered_df['control_score'].mean()
                        st.metric("Avg Control Score", f"{avg_control:.1f}/100")
                    
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                    st.markdown("##### 📈 False Shot Score Distribution")
                    false_shot_dist = filtered_df['false_shot_score'].value_counts().sort_index()
                    
                    fig_dist = px.bar(
                        x=false_shot_dist.index,
                        y=false_shot_dist.values,
                        title='Distribution of False Shot Scores',
                        labels={'x': 'False Shot Score', 'y': 'Frequency'},
                        color=false_shot_dist.index,
                        color_continuous_scale='RdYlGn_r'
                    )
                    st.plotly_chart(fig_dist, use_container_width=True)
                
                else:
                    st.info("Connection data not available for false shot analysis")
            
            except Exception as e:
                st.error(f"Error in false shot analysis: {e}")
    
    with tab7:
        st.subheader("🎯 Bowler Pressure Analysis")
        st.markdown(
            """
            This section provides a deep-dive into bowlers who consistently induce false shots,
            making them a high-pressure threat to batsmen.
            """
        )

        pressure_tab1, pressure_tab2 = st.tabs([
            "📊 Bowler Pressure Index",
            "🔍 False Shot Patterns"
        ])

        with pressure_tab1:
            st.markdown("##### 📈 Bowler Pressure Index")
            st.markdown("Ranking bowlers by their ability to induce false shots (a high score indicates more pressure).")

            try:
                min_balls_bowled = 20
                
                if 'bowler' in filtered_df.columns and 'false_shot_score' in filtered_df.columns:
                    bowler_pressure_df = filtered_df.groupby('bowler').agg(
                        total_false_shots=('false_shot_score', 'sum'),
                        balls_bowled=('totalBallNumber', 'count'),
                        runs_conceded=('runs', 'sum'),
                        avg_control_against=('control_score', 'mean')
                    ).round(2).reset_index()

                    bowler_pressure_df = bowler_pressure_df[bowler_pressure_df['balls_bowled'] >= min_balls_bowled]

                    if not bowler_pressure_df.empty:
                        bowler_pressure_df['Pressure Index'] = bowler_pressure_df['total_false_shots'] / bowler_pressure_df['balls_bowled']
                        bowler_pressure_df['Economy'] = bowler_pressure_df['runs_conceded'] / bowler_pressure_df['balls_bowled']
                        
                        bowler_pressure_df = bowler_pressure_df[[
                            'bowler', 'Pressure Index', 'Economy', 'balls_bowled', 'avg_control_against'
                        ]]
                        bowler_pressure_df.columns = [
                            'Bowler', 'Pressure Index', 'Economy', 'Balls Bowled', 'Avg Control Against'
                        ]
                        
                        bowler_pressure_df = bowler_pressure_df.sort_values('Pressure Index', ascending=False)

                        st.dataframe(bowler_pressure_df, use_container_width=True)

                        st.markdown("""
                            **Interpretation:**
                            - **High Pressure Index:** The bowler consistently forces batsmen into mistakes (e.g., edges, play-and-misses).
                            - **Low Economy + High Pressure Index:** This is a lethal combination. The bowler is a genuine threat to take wickets.
                        """)
                    else:
                        st.info(f"No bowlers with sufficient data (minimum {min_balls_bowled} balls bowled).")
                else:
                    st.info("Bowler or false shot data not available in the current selection.")

            except Exception as e:
                st.error(f"An error occurred: {e}")

        with pressure_tab2:
            st.markdown("##### 🔍 False Shot Patterns by Bowler")
            st.markdown("Visualizing which types of shots each bowler is most effective at causing errors in.")

            try:
                if 'bowler' in filtered_df.columns and 'false_shot_score' in filtered_df.columns and 'battingShotTypeId' in filtered_df.columns:
                    
                    available_bowlers = filtered_df['bowler'].unique()
                    if len(available_bowlers) > 1:
                        selected_bowler = st.selectbox("Select a Bowler", options=sorted(available_bowlers))
                    elif len(available_bowlers) == 1:
                        selected_bowler = available_bowlers[0]
                        st.write(f"Analyzing false shot patterns for **{selected_bowler}**")
                    else:
                        st.info("No bowler data available for this analysis.")
                        selected_bowler = None

                    if selected_bowler:
                        bowler_data = filtered_df[filtered_df['bowler'] == selected_bowler]
                        
                        if not bowler_data.empty and len(bowler_data) >= 10:
                            
                            false_shot_dist = bowler_data['false_shot_score'].value_counts().sort_index()
                            
                            fig_dist_bowler = px.bar(
                                x=false_shot_dist.index,
                                y=false_shot_dist.values,
                                title=f'False Shot Score Distribution for {selected_bowler}',
                                labels={'x': 'False Shot Score', 'y': 'Frequency'},
                                color=false_shot_dist.index,
                                color_continuous_scale='RdYlGn_r'
                            )
                            st.plotly_chart(fig_dist_bowler, use_container_width=True)

                            st.markdown("###### Most Vulnerable Shot Types Against This Bowler")
                            
                            shot_vulnerability = bowler_data.groupby('battingShotTypeId').agg({
                                'false_shot_score': ['mean', 'count'],
                                'control_score': 'mean',
                                'runs': 'mean'
                            }).round(2)
                            
                            shot_vulnerability.columns = ['Avg False Shot Score', 'Frequency', 'Avg Control Score', 'Avg Runs']
                            shot_vulnerability = shot_vulnerability[shot_vulnerability['Frequency'] >= 3]
                            shot_vulnerability = shot_vulnerability.sort_values('Avg False Shot Score', ascending=False)
                            st.dataframe(shot_vulnerability, use_container_width=True)

                        else:
                            st.info(f"Not enough data for {selected_bowler} to perform a detailed analysis (minimum 10 balls).")

                else:
                    st.info("Required data columns ('bowler', 'false_shot_score', 'battingShotTypeId') are not available.")

            except Exception as e:
                st.error(f"An error occurred: {e}")
