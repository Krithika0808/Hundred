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
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data_from_github(github_url=None):
    """Load cricket data from GitHub repository"""
    try:
        # Default GitHub URL for your repository
        if github_url is None:
            github_url = "https://raw.githubusercontent.com/Krithika0808/Hundred/main/Hundred.csv"
        
        # Fetch data from GitHub
        response = requests.get(github_url)
        response.raise_for_status()  # Raise an exception for bad status codes
        
        # Read CSV from the response content
        csv_content = StringIO(response.text)
        df = pd.read_csv(csv_content)
        
        # Data cleaning and preprocessing
        if df.empty:
            return df
            
        # Handle missing values in key columns
        numeric_columns = ['shotAngle', 'shotMagnitude', 'runs', 'totalBallNumber']
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Convert boolean columns
        bool_columns = ['isWicket', 'isBoundary', 'isAirControlled', 'isWide', 'isNoBall']
        for col in bool_columns:
            if col in df.columns:
                df[col] = df[col].astype(bool)
        
        # Clean player names (remove extra spaces)
        string_columns = ['batsman', 'bowler', 'battingShotTypeId', 'battingConnectionId', 'commentary']
        for col in string_columns:
            if col in df.columns:
                df[col] = df[col].astype(str).str.strip()
        
        # Filter out invalid data
        essential_columns = [col for col in ['batsman', 'runs', 'totalBallNumber'] if col in df.columns]
        if essential_columns:
            df = df.dropna(subset=essential_columns)
        
        # Add match context
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
    shot_types = ['Drive', 'Pull', 'Cut', 'Sweep', 'Flick', 'Hook', 'Reverse Sweep', 'Loft']
    connection_types = ['Middled', 'WellTimed', 'Undercontrol', 'MisTimed', 'Missed', 'HitBody']
    
    # Bowling-related data
    length_types = ['Yorker', 'Full', 'Good Length', 'Short', 'Bouncer']
    line_types = ['Off Stump', 'Middle Stump', 'Leg Stump', 'Wide Outside Off', 'DownLeg']
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
        # Bowling-related columns
        'lengthTypeId': np.random.choice(length_types, n_rows),
        'lineTypeId': np.random.choice(line_types, n_rows),
        'bowlingTypeId': np.random.choice(bowling_types, n_rows),
        'bowlingFromId': np.random.choice(bowling_from, n_rows),
        'bowlingHandId': np.random.choice(bowling_hands, n_rows),
        'fieldingPosition': np.random.choice(fielding_positions, n_rows)
    }
    
    df = pd.DataFrame(data)
    
    # Make boundaries more realistic based on runs
    df.loc[df['runs'] == 4, 'isBoundary'] = True
    df.loc[df['runs'] == 6, 'isBoundary'] = True
    df.loc[df['runs'] < 4, 'isBoundary'] = False
    
    # Update commentary based on runs and connection
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

def calculate_shot_intelligence_metrics(df):
    """Calculate advanced shot intelligence metrics"""
    
    if df.empty:
        return df
    
    # Handle missing values and ensure proper data types
    df = df.copy()
    
    # Fill missing values with appropriate defaults
    df['shotAngle'] = df['shotAngle'].fillna(0)
    df['shotMagnitude'] = df['shotMagnitude'].fillna(100)
    df['isAirControlled'] = df['isAirControlled'].fillna(False)
    df['battingConnectionId'] = df['battingConnectionId'].fillna('Unknown')
    df['commentary'] = df['commentary'].fillna('')
    
    # BOUNDARY DETECTION FROM COMMENTARY AND RUNS
    def detect_boundary_from_data(row):
        runs = row.get('runs', 0)
        commentary = str(row['commentary']).lower()
        is_boundary_col = row.get('isBoundary', False)
        
        # Primary check: runs = 4 or 6
        if runs == 4 or runs == 6:
            return True
        
        # Secondary check: isBoundary column
        if is_boundary_col:
            return True
        
        # Tertiary check: commentary mentions boundary
        boundary_keywords = ['four', '4 runs', 'boundary', 'reaches the rope', 'to the fence', 
                           'six', '6 runs', 'maximum', 'over the rope', 'into the stands',
                           'hits it for four', 'hits it for six']
        
        for keyword in boundary_keywords:
            if keyword in commentary:
                return True
                
        return False
    
    df['is_boundary'] = df.apply(detect_boundary_from_data, axis=1)
    
    # 1. TRUE CONTROL MASTERY - Based on battingConnectionId
    def determine_true_control(row):
        connection = str(row['battingConnectionId']).strip()
        
        # High Control Shots
        if connection in ['Middled', 'WellTimed', 'Undercontrol', 'Left']:
            return 'High Control'
        
        # Poor Control Shots  
        elif connection in ['MisTimed', 'BottomEdge', 'TopEdge', 'BatPad', 'Mis-timed', 'InsideEdge', 'LeadingEdge', 'OutsideEdge', 'Gloved', 'ThickEdge', 'TopEdge']:
            return 'Less Control'
        
        # Unknown/Other
        else:
            return 'Poor Control'
    
    df['true_control_category'] = df.apply(determine_true_control, axis=1)
    
    # 2. ADVANCED CONTROL METRICS
    
    # Control Quality Score (0-3 scale)
    control_scores = {
        'Middled': 3,      # Perfect connection
        'WellTimed': 3,  # Excellent timing
        'Undercontrol': 3, # Good control
        'MisTimed': 2,     # Poor timing
        'Missed': 0,       # Complete miss
        'HitBody': 0.5,    # Hit body/padding
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
        'TopEdge': 2      # Default middle value
    }
    
    df['control_quality_score'] = df['battingConnectionId'].map(control_scores).fillna(1.5)
    
    # Create angle_zone column early
    valid_angles = (df['shotAngle'] >= 0) & (df['shotAngle'] <= 360)
    df.loc[~valid_angles, 'shotAngle'] = 0
    
    angle_bins = [0, 45, 90, 135, 180, 225, 270, 315, 360]
    angle_labels = ['Long Off', 'Cover', 'Point', 'Third Man', 'Fine Leg', 'Square Leg', 'Mid Wicket', 'Long On']
    
    df['angle_zone'] = pd.cut(df['shotAngle'], bins=angle_bins, labels=angle_labels, include_lowest=True)
    
    # ENHANCED CONTROL RATE CALCULATION - HYBRID APPROACH
    # Start with base control quality (0-100 scale)
    df['control_score'] = df['control_quality_score'] * 33.33
    
    # Add outcome factors
    df['control_score'] += df['runs'] * 5
    df['control_score'] += df['is_boundary'] * 20
    
    # Add placement factors
    good_placements = ['Cover', 'Mid Wicket', 'Long On', 'Long Off']
    df['control_score'] += df['angle_zone'].isin(good_placements) * 10
    
    # Cap at 100
    df['control_score'] = df['control_score'].clip(0, 100)
    
    # Define controlled shots as score >= 50
    df['is_controlled_shot'] = (df['control_score'] >= 50).astype(int)
    
    # 3. Shot Execution Intelligence
    df['execution_intelligence'] = df['control_quality_score'] * (df['runs'] + 1) / 2
    
    # 4. Risk-Reward with True Control
    df['true_risk_reward'] = np.where(
        df['is_controlled_shot'] == 1,
        df['runs'] * 1.3,  # Bonus for controlled shots
        df['runs'] * 0.7   # Penalty for poor shots
    )
    
    # 5. Shot Efficiency based on True Control
    df['true_shot_efficiency'] = np.where(
        df['shotMagnitude'] > 0,
        (df['runs'] * df['control_quality_score']) / (df['shotMagnitude'] / 100),
        df['runs'] * df['control_quality_score']
    )
    
    # 6. Aerial vs Ground Control Analysis
    df['shot_category'] = np.where(
        df['isAirControlled'] == True,
        'Aerial_' + df['true_control_category'],
        'Ground_' + df['true_control_category']
    )
    
    # 7. Match situation metrics - FIXED BIN CREATION
    if 'totalBallNumber' in df.columns:
        max_ball = int(df['totalBallNumber'].max())
        
        # Create match phase bins ensuring they are monotonically increasing
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
    
    # 8. Pressure Performance with True Control
    if 'totalBallNumber' in df.columns:
        df['is_death_phase'] = (df['totalBallNumber'] >= 76).astype(int)
        df['death_control_performance'] = df['is_death_phase'] * df['is_controlled_shot']
    
    # 9. Shot selection intelligence
    if 'battingShotTypeId' in df.columns:
        shot_control_avg = df.groupby('battingShotTypeId')['control_quality_score'].mean()
        df['shot_type_control_avg'] = df['battingShotTypeId'].map(shot_control_avg)
        
        df['selection_wisdom'] = np.where(
            df['control_quality_score'] >= df['shot_type_control_avg'],
            'Smart Selection',
            'Risky Selection'
        )
    
    return df

def get_player_insights(player_data):
    """Generate player insights including dismissal patterns, favorite shots, and bowling recommendations"""
    insights = {}
    
    # 1. Favorite Shot
    if 'battingShotTypeId' in player_data.columns:
        shot_counts = player_data['battingShotTypeId'].value_counts()
        if not shot_counts.empty:
            favorite_shot = shot_counts.index[0]
            favorite_count = shot_counts.iloc[0]
            total_shots = len(player_data)
            favorite_percentage = (favorite_count / total_shots) * 100
            insights['favorite_shot'] = f"{favorite_shot} ({favorite_percentage:.1f}% of shots)"
    
    # 2. Dismissal Patterns (based on poor control areas)
    dismissal_zone = None
    if 'angle_zone' in player_data.columns and 'true_control_category' in player_data.columns:
        # Identify areas with poor control
        poor_control_shots = player_data[player_data['true_control_category'].isin(['Poor Control', 'Less Control'])]
        if not poor_control_shots.empty:
            dismissal_zones = poor_control_shots['angle_zone'].value_counts()
            if not dismissal_zones.empty:
                dismissal_zone = dismissal_zones.index[0]
                dismissal_count = dismissal_zones.iloc[0]
                total_poor = len(poor_control_shots)
                dismissal_percentage = (dismissal_count / total_poor) * 100
                insights['dismissal_pattern'] = f"{dismissal_zone} ({dismissal_percentage:.1f}% of poor control shots)"
    
    # 3. Bowling Recommendations
    bowling_columns = ['lengthTypeId', 'lineTypeId', 'bowlingTypeId', 'bowlingFromId', 'bowlingHandId']
    available_bowling_cols = [col for col in bowling_columns if col in player_data.columns]
    
    if available_bowling_cols and 'control_score' in player_data.columns:
        bowling_weaknesses = []
        
        for col in available_bowling_cols:
            if col in player_data.columns:
                # Group by the bowling attribute and calculate average control score
                attr_control = player_data.groupby(col)['control_score'].mean()
                if not attr_control.empty:
                    # Find the attribute value with lowest control score
                    weakest_attr = attr_control.idxmin()
                    weakest_score = attr_control.min()
                    
                    # Only consider it a weakness if control score is below 60
                    if weakest_score < 60:
                        # Format the attribute name for display
                        col_names = {
                            'lengthTypeId': 'Length',
                            'lineTypeId': 'Line',
                            'bowlingTypeId': 'Type',
                            'bowlingFromId': 'From',
                            'bowlingHandId': 'Hand'
                        }
                        
                        display_name = col_names.get(col, col)
                        bowling_weaknesses.append((display_name, weakest_attr, weakest_score))
        
        # Sort by control score to find the biggest weaknesses
        if bowling_weaknesses:
            bowling_weaknesses.sort(key=lambda x: x[2])  # Sort by control score
            
            # Take top 2-3 weaknesses
            top_weaknesses = bowling_weaknesses[:3]
            
            # Format the recommendations
            recommendations = []
            for name, value, score in top_weaknesses:
                recommendations.append(f"{name}: {value} ({score:.1f}/100)")
            
            insights['bowl_to'] = recommendations
    
    # 4. Strength Areas
    if 'angle_zone' in player_data.columns and 'control_score' in player_data.columns:
        zone_control = player_data.groupby('angle_zone')['control_score'].mean().sort_values(ascending=False)
        if not zone_control.empty:
            strongest_zone = zone_control.index[0]
            strongest_score = zone_control.iloc[0]
            insights['strength_area'] = f"{strongest_zone} (avg control: {strongest_score:.1f}/100)"
    
    # 5. Most Effective Shot
    if 'battingShotTypeId' in player_data.columns and 'runs' in player_data.columns:
        shot_runs = player_data.groupby('battingShotTypeId')['runs'].agg(['mean', 'count'])
        shot_runs = shot_runs[shot_runs['count'] >= 5]  # Only consider shots with at least 5 attempts
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
    
    # Remove invalid angles and magnitudes
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
    
    # Add scatter plot for shots
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
    
    # Filter out shot types with very few attempts
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
    
    # Add reference lines
    fig.add_hline(y=shot_analysis['Avg Runs'].mean(), line_dash="dash", line_color="gray", opacity=0.5)
    fig.add_vline(x=shot_analysis['True Control Rate'].mean(), line_dash="dash", line_color="gray", opacity=0.5)
    
    # Add quadrant labels
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
    
    metrics = []
    categories = ['Control Rate', 'Avg Runs/Shot', 'Boundary %', 'Control Quality', 'Shot Efficiency']
    
    for player in selected_players[:4]:  # Limit to 4 players for readability
        player_data = df[df['batsman'] == player]
        if not player_data.empty:
            control_rate = player_data['is_controlled_shot'].mean() * 100
            avg_runs = player_data['runs'].mean() * 25  # Scale for radar
            boundary_pct = player_data['is_boundary'].mean() * 100
            control_quality = player_data['control_score'].mean()
            shot_efficiency = min(player_data['true_shot_efficiency'].mean() * 10, 100)  # Cap at 100
            
            metrics.append([control_rate, avg_runs, boundary_pct, control_quality, shot_efficiency])
    
    fig = go.Figure()
    
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
    
    for i, (player, metric_values) in enumerate(zip(selected_players[:4], metrics)):
        fig.add_trace(go.Scatterpolar(
            r=metric_values + [metric_values[0]],  # Close the polygon
            theta=categories + [categories[0]],
            fill='toself',
            name=player,
            line_color=colors[i],
            fillcolor=colors[i],
            opacity=0.3
        ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100]
            )
        ),
        title="Player Comparison - Multi-Dimensional Analysis",
        height=500
    )
    
    return fig

def create_dismissal_analysis(df, selected_batter):
    """Create dismissal analysis charts"""
    dismissals = df[(df['batsman'] == selected_batter) & (df['isWicket'] == True)]
    
    if dismissals.empty:
        return None, None, None, None
    
    # Timing Mapping
    timing_map = {
        'WellTimed': 'WellTimed',
        'Undercontrol': 'Controlled',
        'Missed': 'Missed',
        'Edge': 'Edged',
        'NotApplicable': 'Unknown',
        np.nan: 'Unknown'
    }
    dismissals['Timing'] = dismissals['battingConnectionId'].map(timing_map).fillna('Unknown')
    
    # Summary Table
    summary = dismissals.groupby(
        ['fieldingPosition', 'lineTypeId', 'lengthTypeId', 'bowlingTypeId', 'Timing']
    ).size().reset_index(name='Dismissals') if 'fieldingPosition' in dismissals.columns else pd.DataFrame()
    
    # Dismissals by fielding position
    if 'fieldingPosition' in dismissals.columns:
        zone_counts = dismissals['fieldingPosition'].value_counts().reset_index()
        zone_counts.columns = ['Fielding Position', 'Dismissals']
        fig1 = px.bar(
            zone_counts,
            x='Fielding Position', y='Dismissals',
            color='Fielding Position', title='Dismissals by Fielding Position'
        )
    else:
        fig1 = go.Figure()
    
    # Dismissal timing pie
    fig2 = px.pie(
        dismissals,
        names='Timing',
        title='Dismissal Timing Distribution',
        hole=0.4
    )
    
    # Line vs Length Heatmap
    if 'lineTypeId' in dismissals.columns and 'lengthTypeId' in dismissals.columns:
        heatmap_data = dismissals.groupby(['lineTypeId', 'lengthTypeId']).size().unstack(fill_value=0)
        fig3 = go.Figure(data=go.Heatmap(
            z=heatmap_data.values,
            x=heatmap_data.columns,
            y=heatmap_data.index,
            colorscale='Reds',
            hoverongaps=False
        ))
        fig3.update_layout(xaxis_title='Length', yaxis_title='Line', title='Line vs Length Dismissals Heatmap')
    else:
        fig3 = go.Figure()
    
    return summary, fig1, fig2, fig3

def main():
    """Main Streamlit app"""
    
    st.markdown('<h1 class="main-header">🏏 Women\'s Cricket Shot Intelligence Matrix</h1>', unsafe_allow_html=True)
    
    # Load data automatically from GitHub repository
    with st.spinner("Loading data from GitHub repository..."):
        df = load_data_from_github()
    
    if df.empty:
        st.error("⚠️ No data could be loaded. Using sample data for demonstration.")
        df = create_sample_data()
    
    # Process the data
    df = calculate_shot_intelligence_metrics(df)
    
    # Sidebar filters
    st.sidebar.header("🎯 Analysis Filters")
    
    # Get available players
    available_players = df['batsman'].unique() if 'batsman' in df.columns else []
    if len(available_players) == 0:
        st.error("❌ No batsman data found in the dataset.")
        return
        
    selected_players = st.sidebar.multiselect(
        "Select Players",
        options=available_players,
        default=list(available_players)[:min(5, len(available_players))]
    )
    
    # Shot types filter
    available_shots = df['battingShotTypeId'].unique() if 'battingShotTypeId' in df.columns else []
    if len(available_shots) > 0:
        selected_shots = st.sidebar.multiselect(
            "Select Shot Types",
            options=available_shots,
            default=list(available_shots)
        )
    else:
        selected_shots = []
    
    # Ball range filter
    if 'totalBallNumber' in df.columns and df['totalBallNumber'].notna().any():
        max_balls = int(df['totalBallNumber'].max())
        min_balls = int(df['totalBallNumber'].min())
        ball_range = st.sidebar.slider(
            "Ball Range",
            min_value=min_balls,
            max_value=max_balls,
            value=(min_balls, max_balls)
        )
    else:
        ball_range = (1, 100)
    
    # Apply filters
    filtered_df = df[df['batsman'].isin(selected_players)] if selected_players else df
    
    if selected_shots and 'battingShotTypeId' in df.columns:
        filtered_df = filtered_df[filtered_df['battingShotTypeId'].isin(selected_shots)]
    
    if 'totalBallNumber' in df.columns:
        filtered_df = filtered_df[filtered_df['totalBallNumber'].between(ball_range[0], ball_range[1])]
    
    if filtered_df.empty:
        st.warning("⚠️ No data matches your current filters. Please adjust your selection.")
        return
    
    # Main dashboard tabs
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "🎯 Shot Placement", 
        "⚡ Control vs Aggression", 
        "📊 Match Phase Analysis",
        "🏆 Player Intelligence",
        "📈 Player Comparison",
        "🔍 Advanced Analytics"
    ])
    
    with tab1:
        st.subheader("360° Shot Placement Intelligence")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            if selected_players:
                selected_player = st.selectbox("Select Player for Detailed Analysis", selected_players)
                if selected_player:
                    fig = create_shot_angle_heatmap(filtered_df, selected_player)
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Please select at least one player from the sidebar.")
        
        with col2:
            st.markdown("##### 🧠 Interpretation")
            st.markdown("""
                - **Green markers** indicate excellent control (80-100).
                - **Orange markers** indicate moderate control (50-79).
                - **Red markers** indicate poor control (0-49).
                - **Marker size** increases with runs scored.
                - Hover to see shot type, runs, connection, and control score.
            """)
    
    with tab2:
        st.subheader("Control vs Aggression Matrix")
        fig = create_control_vs_aggression_chart(filtered_df)
        st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.subheader("Match Phase Shot Analysis")
        fig = create_match_phase_analysis(filtered_df)
        st.plotly_chart(fig, use_container_width=True)
    
    with tab4:
        st.subheader("Player Intelligence Cards")
        
        if selected_players:
            for player in selected_players:
                player_data = filtered_df[filtered_df['batsman'] == player]
                if not player_data.empty:
                    # Basic metrics card
                    st.markdown(f"#### {player}")
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Control Rate", f"{player_data['is_controlled_shot'].mean() * 100:.1f}%")
                    with col2:
                        st.metric("Avg Runs/Shot", f"{player_data['runs'].mean():.2f}")
                    with col3:
                        st.metric("Boundary %", f"{player_data['is_boundary'].mean() * 100:.1f}%")
                    st.progress(min(player_data['control_score'].mean() / 100, 1.0))
                    st.caption("Control Score Progress (0-100)")
                    
                    # Player insights card
                    insights = get_player_insights(player_data)
                    if insights:
                        st.markdown('<div class="insight-card">', unsafe_allow_html=True)
                        st.markdown("##### 🎯 Player Insights")
                        
                        # Display insights in a grid
                        insight_cols = st.columns(2)
                        
                        with insight_cols[0]:
                            if 'favorite_shot' in insights:
                                st.markdown(f'<div class="insight-title">🏏 Favorite Shot</div>', unsafe_allow_html=True)
                                st.markdown(f'<div class="insight-content">{insights["favorite_shot"]}</div>', unsafe_allow_html=True)
                            
                            if 'dismissal_pattern' in insights:
                                st.markdown(f'<div class="insight-title">⚠️ Dismissal Pattern</div>', unsafe_allow_html=True)
                                st.markdown(f'<div class="insight-content">{insights["dismissal_pattern"]}</div>', unsafe_allow_html=True)
                        
                        with insight_cols[1]:
                            if 'bowl_to' in insights:
                                st.markdown(f'<div class="insight-title">🎯 Bowl To</div>', unsafe_allow_html=True)
                                # Display multiple bowling recommendations
                                st.markdown('<div class="bowling-recommendation">', unsafe_allow_html=True)
                                for rec in insights['bowl_to']:
                                    st.markdown(f'<div class="recommendation-item">• {rec}</div>', unsafe_allow_html=True)
                                st.markdown('</div>', unsafe_allow_html=True)
                            
                            if 'strength_area' in insights:
                                st.markdown(f'<div class="insight-title">💪 Strength Area</div>', unsafe_allow_html=True)
                                st.markdown(f'<div class="insight-content">{insights["strength_area"]}</div>', unsafe_allow_html=True)
                            
                            if 'most_effective' in insights:
                                st.markdown(f'<div class="insight-title">🚀 Most Effective</div>', unsafe_allow_html=True)
                                st.markdown(f'<div class="insight-content">{insights["most_effective"]}</div>', unsafe_allow_html=True)
                        
                        st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.warning("Please select at least one player from the sidebar to view intelligence cards.")
    
    with tab5:
        st.subheader("Radar Comparison of Selected Players")
        if selected_players and len(selected_players) >= 2:
            radar_fig = create_player_comparison_radar(filtered_df, selected_players)
            st.plotly_chart(radar_fig, use_container_width=True)
        else:
            st.warning("Please select at least 2 players from the sidebar for comparison.")
    with tab6:
            st.header("📉 Bowl-To Strategy: Dismissal Analysis")
            if selected_players:
               selected_batter = st.selectbox(
                "Select Batter for Dismissal Analysis",
                 options=selected_players,
            key="dismissal_batter"
        )
        
        # Get dismissal analysis data
        dismissals = df[(df['batsman'] == selected_batter) & (df['isWicket'] == True)]
        if dismissals.empty:
            st.info("No dismissal data available for this player")
            return
        
        # Timing Mapping
        timing_map = {
            'WellTimed': 'Well Timed',
            'Undercontrol': 'Controlled',
            'Missed': 'Missed',
            'Edge': 'Edged',
            'NotApplicable': 'Unknown'
        }
        dismissals['Timing'] = dismissals['battingConnectionId'].map(timing_map).fillna('Unknown')
        
        # Summary Table
        st.subheader("🧠 Dismissal Zones Summary")
        summary = dismissals.groupby(
            ['fieldingPosition', 'lineTypeId', 'lengthTypeId', 'bowlingTypeId', 'Timing']
        ).size().reset_index(name='dismissals')
        st.dataframe(summary.sort_values(by='dismissals', ascending=False), use_container_width=True)
        
        # Create 2-column layout
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📊 Dismissals by Fielding Zone")
            zone_counts = dismissals['fieldingPosition'].value_counts().reset_index()
            zone_counts.columns = ['Fielding Position', 'Dismissals']
            st.plotly_chart(
                px.bar(zone_counts, x='Fielding Position', y='dismissals'),
                use_container_width=True
            )
            
            st.subheader("🥧 Dismissal Timing Breakdown")
            st.plotly_chart(
                px.pie(dismissals, names='Timing', hole=0.4),
                use_container_width=True
            )
        
        with col2:
            st.subheader("🔥 Heatmap: Line vs Length Dismissals")
            heatmap_data = dismissals.groupby(['lineTypeId', 'lengthTypeId']).size().unstack().fillna(0)
            st.plotly_chart(
                go.Figure(data=go.Heatmap(
                    z=heatmap_data.values,
                    x=heatmap_data.columns,
                    y=heatmap_data.index,
                    colorscale='Reds'
                )),
                use_container_width=True
            )
        
        # Suggested Bowling Plan
        st.subheader("🧾 Suggested Bowling Plan")
        if not summary.empty:
            top_row = summary.sort_values(by='dismissals', ascending=False).iloc[0]
            suggestion = f"""
            🧲 **Bowling Type**: {top_row.get('bowlingTypeId', 'N/A')}  
            🎯 **Line**: {top_row.get('lineTypeId', 'N/A')}  
            📏 **Length**: {top_row.get('lengthTypeId', 'N/A')}  
            🧲 **Target Zone**: {top_row.get('fieldingPosition', 'N/A')}  
            ⌛ **Likely Timing**: {top_row.get('Timing', 'N/A')}
            """
            st.markdown(suggestion)
    else:
        st.warning("Please select at least one player from the sidebar for dismissal analysis.")

if __name__ == "__main__":
    main()




