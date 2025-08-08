import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
from io import StringIO

# Page config
st.set_page_config(
    page_title="Cricket Analysis Dashboard",
    page_icon="🏏",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #f0f2f6 0%, #e8f4fd 100%);
        padding: 2rem;
        border-radius: 1rem;
        border-left: 5px solid #1f77b4;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Data loading function
@st.cache_data
def load_cricket_data():
    """Load cricket data from GitHub or create sample data"""
    try:
        url = "https://raw.githubusercontent.com/Krithika0808/Hundred/main/Hundred.csv"
        response = requests.get(url, timeout=10)
        df = pd.read_csv(StringIO(response.text))
        
        # Basic data cleaning
        df['runs'] = pd.to_numeric(df['runs'], errors='coerce').fillna(0)
        df['shotAngle'] = pd.to_numeric(df['shotAngle'], errors='coerce').fillna(0)
        df['shotMagnitude'] = pd.to_numeric(df['shotMagnitude'], errors='coerce').fillna(50)
        
        return df
    except:
        # Create sample data if GitHub fails
        return create_sample_data()

def create_sample_data():
    """Create sample cricket data"""
    np.random.seed(42)
    
    batsmen = ['Smriti Mandhana', 'Harmanpreet Kaur', 'Beth Mooney', 'Alyssa Healy', 'Meg Lanning']
    bowlers = ['Jess Jonassen', 'Sophie Ecclestone', 'Ashleigh Gardner', 'Shabnim Ismail', 'Marizanne Kapp']
    shot_types = ['Drive', 'Pull', 'Cut', 'Sweep', 'Flick', 'Hook', 'Loft']
    connections = ['Perfect', 'Good', 'Average', 'Poor', 'Missed']
    
    n = 1000
    data = {
        'batsman': np.random.choice(batsmen, n),
        'bowler': np.random.choice(bowlers, n),
        'battingShotTypeId': np.random.choice(shot_types, n),
        'battingConnectionId': np.random.choice(connections, n, p=[0.25, 0.3, 0.25, 0.15, 0.05]),
        'runs': np.random.choice([0, 1, 2, 3, 4, 6], n, p=[0.3, 0.25, 0.2, 0.1, 0.1, 0.05]),
        'shotAngle': np.random.uniform(0, 360, n),
        'shotMagnitude': np.random.uniform(30, 100, n),
        'isWicket': np.random.choice([True, False], n, p=[0.05, 0.95]),
        'bowlingType': np.random.choice(['Fast', 'Spin', 'Medium'], n),
        'ballNumber': np.random.randint(1, 101, n)
    }
    
    return pd.DataFrame(data)

# Core analysis functions
def calculate_control_percentage(df):
    """Calculate control percentage for each batsman"""
    control_map = {'Perfect': 100, 'Good': 80, 'Average': 60, 'Poor': 30, 'Missed': 0}
    df['control_score'] = df['battingConnectionId'].map(control_map).fillna(50)
    
    control_stats = df.groupby('batsman').agg({
        'control_score': 'mean',
        'runs': ['sum', 'count', 'mean'],
        'isWicket': 'sum'
    }).round(1)
    
    control_stats.columns = ['Control%', 'Total_Runs', 'Balls_Faced', 'Avg_Runs', 'Wickets_Lost']
    control_stats = control_stats.reset_index()
    return control_stats

def create_wagon_wheel(df, player):
    """Create wagon wheel for a player"""
    player_data = df[df['batsman'] == player].copy()
    
    # Convert angle to cricket field positions
    def angle_to_position(angle):
        positions = {
            (0, 22.5): 'Long Off', (22.5, 67.5): 'Cover', (67.5, 112.5): 'Point',
            (112.5, 157.5): 'Third Man', (157.5, 202.5): 'Fine Leg', 
            (202.5, 247.5): 'Square Leg', (247.5, 292.5): 'Mid Wicket',
            (292.5, 337.5): 'Long On', (337.5, 360): 'Long Off'
        }
        for (start, end), pos in positions.items():
            if start <= angle < end:
                return pos
        return 'Long Off'
    
    player_data['field_position'] = player_data['shotAngle'].apply(angle_to_position)
    
    # Create polar plot
    fig = go.Figure()
    
    # Add runs as scatter points
    colors = {0: 'gray', 1: 'blue', 2: 'green', 3: 'orange', 4: 'red', 6: 'purple'}
    
    for runs, color in colors.items():
        data = player_data[player_data['runs'] == runs]
        if not data.empty:
            fig.add_trace(go.Scatterpolar(
                r=data['shotMagnitude'],
                theta=data['shotAngle'],
                mode='markers',
                name=f'{runs} runs',
                marker=dict(color=color, size=8 + runs*2),
                hovertemplate=f'<b>{runs} runs</b><br>Angle: %{{theta}}°<br>Distance: %{{r}}<extra></extra>'
            ))
    
    fig.update_layout(
        title=f'{player} - Wagon Wheel',
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 100]),
            angularaxis=dict(
                tickvals=[0, 45, 90, 135, 180, 225, 270, 315],
                ticktext=['Long Off', 'Cover', 'Point', 'Third Man', 'Fine Leg', 'Square Leg', 'Mid Wicket', 'Long On']
            )
        ),
        height=500
    )
    
    return fig

def risk_vs_reward_analysis(df):
    """Analyze risk vs reward for batsmen"""
    # Define risk levels
    def calculate_risk(row):
        risk_score = 0
        if row['runs'] >= 4:  # Boundaries are risky
            risk_score += 3
        if row['battingConnectionId'] in ['Poor', 'Missed']:
            risk_score += 4
        if row['shotMagnitude'] > 80:  # Big shots are risky
            risk_score += 2
        return min(risk_score, 10)  # Cap at 10
    
    df['risk_score'] = df.apply(calculate_risk, axis=1)
    
    risk_reward = df.groupby('batsman').agg({
        'risk_score': 'mean',
        'runs': 'mean',
        'isWicket': lambda x: (x.sum() / len(x)) * 100  # Wicket percentage
    }).round(2)
    
    risk_reward.columns = ['Avg_Risk', 'Avg_Reward', 'Wicket_Rate%']
    
    # Create scatter plot
    fig = px.scatter(
        risk_reward.reset_index(),
        x='Avg_Risk',
        y='Avg_Reward',
        size='Wicket_Rate%',
        color='Wicket_Rate%',
        hover_name='batsman',
        title='Risk vs Reward Analysis',
        labels={'Avg_Risk': 'Average Risk Score', 'Avg_Reward': 'Average Runs per Ball'},
        color_continuous_scale='RdYlGn_r'
    )
    
    return fig, risk_reward.reset_index()

def bowler_strength_analysis(df):
    """Analyze bowler strengths"""
    bowler_stats = df.groupby('bowler').agg({
        'runs': ['sum', 'count'],
        'isWicket': 'sum',
        'battingConnectionId': lambda x: (x.isin(['Poor', 'Missed']).sum() / len(x)) * 100
    }).round(2)
    
    bowler_stats.columns = ['Runs_Conceded', 'Balls_Bowled', 'Wickets', 'Pressure%']
    bowler_stats = bowler_stats[bowler_stats['Balls_Bowled'] >= 20]  # Minimum 20 balls
    
    bowler_stats['Economy'] = (bowler_stats['Runs_Conceded'] / bowler_stats['Balls_Bowled'] * 6).round(2)
    bowler_stats['Strike_Rate'] = (bowler_stats['Balls_Bowled'] / bowler_stats['Wickets']).round(1)
    bowler_stats.loc[bowler_stats['Wickets'] == 0, 'Strike_Rate'] = 999  # No wickets
    
    return bowler_stats.reset_index().sort_values('Economy')

# Main app
def main():
    st.markdown("<h1 class='main-header'>🏏 Cricket Analysis Dashboard</h1>", unsafe_allow_html=True)
    
    # Load data
    with st.spinner('Loading cricket data...'):
        df = load_cricket_data()
    
    st.success(f"✅ Loaded {len(df)} ball records")
    
    # Sidebar for selections
    st.sidebar.header("📊 Analysis Options")
    
    batsmen = sorted(df['batsman'].unique())
    selected_batsman = st.sidebar.selectbox("Select Batsman", batsmen)
    
    # Create tabs
    tab1, tab2, tab3, tab4 = st.tabs(["🎯 Wagon Wheel", "🎮 Control %", "⚡ Risk vs Reward", "🎳 Bowler Strength"])
    
    with tab1:
        st.subheader(f"🎯 {selected_batsman} - Wagon Wheel")
        wagon_fig = create_wagon_wheel(df, selected_batsman)
        st.plotly_chart(wagon_fig, use_container_width=True)
        
        # Shot distribution
        player_data = df[df['batsman'] == selected_batsman]
        shot_dist = player_data['battingShotTypeId'].value_counts()
        
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Shot Distribution")
            fig_bar = px.bar(x=shot_dist.index, y=shot_dist.values, 
                           title="Most Played Shots")
            st.plotly_chart(fig_bar, use_container_width=True)
        
        with col2:
            st.subheader("Runs Distribution")
            runs_dist = player_data['runs'].value_counts().sort_index()
            fig_pie = px.pie(values=runs_dist.values, names=runs_dist.index,
                           title="Runs Scored Distribution")
            st.plotly_chart(fig_pie, use_container_width=True)
    
    with tab2:
        st.subheader("🎮 Batsmen Control Percentage")
        control_stats = calculate_control_percentage(df)
        
        # Display as metrics
        st.subheader(f"{selected_batsman} - Control Stats")
        player_stats = control_stats[control_stats['batsman'] == selected_batsman].iloc[0]
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Control %", f"{player_stats['Control%']:.1f}%")
        with col2:
            st.metric("Total Runs", f"{int(player_stats['Total_Runs'])}")
        with col3:
            st.metric("Balls Faced", f"{int(player_stats['Balls_Faced'])}")
        with col4:
            st.metric("Strike Rate", f"{(player_stats['Avg_Runs'] * 100):.1f}")
        
        # Control comparison chart
        fig_control = px.bar(
            control_stats,
            x='batsman',
            y='Control%',
            title='Control Percentage Comparison - All Batsmen',
            color='Control%',
            color_continuous_scale='RdYlGn'
        )
        fig_control.update_layout(xaxis_tickangle=45)
        st.plotly_chart(fig_control, use_container_width=True)
        
        # Detailed table
        st.subheader("Detailed Control Statistics")
        st.dataframe(control_stats, use_container_width=True)
    
    with tab3:
        st.subheader("⚡ Risk vs Reward Analysis")
        risk_fig, risk_data = risk_vs_reward_analysis(df)
        st.plotly_chart(risk_fig, use_container_width=True)
        
        # Insights
        st.subheader("📈 Risk-Reward Insights")
        col1, col2 = st.columns(2)
        
        with col1:
            safest = risk_data.loc[risk_data['Avg_Risk'].idxmin()]
            st.info(f"**Safest Player**: {safest['batsman']} (Risk: {safest['Avg_Risk']:.1f})")
            
            highest_reward = risk_data.loc[risk_data['Avg_Reward'].idxmax()]
            st.success(f"**Highest Reward**: {highest_reward['batsman']} ({highest_reward['Avg_Reward']:.2f} runs/ball)")
        
        with col2:
            riskiest = risk_data.loc[risk_data['Avg_Risk'].idxmax()]
            st.warning(f"**Riskiest Player**: {riskiest['batsman']} (Risk: {riskiest['Avg_Risk']:.1f})")
            
            most_wickets = risk_data.loc[risk_data['Wicket_Rate%'].idxmax()]
            st.error(f"**Highest Wicket Rate**: {most_wickets['batsman']} ({most_wickets['Wicket_Rate%']:.1f}%)")
        
        st.subheader("Risk vs Reward Data")
        st.dataframe(risk_data, use_container_width=True)
    
    with tab4:
        st.subheader("🎳 Bowler Strength Analysis")
        bowler_stats = bowler_strength_analysis(df)
        
        # Top performers
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🏆 Best Economy Rates")
            top_economy = bowler_stats.head(5)[['bowler', 'Economy', 'Wickets']]
            st.dataframe(top_economy, use_container_width=True)
        
        with col2:
            st.subheader("🎯 Most Wickets")
            top_wickets = bowler_stats.nlargest(5, 'Wickets')[['bowler', 'Wickets', 'Economy']]
            st.dataframe(top_wickets, use_container_width=True)
        
        # Bowling analysis charts
        fig_economy = px.bar(
            bowler_stats.head(10),
            x='bowler',
            y='Economy',
            title='Economy Rate - Top 10 Bowlers',
            color='Economy',
            color_continuous_scale='RdYlGn_r'
        )
        fig_economy.update_layout(xaxis_tickangle=45)
        st.plotly_chart(fig_economy, use_container_width=True)
        
        # Pressure vs Economy scatter
        fig_pressure = px.scatter(
            bowler_stats,
            x='Pressure%',
            y='Economy',
            size='Wickets',
            hover_name='bowler',
            title='Bowler Pressure vs Economy Rate',
            labels={'Pressure%': 'Pressure Created (%)', 'Economy': 'Economy Rate'}
        )
        st.plotly_chart(fig_pressure, use_container_width=True)
        
        st.subheader("Complete Bowler Statistics")
        st.dataframe(bowler_stats, use_container_width=True)

if __name__ == "__main__":
    main()
