import streamlit as st
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Use non-GUI backend
import matplotlib.pyplot as plt
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# Page config
st.set_page_config(
    page_title="Hundred Women Cricket Analysis",
    page_icon="🏏",
    layout="wide"
)

@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/Krithika0808/Hundred/main/Hundred.csv"
    df = pd.read_csv(url)
    
    # Clean the data
    df['runs'] = pd.to_numeric(df['runs'], errors='coerce').fillna(0)
    df['shotAngle'] = pd.to_numeric(df['shotAngle'], errors='coerce')
    df['shotMagnitude'] = pd.to_numeric(df['shotMagnitude'], errors='coerce')
    
    return df

# Load data
df = load_data()

# Main title
st.title("🏏 Hundred Women Cricket Analysis Dashboard")

# Create tabs for the 4 main features
tab1, tab2, tab3, tab4 = st.tabs(["🎯 Wagon Wheel", "📊 Control Percentage", "⚖️ Risk vs Reward", "🎳 Bowler Strength"])

with tab1:
    st.header("🎯 Wagon Wheel Analysis")
    
    # Player selection
    players = df["batsman"].dropna().unique()
    selected_player = st.selectbox("Select Batter", sorted(players))
    
    # Filter data
    player_df = df[(df["batsman"] == selected_player) & (df["shotAngle"].notnull())]
    
    if player_df.empty:
        st.warning("No data available for this player.")
    else:
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Your original wagon wheel code
            angles = np.deg2rad(player_df["shotAngle"])
            magnitudes = player_df["shotMagnitude"]
            
            fig, ax = plt.subplots(subplot_kw={'projection': 'polar'}, figsize=(8, 8))
            ax.set_theta_zero_location("N")   # 0 degrees is straight ahead
            ax.set_theta_direction(-1)        # Clockwise
            
            # Color by runs scored
            colors = player_df['runs'].fillna(0)
            scatter = ax.scatter(angles, magnitudes, c=colors, alpha=0.7, cmap='RdYlGn', s=50)
            ax.set_title(f"Wagon Wheel – {selected_player}", va='bottom', fontsize=14)
            
            # Add colorbar
            plt.colorbar(scatter, ax=ax, label='Runs Scored')
            st.pyplot(fig)
        
        with col2:
            st.subheader("Player Stats")
            total_runs = int(player_df['runs'].sum())
            total_balls = len(player_df)
            boundaries = len(player_df[player_df['runs'] >= 4])
            strike_rate = (total_runs / total_balls * 100) if total_balls > 0 else 0
            
            st.metric("Total Runs", total_runs)
            st.metric("Balls Faced", total_balls)
            st.metric("Strike Rate", f"{strike_rate:.1f}")
            st.metric("Boundaries", boundaries)

with tab2:
    st.header("📊 Control Percentage Analysis")
    
    # Calculate control percentage based on connection quality
    def calculate_control(df):
        # Map connection types to control scores
        control_mapping = {
            'Middled': 100, 'WellTimed': 90, 'Undercontrol': 85,
            'MisTimed': 40, 'TopEdge': 30, 'BottomEdge': 30,
            'InsideEdge': 35, 'OutsideEdge': 35, 'Missed': 0
        }
        
        if 'battingConnectionId' in df.columns:
            df['control_score'] = df['battingConnectionId'].map(control_mapping).fillna(60)
        else:
            # If no connection data, estimate from runs and other factors
            df['control_score'] = np.where(df['runs'] >= 4, 80, 70)  # Boundaries suggest good control
            df['control_score'] = np.where(df['runs'] == 0, df['control_score'] - 20, df['control_score'])
        
        return df
    
    df_with_control = calculate_control(df.copy())
    
    # Control stats by player
    control_stats = df_with_control.groupby('batsman').agg({
        'control_score': 'mean',
        'runs': ['sum', 'count'],
    }).round(1)
    
    control_stats.columns = ['Control%', 'Total_Runs', 'Balls_Faced']
    control_stats = control_stats.reset_index()
    control_stats = control_stats.sort_values('Control%', ascending=False)
    
    # Display top players
    st.subheader("🏆 Top Players by Control Percentage")
    fig_control = px.bar(
        control_stats.head(10),
        x='batsman',
        y='Control%',
        title='Top 10 Players - Control Percentage',
        color='Control%',
        color_continuous_scale='RdYlGn'
    )
    fig_control.update_layout(xaxis_tickangle=45)
    st.plotly_chart(fig_control, use_container_width=True)
    
    # Player selector for detailed view
    selected_control_player = st.selectbox("Select Player for Detailed Control Analysis", sorted(df['batsman'].unique()), key="control_player")
    
    player_control_data = df_with_control[df_with_control['batsman'] == selected_control_player]
    if not player_control_data.empty:
        avg_control = player_control_data['control_score'].mean()
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Average Control %", f"{avg_control:.1f}%")
        with col2:
            good_shots = len(player_control_data[player_control_data['control_score'] >= 70])
            st.metric("Well Controlled Shots", f"{good_shots}/{len(player_control_data)}")
        with col3:
            control_efficiency = (avg_control * player_control_data['runs'].mean()) / 100
            st.metric("Control Efficiency", f"{control_efficiency:.2f}")

with tab3:
    st.header("⚖️ Risk vs Reward Analysis")
    
    # Calculate risk score
    def calculate_risk_score(row):
        risk = 0
        # High magnitude shots are riskier
        if pd.notna(row['shotMagnitude']) and row['shotMagnitude'] > 80:
            risk += 3
        # Boundaries are riskier but rewarding
        if row['runs'] >= 4:
            risk += 2
        # Certain shot types are riskier
        if pd.notna(row['battingShotTypeId']):
            risky_shots = ['Hook', 'Pull', 'Sweep', 'Reverse Sweep']
            if any(shot in str(row['battingShotTypeId']) for shot in risky_shots):
                risk += 2
        return min(risk, 10)  # Cap at 10
    
    df['risk_score'] = df.apply(calculate_risk_score, axis=1)
    
    # Risk vs Reward by player
    risk_reward = df.groupby('batsman').agg({
        'risk_score': 'mean',
        'runs': 'mean',
        'runs': 'sum'
    }).round(2)
    
    risk_reward.columns = ['Avg_Risk', 'Avg_Reward_per_Ball', 'Total_Runs']
    risk_reward = risk_reward.reset_index()
    
    # Create scatter plot
    fig_risk = px.scatter(
        risk_reward,
        x='Avg_Risk',
        y='Avg_Reward_per_Ball',
        size='Total_Runs',
        hover_name='batsman',
        title='Risk vs Reward Analysis - All Players',
        labels={'Avg_Risk': 'Average Risk Score', 'Avg_Reward_per_Ball': 'Average Runs per Ball'},
        color='Avg_Reward_per_Ball',
        color_continuous_scale='RdYlGn'
    )
    st.plotly_chart(fig_risk, use_container_width=True)
    
    # Insights
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("🛡️ Safest Players")
        safest = risk_reward.nsmallest(5, 'Avg_Risk')[['batsman', 'Avg_Risk', 'Avg_Reward_per_Ball']]
        st.dataframe(safest, use_container_width=True)
    
    with col2:
        st.subheader("🚀 Most Rewarding Players")
        most_rewarding = risk_reward.nlargest(5, 'Avg_Reward_per_Ball')[['batsman', 'Avg_Reward_per_Ball', 'Avg_Risk']]
        st.dataframe(most_rewarding, use_container_width=True)

with tab4:
    st.header("🎳 Bowler Strength Analysis")
    
    if 'bowler' in df.columns:
        # Bowler statistics
        bowler_stats = df.groupby('bowler').agg({
            'runs': ['sum', 'count'],
            'batsman': 'count'  # balls bowled
        }).round(2)
        
        bowler_stats.columns = ['Runs_Conceded', 'Balls_Bowled', 'Total_Deliveries']
        bowler_stats = bowler_stats[bowler_stats['Balls_Bowled'] >= 10]  # Minimum 10 balls
        
        if len(bowler_stats) > 0:
            # Calculate economy rate (runs per 6 balls)
            bowler_stats['Economy_Rate'] = (bowler_stats['Runs_Conceded'] / bowler_stats['Balls_Bowled'] * 6).round(2)
            
            # Add pressure metric (percentage of dot balls)
            bowler_pressure = df.groupby('bowler').agg({
                'runs': lambda x: (x == 0).sum() / len(x) * 100  # Dot ball percentage
            }).round(1)
            bowler_pressure.columns = ['Dot_Ball_%']
            
            bowler_stats = bowler_stats.merge(bowler_pressure, left_index=True, right_index=True)
            bowler_stats = bowler_stats.reset_index().sort_values('Economy_Rate')
            
            # Display best bowlers
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("🏆 Best Economy Rates")
                best_economy = bowler_stats.head(10)[['bowler', 'Economy_Rate', 'Balls_Bowled']]
                fig_economy = px.bar(
                    best_economy,
                    x='bowler',
                    y='Economy_Rate',
                    title='Top 10 Bowlers - Economy Rate',
                    color='Economy_Rate',
                    color_continuous_scale='RdYlGn_r'
                )
                fig_economy.update_layout(xaxis_tickangle=45)
                st.plotly_chart(fig_economy, use_container_width=True)
            
            with col2:
                st.subheader("🎯 Pressure Creators")
                pressure_creators = bowler_stats.nlargest(10, 'Dot_Ball_%')[['bowler', 'Dot_Ball_%', 'Economy_Rate']]
                fig_pressure = px.bar(
                    pressure_creators,
                    x='bowler',
                    y='Dot_Ball_%',
                    title='Top 10 Bowlers - Dot Ball Percentage',
                    color='Dot_Ball_%',
                    color_continuous_scale='Blues'
                )
                fig_pressure.update_layout(xaxis_tickangle=45)
                st.plotly_chart(fig_pressure, use_container_width=True)
            
            # Overall bowler performance
            st.subheader("📊 Complete Bowler Analysis")
            
            # Economy vs Pressure scatter
            fig_bowler_scatter = px.scatter(
                bowler_stats,
                x='Dot_Ball_%',
                y='Economy_Rate',
                size='Balls_Bowled',
                hover_name='bowler',
                title='Bowler Performance: Pressure vs Economy',
                labels={'Dot_Ball_%': 'Dot Ball Percentage', 'Economy_Rate': 'Economy Rate'},
                color='Economy_Rate',
                color_continuous_scale='RdYlGn_r'
            )
            st.plotly_chart(fig_bowler_scatter, use_container_width=True)
            
            # Detailed bowler stats table
            st.subheader("Detailed Bowler Statistics")
            st.dataframe(bowler_stats, use_container_width=True)
        else:
            st.info("Not enough bowling data available (minimum 10 balls required per bowler).")
    else:
        st.warning("No bowler data available in the dataset.")

# Footer
st.markdown("---")
st.markdown("**🏏 Hundred Women Cricket Analysis Dashboard** | Data from The Hundred Women's Competition")
