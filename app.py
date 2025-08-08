import streamlit as st
import pandas as pd
import plotly.graph_objects as go

# Load data
@st.cache_data
def load_data():
    return pd.read_csv("hundred.csv")

df = load_data()

# Tab Setup
st.title("🏏 The Hundred Shot Map Explorer")
st.subheader("Tab 1: Wagon Wheel by Batter")

# Sidebar Filters
batter_list = sorted(df['batsman'].dropna().unique())
selected_batter = st.selectbox("Select Batter", batter_list)

# Filter player data
player_df = df[df['batsman'] == selected_batter].copy()

# Fill missing values
player_df['shotAngle'] = player_df['shotAngle'].fillna(0)
player_df['shotMagnitude'] = player_df['shotMagnitude'].fillna(40)
player_df['runs'] = player_df['runs'].fillna(0)
player_df['fieldingPosition'] = player_df['fieldingPosition'].fillna("Unknown")
player_df['battingShotTypeId'] = player_df['battingShotTypeId'].fillna("Unknown")

# Define run color mapping
colors = {
    0: 'gray',
    1: 'blue',
    2: 'green',
    3: 'orange',
    4: 'red',
    6: 'purple'
}

# Create Wagon Wheel Plot
fig = go.Figure()

# Plot all non-dismissal shots grouped by run value
for run_value in sorted(player_df['runs'].unique()):
    run_df = player_df[(player_df['runs'] == run_value) & (player_df['isWicket'] != 1)]
    fig.add_trace(go.Scatterpolar(
        r=run_df['shotMagnitude'],
        theta=run_df['shotAngle'],
        mode='markers',
        name=f'{run_value} runs',
        marker=dict(
            color=colors.get(run_value, 'black'),
            size=8 + run_value * 1.5,
            opacity=0.75,
            line=dict(width=1, color='white')
        ),
        text=[
            f"Shot: {s}<br>Fielded at: {f}" 
            for s, f in zip(run_df['battingShotTypeId'], run_df['fieldingPosition'])
        ],
        hovertemplate='<b>%{text}</b><br>Angle: %{theta}°<br>Distance: %{r}<extra></extra>'
    ))

# Add dismissals as separate red X markers
wicket_df = player_df[player_df['isWicket'] == 1]
fig.add_trace(go.Scatterpolar(
    r=wicket_df['shotMagnitude'],
    theta=wicket_df['shotAngle'],
    mode='markers',
    name='Wickets',
    marker=dict(
        symbol='x',
        color='crimson',
        size=12,
        line=dict(width=2, color='black')
    ),
    text=[
        f"Dismissal: {s}<br>Fielded at: {f}" 
        for s, f in zip(wicket_df['battingShotTypeId'], wicket_df['fieldingPosition'])
    ],
    hovertemplate='<b>%{text}</b><br>Angle: %{theta}°<br>Distance: %{r}<extra></extra>'
))

# Layout settings
fig.update_layout(
    title=f"Wagon Wheel – {selected_batter}",
    polar=dict(
        radialaxis=dict(visible=True, range=[0, 100]),
        angularaxis=dict(
            tickmode='array',
            tickvals=[0, 45, 90, 135, 180, 225, 270, 315],
            ticktext=['Straight', 'Cover', 'Point', 'Third Man', 'Fine Leg', 'Square Leg', 'Midwicket', 'Long On']
        )
    ),
    showlegend=True,
    height=650
)

st.plotly_chart(fig, use_container_width=True)
