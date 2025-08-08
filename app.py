import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load data from GitHub
@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/Krithika0808/Hundred/main/Hundred.csv"
    return pd.read_csv(url)

df = load_data()

# Sidebar filters
batter = st.sidebar.selectbox("Select Batter", sorted(df['batsman'].dropna().unique()))
is_wicket = st.sidebar.selectbox("Wicket Only?", ["No", "Yes"])
fielding_pos = st.sidebar.multiselect("Fielding Position", sorted(df['fieldingPosition'].dropna().unique()))

# Filter dataframe
data = df[df['batsman'] == batter]

if is_wicket == "Yes":
    data = data[data['isWicket'] == 1]

if fielding_pos:
    data = data[data['fieldingPosition'].isin(fielding_pos)]

# Convert polar-like data (angle, magnitude) into Cartesian
angles_rad = np.deg2rad(data['shotAngle'])
x = data['shotMagnitude'] * np.cos(angles_rad)
y = data['shotMagnitude'] * np.sin(angles_rad)

# Plot wagon wheel without polar projection
fig, ax = plt.subplots(figsize=(6, 6))
ax.scatter(x, y, alpha=0.6, c="green", edgecolors="black")

# Styling to look like wagon wheel
ax.set_aspect('equal')
ax.set_xlim(-100, 100)
ax.set_ylim(-100, 100)
ax.axhline(0, color="black", lw=1)
ax.axvline(0, color="black", lw=1)
ax.set_xticks([])
ax.set_yticks([])
ax.set_title(f"Shot Map - {batter}")

st.pyplot(fig)
