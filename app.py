import streamlit as st
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Use non-GUI backend
import matplotlib.pyplot as plt
import numpy as np

@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/Krithika0808/Hundred/main/Hundred.csv"
    return pd.read_csv(url)

df = load_data()

st.title("Hundred Women – Wagon Wheel")

# Player selection
players = df["batsman"].dropna().unique()
selected_player = st.selectbox("Select Batter", sorted(players))

# Filter data
player_df = df[(df["batsman"] == selected_player) & (df["shotAngle"].notnull())]

if player_df.empty:
    st.warning("No data available for this player.")
else:
    # Convert angles to radians for polar plot
    angles = np.deg2rad(player_df["shotAngle"])
    magnitudes = player_df["shotMagnitude"]

    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'}, figsize=(6, 6))
    ax.set_theta_zero_location("N")   # 0 degrees is straight ahead
    ax.set_theta_direction(-1)        # Clockwise
    ax.scatter(angles, magnitudes, alpha=0.6)

    ax.set_title(f"Wagon Wheel – {selected_player}", va='bottom')
    st.pyplot(fig)
