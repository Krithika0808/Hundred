import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

st.set_page_config(page_title="Hundred Batting Dashboard", layout="wide")

@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/Krithika0808/Hundred/main/Hundred.csv"
    return pd.read_csv(url)

df = load_data()

# Tab layout
tab1, tab2, tab3 = st.tabs(["🧠 Batter's Blueprint", "📊 Other Tabs", "🔍 More Coming"])

with tab1:
    st.title("🧠 Batter's Blueprint")
    st.markdown("Explore a batter’s shot direction, dismissal zones, and strike pattern.")

    # Filters
    batters = df['batsman'].dropna().unique()
    selected_batter = st.selectbox("Select Batter", sorted(batters))

    bowling_types = df['bowlingTypeId'].dropna().unique()
    selected_bowling = st.multiselect("Select Bowling Type", sorted(bowling_types), default=sorted(bowling_types))

    phase = st.radio("Select Over Phase", ["Powerplay", "Middle", "Death"])

    # Phase filter using totalBallNumber (each inning has 100 balls)
    if phase == "Powerplay":
        df_phase = df[df["totalBallNumber"] <= 25]
    elif phase == "Middle":
        df_phase = df[(df["totalBallNumber"] > 25) & (df["totalBallNumber"] <= 75)]
    else:
        df_phase = df[df["totalBallNumber"] > 75]

    # Apply filters
    filtered_df = df_phase[
        (df_phase["batsman"] == selected_batter) &
        (df_phase["bowlingTypeId"].isin(selected_bowling)) &
        (~df_phase["shotAngle"].isna()) &
        (~df_phase["shotMagnitude"].isna())
    ]

    st.markdown(f"### Shot Map for {selected_batter} ({phase})")

    if filtered_df.empty:
        st.warning("No data for selected filters.")
    else:
        # Plotting the wagon wheel
        fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
        angles = np.deg2rad(filtered_df['shotAngle'])
        magnitudes = filtered_df['shotMagnitude']

        # Color by dismissal
        colors = ['red' if w == 1 else 'blue' for w in filtered_df['isWicket']]

        ax.scatter(angles, magnitudes, c=colors, alpha=0.7)

        ax.set_theta_zero_location("N")
        ax.set_theta_direction(-1)
        ax.set_rlim(0, max(magnitudes) + 10)
        ax.set_title(f"{selected_batter} – Shot Map", va='bottom')
        st.pyplot(fig)

        st.caption("🔴 = Dismissal | 🔵 = Normal shot")
