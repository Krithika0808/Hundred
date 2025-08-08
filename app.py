import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# Page config
st.set_page_config(page_title="Hundred — Wagon Wheel", layout="wide", page_icon="🏏")

RAW_URL = "https://raw.githubusercontent.com/Krithika0808/Hundred/main/Hundred.csv"

@st.cache_data(ttl=3600)
def load_data(url=RAW_URL):
    df = pd.read_csv(url)
    # basic coercions
    for col in ["shotAngle", "shotMagnitude", "runs", "isWicket", "totalBallNumber"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    # clean strings
    for s in ["fieldingPosition", "battingShotTypeId", "battingConnectionId", "batsman", "bowler", "bowlingTypeId"]:
        if s in df.columns:
            df[s] = df[s].fillna("Unknown").astype(str)
    # normalize angles into [0,360)
    if "shotAngle" in df.columns:
        df["shotAngle"] = df["shotAngle"].fillna(0).mod(360)
    # guard for magnitude
    if "shotMagnitude" in df.columns:
        df["shotMagnitude"] = df["shotMagnitude"].fillna(df["shotMagnitude"].median() if df["shotMagnitude"].notna().any() else 40)
    # runs integer-ish
    if "runs" in df.columns:
        df["runs"] = df["runs"].fillna(0).astype(int)
    # isWicket to bool/int
    if "isWicket" in df.columns:
        df["isWicket"] = df["isWicket"].fillna(0).astype(int)
    return df

df = load_data()

st.title("🏏 The Hundred — Wagon Wheel (Plotly)")
st.markdown("Select a batter and phase. Red X = dismissal. Hover for shot type & fielding position.")

# --- Sidebar filters ---
players = sorted(df['batsman'].dropna().unique())
selected_batter = st.sidebar.selectbox("Select Batter", players)

phase_opt = st.sidebar.selectbox("Phase", ["All", "Powerplay (1-25)", "Middle (26-75)", "Death (76+)"])
wicket_only = st.sidebar.checkbox("Show only dismissals (Wickets)", value=False)
boundaries_only = st.sidebar.checkbox("Show only boundaries (4s & 6s)", value=False)

# optional: filter by bowling type
if 'bowlingTypeId' in df.columns:
    all_bowling_types = sorted(df['bowlingTypeId'].dropna().unique())
    selected_bowling = st.sidebar.multiselect("Bowling Type (optional)", all_bowling_types, default=all_bowling_types)
else:
    selected_bowling = None

# --- Filter dataframe ---
f = df[df['batsman'] == selected_batter].copy()

# phase filter based on totalBallNumber
if 'totalBallNumber' in f.columns:
    if phase_opt.startswith("Powerplay"):
        f = f[f['totalBallNumber'] <= 25]
    elif phase_opt.startswith("Middle"):
        f = f[(f['totalBallNumber'] > 25) & (f['totalBallNumber'] <= 75)]
    elif phase_opt.startswith("Death"):
        f = f[f['totalBallNumber'] > 75]

if wicket_only:
    f = f[f['isWicket'] == 1]

if boundaries_only:
    f = f[f['runs'].isin([4,6])]

if selected_bowling is not None:
    f = f[f['bowlingTypeId'].isin(selected_bowling)]

if f.empty:
    st.warning("No records for selected filters. Try different filters.")
    st.stop()

# --- Plot params ---
max_r = max( (f['shotMagnitude'].max() or 100), 100 )  # radial max
min_r = 0

# run color mapping (customize as you like)
run_colors = {0: "#9e9e9e", 1: "#1f77b4", 2: "#2ca02c", 3: "#ff7f0e", 4: "#d62728", 6: "#9467bd"}
# fallback
default_color = "#111111"

fig = go.Figure()

# Plot non-wicket shots grouped by run value (so legend makes sense)
non_wickets = f[f['isWicket'] == 0]
for rv in sorted(non_wickets['runs'].unique()):
    rv_df = non_wickets[non_wickets['runs'] == rv]
    if rv_df.empty:
        continue
    sizes = np.clip(((rv_df['shotMagnitude'] - rv_df['shotMagnitude'].min()) /
                     (rv_df['shotMagnitude'].max() - rv_df['shotMagnitude'].min() + 1e-6)) * 18 + 6, 6, 30)
    color = run_colors.get(int(rv), default_color)
    hover_text = (
        "Shot: " + rv_df['battingShotTypeId'] + "<br>" +
        "Fielded at: " + rv_df['fieldingPosition'] + "<br>" +
        "Runs: " + rv_df['runs'].astype(str) + "<br>" +
        "Connection: " + rv_df['battingConnectionId'] + "<br>" +
        "Ball#: " + rv_df.get('ballNumber', pd.Series(["?"]*len(rv_df))).astype(str)
    )
    fig.add_trace(go.Scatterpolar(
        r=rv_df['shotMagnitude'],
        theta=rv_df['shotAngle'],
        mode='markers',
        name=f"{rv} runs",
        marker=dict(size=sizes, color=color, line=dict(width=0.5, color='white')),
        hoverinfo='text',
        hovertext=hover_text,
        opacity=0.8,
    ))

# Add wickets as distinct trace (big red X)
w_df = f[f['isWicket'] == 1]
if not w_df.empty:
    sizes_w = np.clip(((w_df['shotMagnitude'] - w_df['shotMagnitude'].min()) /
                       (w_df['shotMagnitude'].max() - w_df['shotMagnitude'].min() + 1e-6)) * 24 + 10, 10, 40)
    hover_text_w = (
        "DISMISSAL<br>" +
        "Shot: " + w_df['battingShotTypeId'] + "<br>" +
        "Fielded at: " + w_df['fieldingPosition'] + "<br>" +
        "Runs: " + w_df['runs'].astype(str) + "<br>" +
        "Connection: " + w_df['battingConnectionId'] + "<br>" +
        "Ball#: " + w_df.get('ballNumber', pd.Series(["?"]*len(w_df))).astype(str)
    )
    fig.add_trace(go.Scatterpolar(
        r=w_df['shotMagnitude'],
        theta=w_df['shotAngle'],
        mode='markers',
        name='Wicket',
        marker=dict(symbol='x', size=sizes_w, color='crimson', line=dict(width=1, color='black')),
        hoverinfo='text',
        hovertext=hover_text_w,
        opacity=0.95,
    ))

# Layout: make polar look like a wagon wheel
fig.update_layout(
    title=f"Wagon Wheel — {selected_batter} (n={len(f)})",
    polar=dict(
        radialaxis=dict(range=[min_r, max_r], showticklabels=False, ticks=''),
        angularaxis=dict(
            direction="clockwise",
            rotation=90,  # 0 degrees at top (north)
            tickvals=[0, 45, 90, 135, 180, 225, 270, 315],
            ticktext=
