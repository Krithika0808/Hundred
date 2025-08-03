import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os
import secrets

# Set page config
st.set_page_config(
    page_title="Women's Cricket Shot Intelligence Matrix",
    page_icon="🏏",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Password protection
def check_password():
    """Returns True if the user has the correct password."""
    
    def password_entered():
        """Checks whether a password entered by the user is correct."""
        return st.session_state["password"] in st.secrets["passwords"]
    
    st.title("🏏 Women's Cricket Shot Intelligence Matrix")
    
    # First time or not logged in
    if "password" not in st.session_state:
        st.session_state["password"] = ""
        st.session_state["password_correct"] = False
    
    if not st.session_state["password_correct"]:
        st.subheader("Please enter the password to access the app")
        
        # Password input
        password_input = st.text_input(
            "Password", 
            type="password", 
            key="password_input"
        )
        
        if st.button("Login"):
            if password_input in st.secrets["passwords"]:
                st.session_state["password"] = password_input
                st.session_state["password_correct"] = True
                st.success("Login successful! Please wait while the app loads...")
                st.experimental_rerun()
            else:
                st.error("Incorrect password. Please try again.")
        
        st.stop()
    else:
        return True

# Create a secrets file if it doesn't exist
def create_secrets_file():
    """Create a secrets.toml file with a random password if it doesn't exist."""
    secrets_file = ".streamlit/secrets.toml"
    
    if not os.path.exists(".streamlit"):
        os.makedirs(".streamlit")
    
    if not os.path.exists(secrets_file):
        # Generate a random password
        password = secrets.token_urlsafe(10)
        
        with open(secrets_file, "w") as f:
            f.write('# Secrets for the Cricket Analytics App\n\n')
            f.write('# Password for accessing the app\n')
            f.write('passwords = ["' + password + '"]\n')
        
        return password
    else:
        # Read existing password
        with open(secrets_file, "r") as f:
            content = f.read()
            # Extract password from the file
            for line in content.split('\n'):
                if 'passwords = [' in line:
                    password = line.split('["')[1].split('"]')[0]
                    return password
    return None

# Simplified data loading
@st.cache_data
def load_data():
    """Load sample cricket data for demonstration"""
    np.random.seed(42)
    
    players = ['Smriti Mandhana', 'Harmanpreet Kaur', 'Beth Mooney', 'Alyssa Healy', 'Meg Lanning']
    shot_types = ['Drive', 'Pull', 'Cut', 'Sweep', 'Flick', 'Hook', 'Reverse Sweep', 'Loft']
    
    n_rows = 500
    
    data = {
        'batsman': np.random.choice(players, n_rows),
        'battingShotTypeId': np.random.choice(shot_types, n_rows),
        'runs': np.random.choice([0, 1, 2, 3, 4, 6], n_rows, p=[0.3, 0.25, 0.2, 0.1, 0.1, 0.05]),
        'shotAngle': np.random.uniform(0, 360, n_rows),
        'shotMagnitude': np.random.uniform(50, 200, n_rows),
        'isBoundary': np.random.choice([True, False], n_rows, p=[0.15, 0.85]),
    }
    
    return pd.DataFrame(data)

# Main app function
def main():
    # Check password first
    if not check_password():
        return
    
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
        .access-info {
            background-color: #e7f3ff;
            padding: 1rem;
            border-radius: 0.5rem;
            margin-bottom: 1rem;
            border-left: 4px solid #0066cc;
        }
        .share-link {
            background-color: #f8f9fa;
            padding: 0.75rem;
            border-radius: 0.375rem;
            font-family: monospace;
            word-break: break-all;
        }
    </style>
    """, unsafe_allow_html=True)
    
    # Display access information
    st.markdown('<div class="access-info">', unsafe_allow_html=True)
    st.markdown("### 🔐 Private Access Information")
    st.markdown("This app is password protected. Share the password only with authorized users.")
    st.markdown("</div>", unsafe_allow_html=True)
    
    st.markdown('<h1 class="main-header">🏏 Women\'s Cricket Shot Intelligence Matrix</h1>', unsafe_allow_html=True)
    
    # Load data
    df = load_data()
    
    # Calculate control metrics
    df['control_score'] = np.random.uniform(30, 100, len(df))
    df['is_controlled_shot'] = (df['control_score'] > 60).astype(int)
    
    # Sidebar filters
    st.sidebar.header("🎯 Analysis Filters")
    
    # Get available players
    available_players = df['batsman'].unique()
    selected_players = st.sidebar.multiselect(
        "Select Players",
        options=available_players,
        default=list(available_players)
    )
    
    # Apply filters
    filtered_df = df[df['batsman'].isin(selected_players)]
    
    # Main dashboard tabs
    tab1, tab2, tab3 = st.tabs([
        "🎯 Shot Placement", 
        "⚡ Control vs Aggression", 
        "🏆 Player Intelligence"
    ])
    
    with tab1:
        st.subheader("360° Shot Placement Intelligence")
        
        selected_player = st.selectbox("Select Player for Detailed Analysis", selected_players)
        if selected_player:
            player_data = filtered_df[filtered_df['batsman'] == selected_player]
            
            fig = go.Figure()
            
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
                name='Shots'
            ))
            
            fig.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, 200],
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
                title=f"{selected_player} - Shot Placement Intelligence",
                height=600,
                showlegend=False
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.subheader("Control vs Aggression Matrix")
        
        shot_analysis = filtered_df.groupby('battingShotTypeId').agg({
            'is_controlled_shot': 'mean',
            'runs': 'mean'
        }).reset_index()
        
        shot_counts = filtered_df.groupby('battingShotTypeId').size().reset_index(name='shot_count')
        shot_analysis = shot_analysis.merge(shot_counts, on='battingShotTypeId')
        
        fig = px.scatter(
            shot_analysis,
            x='is_controlled_shot',
            y='runs',
            size='shot_count',
            color='battingShotTypeId',
            text='battingShotTypeId',
            title='Control vs Aggression Matrix',
            labels={
                'is_controlled_shot': 'Control Rate',
                'runs': 'Average Runs',
                'shot_count': 'Number of Shots'
            }
        )
        
        fig.update_traces(textposition="top center")
        st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.subheader("Player Intelligence Cards")
        
        for player in selected_players:
            player_data = filtered_df[filtered_df['batsman'] == player]
            
            st.markdown(f"#### {player}")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Control Rate", f"{player_data['is_controlled_shot'].mean() * 100:.1f}%")
            with col2:
                st.metric("Avg Runs/Shot", f"{player_data['runs'].mean():.2f}")
            with col3:
                st.metric("Boundary %", f"{player_data['is_boundary'].mean() * 100:.1f}%")
            
            st.progress(player_data['control_score'].mean() / 100)
            st.caption("Control Quality Score")

if __name__ == "__main__":
    # Check if secrets file exists, create if not
    password = create_secrets_file()
    
    if password:
        st.info(f"🔑 A new password has been generated: `{password}`")
        st.info("Please save this password and share it only with authorized users.")
        st.info("The app will restart in a moment...")
        st.experimental_rerun()
    
    # Run the main app
    main()
