import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler
import joblib
import os

# --- Page Configuration ---
st.set_page_config(
    page_title="USD/PEN Algorithmic Trading Model",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom CSS for "Premium" Look ---
st.markdown("""
<style>
    .stApp {
        background-color: #0e1117;
        color: #fafafa;
    }
    .metric-card {
        background-color: #262730;
        padding: 20px;
        border-radius: 10px;
        border: 1px solid #41444b;
        text-align: center;
    }
    .metric-value {
        font-size: 24px;
        font-weight: bold;
        color: #00ff00;
    }
    .metric-label {
        font-size: 14px;
        color: #a0a0a0;
    }
</style>
""", unsafe_allow_html=True)

# --- Helper Functions ---

@st.cache_data
def load_data(ticker, start_date, end_date):
    """Fetches data from Yahoo Finance."""
    try:
        df = yf.download(ticker, start=start_date, end=end_date)
        if df.empty:
            return None
        
        # Flatten MultiIndex if present
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
            
        df = df.reset_index()
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

def calculate_features(df):
    """Calculates necessary features for HMM and Analysis."""
    df = df.copy()
    # Log Returns
    df['Log_Ret'] = np.log(df['Close'] / df['Close'].shift(1))
    
    # Volatility (10-day rolling std of log returns)
    df['Vol_10d'] = df['Log_Ret'].rolling(window=10).std()
    
    # RSI (14-day)
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    df.dropna(inplace=True)
    return df

def train_hmm(df, n_components=3):
    """Trains Gaussian HMM and predicts regimes."""
    X = df[['Log_Ret', 'Vol_10d']].values
    
    # Scale data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Train HMM
    model = GaussianHMM(n_components=n_components, covariance_type="full", n_iter=100, random_state=42)
    model.fit(X_scaled)
    
    # Predict States
    hidden_states = model.predict(X_scaled)
    df['HMM_State'] = hidden_states
    
    # Map States to Regimes (Heuristic based on Volatility/Return)
    state_stats = df.groupby('HMM_State')[['Log_Ret', 'Vol_10d']].mean()
    
    # Sort by Volatility: Low Vol -> Bullish (usually), High Vol -> Bearish/Crisis, Medium -> Sideways
    # This is a simplification. We can also look at Mean Return.
    # Let's try: 
    # Lowest Volatility = Bullish/Stable (Green)
    # Highest Volatility = Bearish/Volatile (Red)
    # Middle = Sideways (Grey)
    
    sorted_states = state_stats.sort_values('Vol_10d')
    state_map = {
        sorted_states.index[0]: 'Low Vol (Bullish)',
        sorted_states.index[1]: 'Neutral (Sideways)',
        sorted_states.index[2]: 'High Vol (Bearish)'
    }
    
    df['Regime'] = df['HMM_State'].map(state_map)
    return df, model, state_map

# --- Main App Layout ---

def main():
    st.sidebar.title("⚙️ Configuration")
    
    # Sidebar Inputs
    ticker = st.sidebar.text_input("Ticker Symbol", value="PEN=X")
    start_date = st.sidebar.date_input("Start Date", value=pd.to_datetime("2015-01-01"))
    end_date = st.sidebar.date_input("End Date", value=pd.to_datetime("2025-12-18"))
    
    st.title(f"📊 Algorithmic Trading Analysis: {ticker}")
    st.markdown("### HMM Regime Detection & RL Agent Analysis")
    
    # Load Data
    with st.spinner('Loading Data...'):
        raw_df = load_data(ticker, start_date, end_date)
        
    if raw_df is not None:
        # Preprocessing
        df = calculate_features(raw_df)
        
        # HMM Training
        df, hmm_model, state_map = train_hmm(df)
        
        # --- Dashboard Metrics ---
        col1, col2, col3, col4 = st.columns(4)
        
        total_return = (df['Close'].iloc[-1] / df['Close'].iloc[0] - 1) * 100
        volatility = df['Log_Ret'].std() * np.sqrt(252) * 100
        sharpe = (df['Log_Ret'].mean() / df['Log_Ret'].std()) * np.sqrt(252)
        current_price = df['Close'].iloc[-1]
        
        col1.metric("Current Price", f"{current_price:.4f}")
        col2.metric("Total Return", f"{total_return:.2f}%", delta_color="normal")
        col3.metric("Annualized Volatility", f"{volatility:.2f}%")
        col4.metric("Sharpe Ratio", f"{sharpe:.2f}")
        
        # --- Interactive Plotly Chart ---
        st.subheader("Market Regimes & Price Action")
        
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                            vertical_spacing=0.03, subplot_titles=('Price & Regimes', 'RSI'),
                            row_heights=[0.7, 0.3])
        
        # Candlestick
        fig.add_trace(go.Candlestick(x=df['Date'],
                        open=df['Open'], high=df['High'],
                        low=df['Low'], close=df['Close'],
                        name='Price'), row=1, col=1)
        
        # Regime Background Colors
        # We create a separate trace for markers or use shapes. 
        # For simplicity and performance in Plotly, we can use a Scatter plot with color mapped to regime.
        # But overlaying background is better. Let's use Scatter markers behind or on top.
        
        colors = {'Low Vol (Bullish)': '#00ff00', 'Neutral (Sideways)': '#808080', 'High Vol (Bearish)': '#ff0000'}
        
        # Create a scatter plot for regimes to show in legend
        for regime, color in colors.items():
            mask = df['Regime'] == regime
            fig.add_trace(go.Scatter(
                x=df.loc[mask, 'Date'], 
                y=df.loc[mask, 'Close'],
                mode='markers',
                marker=dict(size=4, color=color),
                name=regime,
                opacity=0.6
            ), row=1, col=1)

        # RSI
        fig.add_trace(go.Scatter(x=df['Date'], y=df['RSI'], name='RSI', line=dict(color='purple')), row=2, col=1)
        fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
        
        fig.update_layout(height=800, xaxis_rangeslider_visible=False, template="plotly_dark")
        st.plotly_chart(fig, use_container_width=True)
        
        # --- RL Agent Section ---
        st.subheader("🤖 Reinforcement Learning Agent (Inference)")
        
        # Try to load the agent
        try:
            from rl_agent import DQNAgent
            # We need to know state_size and action_size. 
            # Based on training_env.py: State=8, Action=3
            agent = DQNAgent(state_size=8, action_size=3)
            
            model_path = "dqn_model_final.keras"
            if os.path.exists(model_path):
                agent.load(model_path.replace(".keras", "")) # .pkl is added by joblib in load if using sklearn backend logic in user's code? 
                # User's code: joblib.dump(self.model, name + ".pkl")
                # Agent load: joblib.load(name + ".pkl") if backend is sklearn
                # So we pass "dqn_model_final"
                
                st.success(f"RL Agent loaded from {model_path}")
                
                # Predict Action for the last day
                # Construct state: [Price_Norm, RSI_Norm, PCA1, PCA2, PCA3, HMM_OneHot...]
                # We need PCA. In app.py we didn't do PCA. 
                # To fully support RL inference, we need to replicate the EXACT preprocessing pipeline (PCA).
                # This is complex because PCA needs to be fitted on the same training data.
                # For this demo, we will skip PCA or load the saved PCA model if we had one.
                # Since we don't have a saved PCA model, we will warn the user.
                
                st.warning("⚠️ RL Agent requires PCA components which are not fully replicated in this live dashboard yet. Showing HMM-based signals instead.")
                
                # HMM-based signal (Heuristic)
                last_regime = df['Regime'].iloc[-1]
                if "Low Vol" in last_regime:
                    st.metric("Recommended Action (HMM)", "BUY 🟢", "Bullish Regime")
                elif "High Vol" in last_regime:
                    st.metric("Recommended Action (HMM)", "SELL 🔴", "Bearish Regime")
                else:
                    st.metric("Recommended Action (HMM)", "HOLD ⚪", "Neutral Regime")
                    
            else:
                st.info("RL Model not found. Run `rl_agent.py` to train.")
                
        except Exception as e:
            st.error(f"Could not load RL Agent: {e}")
            
        # Show raw data option
        with st.expander("View Processed Data"):
            st.dataframe(df)
            
    else:
        st.warning("No data found. Please check the ticker symbol.")

if __name__ == "__main__":
    main()
