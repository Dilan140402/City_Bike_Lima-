import pandas as pd
import numpy as np
from hmmlearn.hmm import GaussianHMM
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# Set plot style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("viridis")

def calculate_rsi(data, window=14):
    """Calculates the Relative Strength Index (RSI)."""
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50) # Fill NaN with 50 (neutral)

def fit_hmm(df, n_components=3):
    """Fits a Gaussian HMM to detect market regimes."""
    print(f"Fitting Gaussian HMM with {n_components} components...")
    
    # Use Log Returns and Volatility for HMM
    # Reshape for hmmlearn (n_samples, n_features)
    X = df[['Log_Ret', 'Vol_10d']].values
    
    # Train HMM
    model = GaussianHMM(n_components=n_components, covariance_type="full", n_iter=1000, random_state=42)
    model.fit(X)
    
    # Predict states
    hidden_states = model.predict(X)
    
    # Analyze states to map them to regimes
    # We assume:
    # High Volatility -> Crisis/Bearish
    # Low Volatility -> Bullish/Stable
    # Medium -> Sideways
    
    state_stats = []
    for i in range(n_components):
        mask = (hidden_states == i)
        vol_mean = X[mask, 1].mean() # Mean of Vol_10d for this state
        ret_mean = X[mask, 0].mean() # Mean of Log_Ret
        state_stats.append({'state': i, 'vol_mean': vol_mean, 'ret_mean': ret_mean})
    
    stats_df = pd.DataFrame(state_stats)
    print("State Statistics:\n", stats_df)
    
    # Sort by volatility
    sorted_stats = stats_df.sort_values('vol_mean')
    
    # Mapping
    # Lowest Vol -> Low Volatility (0)
    # Middle Vol -> Lateral (2) - Wait, prompt says: 1. Baja, 2. Alta, 3. Lateral.
    # Let's map to 0, 1, 2 integers first, then we can interpret.
    # Let's assign:
    # 0: Low Volatility (Alcista)
    # 1: High Volatility (Bajista)
    # 2: Lateral
    
    # Sorted indices
    low_vol_state = sorted_stats.iloc[0]['state']
    high_vol_state = sorted_stats.iloc[-1]['state']
    lateral_state = sorted_stats.iloc[1]['state']
    
    mapping = {
        low_vol_state: 0,   # Low Vol
        high_vol_state: 1,  # High Vol
        lateral_state: 2    # Lateral
    }
    
    print(f"State Mapping (Original -> New): {mapping}")
    
    # Apply mapping
    df['HMM_State'] = [mapping[s] for s in hidden_states]
    
    return df, model

def plot_regimes(df):
    """Plots the price colored by HMM regime."""
    print("Plotting regimes...")
    plt.figure(figsize=(15, 8))
    
    colors = {0: 'green', 1: 'red', 2: 'blue'}
    labels = {0: 'Low Vol (Bullish)', 1: 'High Vol (Bearish)', 2: 'Lateral'}
    
    for state in [0, 1, 2]:
        mask = df['HMM_State'] == state
        plt.plot(df.index[mask], df['TC_Yahoo'][mask], '.', markersize=5, 
                 color=colors[state], label=labels[state], alpha=0.6)
    
    plt.plot(df.index, df['TC_Yahoo'], color='gray', alpha=0.3, linewidth=1)
    
    plt.title('USD/PEN Market Regimes Detected by HMM')
    plt.xlabel('Date')
    plt.ylabel('USD/PEN')
    plt.legend()
    plt.tight_layout()
    plt.savefig('output_images/hmm_regimes.png')
    plt.close()

if __name__ == "__main__":
    # Load processed data
    df = pd.read_csv("processed_data.csv")
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Calculate RSI
    df['RSI'] = calculate_rsi(df['TC_Yahoo'])
    
    # Fit HMM
    df_hmm, model = fit_hmm(df)
    
    # Plot
    plot_regimes(df_hmm)
    
    # Save model and data
    joblib.dump(model, "hmm_model.pkl")
    df_hmm.to_csv("processed_data_with_hmm.csv", index=False)
    print("HMM Phase complete. Data saved to 'processed_data_with_hmm.csv'.")
