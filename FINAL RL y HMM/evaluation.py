import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from trading_env import TradingEnv
from rl_agent import DQNAgent
import os

# Set plot style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("viridis")

def calculate_metrics(returns):
    """Calculates Sharpe, Sortino, Max Drawdown, and Cumulative Return."""
    if len(returns) < 2:
        return 0, 0, 0, 0
    
    # Annualized metrics (assuming daily data, 252 trading days)
    mean_ret = returns.mean() * 252
    std_ret = returns.std() * np.sqrt(252)
    
    sharpe = mean_ret / std_ret if std_ret != 0 else 0
    
    # Sortino
    downside_returns = returns[returns < 0]
    std_downside = downside_returns.std() * np.sqrt(252)
    sortino = mean_ret / std_downside if std_downside != 0 else 0
    
    # Cumulative Return
    cum_ret = (1 + returns).cumprod()
    total_ret = cum_ret.iloc[-1] - 1
    
    # Max Drawdown
    running_max = cum_ret.cummax()
    drawdown = (cum_ret - running_max) / running_max
    max_drawdown = drawdown.min()
    
    return sharpe, sortino, max_drawdown, total_ret

def run_strategy(env, agent=None, strategy='agent'):
    """Runs a strategy on the environment and returns portfolio history."""
    state, _ = env.reset()
    done = False
    
    portfolio_values = [env.initial_balance]
    
    while not done:
        if strategy == 'agent':
            action = agent.act(state)
        elif strategy == 'buy_and_hold':
            action = 1 # Always Long
        elif strategy == 'random':
            action = env.action_space.sample()
        elif strategy == 'hmm_heuristic':
            # 0: Low Vol (Buy), 1: High Vol (Sell/Cash), 2: Lateral (Hold)
            # State index 5, 6, 7 are HMM one-hot
            # Reconstruct HMM state
            hmm_one_hot = state[5:]
            hmm_state = np.argmax(hmm_one_hot)
            
            if hmm_state == 0: # Low Vol -> Buy
                action = 1
            elif hmm_state == 1: # High Vol -> Sell/Cash
                action = 2
            else: # Lateral -> Hold
                action = 0
        else:
            action = 0 # Default Hold
            
        next_state, _, done, truncated, _ = env.step(action)
        done = done or truncated
        state = next_state
        portfolio_values.append(env.portfolio_value)
        
    return pd.Series(portfolio_values)

def evaluate_models():
    print("Evaluating models...")
    # Load data
    df = pd.read_csv("processed_data_with_hmm.csv")
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Split (Same as training)
    train_df = df[df['Date'] < '2025-01-01']
    test_df = df[df['Date'] >= '2025-01-01']
    
    if len(test_df) == 0:
        split_idx = int(len(df) * 0.8)
        test_df = df.iloc[split_idx:]
    
    # Initialize Environment
    env = TradingEnv(test_df)
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    
    # Load Agent
    agent = DQNAgent(state_size, action_size)
    
    model_path_tf = "dqn_model.weights.h5"
    model_path_sklearn = "dqn_model.pkl"
    
    if os.path.exists(model_path_tf):
        try:
            agent.load(model_path_tf)
            print(f"Agent loaded from {model_path_tf}.")
        except Exception as e:
            print(f"Failed to load TF model: {e}")
    elif os.path.exists(model_path_sklearn):
        try:
            agent.load(model_path_sklearn)
            print(f"Agent loaded from {model_path_sklearn}.")
        except Exception as e:
            print(f"Failed to load Sklearn model: {e}")
    else:
        print("No trained model found. Using random agent.")
    
    # Run Strategies
    results = {}
    
    # 1. RL Agent
    print("Running RL Agent...")
    results['RL Agent'] = run_strategy(env, agent, strategy='agent')
    
    # 2. Buy & Hold
    print("Running Buy & Hold...")
    results['Buy & Hold'] = run_strategy(env, strategy='buy_and_hold')
    
    # 3. Random
    print("Running Random...")
    results['Random'] = run_strategy(env, strategy='random')
    
    # 4. HMM Heuristic
    print("Running HMM Heuristic...")
    results['HMM Heuristic'] = run_strategy(env, strategy='hmm_heuristic')
    
    # Calculate Metrics
    metrics = []
    for name, equity_curve in results.items():
        returns = equity_curve.pct_change().dropna()
        sharpe, sortino, max_dd, total_ret = calculate_metrics(returns)
        metrics.append({
            'Model': name,
            'Sharpe Ratio': sharpe,
            'Sortino Ratio': sortino,
            'Max Drawdown': max_dd,
            'Cumulative Return': total_ret
        })
    
    metrics_df = pd.DataFrame(metrics)
    print("\nPerformance Metrics:")
    print(metrics_df)
    metrics_df.to_csv("evaluation_metrics.csv", index=False)
    
    # Plot Equity Curves
    plt.figure(figsize=(12, 8))
    for name, equity_curve in results.items():
        plt.plot(equity_curve, label=name)
    
    plt.title('Equity Curve Comparison')
    plt.xlabel('Trading Days')
    plt.ylabel('Portfolio Value')
    plt.legend()
    plt.tight_layout()
    plt.savefig('output_images/equity_curve_comparison.png')
    plt.close()
    print("Evaluation complete. Metrics saved to 'evaluation_metrics.csv' and plot to 'output_images/equity_curve_comparison.png'.")

if __name__ == "__main__":
    evaluate_models()
