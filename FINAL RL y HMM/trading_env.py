import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd

class TradingEnv(gym.Env):
    """Custom Trading Environment that follows gymnasium interface."""
    metadata = {'render.modes': ['human']}

    def __init__(self, df, initial_balance=10000, max_steps=None):
        super(TradingEnv, self).__init__()
        
        self.df = df.reset_index(drop=True)
        self.initial_balance = initial_balance
        self.max_steps = max_steps if max_steps else len(df) - 1
        
        # Actions: 0=Hold, 1=Buy (Long), 2=Sell (Short/Cash)
        self.action_space = spaces.Discrete(3)
        
        # State: [Price_Norm, RSI_Norm, PCA1, PCA2, PCA3, HMM_State (One-Hot x3)]
        # Total features: 1 + 1 + 3 + 3 = 8
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(8,), dtype=np.float32)
        
        self.current_step = 0
        self.balance = initial_balance
        self.position = 0 # 0: Neutral, 1: Long, -1: Short
        self.entry_price = 0
        self.portfolio_value = initial_balance
        self.history = []
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.balance = self.initial_balance
        self.position = 0
        self.entry_price = 0
        self.portfolio_value = self.initial_balance
        self.history = []
        return self._next_observation(), {}
    
    def _next_observation(self):
        # Get current row
        obs = self.df.iloc[self.current_step]
        
        # Normalize Price
        price_norm = obs['TC_Yahoo'] / self.df.iloc[0]['TC_Yahoo']
        
        # RSI (0-100) -> 0-1
        rsi_norm = obs['RSI'] / 100.0
        
        # PCA
        pca1 = obs['PCA_1']
        pca2 = obs['PCA_2']
        pca3 = obs['PCA_3']
        
        # HMM One-Hot
        hmm_state = int(obs['HMM_State'])
        hmm_one_hot = [0, 0, 0]
        if 0 <= hmm_state <= 2:
            hmm_one_hot[hmm_state] = 1
            
        return np.array([price_norm, rsi_norm, pca1, pca2, pca3] + hmm_one_hot, dtype=np.float32)

    def step(self, action):
        current_price = self.df.iloc[self.current_step]['TC_Yahoo']
        
        prev_portfolio_value = self.portfolio_value
        
        if action == 1: # Long
            if self.position == 0:
                self.position = 1
                self.entry_price = current_price
            elif self.position == -1:
                # Close Short
                pnl = (self.entry_price - current_price) / self.entry_price
                self.balance *= (1 + pnl)
                # Open Long
                self.position = 1
                self.entry_price = current_price
                
        elif action == 2: # Short
            if self.position == 0:
                self.position = -1
                self.entry_price = current_price
            elif self.position == 1:
                # Close Long
                pnl = (current_price - self.entry_price) / self.entry_price
                self.balance *= (1 + pnl)
                # Open Short
                self.position = -1
                self.entry_price = current_price
        
        # Update Portfolio Value
        if self.position == 1:
            unrealized_pnl = (current_price - self.entry_price) / self.entry_price
            self.portfolio_value = self.balance * (1 + unrealized_pnl)
        elif self.position == -1:
            unrealized_pnl = (self.entry_price - current_price) / self.entry_price
            self.portfolio_value = self.balance * (1 + unrealized_pnl)
        else:
            self.portfolio_value = self.balance
            
        # Calculate Reward
        reward = (self.portfolio_value - prev_portfolio_value) / prev_portfolio_value
        
        self.current_step += 1
        done = self.current_step >= self.max_steps
        truncated = False # Add truncated for gymnasium
        
        obs = self._next_observation()
        
        return obs, reward, done, truncated, {}

    def render(self, mode='human'):
        print(f"Step: {self.current_step}, Price: {self.df.iloc[self.current_step]['TC_Yahoo']:.4f}, "
              f"Position: {self.position}, Value: {self.portfolio_value:.2f}")
