import gym
from gym import spaces
import numpy as np
import pandas as pd

class TradeHistoryEnv(gym.Env):
    def __init__(self, csv_file):
        super(TradeHistoryEnv, self).__init__()

        self.df = pd.read_csv(csv_file)
        self.current_step = 0

        self.action_space = spaces.Discrete(3)

        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32)

    def _get_obs(self):
        row = self.df.iloc[self.current_step]
        obs = np.array([
            1 if row["bias"] == "Bullish" else -1 if row["bias"] == "Bearish" else 0,
            1 if row["action"] == "buy" else -1,
            float(row["entry_price"]),
            float(row["sl"]),
            float(row["tp"]),
            1 if row["result"] == "win" else -1
        ])
        return obs

    def step(self, action):
        row = self.df.iloc[self.current_step]

        reward = 0
        done = False
        info = {}

        if action == 0:  # Hold
            reward = -0.1
        elif (action == 1 and row["result"] == "win" and row["action"] == "buy") or \
            (action == 2 and row["result"] == "win" and row["action"] == "sell"):
            reward = 1
        else:
            reward = -1

        self.current_step += 1
        if self.current_step >= len(self.df) - 1:
            done = True

        return self._get_obs(), reward, done, False, info

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        return self._get_obs(), {}

    
