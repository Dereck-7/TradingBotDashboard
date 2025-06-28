from rl_env import TradeHistoryEnv
from stable_baselines3 import PPO

env = TradeHistoryEnv("trade_log.csv")
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=10000)
model.save("ppo_rl_trading_agent")

print("✅ Training complete!")
