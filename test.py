import gymnasium as gym
import highway_env
from stable_baselines3 import PPO, DQN
import matplotlib.pyplot as plt

env = gym.make('highway-v0', render_mode='rgb_array')
model = PPO.load("highway_ppo/final_model.zip")

plt.ion() 

while True:
    done = truncated = False
    obs, info = env.reset()
    total_reward = 0

    while not (done or truncated):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = env.step(action)
        total_reward += reward

        frame = env.render()

    print("Episode Reward:", total_reward)