import gymnasium as gym
# import highway_env
from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
import pprint
import OTW

def make_env():
    env = gym.make("street-v1")#, render_mode="rgb_array")
    
    env.unwrapped.config["lanes_count"] = 5
    pprint.pprint(env.unwrapped.config)
    env = Monitor(env)  # Write log reward to TensorBoard
    return env


env = DummyVecEnv([make_env])

model = DQN(
    'MlpPolicy',
    env,
    policy_kwargs=dict(net_arch=[256, 256]),
    learning_rate=5e-4,
    buffer_size=15000,
    learning_starts=200,
    batch_size=32,
    gamma=0.8,
    train_freq=1,
    gradient_steps=1,
    target_update_interval=50,
    verbose=1,
    tensorboard_log="log_dir/"
)

model.learn(total_timesteps=int(2e4), tb_log_name="dqn_run")
model.save("highway_dqn/model")