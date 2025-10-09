import gymnasium as gym
import highway_env
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
import pprint
# === Tạo environment ===
def make_env():
    env = gym.make("highway-v0", render_mode="rgb_array")
    
    env.unwrapped.config["lanes_count"] = 5
    env.unwrapped.config["policy_frequency"] = 15
    pprint.pprint(env.unwrapped.config)
    env = Monitor(env)  # Write log reward to TensorBoard
    return env

env = DummyVecEnv([make_env])

eval_env = DummyVecEnv([make_env])
eval_callback = EvalCallback(
    eval_env,
    best_model_save_path="./highway_ppo/best_model",
    log_path="./highway_ppo/logs",
    eval_freq=5000,
    deterministic=True,
    render=False,
    callback_on_new_best=StopTrainingOnRewardThreshold(reward_threshold=30, verbose=1)
)

model = PPO(
    "MlpPolicy",
    env,
    policy_kwargs=dict(net_arch=[256, 256]),
    learning_rate=5e-4,
    n_steps=1024,
    batch_size=64,
    gamma=0.8,
    ent_coef=0.01,
    verbose=1,
    tensorboard_log="log_dir/",
)

model.learn(total_timesteps=int(2e4), tb_log_name="ppo_run")
model.save("highway_ppo/model")
