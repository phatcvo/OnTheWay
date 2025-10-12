import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecVideoRecorder
import pprint
from pathlib import Path
import otw_env

if __name__ == "__main__":
    # Create the environment
    env = gym.make("street-v1")
    try:
        env = gym.make("street-v1")
    except Exception as e:
        print(f"Error: {e}")
        # Debug: kiểm tra env có gì
        from otw_env.envs.street_env import StreetEnv
        env_instance = StreetEnv()
        print(f"env.vehicles: {env_instance.vehicles if hasattr(env_instance, 'vehicles') else 'N/A'}")
        print(f"env.vehicle: {env_instance.vehicle if hasattr(env_instance, 'vehicle') else 'N/A'}")
        print(f"Type of env.vehicle: {type(env_instance.vehicle) if hasattr(env_instance, 'vehicle') else 'N/A'}")
    
    env.unwrapped.config["lanes_count"] = 5
    pprint.pprint(env.unwrapped.config)
    env = Monitor(env)  # Write log reward to TensorBoard
    obs, info = env.reset()
    
    # Train
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
        tensorboard_log="log/tensorboard"
    )

    model.learn(total_timesteps=int(2e4), tb_log_name="change_lane_dqn")
    Path("models").mkdir(exist_ok=True)
    model.save("models/change_lane_dqn.zip")
    print("✅ Saved models/change_lane_dqn.zip")
    
        # Record video
    model = DQN.load("models/change_lane_dqn")

    video_length = 2 * env.envs[0].config["duration"]
    env = VecVideoRecorder(
        env,
        "videos/",
        record_video_trigger=lambda x: x == 0,
        video_length=video_length,
        name_prefix="dqn-agent",
    )
    obs, info = env.reset()
    for _ in range(video_length + 1):
        action, _ = model.predict(obs)
        obs, _, _, _, _ = env.step(action)
    env.close()