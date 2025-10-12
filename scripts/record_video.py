import argparse
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecVideoRecorder
import gymnasium as gym
import numpy as np
import os
import time
import otw_env
import warnings

# === Hàm tạo environment ===
def make_env(render_mode=None):
    env = gym.make("street-v1", render_mode=render_mode)
    env.unwrapped.configure({
        "lanes_count": 4,
        "vehicles_count": 10,
        "duration": 30,
        "simulation_frequency": 15,
        "policy_frequency": 5,
        "offscreen_rendering": False,
        "screen_width": 800,
        "screen_height": 300,
    })
    return env


# === Hàm chạy hoặc ghi video ===
def record_video(model_path, render_mode="none", video_prefix="ppo_best_agent"):
    print(f"🎥 Loading model from: {model_path}")
    model = PPO.load(model_path)

    # Lấy chế độ render
    render_mode = render_mode.lower()
    os.makedirs("videos", exist_ok=True)

    if render_mode == "human":
        warnings.filterwarnings("ignore", category=UserWarning, message="Overriding environment")
        print("🧭 Running in HUMAN mode (display only, with trajectory)...")

        env = DummyVecEnv([lambda: make_env(render_mode="human")])
        obs = env.reset()
        base_env = env.envs[0].unwrapped

        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)

            try:
                if hasattr(base_env, "viewer") and hasattr(base_env.viewer, "set_agent_action_sequence"):
                    if np.isscalar(action[0]):
                        plan = [int(action[0])] * 5
                    else:
                        plan = [action[0].tolist()] * 5
                    base_env.viewer.set_agent_action_sequence(plan)
            except Exception as e:
                print(f"[WARN] set_agent_action_sequence failed: {e}")

            base_env.render()     # 🟢 Thêm dòng này để update GUI
            time.sleep(0.05)

            if done.any():
                obs = env.reset()

    else:
        # =========================
        # 🎥 Chế độ ghi video (offscreen, không GUI)
        # =========================
        warnings.filterwarnings("ignore", category=UserWarning, message="Overriding environment")
        print("🎞 Running in RGB_ARRAY mode (recording video)...")
        env = DummyVecEnv([lambda: make_env(render_mode=None)])
        base_env = env.envs[0].unwrapped
        duration = base_env.config.get("duration", 30)
        video_length = 2 * duration

        env = VecVideoRecorder(
            env,
            "videos/",
            record_video_trigger=lambda step: step == 0,
            video_length=video_length,
            name_prefix=video_prefix,
        )

        obs = env.reset()
        for _ in range(video_length):
            action, _ = model.predict(obs, deterministic=True)
            obs, _, done, info = env.step(action)
            time.sleep(0.1)
            if done.any():
                obs = env.reset()

        env.close()
        print(f"✅ Video saved to ./videos/{video_prefix}-step-0-to-{video_length}.mp4")


# === Entry point ===
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Record or View PPO StreetEnv")
    parser.add_argument("--model", type=str, required=True, help="Path to trained model (.zip)")
    parser.add_argument("--render", type=str, default="none", choices=["none", "human"], help="Render mode (human=view / none=record)")
    parser.add_argument("--prefix", type=str, default="ppo_best_agent", help="Video file name prefix")
    args = parser.parse_args()

    record_video(args.model, render_mode=args.render, video_prefix=args.prefix)