import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecVideoRecorder, SubprocVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import VecEnv
import numpy as np
import os, csv
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
import otw_env

# === Config ===
TOTAL_TIMESTEPS = 100000   
EVAL_EPISODES = 15        # episode evaluate
N_MODELS = 5             
TENSORBOARD_DIR = "logs/tensorboard"
REPORT_FILE = "reports/summary_gating.csv"

# === Gating thresholds ===
TTC_MIN_P95 = 1.5
JERK_MAX_P95 = 5.0
COLLISION_MAX_RATE = 0.05

# === Helper: create environment ===
def make_env(render_mode=None):
    env = gym.make("street-v1")
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
    env = Monitor(env)
    return env

# === Helper: compute metrics ===
def compute_metrics(ep_rewards, ttc, jerk, collisions):
    def p(x, q): return np.percentile(x, q) if len(x) else 0
    return {
        "reward_mean": np.mean(ep_rewards),
        "reward_std": np.std(ep_rewards),
        "ttc_p50": p(ttc, 50),
        "ttc_p95": p(ttc, 95),
        "jerk_p50": p(jerk, 50),
        "jerk_p95": p(jerk, 95),
        "collision_rate": np.mean(collisions),
        "collision_count": np.sum(collisions)
    }


def safe_reset(env):
    if isinstance(env, VecEnv):
        obs = env.reset()
    else:
        reset_out = env.reset()
        if isinstance(reset_out, tuple):
            obs, _ = reset_out
        else:
            obs = reset_out
    return obs


def run_eval(env, model, episodes=EVAL_EPISODES, dt=0.1):
    rewards, ttc_list, jerk_list, collisions = [], [], [], []
    for ep in range(episodes):
        obs = safe_reset(env)
        done, ep_r = False, 0
        prev_speed, prev_accel = None, None

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)  
            ep_r += reward

            base_env = env.envs[0].unwrapped if hasattr(env, "envs") else env.unwrapped
            ego = base_env.vehicle
            others = base_env.road.vehicles if hasattr(base_env, "road") else []

            speed = np.linalg.norm(ego.velocity)
            if prev_speed is not None:
                accel = (speed - prev_speed) / dt
                if prev_accel is not None:
                    jerk = abs((accel - prev_accel) / dt)
                    jerk = abs((accel - prev_accel) / dt)
                    jerk = min(jerk, 10.0)   # limit max value
                    jerk_list.append(jerk)
                prev_accel = accel
            prev_speed = speed

            ttc_min = np.inf
            for other in others:
                if other is ego:
                    continue
                dx = other.position[0] - ego.position[0]
                dv = ego.velocity[0] - other.velocity[0]
                if dx > 0 and dv > 0:
                    ttc_min = min(ttc_min, dx / dv)
            if np.isfinite(ttc_min):
                ttc_list.append(ttc_min)

            collision = any(ego.crashed or getattr(v, "crashed", False) for v in others)
            collisions.append(1 if collision else 0)

        rewards.append(ep_r)

    return compute_metrics(rewards, ttc_list, jerk_list, collisions)


# === Helper: write CSV report ===
def save_report(metrics, model_idx, path=REPORT_FILE):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    header = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.writer(f)
        if header:
            w.writerow(list(metrics.keys()) + ["model_idx"])
        w.writerow(list(metrics.values()) + [model_idx])

# === Main training ===
if __name__ == "__main__":
    Path("models").mkdir(exist_ok=True)
    Path("videos").mkdir(exist_ok=True)
    Path("logs").mkdir(exist_ok=True)
    Path("reports").mkdir(exist_ok=True)

    writer = SummaryWriter(TENSORBOARD_DIR)

    best_reward = -np.inf
    best_model_path = None

    for i in range(1, N_MODELS + 1):
        print(f"\n🚀 Training PPO model {i}/{N_MODELS} ...")

        env = DummyVecEnv([lambda: make_env(render_mode=None)])
        # env = SubprocVecEnv([lambda: make_env(render_mode=None)]*4) # 
        model = PPO(
            "MlpPolicy",                            # fully-connected with Kinematics vector
            env,
            policy_kwargs=dict(net_arch=[128, 128]),# network actor/critic iwth hidden N layers.
            learning_rate=3e-4,                     # High → fast learn but easy viration/diverge; low → stable
            n_steps=1024,                            # step to rollout before updating; High → advantage estimation more smooth (low noise) ⇒ more stable but high RAM & latency. test: 128-512; train: 1024-2048
            batch_size=64,                          # num of samples per minibatch; batch_size ≤ n_steps * n_envs
            n_epochs=10,                             # High (5–10) improve by take an advantage data but easy overfit clip; low is fast train. Test: 1–3; train: 5–10.
            gamma=0.97,                             # discount factor. High (0.99) → far-sighted agent; low (0.95) → short-sighted agent; Test: 0.95; train highway: 0.97-0.99
            gae_lambda=0.95,                         # bias-variance trade-off near 1.0 → a small variance (smooth) but high bias; low → small bias but noisy. Train 0.95–0.98; 0.9 for test.
            clip_range=0.2,                         # threshold PPO clipping (|ratio−1| ≤ clip); small (0.1–0.2) → stable, small step learn; high → fast learn but easy to distroy policy. 0.2 is good default.
            ent_coef=0.008,                          # coeff for entropy bonus; high → explore more (noisy, stochastic); low → exploit more (deterministic). test: 0.01; train: 0.001-0.01
            verbose=1,                              # log in terminal
            tensorboard_log=TENSORBOARD_DIR,
            device="cpu",                           # force CPU
        )

        model.learn(total_timesteps=TOTAL_TIMESTEPS, tb_log_name=f"ppo_street_{i}")
        metrics = run_eval(env, model, episodes=EVAL_EPISODES)
        print(f"Model {i} metrics:")
        for k, v in metrics.items():
            print(f"  {k}: {v:.3f}")

        passed = (
            metrics["ttc_p95"] >= TTC_MIN_P95 and
            metrics["jerk_p95"] <= JERK_MAX_P95 and
            metrics["collision_rate"] <= COLLISION_MAX_RATE
        )

        writer.add_scalar("eval/reward_mean", metrics["reward_mean"], i)
        writer.add_scalar("eval/ttc_p95", metrics["ttc_p95"], i)
        writer.add_scalar("eval/jerk_p95", metrics["jerk_p95"], i)
        writer.add_scalar("eval/collision_rate", metrics["collision_rate"], i)

        save_report(metrics, model_idx=i)

        if passed and metrics["reward_mean"] > best_reward:
            best_reward = metrics["reward_mean"]
            best_model_path = f"models/ppo_street_best.zip"
            model.save(best_model_path)
            print(f"✅ New best model (reward={best_reward:.2f}) passed gating!")

        env.close()

    writer.close()

    print(f"\n🏆 Best model: {best_model_path} (mean_reward={best_reward:.2f})")
