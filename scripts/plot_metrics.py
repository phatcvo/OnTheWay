import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# === Đọc file metrics ===
path = "reports/summary_gating.csv"
if not os.path.exists(path):
    raise FileNotFoundError(f"{path} không tồn tại — hãy chắc bạn đã train PPO và có file metrics CSV!")

df = pd.read_csv(path)

# === Chuẩn hóa dữ liệu ===
# loại bỏ các dòng NaN hoặc trùng model_idx
df = df.dropna(subset=["reward_mean", "ttc_p95", "jerk_p95"])
df = df.sort_values(by="model_idx").reset_index(drop=True)

# Nếu model_idx trùng nhau nhiều lần => lấy trung bình
df_group = df.groupby("model_idx").agg({
    "reward_mean": "mean",
    "reward_std": "mean",
    "ttc_p95": "mean",
    "jerk_p95": "mean",
    "collision_rate": "mean"
}).reset_index()

# === Vẽ biểu đồ ===
fig, axs = plt.subplots(4, 1, figsize=(10, 12), sharex=True)

# --- 1. Reward ---
axs[0].plot(df_group["model_idx"], df_group["reward_mean"], "o-", color="tab:blue", label="Reward mean")
axs[0].fill_between(df_group["model_idx"],
                    df_group["reward_mean"] - df_group["reward_std"],
                    df_group["reward_mean"] + df_group["reward_std"],
                    color="tab:blue", alpha=0.2)
axs[0].set_ylabel("Reward")
axs[0].set_title("Training metrics over PPO models")
axs[0].legend()
axs[0].grid(True)

# --- 2. TTC ---
axs[1].plot(df_group["model_idx"], df_group["ttc_p95"], "o-", color="tab:green", label="TTC p95")
axs[1].axhline(1.5, color="red", linestyle="--", label="safety limit = 1.5 s")
axs[1].set_ylabel("TTC (p95)")
axs[1].legend()
axs[1].grid(True)

# --- 3. Jerk ---
axs[2].plot(df_group["model_idx"], df_group["jerk_p95"], "o-", color="tab:orange", label="Jerk p95")
axs[2].axhline(5.0, color="red", linestyle="--", label="comfort limit = 5.0 m/s³")
axs[2].set_ylabel("Jerk (p95)")
axs[2].set_yscale("log")  # log scale để thấy rõ các giá trị lớn
axs[2].legend()
axs[2].grid(True)

# --- 4. Collision ---
axs[3].plot(df_group["model_idx"], df_group["collision_rate"], "o-", color="tab:red", label="Collision rate")
axs[3].set_ylabel("Collision rate")
axs[3].set_xlabel("Model index")
axs[3].legend()
axs[3].grid(True)

plt.tight_layout()
plt.show()


plt.figure(figsize=(10,6))
# Normalize
def normalize(series):
    return (series - np.min(series)) / (np.max(series) - np.min(series) + 1e-6)

plt.plot(df_group["model_idx"], normalize(df_group["reward_mean"]), "o-", label="Reward (norm)")
plt.plot(df_group["model_idx"], normalize(df_group["ttc_p95"]), "o-", label="TTC p95 (norm)")
plt.plot(df_group["model_idx"], 1 - normalize(df_group["jerk_p95"]), "o-", label="1 - Jerk p95 (norm)")
plt.plot(df_group["model_idx"], 1 - normalize(df_group["collision_rate"]), "o-", label="1 - Collision (norm)")

plt.title("Normalized metrics comparison")
plt.xlabel("Model index")
plt.ylabel("Normalized score")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

