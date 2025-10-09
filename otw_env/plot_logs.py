import json, sys
import matplotlib.pyplot as plt

def plot_log(file_path):
    with open(file_path, "r") as f:
        data = json.load(f)

    t = [d["time"] for d in data]
    reward = [d["reward"] for d in data]

    x = [d.get("obs_0", d.get("x", None)) for d in data]
    y = [d.get("obs_1", d.get("y", None)) for d in data]
    v = [d.get("obs_2", d.get("v", None)) for d in data]

    fig, ax = plt.subplots(3, 1, figsize=(8, 8), sharex=True)
    ax[0].plot(t, x, label="x (m)")
    ax[0].plot(t, y, label="y (m)")
    ax[0].legend(); ax[0].grid(); ax[0].set_ylabel("Position")

    ax[1].plot(t, v, label="velocity (m/s)")
    ax[1].legend(); ax[1].grid(); ax[1].set_ylabel("Velocity")

    ax[2].plot(t, reward, label="reward", color="orange")
    ax[2].legend(); ax[2].grid(); ax[2].set_xlabel("time (s)")
    ax[2].set_ylabel("Reward")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python plot_logs.py <log_file.json>")
    else:
        plot_log(sys.argv[1])
