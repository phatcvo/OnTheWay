import json, csv, os

class Logger:
    def __init__(self, path="logs.json"):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self.path = path
        self.data = []

    def log(self, t, obs, reward):
        """
        obs: list or dict of observed state
        reward: float
        """
        if isinstance(obs, list):
            record = {"time": t, "reward": reward}
            for i, val in enumerate(obs):
                record[f"obs_{i}"] = val
        elif isinstance(obs, dict):
            record = {"time": t, "reward": reward, **obs}
        else:
            record = {"time": t, "reward": reward}
        self.data.append(record)

    def save_json(self):
        with open(self.path, "w") as f:
            json.dump(self.data, f, indent=2)
        print(f"✅ Log saved to {self.path}")

    def save_csv(self):
        csv_path = os.path.splitext(self.path)[0] + ".csv"
        if not self.data:
            print("⚠️ No data to save.")
            return
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.data[0].keys())
            writer.writeheader()
            writer.writerows(self.data)
        print(f"✅ CSV saved to {csv_path}")
