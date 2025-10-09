def compute_reward(obs):
    """Simple reward: encourage forward motion, penalize deviation."""
    x, y, v = obs
    return v - 0.1 * abs(y)
