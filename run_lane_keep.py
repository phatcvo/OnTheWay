from otw_env.env_base import OTWEnv
from otw_env.logger import Logger
import numpy as np

env = OTWEnv()
logger = Logger("logs/log_lane_keep.json")

obs = env.reset()
for t in range(200):
    delta = -0.1 * obs[1]
    a = 0.0
    obs, r, done, _ = env.step((a, delta))
    logger.log(env.time, obs, r)
    if done:
        break

logger.save_json()