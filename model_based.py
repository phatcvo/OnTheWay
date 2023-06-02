import gym
import OTW
import pprint

# IO
from pathlib import Path
# Models and computation
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import namedtuple

# Visualization
import matplotlib
import matplotlib.pyplot as plt
import tqdm
from tqdm.notebook import trange

# Create environment
# env = gym.make("street-v1")
env = gym.make("inter-v1")
# env = gym.make("Troad-v1")
# env = gym.make("park-v1")
env.reset()
pprint.pprint(env.config)

while True:
    done = False
    obs = env.reset()
    while not done:
        # Test environment
        # action = env.action_type.actions_indexes["IDLE"] 
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        # obs, reward, done, truncated, info = env.step(action)
        
        print(obs)
        env.render()