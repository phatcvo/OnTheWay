import gym
import OTW

import importlib
import json
import logging
import gym
import numpy as np
import datetime
import logging
import time
import collections
import seaborn as sns
sns.set()
import pprint
# from utils import record_videos, show_videos
# logger = logging.getLogger(__name__)
import RobustPlanner
from RobustPlanner.trainer.evaluation import Evaluation
from RobustPlanner.common.factory import agent_factory, load_agent, load_environment
# from RobustPlanner.agents.robust.graphics.robust_graphics import IntervalRobustPlannerGraphics
# from RobustPlanner.agents.robust.robust import IntervalRobustPlannerAgent

#############################################################
# class Configurable(object):
#     def __init__(self, config=None):
#         self.config = self.default_config()
#         if config:
#             # Override default config with variant
#             Configurable.rec_update(self.config, config)
#             # Override incomplete variant with completed variant
#             Configurable.rec_update(config, self.config)

#     def update_config(self, config):
#         Configurable.rec_update(self.config, config)

#     @classmethod
#     def default_config(cls):
#         return {}

#     @staticmethod
#     def rec_update(d, u):
#         for k, v in u.items():
#             if isinstance(v, collections.Mapping):
#                 d[k] = Configurable.rec_update(d.get(k, {}), v)
#             else:
#                 d[k] = v
#         return d
# ################################################################
# def agent_factory(environment, config):
#     if "__class__" in config:
#         path = config['__class__'].split("'")[1]
#         module_name, class_name = path.rsplit(".", 1)
#         agent_class = getattr(importlib.import_module(module_name), class_name)
#         agent = agent_class(environment, config)
#         return agent
#     else:
#         raise ValueError("The configuration should specify the agent __class__")
    
# def load_agent(agent_config, env):
#     # Load config from file
#     if not isinstance(agent_config, dict):
#         agent_config = load_agent_config(agent_config)
#     return agent_factory(env, agent_config)

# def load_agent_config(config_path):
#     with open(config_path) as f:
#         agent_config = json.loads(f.read())
#     if "base_config" in agent_config:
#         base_config = load_agent_config(agent_config["base_config"])
#         del agent_config["base_config"]
#         agent_config = Configurable.rec_update(base_config, agent_config)
#     return agent_config

# def load_environment(env_config):
#     # Load the environment config from file
#     if not isinstance(env_config, dict):
#         with open(env_config) as f:
#             env_config = json.loads(f.read())

#     # Make the environment
#     if env_config.get("import_module", None):
#         __import__(env_config["import_module"])
#     try:
#         env = gym.make(env_config['id'])
#         # Save env module in order to be able to import it again
#         env.import_module = env_config.get("import_module", None)
#     except KeyError:
#         raise ValueError("The gym register id of the environment must be provided")
#     # except gym.error.UnregisteredEnv:
#     #     # The environment is unregistered.
#     #     print("import_module", env_config["import_module"])
#     #     raise gym.error.UnregisteredEnv('Environment {} not registered. The environment module should be specified by '
#     #                                     'the "import_module" key of the environment configuration'.format(
#     #                                         env_config['id']))

#     # Configure the environment, if supported
#     try:
#         env.unwrapped.configure(env_config)
#         # Reset the environment to ensure configuration is applied
#         env.reset()
#     except AttributeError as e:
#         logger.info("This environment does not support configuration. {}".format(e))
#     return env
# ################################################################

# Create environment
env = gym.make("street-v1")
# env = gym.make("inter-v1")
# env = gym.make("Troad-v1")
env.reset()


# Make agent

agent_config2 = {
    "__class__": "<class 'RobustPlanner.agents.robust.robust.IntervalRobustPlannerAgent'>",
    "env_preprocessors": [{"method":"simplify"}],
    "budget": 50,
    "gamma": 0.7,
}

agent_config1 = {
    "__class__": "<class 'RobustPlanner.agents.robust.robust.IntervalRobustPlannerAgent'>",
    "env_preprocessors": [{"method": "simplify"},
        {"method": "change_vehicles", "args": "OTW.vehicle.prediction.IntervalVehicle"}]
}

agent_config = {
    "__class__": "<class 'RobustPlanner.agents.tree_search.deterministic.DeterministicPlannerAgent'>",
    "env_preprocessors": [{"method":"simplify"}],
    "budget": 50,
    "gamma": 0.7,
}
agent = agent_factory(env, agent_config)
pprint.pprint(env.config)

# Run episode
while True:
    done = False
    obs = env.reset()
    while not done:
        # Test environment
        # action = env.action_type.actions_indexes["IDLE"] 
        action = agent.act(obs)
        obs, reward, done, info = env.step(action)
        
        #print(obs)
        env.render()




