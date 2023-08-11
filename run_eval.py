from __future__ import division, print_function
import os
import sys

print("Current Working Directory:", os.getcwd())

# from rl_agents.trainer import logger
# sys.path.append('/home/rml-phat/.local/lib/python3.8/site-packages/rl_agents') 

# file_path ='/home/rml-phat/Documents/RML-Work/Code/OTW-Project-v2.2/rl-agents'
# sys.path.append(file_path)
# print("File Path:", file_path)

# from rl_agents.trainer.evaluation1 import Evaluation
# from rl_agents.agents.common.factory import load_agent, load_environment
# from rl_agents.trainer.graphics import RewardViewer
# script_path = os.path.abspath(__file__)
# print("Script Path:", script_path)

import copy
import importlib
import json
import logging
import gym
import numpy as np
import datetime
import logging
import time
import collections
# import seaborn as sns
# sns.set()

from pathlib import Path
# from tensorboardX import SummaryWriter
from gym.wrappers import RecordVideo, RecordEpisodeStatistics, capped_cubic_video_schedule
import sys
# sys.path.append('/home/rml-phat/Documents/OTW-RML')
# from rl_agents.agents.common.graphics import AgentGraphics
# from rl_agents.trainer.graphics import RewardViewer
# from rl_agents.configuration import Configurable
import RobustPlanner
from RobustPlanner.trainer.evaluation import Evaluation
from RobustPlanner.common.factory import load_agent, load_environment
logger = logging.getLogger(__name__)


# from RobustPlanner.agents.robust.graphics.robust_graphics import IntervalRobustPlannerGraphics
# from RobustPlanner.agents.robust.robust import IntervalRobustPlannerAgent

# ################################################################
# class AgentGraphics(object):
#     """
#         Graphical visualization of any Agent implementing AbstractAgent.
#     """
#     @classmethod
#     def display(cls, agent, agent_surface, sim_surface=None):
#         """
#             Display an agent visualization on a pygame surface.

#         :param agent: the agent to be displayed
#         :param agent_surface: the pygame surface on which the agent is displayed
#         :param sim_surface: the pygame surface on which the environment is displayed
#         """

#         if isinstance(agent, IntervalRobustPlannerAgent):
#             IntervalRobustPlannerGraphics.display(agent, agent_surface, sim_surface)


# ################################################################

# global rewards
# rewards = []

# class Evaluation(object):
#     """
#         The evaluation of an agent interacting with an environment to maximize its expected reward.
#     """

#     OUTPUT_FOLDER = 'out'
#     SAVED_MODELS_FOLDER = 'saved_models'
#     RUN_FOLDER = 'run_{}_{}'
#     METADATA_FILE = 'metadata.{}.json'
#     LOGGING_FILE = 'logging.{}.log'

#     def __init__(self,
#                  env,
#                  agent,
#                  num_episodes=1000,
#                  sim_seed=None,
#                  recover=None,
#                  display_env=True,
#                  display_agent=True,
#                  display_rewards=True,
#                  close_env=True):

#         self.env = env
#         self.agent = agent
#         self.num_episodes = num_episodes
#         # self.training = training
#         self.sim_seed = sim_seed
#         self.close_env = close_env
#         self.display_env = display_env

#         self.directory = self.default_directory # or Path(directory)
#         self.run_directory = self.directory / self.default_run_directory # or run_directory
#         self.wrapped_env = RecordVideo(env,
#                                        self.run_directory,
#                                        episode_trigger=(None if self.display_env else lambda e: False))
#         try:
#             self.wrapped_env.unwrapped.set_record_video_wrapper(self.wrapped_env)
#         except AttributeError:
#             pass
#         self.wrapped_env = RecordEpisodeStatistics(self.wrapped_env)
#         self.episode = 0
#         self.writer = SummaryWriter(str(self.run_directory))
#         self.agent.set_writer(self.writer)
#         self.agent.evaluation = self


#         self.recover = recover
#         if self.recover:
#             self.load_agent_model(self.recover)

#         if display_agent:
#             try:
#                 # Render the agent within the environment viewer, if supported
#                 self.env.render()
#                 self.env.unwrapped.viewer.directory = self.run_directory
#                 self.env.unwrapped.viewer.set_agent_display(
#                     lambda agent_surface, sim_surface: AgentGraphics.display(self.agent, agent_surface, sim_surface))
#                 self.env.unwrapped.viewer.directory = self.run_directory
#             except AttributeError:
#                 logger.info("The environment viewer doesn't support agent rendering.")
#         self.reward_viewer = None
#         # if display_rewards:
#         #     self.reward_viewer = RewardViewer()
            
#         self.observation = None

#     def run_episodes(self):
#         for self.episode in range(self.num_episodes):
#             # Run episode
#             terminal = False
#             self.seed(self.episode)
#             self.reset()
#             rewards = []
#             # start_time = time.time()
#             while not terminal:
#                 # Step until a terminal step is reached
#                 reward, terminal = self.step()
#                 rewards.append(reward)
#                 # Catch interruptions
#                 try:
#                     if self.env.unwrapped.done:
#                         break
#                 except AttributeError:
#                     pass

#             # End of episode
#             # duration = time.time() - start_time
#             # self.after_all_episodes(self.episode, rewards, duration)
#             # self.after_some_episodes(self.episode, rewards)
#             # RewardViewer.update(reward)


#     def step(self):
#         """
#             Plan a sequence of actions according to the agent policy, and step the environment accordingly.
#         """
#         # Query agent for actions sequence
#         actions = self.agent.plan(self.observation)
#         if not actions:
#             raise Exception("The agent did not plan any action")

#         # Forward the actions to the environment viewer
#         try:
#             self.env.unwrapped.viewer.set_agent_action_sequence(actions)
#         except AttributeError:
#             pass

#         # Step the environment
#         previous_observation, action = self.observation, actions[0]
#         # print("=============10")
#         self.observation, reward, terminal, info = self.wrapped_env.step(action)


#         rewards.append(reward)

        
        
#         # print("===============================")
#         # plt.figure(num='Rewards')
#         # plt.clf()
#         # plt.title('Total reward')
#         # plt.xlabel('Episode')
#         # plt.ylabel('Reward')

#         # rewards1 = pd.Series(rewards)
#         # means = rewards1.rolling(window=100).mean()
#         # plt.plot(rewards1)
#         # plt.plot(means)
#         # plt.pause(0.001)
#         # plt.plot(block=False)

#         return reward, terminal



#     def load_agent_model(self, model_path):
#         if model_path is True:
#             model_path = self.directory / self.SAVED_MODELS_FOLDER / "latest.tar"
#         if isinstance(model_path, str):
#             model_path = Path(model_path)
#             if not model_path.exists():
#                 model_path = self.directory / self.SAVED_MODELS_FOLDER / model_path
#         try:
#             model_path = self.agent.load(filename=model_path)
#             if model_path:
#                 logger.info("Loaded {} model from {}".format(self.agent.__class__.__name__, model_path))
#         except FileNotFoundError:
#             logger.warning("No pre-trained model found at the desired location.")
#         except NotImplementedError:
#             pass

#     def after_all_episodes(self, episode, rewards, duration):
#         rewards = np.array(rewards)
#         gamma = self.agent.config.get("gamma", 1)
#         self.writer.add_scalar('episode/length', len(rewards), episode)
#         self.writer.add_scalar('episode/total_reward', sum(rewards), episode)
#         self.writer.add_scalar('episode/return', sum(r*gamma**t for t, r in enumerate(rewards)), episode)
#         self.writer.add_scalar('episode/fps', len(rewards) / duration, episode)
#         self.writer.add_histogram('episode/rewards', rewards, episode)
#         logger.info("Episode {} score: {:.1f}".format(episode, sum(rewards)))

#     @property
#     def default_directory(self):
#         return Path(self.OUTPUT_FOLDER) / self.env.unwrapped.__class__.__name__ / self.agent.__class__.__name__

#     @property
#     def default_run_directory(self):
#         return self.RUN_FOLDER.format(datetime.datetime.now().strftime('%Y%m%d-%H%M%S'), os.getpid())


#     def seed(self, episode=0):
#         seed = self.sim_seed + episode if self.sim_seed is not None else None
#         seed = self.wrapped_env.seed(seed)
#         self.agent.seed(seed[0])  # Seed the agent with the main environment seed
#         return seed

#     def reset(self):
#         self.observation = self.wrapped_env.reset()
#         self.agent.reset()

#     def close(self):
#         self.wrapped_env.close()
#         self.writer.close()
#         if self.close_env:
#             self.env.close()

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

def main():    
    repeat_count = 5  # Set the number of repetitions
    test_mode = True  # Set to True if you want to perform testing

    environment_config = "/home/rml-phat/Documents/RML-Work/OTW-RML/RobustPlanner/config/env_linear.json"
    agent_config = "/home/rml-phat/Documents/RML-Work/OTW-RML/RobustPlanner/config/baseline.json"
    print("Start to test >>>>>")
    evaluate(environment_config, agent_config, repeat_count, test_mode)

def evaluate(environment_config, agent_config, repeat_count, test):
    env = load_environment(environment_config)
    agent = load_agent(agent_config, env)
    evaluation = Evaluation(env, agent)

    for _ in range(repeat_count):
        if test:
            evaluation.run_episodes()

if __name__ == "__main__":
    main()