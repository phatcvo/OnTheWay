import logging
from collections import defaultdict

import gym
import numpy as np
from gym.utils import seeding

from RobustPlanner.common.abstract import AbstractAgent
from RobustPlanner.common.factory import preprocess_env, safe_deepcopy_env
from RobustPlanner.common.configuration import Configurable
from RobustPlanner.agents.tree_search.graphics import TreePlot

logger = logging.getLogger(__name__)

# A new Tree Search agent. The agent configuration. Use default if None.
class AbstractTreeSearchAgent(AbstractAgent):
    PLANNER_TYPE = None
    NODE_TYPE = None

    def __init__(self, env, config=None):

        super(AbstractTreeSearchAgent, self).__init__(config)
        self.env = env
        self.planner = self.make_planner()
        self.previous_actions = []
        self.remaining_horizon = 0
        self.steps = 0

    @classmethod
    def default_config(cls):
        return {
            "env_preprocessors": [],
            "display_tree": False,
            "receding_horizon": 1,
            "terminal_reward": 0
        }

    def make_planner(self):
        if self.PLANNER_TYPE:
            return self.PLANNER_TYPE(self.env, self.config)
        else:
            raise NotImplementedError()

    # Plan an optimal sequence of actions. Start by updating the previously found tree with the last action performed.
    def plan(self, observation):
        #observation: the current state

        self.steps += 1
        replanning_required = self.step(self.previous_actions)
        if replanning_required:
            env = preprocess_env(self.env, self.config["env_preprocessors"])
            actions = self.planner.plan(state=env, observation=observation)
        else:
            actions = self.previous_actions[1:]
        self.write_tree()

        self.previous_actions = actions
        return actions # the list of actions

    # Handle receding horizon mechanism
    def step(self, actions):
        replanning_required = self.remaining_horizon == 0 or len(actions) <= 1
        if replanning_required:
            self.remaining_horizon = self.config["receding_horizon"] - 1
        else:
            self.remaining_horizon -= 1

        self.planner.step_tree(actions)
        return replanning_required # whether a replanning is required

    def reset(self):
        self.planner.step_by_reset()
        self.remaining_horizon = 0
        self.steps = 0

    def seed(self, seed=None):
        return self.planner.seed(seed)

    def record(self, state, action, reward, next_state, done, info):
        pass

    def act(self, state):
        return self.plan(state)[0]

    def save(self, filename):
        return False

    def load(self, filename):
        return False

    def write_tree(self):
        if self.config["display_tree"] and self.writer:
            TreePlot(self.planner, max_depth=6).plot_to_writer(self.writer, epoch=self.steps, show=True)


class AbstractPlanner(Configurable):
    def __init__(self, config=None):
        super().__init__(config)
        self.np_random = None
        self.root = None
        self.observations = []
        self.reset()
        self.seed()

    @classmethod
    def default_config(cls):
        return dict(budget=500,
                    gamma=0.8,
                    step_strategy="reset")

    # Seed the planner randomness source, e.g. for rollout policy
    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed] # the used seed

    # Plan an optimal sequence of actions.
    def plan(self, state, observation):
        raise NotImplementedError()

    # Get the optimal action sequence of the current tree by recursively selecting the best action within each node with no exploration.
    def get_plan(self):
        actions = []
        node = self.root
        while node.children:
            action = node.selection_rule()
            actions.append(action)
            node = node.children[action]
        return actions # the list of actions

    def step(self, state, action):
        step_data = state.step(action)
        self.observations.append(step_data[0])
        return step_data

    def get_visits(self):
        visits = defaultdict(int)
        for observation in self.observations:
            visits[str(observation)] += 1
        return visits

    def get_updates(self):
        return defaultdict(int)

    # Update the planner tree when the agent performs an action
    def step_tree(self, actions):
        if self.config["step_strategy"] == "reset":
            self.step_by_reset()
        elif self.config["step_strategy"] == "subtree":
            if actions: # a sequence of actions to follow from the root node
                self.step_by_subtree(actions[0])
            else:
                self.step_by_reset()
        else:
            logger.warning("Unknown step strategy: {}".format(self.config["step_strategy"]))
            self.step_by_reset()

    # Reset the planner tree to a root node for the new state.
    def step_by_reset(self):
        self.reset()

    # Replace the planner tree by its subtree corresponding to the chosen action.
    def step_by_subtree(self, action):
        # action: a chosen action from the root node
        if action in self.root.children:
            self.root = self.root.children[action]
            self.root.parent = None
        else:
            # The selected action was never explored, start a new tree.
            self.step_by_reset()

    def reset(self):
        raise NotImplementedError

# A tree node ========================================
class Node(object):

    # New node.
    def __init__(self, parent, planner):

        self.parent = parent # its parent node
        self.planner = planner # the planner using the node

        # Dict of children nodes, indexed by action labels
        self.children = {}

        # Number of times the node was visited.
        self.count = 0

    # Return an estimate of the node value.
    def get_value(self) -> float:
        raise NotImplementedError()

    def expand(self, branching_factor):
        for a in range(branching_factor):
            self.children[a] = type(self)(self, self.planner)

    def selection_rule(self):
        raise NotImplementedError()

    # Breadth-first search of all paths to nodes that meet a given condition
    @staticmethod
    def breadth_first_search(root, operator=None, condition=None, condition_blocking=True):

        # root: starting node
        # operator: will be applied to all traversed nodes
        # condition: nodes meeting that condition will be returned
        # condition_blocking: do not explore a node which met the condition
        # list of paths to nodes that met the condition

        queue = [(root, [])]
        while queue:
            (node, path) = queue.pop(0)
            if (condition is None) or condition(node):
                returned = operator(node, path) if operator else (node, path)
                yield returned
            if (condition is None) or not condition_blocking or not condition(node):
                for next_key, next_node in node.children.items():
                    queue.append((next_node, path + [next_key]))

    def is_leaf(self):
        return not self.children

    def path(self):
        node = self
        path = []
        while node.parent:
            for a in node.parent.children:
                if node.parent.children[a] == node:
                    path.append(a)
                    break
            node = node.parent
        return reversed(path) # sequence of action labels from the root to the node

    def sequence(self):
        node = self
        path = [node]
        while node.parent:
            path.append(node.parent)
            node = node.parent
        return reversed(path) # sequence of nodes from the root to the node

    @staticmethod
    def all_argmax(x): # a set
        m = np.amax(x)
        return np.nonzero(x == m)[0] # the list of indexes of all maximums of x

    # Randomly tie-breaking arg max
    def random_argmax(self, x): # x: an array
        indices = Node.all_argmax(x)
        return self.planner.np_random.choice(indices) # a random index among the maximums

    def __str__(self):
        return "{} (n:{}, v:{:.2f})".format(list(self.path()), self.count, self.get_value())

    def __repr__(self):
        return '<node {}>'.format(id(self))

    # Get a list of visited nodes corresponding to the node subtree
    def get_trajectories(self, full_trajectories=True, include_leaves=True):
        trajectories = [] # return a list of observation sequences, else a list of observations
        if self.children:
            for action, child in self.children.items():
                child_trajectories = child.get_trajectories(full_trajectories, include_leaves)
                if full_trajectories:
                    trajectories.extend([[self] + trajectory for trajectory in child_trajectories])
                else:
                    trajectories.extend(child_trajectories)
            if not full_trajectories:
                trajectories.append(self)
        elif include_leaves: # include leaves or only expanded nodes
            trajectories = [[self]] if full_trajectories else [self]
        return trajectories # the list of trajectories

    def get_obs_visits(self, state=None):
        visits = defaultdict(int)
        updates = defaultdict(int)
        if hasattr(self, "observation"):
            for node in self.get_trajectories(full_trajectories=False,
                                              include_leaves=False):
                if hasattr(node, "observation"):
                    visits[str(node.observation)] += 1
                    if hasattr(node, "updates_count"):
                        updates[str(node.observation)] += node.updates_count
        else:  # Replay required
            for node in self.get_trajectories(full_trajectories=False,
                                              include_leaves=False):
                replay_state = safe_deepcopy_env(state)
                for action in node.path():
                    observation, _, _, _ = replay_state.step(action)
                visits[str(observation)] += 1
        return visits, updates
