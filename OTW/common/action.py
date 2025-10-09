from itertools import product
from typing import TYPE_CHECKING, Optional, Union, Tuple, Callable
from gymnasium import spaces
import numpy as np

from OTW.common import utils
from OTW.vehicle import kinematics
from OTW.vehicle import controller

if TYPE_CHECKING:
    from OTW.common.abstract import AbstractEnv

Action = Union[int, np.ndarray]


class ActionType(object):

    # A type of action specifies its definition space, and how actions are executed in the environment
    def __init__(self, env: 'AbstractEnv', **kwargs) -> None:
        self.env = env
        self.__controlled_vehicle = None

    def space(self) -> spaces.Space: # The action space.
        raise NotImplementedError

    @property
    def vehicle_class(self) -> Callable: # The class of a vehicle able to execute the action.
        raise NotImplementedError # return a subclass of :py:class:`OTW.vehicle.kinematics.Vehicle`

    # Most of the action mechanics are actually implemented in vehicle.act(action), where
    # vehicle is an instance of the specified :py:class:`OTW.envs.common.action.ActionType.vehicle_class`.
    # Must some pre-processing can be applied to the action based on the ActionType configurations.
    def act(self, action: Action) -> None: # Execute the action on the ego-vehicle.
        raise NotImplementedError

    @property
    def controlled_vehicle(self): # The vehicle acted upon. If not set, the first controlled vehicle is used by default.
        return self.__controlled_vehicle or self.env.vehicle

    @controlled_vehicle.setter
    def controlled_vehicle(self, vehicle):
        self.__controlled_vehicle = vehicle


class DiscreteMetaAction(ActionType):
    # A mapping of action indexes to labels.
    ACTIONS_ALL = {
        0: 'LANE_LEFT',
        1: 'IDLE',
        2: 'LANE_RIGHT',
        3: 'FASTER',
        4: 'SLOWER'
    }
    # A mapping of longitudinal action indexes to labels.
    ACTIONS_LONGITUDINAL = {
        0: 'SLOWER',
        1: 'IDLE',
        2: 'FASTER'
    }
    # A mapping of lateral action indexes to labels.
    ACTIONS_LATERAL = {
        0: 'LANE_LEFT',
        1: 'IDLE',
        2: 'LANE_RIGHT'
    }

    # Create a discrete action space of meta-actions.
    def __init__(self, env: 'AbstractEnv', longitudinal: bool = True, lateral: bool = True, **kwargs) -> None:
        super().__init__(env)
        self.longitudinal = longitudinal
        self.lateral = lateral
        self.actions = self.ACTIONS_ALL \
            if longitudinal and lateral else self.ACTIONS_LONGITUDINAL \
            if longitudinal else self.ACTIONS_LATERAL \
            if lateral else None
        if self.actions is None:
            raise ValueError("At least longitudinal or lateral actions must be included")
        self.actions_indexes = {v: k for k, v in self.actions.items()}

    def space(self) -> spaces.Space:
        return spaces.Discrete(len(self.actions))

    @property
    def vehicle_class(self) -> Callable:
        return controller.MDPVehicle

    def act(self, action: int) -> None:
        self.controlled_vehicle.act(self.actions[action])

def action_factory(env: 'AbstractEnv', config: dict) -> ActionType:
    if config["type"] == "DiscreteMetaAction":
        return DiscreteMetaAction(env, **config)
    else:
        raise ValueError("Unknown action type")
