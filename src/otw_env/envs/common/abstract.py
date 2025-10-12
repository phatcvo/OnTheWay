from __future__ import annotations

import copy
import os
from typing import TypeVar

import gymnasium as gym
import numpy as np
from gymnasium import Wrapper
from gymnasium.utils import RecordConstructorArgs
from gymnasium.wrappers import RecordVideo

from otw_env.utils.utils import class_from_path
from otw_env.envs.common.action import action_factory, Action, DiscreteMetaAction, ActionType
from otw_env.envs.common.observation import observation_factory, ObservationType
from otw_env.envs.common.graphics import EnvViewer
from otw_env.core.vehicle.behavior_controller import IDMVehicle
from otw_env.core.vehicle.kinematics import Vehicle

Observation = np.ndarray

# A generic environment for various tasks involving a vehicle driving on a road.
class AbstractEnv(gym.Env):
    observation_type: ObservationType
    action_type: ActionType
    _record_video_wrapper: RecordVideo | None
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 30,
    }

    # The maximum distance of any vehicle present in the observation [m]
    PERCEPTION_DISTANCE = 6.0 * Vehicle.MAX_SPEED

    def __init__(self, config: dict = None, render_mode: str | None = None) -> None:
        super().__init__()

        # Configuration
        self.config = self.default_config()
        self.configure(config)

        # Scene
        self.road = None
        self.controlled_vehicles = []

        # Spaces
        self.action_type = None
        self.action_space = None
        self.observation_type = None
        self.observation_space = None
        self.define_spaces()

        # Running
        self.time = 0  # Simulation time
        self.steps = 0  # Actions performed
        self.done = False

        # Rendering
        self.viewer = None
        self._record_video_wrapper = None
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        self.enable_auto_render = False

        self.reset()

    @property
    def vehicle(self) -> Vehicle: # First (default) controlled vehicle.
        return self.controlled_vehicles[0] if self.controlled_vehicles else None

    # Set a unique controlled vehicle.
    @vehicle.setter
    def vehicle(self, vehicle: Vehicle) -> None:
        self.controlled_vehicles = [vehicle]

    # Can be overloaded in environment implementations, or by calling configure().
    def default_config(cls) -> dict: # Default environment configuration.
        return {
            "simulation_frequency": 15,  # [Hz]
            "policy_frequency": 1,  # [Hz]
            "offscreen_rendering": os.environ.get("OFFSCREEN_RENDERING", "0") == "1",
            "real_time_rendering": False
        }

    def configure(self, config: dict) -> None:
        if config:
            self.config.update(config)

    def update_metadata(self, video_real_time_ratio = 1):
        frames_freq = self.config["simulation_frequency"] \
            if self._record_video_wrapper else self.config["policy_frequency"]
        self.metadata['video.frames_per_second'] = video_real_time_ratio * frames_freq
        # print("frames_freq", frames_freq)

    # Set the types and spaces of observation and action from config.
    def define_spaces(self) -> None:
        self.observation_type = observation_factory(self, self.config["observation"])
        self.action_type = action_factory(self, self.config["action"])
        self.observation_space = self.observation_type.space()
        self.action_space = self.action_type.space()

    # Return the reward associated with performing a given action and ending up in the current state.
    def _reward(self, action: Action) -> float: # the last action performed
        raise NotImplementedError # the reward

    # Check whether the current state is a terminal state
    def _is_terminal(self) -> bool:
        raise NotImplementedError # is the state terminal

    # Return a dictionary of additional information
    def _info(self, obs: Observation, action: Action) -> dict: # current observation
        info = {
            "speed": self.vehicle.speed,
            "crashed": self.vehicle.crashed,
            "action": action, # current action
        }
        try:
            info["cost"] = self._cost(action)
            # print('cost:', self._cost(action))
        except NotImplementedError:
            pass
        return info # info dict

    
    def _is_terminated(self):
        """
        Check if the episode should terminate.
        """
        if self.vehicle and self.vehicle.crashed:
            return True
        
        # Ví dụ: Kết thúc nếu vượt quá thời gian
        if self.time >= self.config["duration"]:
            return True
        
        return False

    def _is_truncated(self) -> bool:
        """
        Check we truncate the episode at the current step

        :return: is the episode truncated
        """
        
        return self.time >= self.config["duration"]

    # A constraint metric, for budgeted MDP.
    # If a constraint is defined, it must be used with an alternate reward that doesn't contain it as a penalty.
    def _cost(self, action: Action) -> float: # action: the last action performed
        raise NotImplementedError # the constraint signal, the alternate (constraint-free) reward

    # Reset the environment to it's initial configuration
    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict | None = None,
    ) -> tuple[Observation, dict]:
        """
        Reset the environment to it's initial configuration

        :param seed: The seed that is used to initialize the environment's PRNG
        :param options: Allows the environment configuration to specified through `options["config"]`
        :return: the observation of the reset state
        """
        super().reset(seed=seed, options=options)
        if options and "config" in options:
            self.configure(options["config"])
        self.update_metadata()
        self.define_spaces()  # First, to set the controlled vehicle class depending on action space
        self.time = self.steps = 0
        self.done = False
        self._reset()
        self.define_spaces()  # Second, to link the obs and actions to the vehicles once the scene is created
        obs = self.observation_type.observe()
        info = self._info(obs, action=self.action_space.sample())
        if self.render_mode == "human":
            self.render()
        return obs, info

    # Reset the scene: roads and vehicles. This method must be overloaded by the environments.
    def _reset(self) -> None:
        raise NotImplementedError()

# =================================================================================================
    # Perform an action and step the environment dynamics.
    # The action is executed by the ego-vehicle, and all other vehicles on the road performs their default behavior
    # for several simulation timesteps until the next decision making step.
    def step(self, action: Action) -> tuple[Observation, float, bool, bool, dict]:
        """
        Perform an action and step the environment dynamics.

        The action is executed by the ego-vehicle, and all other vehicles on the road performs their default behaviour
        for several simulation timesteps until the next decision making step.

        :param action: the action performed by the ego-vehicle
        :return: a tuple (observation, reward, terminated, truncated, info)
        """
        if self.road is None or self.vehicle is None:
            raise NotImplementedError(
                "The road and vehicle must be initialized in the environment implementation"
            )

        self.time += 1 / self.config["policy_frequency"]
        self._simulate(action)

        obs = self.observation_type.observe()
        reward = self._reward(action)
        terminated = self._is_terminated()
        truncated = self._is_truncated()
        info = self._info(obs, action)
        if self.render_mode == "human":
            self.render()

        return obs, reward, terminated, truncated, info
    
    
    # def display(self):
    #     plt.figure(num='Rewards')
    #     plt.clf()
    #     plt.title('Total reward')
    #     plt.xlabel('Episode')
    #     plt.ylabel('Reward')

    #     rewards = pd.Series(self.rewards)
    #     means = rewards.rolling(window=100).mean()
    #     plt.plot(rewards)
    #     plt.plot(means)
    #     plt.pause(0.001)
    #     plt.plot(block=False)
#  =================================================================================================
    def _simulate(self, action: Action | None = None) -> None:
        """Perform several steps of simulation with constant action."""
        frames = int(
            self.config["simulation_frequency"] // self.config["policy_frequency"]
        )
        for frame in range(frames):
            # Forward action to the vehicle
            if (
                action is not None
                and not self.config["manual_control"]
                and self.steps
                % int(
                    self.config["simulation_frequency"]
                    // self.config["policy_frequency"]
                )
                == 0
            ):
                self.action_type.act(action)

            self.road.act()
            self.road.step(1 / self.config["simulation_frequency"])
            self.steps += 1

            # Automatically render intermediate simulation steps if a viewer has been launched
            # Ignored if the rendering is done offscreen
            if (
                frame < frames - 1
            ):  # Last frame will be rendered through env.render() as usual
                self._automatic_rendering()

        self.enable_auto_render = False


    # Render the environment. Create a viewer if none exists, and use it to render an image.
    def render(self) -> np.ndarray | None:
        if self.render_mode is None:
            assert self.spec is not None
            gym.logger.warn(
                "You are calling render method without specifying any render mode. "
                "You can specify the render_mode at initialization, "
                f'e.g. gym.make("{self.spec.id}", render_mode="rgb_array")'
            )
            return
        if self.viewer is None:
            self.viewer = EnvViewer(self)

        self.enable_auto_render = True

        self.viewer.display()

        if not self.viewer.offscreen:
            self.viewer.handle_events()
        if self.render_mode == "rgb_array":
            image = self.viewer.get_image()
            return image

    # Close the environment. Will close the environment viewer if it exists.
    def close(self) -> None:
        self.done = True
        if self.viewer is not None:
            self.viewer.close()
        self.viewer = None

    # Get the list of currently available actions.
    # Lane changes are not available on the boundary of the road, and speed changes are not available at
    # maximal or minimal speed.
    def get_available_actions(self) -> list[int]:
        return self.action_type.get_available_actions()

    def set_record_video_wrapper(self, wrapper: RecordVideo):
        self._record_video_wrapper = wrapper
        self.update_metadata()
        self._record_video_wrapper.frames_per_sec = self.metadata["render_fps"]


    # Automatically render the intermediate frames while an action is still ongoing.
    # This allows to render the whole video and not only single steps corresponding to agent decision-making.
    # If a monitor has been set, use its video recorder to capture intermediate frames.
    def _automatic_rendering(self) -> None:
        if self.viewer is not None and self.enable_auto_render:
            if self._record_video_wrapper:
                self._record_video_wrapper._capture_frame()
            else:
                self.render()

    # Return a simplified copy of the environment where distant vehicles have been removed from the road.
    # This is meant to lower the policy computational load while preserving the optimal actions set.
    def simplify(self) -> 'AbstractEnv':
        state_copy = copy.deepcopy(self)
        state_copy.road.vehicles = [state_copy.vehicle] + state_copy.road.close_vehicles_to(
            state_copy.vehicle, self.PERCEPTION_DISTANCE)
        return state_copy # a simplified environment state

    # Change the type of all vehicles on the road
    def change_vehicles(self, vehicle_class_path: str) -> 'AbstractEnv':
        # The path of the class of behavior for other vehicles.  Example: "OTW.vehicle.controller.IDMVehicle"
        vehicle_class = class_from_path(vehicle_class_path)
        # vehicle_class = IntervalVehicle

        env_copy = copy.deepcopy(self)
        vehicles = env_copy.road.vehicles
        for i, v in enumerate(vehicles):
            if v is not env_copy.vehicle:
                vehicles[i] = vehicle_class.create_from(v)
        return env_copy # a new environment with modified behavior model for other vehicles

    def set_preferred_lane(self, preferred_lane: int = None) -> 'AbstractEnv':
        env_copy = copy.deepcopy(self)
        if preferred_lane:
            for v in env_copy.road.vehicles:
                if isinstance(v, IDMVehicle):
                    v.route = [(lane[0], lane[1], preferred_lane) for lane in v.route]
                    # Vehicle with lane preference are also less cautious
                    v.LANE_CHANGE_MAX_BRAKING_IMPOSED = 1000
        return env_copy

    def set_vehicle_field(self, args: Tuple[str, object]) -> 'AbstractEnv':
        field, value = args
        env_copy = copy.deepcopy(self)
        for v in env_copy.road.vehicles:
            if v is not self.vehicle:
                setattr(v, field, value)
        return env_copy

    def call_vehicle_method(self, args: Tuple[str, Tuple[object]]) -> 'AbstractEnv':
        method, method_args = args
        env_copy = copy.deepcopy(self)
        for i, v in enumerate(env_copy.road.vehicles):
            if hasattr(v, method):
                env_copy.road.vehicles[i] = getattr(v, method)(*method_args)
        return env_copy

    def randomize_behavior(self) -> 'AbstractEnv':
        env_copy = copy.deepcopy(self)
        for v in env_copy.road.vehicles:
            if isinstance(v, IDMVehicle):
                v.randomize_behavior()
        return env_copy

    def __deepcopy__(self, memo):
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for k, v in self.__dict__.items():
            if k not in ['viewer', '_record_video_wrapper']:
                setattr(result, k, copy.deepcopy(v, memo))
            else:
                setattr(result, k, None)
        return result
