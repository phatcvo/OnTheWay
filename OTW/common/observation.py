from typing import List, Dict, TYPE_CHECKING, Optional
from functools import partial

from gym import spaces
import numpy as np
import pandas as pd

from OTW.common import utils
from OTW.road.road import AbstractLane
from OTW.vehicle import controller

if TYPE_CHECKING:
    from OTW.common import abstract


class ObservationType(object):
    def __init__(self, env: 'abstract.AbstractEnv', **kwargs) -> None:
        self.env = env
        self.__observer_vehicle = None

# Get the observation space.
    def space(self) -> spaces.Space:
        raise NotImplementedError()

# Get an observation of the environment state.
    def observe(self):
        raise NotImplementedError()

# The vehicle observing the scene. If not set, the first controlled vehicle is used by default.
    @property
    def observer_vehicle(self):
        return self.__observer_vehicle or self.env.vehicle

    @observer_vehicle.setter
    def observer_vehicle(self, vehicle):
        self.__observer_vehicle = vehicle


class KinematicObservation(ObservationType):
    FEATURES: List[str] = ['presence', 'x', 'y', 'vx', 'vy']

    def __init__(self, env: 'abstract.AbstractEnv',
                 features: List[str] = None,
                 vehicles_count: int = 15,
                 features_range: Dict[str, List[float]] = None,
                 absolute: bool = False,
                 order: str = "sorted",
                 normalize: bool = True,
                 clip: bool = True,
                 see_behind: bool = False,
                 observe_intentions: bool = False,
                 **kwargs: dict) -> None:

        super().__init__(env)                       # env: The environment to observe
        self.features = features or self.FEATURES   # features: Names of features used in the observation
        self.vehicles_count = vehicles_count        # vehicles_count: Number of observed vehicles
        self.features_range = features_range
        self.absolute = absolute                    # absolute: Use absolute coordinates
        self.order = order                          # order: Order of observed vehicles. Values: sorted, shuffled
        self.normalize = normalize                  # normalize: Should the observation be normalized
        self.clip = clip                            # clip: Should the value be clipped in the desired range
        self.see_behind = see_behind                # see_behind: Should the observation contains the vehicles behind
        self.observe_intentions = observe_intentions # observe_intentions: Observe the destinations of other vehicles

    def space(self) -> spaces.Space:
        return spaces.Box(shape=(self.vehicles_count, len(self.features)), low=-np.inf, high=np.inf, dtype=np.float32)

    # Normalize the observation values. For now, assume that the road is straight along the x axis.
    def normalize_obs(self, df: pd.DataFrame) -> pd.DataFrame: # Dataframe df: observation data
        if not self.features_range:
            side_lanes = self.env.road.network.all_side_lanes(self.observer_vehicle.lane_index)
            self.features_range = {
                "x": [-5.0 * controller.MDPVehicle.SPEED_MAX, 5.0 * controller.MDPVehicle.SPEED_MAX],
                "y": [-AbstractLane.DEFAULT_WIDTH * len(side_lanes), AbstractLane.DEFAULT_WIDTH * len(side_lanes)],
                "vx": [-2*controller.MDPVehicle.SPEED_MAX, 2*controller.MDPVehicle.SPEED_MAX],
                "vy": [-2*controller.MDPVehicle.SPEED_MAX, 2*controller.MDPVehicle.SPEED_MAX]
            }
        for feature, f_range in self.features_range.items():
            if feature in df:
                df[feature] = utils.lmap(df[feature], [f_range[0], f_range[1]], [-1, 1])
                if self.clip:
                    df[feature] = np.clip(df[feature], -1, 1)
        # print('observation/df', df)
        return df

    def observe(self) -> np.ndarray:
        if not self.env.road:
            return np.zeros(self.space().shape)

        # Add ego-vehicle
        df = pd.DataFrame.from_records([self.observer_vehicle.to_dict()])[self.features]
        # print('ego-vehicle', df)
        # Add nearby traffic
        close_vehicles = self.env.road.close_vehicles_to(self.observer_vehicle,
                                                         self.env.PERCEPTION_DISTANCE,
                                                         count=self.vehicles_count - 1,
                                                         see_behind=self.see_behind,
                                                         sort=self.order == "sorted")
        if close_vehicles:
            origin = self.observer_vehicle if not self.absolute else None
            new_data = pd.DataFrame.from_records([v.to_dict(origin, observe_intentions=self.observe_intentions)
                 for v in close_vehicles[-self.vehicles_count + 1:]])
            df = pd.concat([df, new_data], ignore_index=True)
            # df = df.append(pd.DataFrame.from_records([v.to_dict(origin, observe_intentions=self.observe_intentions) for v in close_vehicles[-self.vehicles_count + 1:]])[self.features], ignore_index=True)
        # Normalize and clip
        if self.normalize:
            df = self.normalize_obs(df)
        # Fill missing rows
        if df.shape[0] < self.vehicles_count:
            rows = np.zeros((self.vehicles_count - df.shape[0], len(self.features)))
            # Assuming df is your original DataFrame
            new_data = pd.DataFrame.from_records(data=rows, columns=self.features)
            # Use pandas.concat to combine the DataFrames
            df = pd.concat([df, new_data], ignore_index=True)
            # df = df.append(pd.DataFrame(data=rows, columns=self.features), ignore_index=True)
        # Reorder
        df = df[self.features]
        obs = df.values.copy()
        if self.order == "shuffled":
            self.env.np_random.shuffle(obs[1:])
        # Flatten
        return obs.astype(self.space().dtype)


def observation_factory(env: 'abstract.AbstractEnv', config: dict) -> ObservationType:
    if config["type"] == "Kinematics":
        return KinematicObservation(env, **config)
    else:
        raise ValueError("Unknown observation type")
