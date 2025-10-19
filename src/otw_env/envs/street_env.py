import numpy as np
from gymnasium.envs.registration import register
from otw_env.utils import utils
from otw_env.envs.common import abstract, action
from otw_env.core.road.road import RoadNetwork, Road
from otw_env.core.vehicle.behavior_controller import ControlledVehicle

# The vehicle is driving on a straight street with several lanes
# and is rewarded for reaching a high speed, staying on the rightmost lanes and avoiding collisions.
class StreetEnv(abstract.AbstractEnv):
    def default_config(cls) -> dict:
        config = super().default_config()
        config.update({
            "observation": {"type": "Kinematics"},
            "action": {"type": "DiscreteMetaAction"},
            "lanes_count": 3,
            "vehicles_count": 50,
            "controlled_vehicles": 1,
            "initial_lane_id": None,
            "duration": 50,  # [s]
            "ego_spacing": 2,
            "vehicles_density": 1,
            "collision_reward": -1,    # The reward received when colliding with a vehicle.
            "right_lane_reward": 0.0,  # The reward received when driving on the right-most lanes, linearly mapped to zero for other lanes.
            "high_speed_reward": 0.4,  # The reward received when driving at full speed, linearly mapped to zero for lower speeds according to config["reward_speed_range"].
            "lane_change_reward": -0.1,   # The reward received at each lane change action.
            "lane_change_cooldown": 30.0,  # seconds
            "lane_change_penalty": -1.0,
            "reward_speed_range": [10, 30], # [m/s] The reward for high speed is mapped linearly from this range to [0, HIGH_SPEED_REWARD].
            "offroad_terminal": False,
            "other_vehicles_type": "otw_env.core.vehicle.behavior_controller.IDMVehicle",
            "screen_width": 600,  # [px]
            "screen_height": 800,  # [px]
            "centering_position": [-1.0, 0.2],
            "scaling": 8.5,
            "show_trajectories": False, #
            "render_agent": True, #
            "normalize_reward": True,
            "offroad_terminal": False,
            "manual_control": False,
            "real_time_rendering": False
        })
        return config


    # Create a road composed of straight adjacent lanes.
    def _create_road(self) -> None:
        self.road = Road(network=RoadNetwork.straight_road_network(self.config["lanes_count"], speed_limit=10),
                         np_random=self.np_random, record_history=self.config["show_trajectories"])

    # Create some new random vehicles of a given type, and add them on the road.
    def _create_vehicles(self) -> None:
        other_vehicles_type = utils.class_from_path(self.config["other_vehicles_type"])
        other_per_controlled = utils.near_split(self.config["vehicles_count"], num_bins=self.config["controlled_vehicles"])
        self.controlled_vehicles = []

        for others in other_per_controlled:
            controlled_vehicle = self.action_type.vehicle_class.create_random(
                self.road, speed = 25,
                lane_id=self.config["initial_lane_id"],
                spacing=self.config["ego_spacing"]
            )
            self.controlled_vehicles.append(controlled_vehicle)
            self.road.vehicles.append(controlled_vehicle)

            for _ in range(others):
                vehicle = other_vehicles_type.create_random(self.road, spacing=3 / self.config["vehicles_density"])
                vehicle.randomize_behavior()
                self.road.vehicles.append(vehicle)
            # print(vehicle)

    # The reward is defined to foster driving at high speed, on the rightmost lanes, and to avoid collisions.    
    def _reward(self, action: action.Action) -> float: # the last action performed
        neighbors = self.road.network.all_side_lanes(self.vehicle.lane_index)
        lane = self.vehicle.target_lane_index[2] if isinstance(self.vehicle, ControlledVehicle) else self.vehicle.lane_index[2]

        forward_speed = self.vehicle.speed * np.cos(self.vehicle.heading)
        scaled_speed = utils.lmap(forward_speed, self.config["reward_speed_range"], [0, 1])
        lane_change = self.config["lane_change_reward"] * lane / max(len(neighbors) - 1, 1)
        reward = \
            + self.config["collision_reward"] * self.vehicle.crashed \
            + self.config["right_lane_reward"] * lane / max(len(neighbors) - 1, 1) \
            + self.config["high_speed_reward"] * np.clip(scaled_speed, 0, 1) \
            + lane_change
        reward = utils.lmap(reward, [self.config["collision_reward"], self.config["high_speed_reward"] + self.config["right_lane_reward"]], [0, 1])
        
        reward = 0 if not self.vehicle.on_road else reward 
        return reward # the corresponding reward
    # def _reward(self, action: action.Action) -> float:
    #     # === Update simulation time ===
    #     self.sim_time += 1.0 / self.config.get("policy_frequency", 5.0)

    #     # === Core reward terms ===
    #     neighbors = self.road.network.all_side_lanes(self.vehicle.lane_index)
    #     lane = (
    #         self.vehicle.target_lane_index[2]
    #         if isinstance(self.vehicle, ControlledVehicle)
    #         else self.vehicle.lane_index[2]
    #     )

    #     forward_speed = self.vehicle.speed * np.cos(self.vehicle.heading)
    #     scaled_speed = utils.lmap(forward_speed, self.config["reward_speed_range"], [0, 1])
    #     lane_change_term = self.config["lane_change_reward"] * lane / max(len(neighbors) - 1, 1)

    #     reward = (
    #         + self.config["collision_reward"] * self.vehicle.crashed
    #         + self.config["right_lane_reward"] * lane / max(len(neighbors) - 1, 1)
    #         + self.config["high_speed_reward"] * np.clip(scaled_speed, 0, 1)
    #         + lane_change_term
    #     )

    #     # Normalize reward range [collision_reward → high_speed_reward + right_lane_reward]
    #     reward = utils.lmap(
    #         reward,
    #         [self.config["collision_reward"], self.config["high_speed_reward"] + self.config["right_lane_reward"]],
    #         [0, 1],
    #     )

    #     # === Lane change cooldown penalty ===
    #     # 1. Detect if ego changed lane
    #     current_lane = self.vehicle.lane_index[2]
    #     if current_lane != getattr(self, "_prev_lane", current_lane):
    #         self.last_lane_change_time = getattr(self, "sim_time", 0.0)
    #     self._prev_lane = current_lane

    #     # 2. Check if cooldown period active
    #     time_since_change = getattr(self, "sim_time", 0.0) - getattr(self, "last_lane_change_time", -np.inf)
    #     on_cooldown = time_since_change < self.config.get("lane_change_cooldown", 2.0)

    #     # 3. Penalize if ego tries to change lane again within cooldown
    #     if on_cooldown and current_lane != getattr(self, "_last_valid_lane", current_lane):
    #         reward += self.config.get("lane_change_penalty", -1.0)
    #     else:
    #         self._last_valid_lane = current_lane

    #     # === Final adjustments ===
    #     if not self.vehicle.on_road:
    #         reward = 0.0

    #     return float(np.clip(reward, -1.0, 1.0))


    # The episode is over if the ego vehicle crashed or the time is out
    def _is_terminal(self) -> bool:
        return self.vehicle.crashed or self.steps >= self.config["duration"] or (self.config["offroad_terminal"] and not self.vehicle.on_road)

    # The cost signal is the occurrence of collision
    def _cost(self, action: int) -> float:
        return float(self.vehicle.crashed)

    def _reset(self) -> None:
        self._create_road()
        self._create_vehicles()
        # === Simulation time & lane change cooldown tracking ===
        self.sim_time = 0.0
        self.last_lane_change_time = -np.inf
        self._prev_lane = None
        self._last_valid_lane = None
        self.config.setdefault("lane_change_cooldown", 30.0)  # seconds
        self.config.setdefault("lane_change_penalty", -1.0)
        # =======================================================

register(
    id='street-v1',
    entry_point='otw_env.envs.street_env:StreetEnv',
)