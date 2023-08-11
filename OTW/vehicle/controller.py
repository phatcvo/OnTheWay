from typing import List, Tuple, Union

import numpy as np
import copy

from OTW.road.road import Road, LaneIndex, Route
from OTW.common.utils import Vector, not_zero, wrap_to_pi, do_every
from OTW.vehicle.kinematics import Vehicle



# A vehicle piloted by two low-level controller, allowing high-level actions such as cruise control and lane changes.
# - The longitudinal controller is a speed controller;
# - The lateral controller is a heading controller cascaded with a lateral position controller.

class ControlledVehicle(Vehicle):

    # Desired velocity.
    target_speed: float

    # Characteristic time
    TAU_ACC = 0.6  # [s]
    TAU_HEADING = 0.2  # [s]
    TAU_LATERAL = 0.6  # [s]

    TAU_PURSUIT = 0.5 * TAU_HEADING  # [s]
    KP_A = 1 / TAU_ACC
    KP_HEADING = 1 / TAU_HEADING
    KP_LATERAL = 1 / TAU_LATERAL  # [1/s]
    MAX_STEERING_ANGLE = np.pi / 3  # [rad]
    DELTA_SPEED = 5  # [m/s]
    ac = np.array([])

    def __init__(self, road: Road, position: Vector, heading: float = 0, speed: float = 0,
                 target_lane_index: LaneIndex = None, target_speed: float = None, route: Route = None):
        super().__init__(road, position, heading, speed)
        self.target_lane_index = target_lane_index or self.lane_index
        self.target_speed = target_speed or self.speed
        self.route = route
        # ac = np.array([])


    # Create a new vehicle from an existing one.
    # The vehicle dynamics and target dynamics are copied, other properties are default.
    @classmethod
    def create_from(cls, vehicle: "ControlledVehicle") -> "ControlledVehicle":
        v = cls(vehicle.road, vehicle.position, heading=vehicle.heading, speed=vehicle.speed,
                target_lane_index=vehicle.target_lane_index, target_speed=vehicle.target_speed,
                route=vehicle.route)
        return v

    # Plan a route to a destination in the road network
    def plan_route_to(self, destination: str) -> "ControlledVehicle":
        try:
            path = self.road.network.shortest_path(self.lane_index[1], destination)
        except KeyError:
            path = []
        if path:
            self.route = [self.lane_index] + [(path[i], path[i + 1], None) for i in range(len(path) - 1)]
        else:
            self.route = [self.lane_index]
        return self


    # Perform a high-level action to change the desired lane or speed.
    # - If a high-level action is provided, update the target speed and lane;
    # - then, perform longitudinal and lateral control.
    def act(self, action: Union[dict, str] = None) -> None:
        self.follow_road()
        if action == "FASTER":
            self.target_speed += self.DELTA_SPEED
        elif action == "SLOWER":
            self.target_speed -= self.DELTA_SPEED
        elif action == "LANE_RIGHT":
            _from, _to, _id = self.target_lane_index
            target_lane_index = _from, _to, np.clip(_id + 1, 0, len(self.road.network.graph[_from][_to]) - 1)
            if self.road.network.get_lane(target_lane_index).is_reachable_from(self.position):
                self.target_lane_index = target_lane_index
        elif action == "LANE_LEFT":
            _from, _to, _id = self.target_lane_index
            target_lane_index = _from, _to, np.clip(_id - 1, 0, len(self.road.network.graph[_from][_to]) - 1)
            if self.road.network.get_lane(target_lane_index).is_reachable_from(self.position):
                self.target_lane_index = target_lane_index

        action = {"steering": self.steering_control(self.target_lane_index),
                  "acceleration": self.speed_control(self.target_speed)}

        action['steering'] = np.clip(action['steering'], -self.MAX_STEERING_ANGLE, self.MAX_STEERING_ANGLE)
        # print('controller/action:', action)

        super().act(action)

    def follow_road(self) -> None:
        """At the end of a lane, automatically switch to a next one."""
        if self.road.network.get_lane(self.target_lane_index).after_end(self.position):
            self.target_lane_index = self.road.network.next_lane(self.target_lane_index,
                                                                 route=self.route,
                                                                 position=self.position,
                                                                 np_random=self.road.np_random)


    # Steer the vehicle to follow the center of an given lane.
    #
    #     1. Lateral position is controlled by a proportional controller yielding a lateral speed command
    #     2. Lateral speed command is converted to a heading reference
    #     3. Heading is controlled by a proportional controller yielding a heading rate command
    #     4. Heading rate command is converted to a steering angle
    def steering_control(self, target_lane_index: LaneIndex) -> float:
        target_lane = self.road.network.get_lane(target_lane_index)
        lane_coords = target_lane.local_coordinates(self.position)
        lane_next_coords = lane_coords[0] + self.speed * self.TAU_PURSUIT
        lane_future_heading = target_lane.heading_at(lane_next_coords)

        # Lateral position control
        lateral_speed_command = - self.KP_LATERAL * lane_coords[1]
        # Lateral speed to heading
        heading_command = np.arcsin(np.clip(lateral_speed_command / not_zero(self.speed), -1, 1))
        heading_ref = lane_future_heading + np.clip(heading_command, -np.pi/4, np.pi/4)
        # Heading control
        heading_rate_command = self.KP_HEADING * wrap_to_pi(heading_ref - self.heading)
        # Heading rate to steering angle
        steering_angle = np.arcsin(np.clip(self.LENGTH / 2 / not_zero(self.speed) * heading_rate_command,
                                           -1, 1))
        steering_angle = np.clip(steering_angle, -self.MAX_STEERING_ANGLE, self.MAX_STEERING_ANGLE)
        # print('steering_angle:', steering_angle, 'position:', self.position,)
        return float(steering_angle) # a steering wheel angle command [rad]


    # Control the speed of the vehicle. Using a simple proportional controller.
    def speed_control(self, target_speed: float) -> float:
        # print(f'controller/speed {self.speed}, target {target_speed}')
        return self.KP_A * (target_speed - self.speed) # an acceleration command [m/s2]



    # Predict the future positions of the vehicle along its planned route, under constant speed
    def predict_trajectory_constant_speed(self, times: np.ndarray) -> Tuple[List[np.ndarray], List[float]]:
        coordinates = self.lane.local_coordinates(self.position)
        route = self.route or [self.lane_index]
        return tuple(zip(*[self.road.network.position_heading_along_route(route, coordinates[0] + self.speed * t, 0)
                     for t in times])) # positions, headings

# A controlled vehicle with a specified discrete range of allowed target speeds.
class MDPVehicle(ControlledVehicle):
    SPEED_COUNT: int = 3  # []
    SPEED_MIN: float = 0  # [m/s]
    SPEED_MAX: float = 30  # [m/s]

    def __init__(self, road: Road, position: List[float], heading: float = 0, speed: float = 0,
                 target_lane_index: LaneIndex = None, target_speed: float = None, route: Route = None) -> None:
        super().__init__(road, position, heading, speed, target_lane_index, target_speed, route)
        self.speed_index = self.speed_to_index(self.target_speed)
        self.target_speed = self.index_to_speed(self.speed_index)


    # Perform a high-level action.
    #     - If the action is a speed change, choose speed from the allowed discrete range.
    #     - Else, forward action to the ControlledVehicle handler.
    def act(self, action: Union[dict, str] = None) -> None:
        if action == "FASTER":
            self.speed_index = self.speed_to_index(self.speed) + 1
        elif action == "SLOWER":
            self.speed_index = self.speed_to_index(self.speed) - 1
        else:
            super().act(action)
            return
        self.speed_index = int(np.clip(self.speed_index, 0, self.SPEED_COUNT - 1))
        self.target_speed = self.index_to_speed(self.speed_index)
        # print(f'controller/speed_index {self.speed_index}')
        super().act()

    # Convert an index among allowed speeds to its corresponding speed
    def index_to_speed(self, index: int) -> float:
        if self.SPEED_COUNT > 1:
            return self.SPEED_MIN + index * (self.SPEED_MAX - self.SPEED_MIN) / (self.SPEED_COUNT - 1)
        else:
            return self.SPEED_MIN # the corresponding speed [m/s]


    # Find the index of the closest speed allowed to a given speed.
    def speed_to_index(self, speed: float) -> int: # an input speed [m/s]
        x = (speed - self.SPEED_MIN) / (self.SPEED_MAX - self.SPEED_MIN)
        # print(f'controller/speed2index {x}')
        return int(np.clip(np.round(x * (self.SPEED_COUNT - 1)), 0, self.SPEED_COUNT - 1)) # the index of the closest speed allowed []

    # Find the index of the closest speed allowed to a given speed.
    @classmethod
    def speed_to_index_default(cls, speed: float) -> int: # an input speed [m/s]
        x = (speed - cls.SPEED_MIN) / (cls.SPEED_MAX - cls.SPEED_MIN)
        # print(f'controller/speed2index_def {x}')
        return np.int(np.clip(np.round(x * (cls.SPEED_COUNT - 1)), 0, cls.SPEED_COUNT - 1)) # the index of the closest speed allowed []

    @classmethod
    def get_speed_index(cls, vehicle: Vehicle) -> int:
        return getattr(vehicle, "speed_index", cls.speed_to_index_default(vehicle.speed))

    # Predict the future trajectory of the vehicle given a sequence of actions.
    # actions: a sequence of future actions.
    # action_duration: the duration of each action.
    # trajectory_timestep: the duration between each save of the vehicle state.
    # dt: the timestep of the simulation
    def predict_trajectory(self, actions: List, action_duration: float, trajectory_timestep: float, dt: float) \
            -> List[ControlledVehicle]:
        states = []
        v = copy.deepcopy(self)
        t = 0
        for action in actions:
            v.act(action)  # High-level decision
            for _ in range(int(action_duration / dt)):
                t += 1
                v.act()  # Low-level control action
                v.step(dt)
                if (t % int(trajectory_timestep / dt)) == 0:
                    states.append(copy.deepcopy(v))
        # print(f'controller/predict_traj {actions}')
        return states # the sequence of future states

# =================================================================================================
class IDMVehicle(ControlledVehicle):

    # Longitudinal policy parameters. Maximum acceleration.
    ACC_MAX = 6.0  # [m/s2]

    # Desired maximum acceleration.
    COMFORT_ACC_MAX = 3.0  # [m/s2]

    #Desired maximum deceleration.
    COMFORT_ACC_MIN = -5.0  # [m/s2]

    # Desired jam distance to the front vehicle.
    DISTANCE_WANTED = 5.0 + ControlledVehicle.LENGTH  # [m]

    # Desired time gap to the front vehicle.
    TIME_WANTED = 1.5  # [s]

    # Exponent of the velocity term.
    DELTA = 4.0  # []

    # Range of delta when chosen randomly.
    DELTA_RANGE = [3.5, 4.5]

    # Lateral policy parameters
    POLITENESS = 0.  # in [0, 1]
    LANE_CHANGE_MIN_ACC_GAIN = 0.2  # [m/s2]
    LANE_CHANGE_MAX_BRAKING_IMPOSED = 2.0  # [m/s2]
    LANE_CHANGE_DELAY = 1.0  # [s]

    def __init__(self, road: Road, position: Vector, heading: float = 0, speed: float = 0,
                 target_lane_index: int = None, target_speed: float = None, route: Route = None,
                 enable_lane_change: bool = True, timer: float = None):
        super().__init__(road, position, heading, speed, target_lane_index, target_speed, route)
        self.enable_lane_change = enable_lane_change
        self.timer = timer or (np.sum(self.position)*np.pi) % self.LANE_CHANGE_DELAY

    def randomize_behavior(self):
        self.DELTA = self.road.np_random.uniform(low=self.DELTA_RANGE[0], high=self.DELTA_RANGE[1])

    # Create a new vehicle from an existing one.
    # The vehicle dynamics and target dynamics are copied, other properties are default.
    @classmethod
    def create_from(cls, vehicle: ControlledVehicle) -> "IDMVehicle":
        v = cls(vehicle.road, vehicle.position, heading=vehicle.heading, speed=vehicle.speed,
                target_lane_index=vehicle.target_lane_index, target_speed=vehicle.target_speed,
                route=vehicle.route, timer=getattr(vehicle, 'timer', None))
        return v # a new vehicle at the same dynamical state

    # Execute an action.
    # For now, no action is supported because the vehicle takes all decisions
    # of acceleration and lane changes on its own, based on the IDM and MOBIL models.
    def act(self, action: Union[dict, str] = None):

        if self.crashed:
            return
        action = {}
        # Lateral: MOBIL
        self.follow_road()
        if self.enable_lane_change:
            self.change_lane_policy()
        action['steering'] = self.steering_control(self.target_lane_index)
        action['steering'] = np.clip(action['steering'], -self.MAX_STEERING_ANGLE, self.MAX_STEERING_ANGLE)

        # Longitudinal: IDM
        front_vehicle, rear_vehicle = self.road.neighbor_vehicles(self, self.lane_index)
        action['acceleration'] = self.acceleration(ego_vehicle=self, front_vehicle=front_vehicle, rear_vehicle=rear_vehicle)
        # When changing lane, check both current and target lanes
        if self.lane_index != self.target_lane_index:
            front_vehicle, rear_vehicle = self.road.neighbor_vehicles(self, self.target_lane_index)
            target_idm_acceleration = self.acceleration(ego_vehicle=self, front_vehicle=front_vehicle, rear_vehicle=rear_vehicle)
            action['acceleration'] = min(action['acceleration'], target_idm_acceleration)
        # action['acceleration'] = self.recover_from_stop(action['acceleration'])
        action['acceleration'] = np.clip(action['acceleration'], -self.ACC_MAX, self.ACC_MAX)
        Vehicle.act(self, action)  # Skip ControlledVehicle.act(), or the command will be overriden.

    # Step the simulation. Increases a timer used for decision policies, and step the vehicle dynamics.
    def step(self, dt: float):
        self.timer += dt
        super().step(dt)


    # Compute an acceleration command with the Intelligent Driver Model.
    # The acceleration is chosen so as to:
    # - reach a target speed;
    # - maintain a minimum safety distance (and safety time) w.r.t the front vehicle.

    def acceleration(self, ego_vehicle: ControlledVehicle, front_vehicle: Vehicle = None,
                     rear_vehicle: Vehicle = None) -> float:

        if not ego_vehicle or not isinstance(ego_vehicle, Vehicle):
            return 0
        ego_target_speed = abs(not_zero(getattr(ego_vehicle, "target_speed", 0)))
        acceleration = self.COMFORT_ACC_MAX * (
                1 - np.power(max(ego_vehicle.speed, 0) / ego_target_speed, self.DELTA))

        if front_vehicle:
            d = ego_vehicle.lane_distance_to(front_vehicle)
            acceleration -= self.COMFORT_ACC_MAX * \
                np.power(self.desired_gap(ego_vehicle, front_vehicle) / not_zero(d), 2)
        return acceleration

    # Compute the desired distance between a vehicle and its leading vehicle.
    def desired_gap(self, ego_vehicle: Vehicle, front_vehicle: Vehicle = None, projected: bool = True) -> float:
        d0 = self.DISTANCE_WANTED
        tau = self.TIME_WANTED
        ab = -self.COMFORT_ACC_MAX * self.COMFORT_ACC_MIN
        dv = np.dot(ego_vehicle.velocity - front_vehicle.velocity, ego_vehicle.direction) if projected \
            else ego_vehicle.speed - front_vehicle.speed
        d_star = d0 + ego_vehicle.speed * tau + ego_vehicle.speed * dv / (2 * np.sqrt(ab))
        return d_star

    # Decide when to change lane.
    # Based on:frequency; closeness of the target lane; MOBIL model.
    def change_lane_policy(self) -> None:
        # If a lane change already ongoing
        if self.lane_index != self.target_lane_index:
            # If we are on correct route but bad lane: abort it if someone else is already changing into the same lane
            if self.lane_index[:2] == self.target_lane_index[:2]:
                for v in self.road.vehicles:
                    if v is not self \
                            and v.lane_index != self.target_lane_index \
                            and isinstance(v, ControlledVehicle) \
                            and v.target_lane_index == self.target_lane_index:
                        d = self.lane_distance_to(v)
                        d_star = self.desired_gap(self, v)
                        if 0 < d < d_star:
                            self.target_lane_index = self.lane_index
                            break
            return

        # else, at a given frequency,
        if not do_every(self.LANE_CHANGE_DELAY, self.timer):
            return
        self.timer = 0

        # decide to make a lane change
        for lane_index in self.road.network.side_lanes(self.lane_index):
            # Is the candidate lane close enough?
            if not self.road.network.get_lane(lane_index).is_reachable_from(self.position):
                continue
            # Does the MOBIL model recommend a lane change?
            if self.mobil(lane_index):
                self.target_lane_index = lane_index

    
    # The vehicle should change lane only if:
    # - after changing it (and/or following vehicles) can accelerate more;
    # - it doesn't impose an unsafe braking on its new following vehicle.
    def mobil(self, lane_index: LaneIndex) -> bool:
        # Is the maneuver unsafe for the new following vehicle?
        new_preceding, new_following = self.road.neighbor_vehicles(self, lane_index)
        new_following_a = self.acceleration(ego_vehicle=new_following, front_vehicle=new_preceding)
        new_following_pred_a = self.acceleration(ego_vehicle=new_following, front_vehicle=self)
        if new_following_pred_a < -self.LANE_CHANGE_MAX_BRAKING_IMPOSED:
            return False

        # Do I have a planned route for a specific lane which is safe for me to access?
        old_preceding, old_following = self.road.neighbor_vehicles(self)
        self_pred_a = self.acceleration(ego_vehicle=self, front_vehicle=new_preceding)
        if self.route and self.route[0][2]:
            # Wrong direction
            if np.sign(lane_index[2] - self.target_lane_index[2]) != np.sign(self.route[0][2] - self.target_lane_index[2]):
                return False
            # Unsafe braking required
            elif self_pred_a < -self.LANE_CHANGE_MAX_BRAKING_IMPOSED:
                return False

        # Is there an acceleration advantage for me and/or my followers to change lane?
        else:
            self_a = self.acceleration(ego_vehicle=self, front_vehicle=old_preceding)
            old_following_a = self.acceleration(ego_vehicle=old_following, front_vehicle=self)
            old_following_pred_a = self.acceleration(ego_vehicle=old_following, front_vehicle=old_preceding)
            jerk = self_pred_a - self_a + self.POLITENESS * (new_following_pred_a - new_following_a
                                                             + old_following_pred_a - old_following_a)
            if jerk < self.LANE_CHANGE_MIN_ACC_GAIN:
                return False
        # All clear, let's go!
        return True

    # If stopped on the wrong lane, try a reversing maneuver.
    def recover_from_stop(self, acceleration: float) -> float: # desired acceleration from IDM
        stopped_speed = 5
        safe_distance = 200
        # Is the vehicle stopped on the wrong lane?
        if self.target_lane_index != self.lane_index and self.speed < stopped_speed:
            _, rear = self.road.neighbor_vehicles(self)
            _, new_rear = self.road.neighbor_vehicles(self, self.road.network.get_lane(self.target_lane_index))
            # Check for free room behind on both lanes
            if (not rear or rear.lane_distance_to(self) > safe_distance) and \
                    (not new_rear or new_rear.lane_distance_to(self) > safe_distance):
                # Reverse
                return -self.COMFORT_ACC_MAX / 2
        return acceleration # suggested acceleration to recover from being stuck

class LinearVehicle(IDMVehicle):

    """A Vehicle whose longitudinal and lateral controllers are linear with respect to parameters."""

    ACCELERATION_PARAMETERS = [0.3, 0.3, 2.0]
    STEERING_PARAMETERS = [ControlledVehicle.KP_HEADING, ControlledVehicle.KP_HEADING * ControlledVehicle.KP_LATERAL]

    ACCELERATION_RANGE = np.array([0.5*np.array(ACCELERATION_PARAMETERS), 1.5*np.array(ACCELERATION_PARAMETERS)])
    STEERING_RANGE = np.array([np.array(STEERING_PARAMETERS) - np.array([0.07, 1.5]),
                               np.array(STEERING_PARAMETERS) + np.array([0.07, 1.5])])

    TIME_WANTED = 2.5

    def __init__(self,
                 road: Road,
                 position: Vector,
                 heading: float = 0,
                 speed: float = 0,
                 target_lane_index: int = None,
                 target_speed: float = None,
                 route: Route = None,
                 enable_lane_change: bool = True,
                 timer: float = None,
                 data: dict = None):
        super().__init__(road, position, heading, speed, target_lane_index, target_speed, route,
                         enable_lane_change, timer)
        self.data = data if data is not None else {}
        self.collecting_data = True

    def act(self, action: Union[dict, str] = None):
        if self.collecting_data:
            self.collect_data()
        super().act(action)

    def randomize_behavior(self):
        ua = self.road.np_random.uniform(size=np.shape(self.ACCELERATION_PARAMETERS))
        self.ACCELERATION_PARAMETERS = self.ACCELERATION_RANGE[0] + ua*(self.ACCELERATION_RANGE[1] -
                                                                        self.ACCELERATION_RANGE[0])
        ub = self.road.np_random.uniform(size=np.shape(self.STEERING_PARAMETERS))
        self.STEERING_PARAMETERS = self.STEERING_RANGE[0] + ub*(self.STEERING_RANGE[1] - self.STEERING_RANGE[0])

    def acceleration(self,
                     ego_vehicle: ControlledVehicle,
                     front_vehicle: Vehicle = None,
                     rear_vehicle: Vehicle = None) -> float:
        """
        Compute an acceleration command with a Linear Model.

        The acceleration is chosen so as to:
        - reach a target speed;
        - reach the speed of the leading (resp following) vehicle, if it is lower (resp higher) than ego's;
        - maintain a minimum safety distance w.r.t the leading vehicle.

        :param ego_vehicle: the vehicle whose desired acceleration is to be computed. It does not have to be an
                            Linear vehicle, which is why this method is a class method. This allows a Linear vehicle to
                            reason about other vehicles behaviors even though they may not Linear.
        :param front_vehicle: the vehicle preceding the ego-vehicle
        :param rear_vehicle: the vehicle following the ego-vehicle
        :return: the acceleration command for the ego-vehicle [m/s2]
        """
        return float(np.dot(self.ACCELERATION_PARAMETERS,
                            self.acceleration_features(ego_vehicle, front_vehicle, rear_vehicle)))

    def acceleration_features(self, ego_vehicle: ControlledVehicle,
                              front_vehicle: Vehicle = None,
                              rear_vehicle: Vehicle = None) -> np.ndarray:
        vt, dv, dp = 0, 0, 0
        if ego_vehicle:
            vt = ego_vehicle.target_speed - ego_vehicle.speed
            d_safe = self.DISTANCE_WANTED + np.maximum(ego_vehicle.speed, 0) * self.TIME_WANTED
            if front_vehicle:
                d = ego_vehicle.lane_distance_to(front_vehicle)
                dv = min(front_vehicle.speed - ego_vehicle.speed, 0)
                dp = min(d - d_safe, 0)
        return np.array([vt, dv, dp])

    def steering_control(self, target_lane_index: LaneIndex) -> float:
        """
        Linear controller with respect to parameters.

        Overrides the non-linear controller ControlledVehicle.steering_control()

        :param target_lane_index: index of the lane to follow
        :return: a steering wheel angle command [rad]
        """
        return float(np.dot(np.array(self.STEERING_PARAMETERS), self.steering_features(target_lane_index)))

    def steering_features(self, target_lane_index: LaneIndex) -> np.ndarray:
        """
        A collection of features used to follow a lane

        :param target_lane_index: index of the lane to follow
        :return: a array of features
        """
        lane = self.road.network.get_lane(target_lane_index)
        lane_coords = lane.local_coordinates(self.position)
        lane_next_coords = lane_coords[0] + self.speed * self.TAU_PURSUIT
        lane_future_heading = lane.heading_at(lane_next_coords)
        features = np.array([wrap_to_pi(lane_future_heading - self.heading) *
                             self.LENGTH / not_zero(self.speed),
                             -lane_coords[1] * self.LENGTH / (not_zero(self.speed) ** 2)])
        return features

    def longitudinal_structure(self):
        # Nominal dynamics: integrate speed
        A = np.array([
            [0, 0, 1, 0],
            [0, 0, 0, 1],
            [0, 0, 0, 0],
            [0, 0, 0, 0]
        ])
        # Target speed dynamics
        phi0 = np.array([
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, -1, 0],
            [0, 0, 0, -1]
        ])
        # Front speed control
        phi1 = np.array([
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, -1, 1],
            [0, 0, 0, 0]
        ])
        # Front position control
        phi2 = np.array([
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [-1, 1, -self.TIME_WANTED, 0],
            [0, 0, 0, 0]
        ])
        # Disable speed control
        front_vehicle, _ = self.road.neighbour_vehicles(self)
        if not front_vehicle or self.speed < front_vehicle.speed:
            phi1 *= 0

        # Disable front position control
        if front_vehicle:
            d = self.lane_distance_to(front_vehicle)
            if d != self.DISTANCE_WANTED + self.TIME_WANTED * self.speed:
                phi2 *= 0
        else:
            phi2 *= 0

        phi = np.array([phi0, phi1, phi2])
        return A, phi

    def lateral_structure(self):
        A = np.array([
            [0, 1],
            [0, 0]
        ])
        phi0 = np.array([
            [0, 0],
            [0, -1]
        ])
        phi1 = np.array([
            [0, 0],
            [-1, 0]
        ])
        phi = np.array([phi0, phi1])
        return A, phi

    def collect_data(self):
        """Store features and outputs for parameter regression."""
        self.add_features(self.data, self.target_lane_index)

    def add_features(self, data, lane_index, output_lane=None):

        front_vehicle, rear_vehicle = self.road.neighbour_vehicles(self)
        features = self.acceleration_features(self, front_vehicle, rear_vehicle)
        output = np.dot(self.ACCELERATION_PARAMETERS, features)
        if "longitudinal" not in data:
            data["longitudinal"] = {"features": [], "outputs": []}
        data["longitudinal"]["features"].append(features)
        data["longitudinal"]["outputs"].append(output)

        if output_lane is None:
            output_lane = lane_index
        features = self.steering_features(lane_index)
        out_features = self.steering_features(output_lane)
        output = np.dot(self.STEERING_PARAMETERS, out_features)
        if "lateral" not in data:
            data["lateral"] = {"features": [], "outputs": []}
        data["lateral"]["features"].append(features)
        data["lateral"]["outputs"].append(output)
