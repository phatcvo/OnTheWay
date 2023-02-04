import numpy as np
import logging
from typing import List, Tuple, Dict, TYPE_CHECKING, Optional

from abc import ABCMeta, abstractmethod
from typing import Tuple, List, Optional
import numpy as np

# from OTW.road.graphics import LineType, StraightLane, AbstractLane
from OTW.vehicle.objects import Landmark
from OTW.common.utils import Vector, wrap_to_pi
if TYPE_CHECKING:
    from OTW.vehicle import kinematics, objects

logger = logging.getLogger(__name__)

LaneIndex = Tuple[str, str, int]
Route = List[LaneIndex]

# =================================================================

# Convert local lane coordinates to a world position.
# A lane on the road, described by its central curve.
class AbstractLane(object):

    metaclass__ = ABCMeta
    DEFAULT_WIDTH: float = 4
    VEHICLE_LENGTH: float = 5
    length: float = 0
    line_types: List["LineType"]

    @abstractmethod
    def position(self, longitudinal: float, lateral: float) -> np.ndarray:
        raise NotImplementedError()

    # Convert a world position to local lane coordinates.
    @abstractmethod
    def local_coordinates(self, position: np.ndarray) -> Tuple[float, float]:
        raise NotImplementedError()

    # Get the lane heading at a given longitudinal lane coordinate.
    @abstractmethod
    def heading_at(self, longitudinal: float) -> float:
        raise NotImplementedError()

    # Get the lane width at a given longitudinal lane coordinate.
    @abstractmethod
    def width_at(self, longitudinal: float) -> float:
        raise NotImplementedError()

    # Whether a given world position is on the lane.

    def on_lane(self, position: np.ndarray, longitudinal: float = None, lateral: float = None, margin: float = 0) \
            -> bool:
        if longitudinal is None or lateral is None:
            longitudinal, lateral = self.local_coordinates(position)
        is_on = np.abs(lateral) <= self.width_at(longitudinal) / 2 + margin and \
            -self.VEHICLE_LENGTH <= longitudinal < self.length + self.VEHICLE_LENGTH
        return is_on # is the position on the lane?

    # Whether the lane is reachable from a given world position
    def is_reachable_from(self, position: np.ndarray) -> bool: # position: the world position [m]
        if self.forbidden:
            return False
        longitudinal, lateral = self.local_coordinates(position)
        is_close = np.abs(lateral) <= 2 * self.width_at(longitudinal) and \
            0 <= longitudinal < self.length + self.VEHICLE_LENGTH
        return is_close # is the lane reachable?

    def after_end(self, position: np.ndarray, longitudinal: float = None, lateral: float = None) -> bool:
        if not longitudinal:
            longitudinal, _ = self.local_coordinates(position)
        return longitudinal > self.length - self.VEHICLE_LENGTH / 2

    # Compute the L1 distance [m] from a position to the lane.
    def distance(self, position: np.ndarray):
        s, r = self.local_coordinates(position)
        return abs(r) + max(s - self.length, 0) + max(0 - s, 0)

    # Compute a weighted distance in position and heading to the lane.
    def distance_with_heading(self, position: np.ndarray, heading: Optional[float], heading_weight: float = 1.0):
        if heading is None:
            return self.distance(position)
        s, r = self.local_coordinates(position)
        angle = np.abs(wrap_to_pi(heading - self.heading_at(s)))
        return abs(r) + max(s - self.length, 0) + max(0 - s, 0) + heading_weight*angle

# A lane side line type.
class LineType():
    NONE = 0
    STRIPED = 1
    CONTINUOUS = 2
    CONTINUOUS_LINE = 3

# A lane going in straight line.
class StraightLane(AbstractLane):

    # New straight lane.
    def __init__(self,
                 start: Vector,
                 end: Vector,
                 width: float = AbstractLane.DEFAULT_WIDTH,
                 line_types: Tuple[LineType, LineType] = None,
                 forbidden: bool = False,
                 speed_limit: float = 15,
                 priority: int = 0) -> None:

        self.start = np.array(start)    # the lane starting position [m]
        self.end = np.array(end)        # the lane ending position [m]
        self.width = width              # the lane width [m]
        self.heading = np.arctan2(self.end[1] - self.start[1], self.end[0] - self.start[0])
        self.length = np.linalg.norm(self.end - self.start)
        self.line_types = line_types or [LineType.STRIPED, LineType.STRIPED] # the type on both sides of the lane
        self.direction = (self.end - self.start) / self.length
        self.direction_lateral = np.array([-self.direction[1], self.direction[0]])
        self.forbidden = forbidden      # is changing to this lane forbidden
        self.priority = priority        # priority level of the lane, for determining who has right of way
        self.speed_limit = speed_limit

    def position(self, longitudinal: float, lateral: float) -> np.ndarray:
        return self.start + longitudinal * self.direction + lateral * self.direction_lateral

    def heading_at(self, longitudinal: float) -> float:
        return self.heading

    def width_at(self, longitudinal: float) -> float:
        return self.width

    def local_coordinates(self, position: np.ndarray) -> Tuple[float, float]:
        delta = position - self.start
        longitudinal = np.dot(delta, self.direction)
        lateral = np.dot(delta, self.direction_lateral)
        return float(longitudinal), float(lateral)

# =================================================================
class RoadNetwork(object):
    graph: Dict[str, Dict[str, List[AbstractLane]]]

    def __init__(self):
        self.graph = {}

    # A lane is encoded as an edge in the road network.
    def add_lane(self, _from: str, _to: str, lane: AbstractLane) -> None:

        if _from not in self.graph:
            self.graph[_from] = {}
        if _to not in self.graph[_from]:
            self.graph[_from][_to] = []
        self.graph[_from][_to].append(lane)

    # Get the lane geometry corresponding to a given index in the road network.
    def get_lane(self, index: LaneIndex) -> AbstractLane:
        _from, _to, _id = index # index: a tuple (origin node, destination node, lane id on the road).
        if _id is None and len(self.graph[_from][_to]) == 1:
            _id = 0
        return self.graph[_from][_to][_id] # the corresponding lane geometry.

    # Get the index of the lane closest to a world position.
    def get_closest_lane_index(self, position: np.ndarray, heading: Optional[float] = None) -> LaneIndex:

        indexes, distances = [], []
        for _from, to_dict in self.graph.items():
            for _to, lanes in to_dict.items():
                for _id, l in enumerate(lanes):
                    distances.append(l.distance_with_heading(position, heading)) # ([m], [rad])
                    indexes.append((_from, _to, _id))
        return indexes[int(np.argmin(distances))] # the index of the closest lane.

    # Get the index of the next lane that should be followed after finishing the current lane.
    def next_lane(self, current_index: LaneIndex, route: Route = None, position: np.ndarray = None,
                  np_random: np.random.RandomState = np.random) -> LaneIndex:
        
        # - If a plan is available and matches with current lane, follow it.
        # - Else, pick next road randomly.
        # - If it has the same number of lanes as current road, stay in the same lane.
        # - Else, pick next road's closest lane.
        
        _from, _to, _id = current_index # the index of the current target lane.
        next_to = next_id = None
        # Pick next road according to planned route
        if route: # the planned route, if any.
            if route[0][:2] == current_index[:2]:  # We just finished the first step of the route, drop it.
                route.pop(0)
            if route and route[0][0] == _to:  # Next road in route is starting at the end of current road.
                _, next_to, next_id = route[0]
            elif route:
                logger.warning("Route {} does not start after current road {}.".format(route[0], current_index))

        # Compute current projected (desired) position
        long, lat = self.get_lane(current_index).local_coordinates(position)
        projected_position = self.get_lane(current_index).position(long, lateral=0)

        # If next route is not known
        if not next_to:
            # Pick the one with the closest lane to projected target position
            try:
                lanes_dists = [(next_to,
                                *self.next_lane_given_next_road(_from, _to, _id, next_to, next_id, projected_position))
                               for next_to in self.graph[_to].keys()]  # (next_to, next_id, distance)
                next_to, next_id, _ = min(lanes_dists, key=lambda x: x[-1])
            except KeyError:
                return current_index
        else:
            # If it is known, follow it and get the closest lane
            next_id, _ = self.next_lane_given_next_road(_from, _to, _id, next_to, next_id, projected_position)
        return _to, next_to, next_id # the index of the next lane to be followed when current lane is finished.

    def next_lane_given_next_road(self, _from: str, _to: str, _id: int,
                                  next_to: str, next_id: int, position: np.ndarray) -> Tuple[int, float]:
        # If next road has same number of lane, stay on the same lane
        if len(self.graph[_from][_to]) == len(self.graph[_to][next_to]):
            if next_id is None:
                next_id = _id
        # Else, pick closest lane
        else:
            lanes = range(len(self.graph[_to][next_to]))
            next_id = min(lanes,
                          key=lambda l: self.get_lane((_to, next_to, l)).distance(position))
        return next_id, self.get_lane((_to, next_to, next_id)).distance(position)

    # Breadth-first search of all routes from start to goal
    def bfs_paths(self, start: str, goal: str) -> List[List[str]]:
        queue = [(start, [start])]
        while queue:
            (node, path) = queue.pop(0)
            if node not in self.graph:
                yield []
            for _next in set(self.graph[node].keys()) - set(path):
                if _next == goal:
                    yield path + [_next]
                elif _next in self.graph:
                    queue.append((_next, path + [_next]))

    # Breadth-first search of shortest path from start to goal.
    def shortest_path(self, start: str, goal: str) -> List[str]:
        return next(self.bfs_paths(start, goal), [])

    def all_side_lanes(self, lane_index: LaneIndex) -> List[LaneIndex]:
        # all lanes belonging to the same road.
        return [(lane_index[0], lane_index[1], i) for i in range(len(self.graph[lane_index[0]][lane_index[1]]))]

    def side_lanes(self, lane_index: LaneIndex) -> List[LaneIndex]:
        _from, _to, _id = lane_index
        lanes = []
        if _id > 0:
            lanes.append((_from, _to, _id - 1))
        if _id < len(self.graph[_from][_to]) - 1:
            lanes.append((_from, _to, _id + 1))
        return lanes # indexes of lanes next to an input lane, to its right or left.

    @staticmethod
    def is_same_road(lane_index_1: LaneIndex, lane_index_2: LaneIndex, same_lane: bool = False) -> bool:
        # Is lane 1 in the same road as lane 2?
        return lane_index_1[:2] == lane_index_2[:2] and (not same_lane or lane_index_1[2] == lane_index_2[2])

    @staticmethod
    def is_leading_to_road(lane_index_1: LaneIndex, lane_index_2: LaneIndex, same_lane: bool = False) -> bool:
        # Is lane 1 leading to of lane 2?
        return lane_index_1[1] == lane_index_2[0] and (not same_lane or lane_index_1[2] == lane_index_2[2])

    def is_connected_road(self, lane_index_1: LaneIndex, lane_index_2: LaneIndex, route: Route = None,
                          same_lane: bool = False, depth: int = 0) -> bool:
        # Is the lane 2 leading to a road within lane 1's route?
        if RoadNetwork.is_same_road(lane_index_2, lane_index_1, same_lane) \
                or RoadNetwork.is_leading_to_road(lane_index_2, lane_index_1, same_lane):
            return True
        if depth > 0:
            if route and route[0][:2] == lane_index_1[:2]:
                # Route is starting at current road, skip it
                return self.is_connected_road(lane_index_1, lane_index_2, route[1:], same_lane, depth)
            elif route and route[0][0] == lane_index_1[1]:
                # Route is continuing from current road, follow it
                return self.is_connected_road(route[0], lane_index_2, route[1:], same_lane, depth - 1)
            else:
                # Recursively search all roads at intersection
                _from, _to, _id = lane_index_1
                return any([self.is_connected_road((_to, l1_to, _id), lane_index_2, route, same_lane, depth - 1)
                            for l1_to in self.graph.get(_to, {}).keys()])
        return False

    def lanes_list(self) -> List[AbstractLane]:
        return [lane for to in self.graph.values() for ids in to.values() for lane in ids]

    @staticmethod
    def straight_road_network(lanes: int = 3, start: float = 0, length: float = 10000, angle: float = 0,
                              speed_limit: float = 15, nodes_str: Optional[Tuple[str, str]] = None,
                              net: Optional['RoadNetwork'] = None) -> 'RoadNetwork':
        net = net or RoadNetwork()
        nodes_str = nodes_str or ("0", "1")
        for lane in range(lanes):
            origin = np.array([start, lane * StraightLane.DEFAULT_WIDTH])
            end = np.array([start + length, lane * StraightLane.DEFAULT_WIDTH])
            rotation = np.array([[np.cos(angle), np.sin(angle)], [-np.sin(angle), np.cos(angle)]])
            origin = rotation @ origin
            end = rotation @ end
            line_types = [LineType.CONTINUOUS_LINE if lane == 0 else LineType.STRIPED,
                          LineType.CONTINUOUS_LINE if lane == lanes - 1 else LineType.NONE]
            net.add_lane(*nodes_str, StraightLane(origin, end, line_types=line_types, speed_limit=speed_limit))
        return net

    # Get the absolute position and heading along a route composed of several lanes at some local coordinates.
    def position_heading_along_route(self, route: Route, longitudinal: float, lateral: float) \
            -> Tuple[np.ndarray, float]:
        while len(route) > 1 and longitudinal > self.get_lane(route[0]).length:
            longitudinal -= self.get_lane(route[0]).length
            route = route[1:]
        return self.get_lane(route[0]).position(longitudinal, lateral), self.get_lane(route[0]).heading_at(longitudinal)

    def random_lane_index(self, np_random: np.random.RandomState) -> LaneIndex:
        _from = np_random.choice(list(self.graph.keys()))
        _to = np_random.choice(list(self.graph[_from].keys()))
        _id = np_random.randint(len(self.graph[_from][_to]))
        return _from, _to, _id

# A road is a set of lanes, and a set of vehicles driving on these lanes.
class Road(object):
    def __init__(self, network: RoadNetwork = None, vehicles: List['kinematics.Vehicle'] = None, 
                road_objects: List['objects.RoadObject'] = None, np_random: np.random.RandomState = None, record_history: bool = False) -> None:

        self.network = network
        self.vehicles = vehicles or []
        self.objects = road_objects or []
        self.np_random = np_random if np_random else np.random.RandomState()
        self.record_history = record_history

    def close_vehicles_to(self, vehicle: 'kinematics.Vehicle', distance: float, count: Optional[int] = None, see_behind: bool = True, sort: bool = True) -> object:
        vehicles = [v for v in self.vehicles
                    if np.linalg.norm(v.position - vehicle.position) < distance
                    and v is not vehicle
                    and (see_behind or -2 * vehicle.LENGTH < vehicle.lane_distance_to(v))]

        if sort:
            vehicles = sorted(vehicles, key=lambda v: abs(vehicle.lane_distance_to(v)))
        if count:
            vehicles = vehicles[:count]
        return vehicles

    # Decide the actions of each entity on the road.
    def act(self) -> None:
        for vehicle in self.vehicles:
            vehicle.act()

    # Step the dynamics of each entity on the road.
    def step(self, dt: float) -> None:
        for vehicle in self.vehicles:
            vehicle.step(dt) # dt: timestep [s]
        for i, vehicle in enumerate(self.vehicles):
            for other in self.vehicles[i+1:]:
                vehicle.handle_collisions(other, dt)
            for other in self.objects:
                vehicle.handle_collisions(other, dt)

    # Find the preceding and following vehicles of a given vehicle.
    def neighbor_vehicles(self, vehicle: 'kinematics.Vehicle', lane_index: LaneIndex = None) \
            -> Tuple[Optional['kinematics.Vehicle'], Optional['kinematics.Vehicle']]:
        lane_index = lane_index or vehicle.lane_index
        # print(lane_index)
        if not lane_index:
            return None, None
        lane = self.network.get_lane(lane_index)
        s = self.network.get_lane(lane_index).local_coordinates(vehicle.position)[0]
        s_front = s_rear = None
        v_front = v_rear = None
        for v in self.vehicles + self.objects:
            if v is not vehicle and not isinstance(v, Landmark): 
                s_v, lat_v = lane.local_coordinates(v.position)
                if not lane.on_lane(v.position, s_v, lat_v, margin=1):
                    continue
                if s <= s_v and (s_front is None or s_v <= s_front):
                    s_front = s_v
                    v_front = v
                if s_v < s and (s_rear is None or s_v > s_rear):
                    s_rear = s_v
                    v_rear = v
        return v_front, v_rear

    def __repr__(self):
        return self.vehicles.__repr__()



