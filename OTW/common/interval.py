import itertools
from typing import Tuple, Union, List, Callable

import numpy as np
from numpy.linalg import LinAlgError

from OTW.road.road import AbstractLane
from OTW.common.utils import Vector, Matrix, Interval


# Compute the product of two intervals
def intervals_product(a: Interval, b: Interval) -> np.ndarray: # a: interval [a_min, a_max], b: interval [b_min, b_max]

    p = lambda x: np.maximum(x, 0)
    n = lambda x: np.maximum(-x, 0)
    return np.array(
        [np.dot(p(a[0]), p(b[0])) - np.dot(p(a[1]), n(b[0])) - np.dot(n(a[0]), p(b[1])) + np.dot(n(a[1]), n(b[1])),
         np.dot(p(a[1]), p(b[1])) - np.dot(p(a[0]), n(b[1])) - np.dot(n(a[1]), p(b[0])) + np.dot(n(a[0]), n(b[0]))])

# Scale an intervals
def intervals_scaling(a: Interval, b: Interval) -> np.ndarray: # a: matrix a, b: interval [b_min, b_max]

    p = lambda x: np.maximum(x, 0)
    n = lambda x: np.maximum(-x, 0)
    return np.array(
        [np.dot(p(a), b[0]) - np.dot(n(a), b[1]),
         np.dot(p(a), b[1]) - np.dot(n(a), b[0])])

# Compute the difference of two intervals
def intervals_diff(a: Interval, b: Interval) -> np.ndarray: # a: interval [a_min, a_max], b: interval [b_min, b_max]
    return np.array([a[0] - b[1], a[1] - b[0]]) # the interval of their difference a - b

# Compute the negative part of an interval
def interval_negative_part(a: Interval) -> np.ndarray: # a: interval [a_min, a_max]
    return np.minimum(a, 0) # the interval of its negative part min(a, 0)

# Compute the interval of an integrator system: dx = -k*x
def integrator_interval(x: Interval, k: Interval) -> np.ndarray: # x: state interval, k: gain interval, must be positive

    if x[0] >= 0:
        interval_gain = np.flip(-k, 0)
    elif x[1] <= 0:
        interval_gain = -k
    else:
        interval_gain = -np.array([k[0], k[0]])
    # interval for dx
    return interval_gain*x  # Note: no flip of x, contrary to using intervals_product(k,interval_minus(x))


def vector_interval_section(v_i: Interval, direction: Vector) -> np.ndarray:
    corners = [[v_i[0, 0], v_i[0, 1]],
               [v_i[0, 0], v_i[1, 1]],
               [v_i[1, 0], v_i[0, 1]],
               [v_i[1, 0], v_i[1, 1]]]
    corners_dist = [np.dot(corner, direction) for corner in corners]
    return np.array([min(corners_dist), max(corners_dist)])

# Converts an interval in absolute x,y coordinates to an interval in local (longiturinal, lateral) coordinates
def interval_absolute_to_local(position_i: Interval, lane: AbstractLane) -> Tuple[np.ndarray, np.ndarray]:
    # position_i: the position interval [x_min, x_max], lane: the lane giving the local frame    
    position_corners = np.array([[position_i[0, 0], position_i[0, 1]],
                                 [position_i[0, 0], position_i[1, 1]],
                                 [position_i[1, 0], position_i[0, 1]],
                                 [position_i[1, 0], position_i[1, 1]]])
    corners_local = np.array([lane.local_coordinates(c) for c in position_corners])
    longitudinal_i = np.array([min(corners_local[:, 0]), max(corners_local[:, 0])])
    lateral_i = np.array([min(corners_local[:, 1]), max(corners_local[:, 1])])
    return longitudinal_i, lateral_i # the corresponding local interval

# Converts an interval in local (longiturinal, lateral) coordinates to an interval in absolute x,y coordinates
def interval_local_to_absolute(longitudinal_i: Interval, lateral_i: Interval, lane: AbstractLane) -> Interval:
    # longitudinal_i: the longitudinal interval [L_min, L_max], lateral_i: the lateral interval [l_min, l_max], lane: the lane giving the local frame

    corners_local = [[longitudinal_i[0], lateral_i[0]],
                     [longitudinal_i[0], lateral_i[1]],
                     [longitudinal_i[1], lateral_i[0]],
                     [longitudinal_i[1], lateral_i[1]]]
    corners_absolute = np.array([lane.position(*c) for c in corners_local])
    position_i = np.array([np.amin(corners_absolute, axis=0), np.amax(corners_absolute, axis=0)])
    return position_i # the corresponding absolute interval

# Get a matrix polytope from a parametrized matrix function and parameter box
def polytope(parametrized_f: Callable[[np.ndarray], np.ndarray], params_intervals: np.ndarray) \
        -> Tuple[np.ndarray, List[np.ndarray]]:
# parametrized_f: parametrized matrix function, params_intervals: axes: [min, max], params

    params_means = params_intervals.mean(axis=0)
    a0 = parametrized_f(params_means)
    vertices_id = itertools.product([0, 1], repeat=params_intervals.shape[1])
    d_a = []
    for vertex_id in vertices_id:
        params_vertex = params_intervals[vertex_id, np.arange(len(vertex_id))]
        d_a.append(parametrized_f(params_vertex) - parametrized_f(params_means))
    d_a = list({d_a_i.tostring(): d_a_i for d_a_i in d_a}.values())
    return a0, d_a # a0, d_a polytope that represents the matrix interval


def is_metzler(matrix: np.ndarray, eps: float = 1e-9) -> bool:
    return (matrix - np.diag(np.diag(matrix)) >= -eps).all()

# A Linear Parameter-Varying system: dx = (a0 + sum(da))(x - center) + bd + c
class LPV(object):
    def __init__(self,
                 x0: Vector,                    # initial state
                 a0: Matrix,                    # nominal dynamics
                 da: List[Vector],              # list of dynamics deviations
                 b: Matrix = None,              # control matrix
                 d: Matrix = None,              # perturbation matrix
                 omega_i: Matrix = None,        # perturbation bounds
                 u: Vector = None,              # constant known control
                 k: Matrix = None,              # linear feedback: a0 x + bu -> (a0+bk)x + b(u-kx), where a0+bk is stable
                 center: Vector = None,         # asymptotic state
                 x_i: Matrix = None) -> None:   # initial state interval

        self.x0 = np.array(x0, dtype=float)
        self.a0 = np.array(a0, dtype=float)
        self.da = [np.array(da_i) for da_i in da]
        self.b = np.array(b) if b is not None else np.zeros((*self.x0.shape, 1))
        self.d = np.array(d) if d is not None else np.zeros((*self.x0.shape, 1))
        self.omega_i = np.array(omega_i) if omega_i is not None else np.zeros((2, 1))
        self.u = np.array(u) if u is not None else np.zeros((1,))
        self.k = np.array(k) if k is not None else np.zeros((self.b.shape[1], self.b.shape[0]))
        self.center = np.array(center) if center is not None else np.zeros(self.x0.shape)

        # Closed-loop dynamics
        self.a0 += self.b @ self.k

        self.coordinates = None

        self.x_t = self.x0
        self.x_i = np.array(x_i) if x_i is not None else np.array([self.x0, self.x0])
        self.x_i_t = None

        self.update_coordinates_frame(self.a0)


    # Ensure that the dynamics matrix A0 is Metzler. If not, design a coordinate transformation and apply it to the model and state interval.
    def update_coordinates_frame(self, a0: np.ndarray) -> None: # a0: the dynamics matrix A0

        self.coordinates = None
        # Rotation
        if not is_metzler(a0):
            eig_v, transformation = np.linalg.eig(a0)
            if np.isreal(eig_v).all():
                try:
                    self.coordinates = (transformation, np.linalg.inv(transformation))
                except LinAlgError:
                    pass
            if not self.coordinates:
                print("Non Metzler A0 with eigenvalues: ", eig_v)
        else:
            self.coordinates = (np.eye(a0.shape[0]), np.eye(a0.shape[0]))

        # Forward coordinates change of states and models
        self.a0 = self.change_coordinates(self.a0, matrix=True)
        self.da = self.change_coordinates(self.da, matrix=True)
        self.b = self.change_coordinates(self.b, offset=False)
        self.x_i_t = np.array(self.change_coordinates([x for x in self.x_i]))

    def set_control(self, control: np.ndarray, state: np.ndarray = None) -> None:
        if state is not None:
            control = control - self.k @ state  # the Kx part of the control is already present in A0.
        self.u = control

    # Perform a change of coordinate: rotation and centering.
    def change_coordinates(self, value: Union[np.ndarray, List[np.ndarray]], matrix: bool = False, back: bool = False,
                           interval: bool = False, offset: bool = True) -> Union[np.ndarray, List[np.ndarray]]:
        # value: the object to transform
        # matrix: is it a matrix or a vector?
        # back: if True, transform back to the original coordinates
        # interval: when transforming an interval, lossy interval arithmetic must be used to preserve the inclusion property.
        # offset: should we apply the centering or not

        if self.coordinates is None:
            return value
        transformation, transformation_inv = self.coordinates
        if interval:
            if back:
                value = intervals_scaling(transformation, value[:, :, np.newaxis]).squeeze() + \
                        offset * np.array([self.center, self.center])
                return value
            else:
                value = value - offset * np.array([self.center, self.center])
                value = intervals_scaling(transformation_inv, value[:, :, np.newaxis]).squeeze()
                return value
        elif matrix:  # Matrix
            if back:
                return transformation @ value @ transformation_inv
            else:
                return transformation_inv @ value @ transformation
        elif isinstance(value, list):  # List
            return [self.change_coordinates(v, back) for v in value]
        else:
            if back:
                value = transformation @ value
                if offset:
                    value += self.center
                return value
            else:
                if offset:
                    value -= self.center
                return transformation_inv @ value
        # return: the transformed object

    def step(self, dt: float) -> None:
        if is_metzler(self.a0):
            self.x_i_t = self.step_interval_predictor(self.x_i_t, dt)
        else:
            self.x_i_t = self.step_naive_predictor(self.x_i_t, dt)
        dx = self.a0 @ self.x_t + self.b @ self.u.squeeze(-1)
        self.x_t = self.x_t + dx * dt

    # Step an interval predictor with box uncertainty.
    def step_naive_predictor(self, x_i: Interval, dt: float) -> np.ndarray: # x_i: state interval at time t, dt: time step

        a0, da, d, omega_i, b, u = self.a0, self.da, self.d, self.omega_i, self.b, self.u
        a_i = a0 + sum(intervals_product([0, 1], [da_i, da_i]) for da_i in da)
        bu = (b @ u).squeeze(-1)
        dx_i = intervals_product(a_i, x_i) + intervals_product([d, d], omega_i) + np.array([bu, bu])
        return x_i + dx_i*dt # state interval at time t+dt

    # Step an interval predictor with polytopic uncertainty.
    def step_interval_predictor(self, x_i: Interval, dt: float) -> np.ndarray: # x_i: state interval at time t, dt: time step
        
        a0, da, d, omega_i, b, u = self.a0, self.da, self.d, self.omega_i, self.b, self.u
        p = lambda x: np.maximum(x, 0)
        n = lambda x: np.maximum(-x, 0)
        da_p = sum(p(da_i) for da_i in da)
        da_n = sum(n(da_i) for da_i in da)
        x_m, x_M = x_i[0, :, np.newaxis], x_i[1, :, np.newaxis]
        o_m, o_M = omega_i[0, :, np.newaxis], omega_i[1, :, np.newaxis]
        dx_m = a0 @ x_m - da_p @ n(x_m) - da_n @ p(x_M) + p(d) @ o_m - n(d) @ o_M + b @ u
        dx_M = a0 @ x_M + da_p @ p(x_M) + da_n @ n(x_m) + p(d) @ o_M - n(d) @ o_m + b @ u
        dx_i = np.array([dx_m.squeeze(axis=-1), dx_M.squeeze(axis=-1)])
        return x_i + dx_i * dt # state interval at time t+dt
