import itertools
from typing import List, Tuple, TYPE_CHECKING

import numpy as np
import pygame
from matplotlib import cm as cm

from otw_env.utils.utils import Vector
from otw_env.core.vehicle.kinematics import Vehicle
from otw_env.core.vehicle.controller import MDPVehicle
from otw_env.core.vehicle.behavior_controller import IDMVehicle
import os
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

if TYPE_CHECKING:
    from otw_env.core.road.road import WorldSurface

pygame.init()

# Display a vehicle on a pygame surface. The vehicle is represented as a colored rotated rectangle.
class VehicleGraphics(object):
    
    RED = (255, 100, 100)
    GREEN = (50, 200, 0)
    BLUE = (100, 200, 255)
    YELLOW = (200, 200, 0)
    BLACK = (60, 60, 60)
    PURPLE = (200, 0, 150)
    DEFAULT_COLOR = YELLOW
    EGO_COLOR = GREEN
    UNCERTAINTY_TIME_COLORMAP = cm.bwr
    MODEL_TRAJ_COLOR = (148, 173, 215)
    EGO_TRAJ_COLOR = (102, 140, 255)
    TRANSPARENCY = 100
    
    @classmethod
    def display(cls, vehicle: Vehicle, surface: "WorldSurface", # the surface to draw the vehicle on
                transparent: bool = False,                  # whether the vehicle should be drawn slightly transparent
                offscreen: bool = False,                    # whether the rendering should be done offscreen or not
                label: bool = True,                        # whether a text label should be rendered
                draw_roof: bool = False) -> None:

        if not surface.is_visible(vehicle.position):
            return

        if vehicle.crashed:
            vehicle_surface = pygame.image.load(os.path.join(ROOT_DIR, 'Image', 'RedCar1.png'))
        elif isinstance(vehicle, IDMVehicle):
            vehicle_surface = pygame.image.load(os.path.join(ROOT_DIR, 'Image', 'WhiteCar1.png'))
        elif isinstance(vehicle, MDPVehicle):
            vehicle_surface = pygame.image.load(os.path.join(ROOT_DIR, 'Image', 'Ego1.png'))


        # Centered rotation
        head = vehicle.heading if abs(vehicle.heading) > 2 * np.pi / 180 else 0 # heading clipping
        position = [*surface.pos2pix(vehicle.position[1], vehicle.position[0])]
        if not offscreen:
            # convert_alpha throws errors in offscreen mode
            vehicle_surface = pygame.Surface.convert_alpha(vehicle_surface)
        cls.blit_rotate(surface, vehicle_surface, position, np.rad2deg(-head)) #This line draws car moving

        # Label
        speed = round(vehicle.speed, 1)
        steering_angle = round(vehicle.steering_control(vehicle.target_lane_index)* 57.2957795, 1)
        if label:
            font = pygame.font.Font(None, 20)
            text = "{} m/s, {} deg".format(speed, steering_angle)
            text = font.render(text, 1, (10, 10, 10), (255, 255, 255))
            surface.blit(text, [10, position[1]])

      

    @staticmethod
    def blit_rotate(surf: pygame.SurfaceType, image: pygame.SurfaceType, pos: Vector, angle: float,
                    origin_pos: Vector = None, show_rect: bool = False) -> None:
        # calculate the axis aligned bounding box of the rotated image
        w, h = image.get_size()
        box = [pygame.math.Vector2(p) for p in [(0, 0), (w, 0), (w, -h), (0, -h)]]
        box_rotate = [p.rotate(angle) for p in box]
        min_box = (min(box_rotate, key=lambda p: p[0])[0], min(box_rotate, key=lambda p: p[1])[1])
        max_box = (max(box_rotate, key=lambda p: p[0])[0], max(box_rotate, key=lambda p: p[1])[1])

        # calculate the translation of the pivot
        if origin_pos is None:
            origin_pos = w / 2, h / 2
        pivot = pygame.math.Vector2(origin_pos[0], -origin_pos[1])
        pivot_rotate = pivot.rotate(angle)
        pivot_move = pivot_rotate - pivot

        # calculate the upper left origin of the rotated image
        origin = (pos[0] - origin_pos[0] + min_box[0] - pivot_move[0], pos[1] - origin_pos[1] - max_box[1] + pivot_move[1])
        # get a rotated image
        rotated_image = pygame.transform.rotate(image, angle)
        # rotate and blit the image
        surf.blit(rotated_image, origin)
        # draw rectangle around the image
        if show_rect:
            pygame.draw.rect(surf, (255, 0, 0), (*origin, *rotated_image.get_size()), 2)


    @classmethod #display trajectory for each agent
    def display_trajectory(cls, states: List[Vehicle], surface: "WorldSurface", offscreen: bool = False) -> None:
        state_points = [] 

        closed = False
        for vehicle in states:
            # cls.display(vehicle, surface, transparent=True, offscreen=offscreen)
            state_points.append(surface.pos2pix(vehicle.position[1], vehicle.position[0]))
            # cls.display(state_points, surface, transparent=True, offscreen=offscreen)
        # print("Horizon steps: ", len(state_points))

        pygame.draw.lines(surface, cls.EGO_TRAJ_COLOR, closed, state_points[0:25], width = 15)
        
    
    @classmethod #display history trajectory for each agent
    def display_history(cls, v: Vehicle, surface: "WorldSurface", frequency: float = 3, duration: float = 2, simulation: int = 15, offscreen: bool = False) -> None:
        
        closed = False

        for v in itertools.islice(v.history, None, int(simulation * duration), int(simulation / v)):
            cls.display(v, surface, transparent=True, offscreen=offscreen)
            
        histories = v.history
        
        if len(histories) < 2:
            return

        history_points = []
        for history in histories:
            history_points.append((surface.pos2pix(history.position[1], history.position[0])))
        #draw history, if exists.
        pygame.draw.lines(surface, cls.BLUE, closed, history_points)
