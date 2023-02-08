import itertools
from typing import List, Tuple, TYPE_CHECKING

import numpy as np
import pygame

from OTW.common.utils import Vector
from OTW.vehicle.kinematics import Vehicle
from OTW.vehicle.controller import MDPVehicle, IDMVehicle

if TYPE_CHECKING:
    from OTW.road.road import WorldSurface

pygame.init()

class VehicleGraphics(object):
    # Display a vehicle on a pygame surface. The vehicle is represented as a colored rotated rectangle.

    @classmethod
    def display(cls, vehicle: Vehicle, surface: "WorldSurface", # the surface to draw the vehicle on
                transparent: bool = False,                  # whether the vehicle should be drawn slightly transparent
                offscreen: bool = False,                    # whether the rendering should be done offscreen or not
                label: bool = True,                        # whether a text label should be rendered
                draw_roof: bool = False) -> None:


        if not surface.is_visible(vehicle.position):
            return

        v = vehicle

        if vehicle.crashed:
            vehicle_surface = pygame.image.load('OTW/vehicle/Image/RedCar1.png')
        elif isinstance(vehicle, IDMVehicle):
            vehicle_surface = pygame.image.load('OTW/vehicle/Image/WhiteCar1.png')
        elif isinstance(vehicle, MDPVehicle):
            vehicle_surface = pygame.image.load('OTW/vehicle/Image/Ego1.png')


        # Centered rotation
        h = v.heading if abs(v.heading) > 2 * np.pi / 180 else 0 # heading clipping
        position = [*surface.pos2pix(v.position[1], v.position[0])]
        if not offscreen:
            # convert_alpha throws errors in offscreen mode
            # convert_alpha throws errors in offscreen mode
            vehicle_surface = pygame.Surface.convert_alpha(vehicle_surface)
        cls.blit_rotate(surface, vehicle_surface, position, np.rad2deg(-h)) #This line draws car moving

        # Label
        speed = round(vehicle.speed, 1)
        steering_angle = round(vehicle.steering_control(vehicle.target_lane_index)* 57.2957795, 1)
        if label:
            font = pygame.font.Font(None, 15)
            # text = "#{}".format(id(v) % 1000)
            # int vel = ControlledVehicle.velocity
            text = "{}m/s, {} deg".format(speed, steering_angle)
            text = font.render(text, 1, (10, 10, 10), (255, 255, 255))
            surface.blit(text, [10, position[1]])
            # surface.blit(text, position)
        # print("speed {:0.03f}".format(speed))
      

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

    @classmethod
    def display_history(cls, v: Vehicle, surface: "WorldSurface", simulation: int, offscreen: bool) -> None:
        #display history trajectory for each agent
        color = (255,0,0)
        closed = False

        histories = v.history
        #draw history, if exists.
        if len(histories) < 2:
            return

        history_points = []
        for history in histories:
            history_points.append((surface.pos2pix(history.position[1], history.position[0])))

        pygame.draw.lines(surface, color, closed, history_points)

