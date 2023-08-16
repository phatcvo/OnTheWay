import os
from typing import TYPE_CHECKING, Callable, List, Optional
import numpy as np
import pygame

from OTW.common import action
from OTW.road.graphics import WorldSurface, RoadGraphics
from OTW.vehicle.graphics import VehicleGraphics

if TYPE_CHECKING:
    from OTW.envs import AbstractEnv
    from OTW.common.abstract import Action


class EnvViewer(object):
    SAVE_IMAGES = False

    def __init__(self, env: 'AbstractEnv', config: Optional[dict] = None) -> None:
        self.env = env
        self.config = config or env.config
        self.offscreen = self.config["offscreen_rendering"]

        pygame.init()
        pygame.display.set_caption("OTW-RML")
        panel_size = (self.config["screen_width"], self.config["screen_height"])

        # A display is not mandatory to draw things. Ignoring the display.set_mode() instruction allows the drawing to be done on surfaces without handling a screen display, useful for e.g. cloud computing
        if not self.offscreen:
            self.screen = pygame.display.set_mode([self.config["screen_width"], self.config["screen_height"]])

        self.sim_surface = WorldSurface(panel_size, 0, pygame.Surface(panel_size))
        self.sim_surface.scaling = self.config.get("scaling", self.sim_surface.INITIAL_SCALING)
        self.sim_surface.centering_position = self.config.get("centering_position", self.sim_surface.INITIAL_CENTERING)
        self.clock = pygame.time.Clock()
        self.enabled = True

        if os.environ.get("SDL_VIDEODRIVER", None) == "dummy":
            self.enabled = False

        self.agent_display = None
        self.agent_surface = None
        self.vehicle_trajectory = None
        self.frame = 0
        self.directory = None

    # Set a display callback provided by an agent
    # So that they can render their behaviour on a dedicated agent surface, or even on the simulation surface.
    def set_agent_display(self, agent_display: Callable) -> None:
        if self.agent_display is None:
            if not self.offscreen:
                if self.config["screen_width"] > self.config["screen_height"]:
                    self.screen = pygame.display.set_mode((self.config["screen_width"], 2 * self.config["screen_height"]))
                else:
                    self.screen = pygame.display.set_mode((2 * self.config["screen_width"], self.config["screen_height"]))
            self.agent_surface = pygame.Surface((self.config["screen_width"], self.config["screen_height"]))
        self.agent_display = agent_display # a callback provided by the agent to display on surfaces

    # Set the sequence of actions chosen by the agent, so that it can be displayed
    def set_agent_action_sequence(self, actions: List['Action']) -> None:
        # list of action, following the env's action space specification
        if isinstance(self.env.action_type, action.DiscreteMetaAction):
            actions = [self.env.action_type.actions[a] for a in actions]
        if len(actions) > 1:
            self.vehicle_trajectory = self.env.vehicle.predict_trajectory(actions,1 / self.env.config["policy_frequency"], 1 / 3 / self.env.config["policy_frequency"], 1 / self.env.config["simulation_frequency"])

    # Handle pygame events by forwarding them to the display and environment vehicle.
    def handle_events(self) -> None:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.env.close()
            self.sim_surface.handle_event(event)
            if self.env.action_type:
                EventHandler.handle_event(self.env.action_type, event)

    # Display the road and vehicles on a pygame window.
    def display(self) -> None:
        if not self.enabled:
            return
        self.sim_surface.move_display_window_to(self.window_position())
        RoadGraphics.display(self.env.road, self.sim_surface)

        if self.vehicle_trajectory:
            VehicleGraphics.display_trajectory(self.vehicle_trajectory, self.sim_surface, offscreen=self.offscreen)
        # RoadGraphics.display_road_objects(self.env.road, self.sim_surface, offscreen=self.offscreen)

        if self.agent_display:
            self.agent_display(self.agent_surface, self.sim_surface)
            if not self.offscreen:
                if self.config["screen_width"] > self.config["screen_height"]:
                    self.screen.blit(self.agent_surface, (0, self.config["screen_height"]))
                else:
                    self.screen.blit(self.agent_surface, (self.config["screen_width"], 0))

        RoadGraphics.display_traffic(self.env.road, self.sim_surface, simulation_frequency=self.env.config["simulation_frequency"],  offscreen=self.offscreen)

        if not self.offscreen:
            self.screen.blit(self.sim_surface, (0, 0))
            if self.env.config["real_time_rendering"]:
                self.clock.tick(self.env.config["simulation_frequency"])
            pygame.display.flip()

        if self.SAVE_IMAGES and self.directory:
            pygame.image.save(self.sim_surface, str(self.directory / "OTW-env_{}.png".format(self.frame)))
            self.frame += 1
            
    # The rendered image as a rgb array. OpenAI gym's channel convention is H x W x C
    def get_image(self) -> np.ndarray:
        surface = self.screen if self.config["render_agent"] and not self.offscreen else self.sim_surface
        data = pygame.surfarray.array3d(surface)  # in W x H x C channel convention
        return np.moveaxis(data, 0, 1)

    # the world position of the center of the displayed window.
    def window_position(self) -> np.ndarray:
        a = self.env.vehicle.position
        if self.env.vehicle:
            return np.array([a[0], 5]) # fix vertical exis
        else:
            return np.array([0, 0])
        # print('vehicle.position', a[0])  # horizontal axis

    # Close the pygame window.
    def close(self) -> None:
        pygame.quit()


# the ActionType that defines how the vehicle is controlled
# Map the pygame keyboard events to control decisions
class EventHandler(object):
    @classmethod
    def handle_event(cls, action_type: action.ActionType, event: pygame.event.EventType) -> None:
        if isinstance(action_type, action.DiscreteMetaAction):
            cls.handle_discrete_action_event(action_type, event)


    @classmethod
    def handle_discrete_action_event(cls, action_type: action.DiscreteMetaAction, event: pygame.event.EventType) -> None:
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_UP and action_type.longitudinal:
                action_type.act(action_type.actions_indexes["FASTER"])
            if event.key == pygame.K_DOWN and action_type.longitudinal:
                action_type.act(action_type.actions_indexes["SLOWER"])
            if event.key == pygame.K_RIGHT and action_type.lateral:
                action_type.act(action_type.actions_indexes["LANE_RIGHT"])
            if event.key == pygame.K_LEFT:
                action_type.act(action_type.actions_indexes["LANE_LEFT"])


