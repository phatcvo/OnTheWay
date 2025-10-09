from otw_env.vehicle import Vehicle
from otw_env.reward import compute_reward
from otw_env.utils import clamp

class OTWEnv:
    """Base environment: deterministic loop"""
    def __init__(self, dt=0.1):
        self.dt = dt
        self.time = 0.0
        self.done = False
        self.ego = Vehicle()

    def reset(self):
        self.ego = Vehicle()
        self.time = 0.0
        self.done = False
        return self.observe()

    def step(self, action):
        a, delta = action
        a = clamp(a, -2, 2)
        delta = clamp(delta, -0.4, 0.4)
        self.ego.step(a, delta, self.dt)
        obs = self.observe()
        reward = compute_reward(obs)
        self.time += self.dt
        self.done = self.check_done()
        return obs, reward, self.done, {}

    def observe(self):
        return [self.ego.x, self.ego.y, self.ego.v]

    def check_done(self):
        return self.ego.x >= 100.0
