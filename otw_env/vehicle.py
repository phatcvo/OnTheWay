import numpy as np

class Vehicle:
    def __init__(self, x=0., y=0., v=10., psi=0., L=2.5):
        self.x, self.y, self.v, self.psi, self.L = x, y, v, psi, L

    def step(self, a, delta, dt):
        """Simple kinematic bicycle model"""
        self.x += self.v * np.cos(self.psi) * dt
        self.y += self.v * np.sin(self.psi) * dt
        self.psi += (self.v / self.L) * np.tan(delta) * dt
        self.v += a * dt

    def state(self):
        return dict(x=self.x, y=self.y, v=self.v, psi=self.psi)
