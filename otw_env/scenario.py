from otw_env.vehicle import Vehicle

class CutInScenario:
    """Ego on main lane, NPC cuts in from side."""
    def __init__(self, dt=0.1):
        self.dt = dt
        self.ego = Vehicle(y=0.)
        self.npc = Vehicle(x=30., y=3., v=10.)
        self.time = 0.0
        self.cut_in_start = 20.0
        self.done = False

    def step(self, action):
        a, delta = action
        self.ego.step(a, delta, self.dt)
        if self.time > self.cut_in_start:
            self.npc.y -= 0.2
        self.npc.step(0, 0, self.dt)
        self.time += self.dt
        collision = self.check_collision()
        return self.observe(), self.reward(collision), collision

    def observe(self):
        dx = self.npc.x - self.ego.x
        dy = self.npc.y - self.ego.y
        return dict(ego=self.ego.state(), npc=self.npc.state(), dx=dx, dy=dy)

    def check_collision(self):
        dx = abs(self.ego.x - self.npc.x)
        dy = abs(self.ego.y - self.npc.y)
        return dx < 2.5 and dy < 1.0

    def reward(self, collision):
        r = self.ego.v
        if collision:
            r -= 100
        return r
