from otw_env.vehicle import Vehicle
from otw_env.utils import ttc

ego = Vehicle(x=0., v=10.)
npc = Vehicle(x=30., v=8.)
dt = 0.1

for t in range(200):
    distance = npc.x - ego.x
    TTC = ttc(ego.v, npc.v, distance)
    if TTC < 2:
        a = -1  # brake
    elif TTC > 5:
        a = 1   # accelerate
    else:
        a = 0
    ego.step(a, 0.0, dt)
    npc.step(0.0, 0.0, dt)
    if t % 20 == 0:
        print(f"t={t*dt:.1f}s | dist={distance:.1f} | TTC={TTC:.2f} | v={ego.v:.1f}")
