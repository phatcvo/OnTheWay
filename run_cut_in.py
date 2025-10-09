from otw_env.scenario import CutInScenario
from otw_env.logger import Logger

env = CutInScenario(dt=0.1)
log = Logger("logs_cutin.json")

for t in range(400):
    obs, reward, collision = env.step((0.0, 0.0))
    log.log(env.time, {"reward": reward, "collision": collision, **obs["ego"]})
    if collision:
        print(f"🚨 Collision at t={env.time:.1f}s, x={obs['ego']['x']:.1f}")
        break
log.save()
print("✅ Log saved: logs_cutin.json")
