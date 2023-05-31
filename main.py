import gym
import OTW
import pprint
from RobustPlanner.common.factory import agent_factory
from tqdm.notebook import trange

# Create environment
env = gym.make("street-v1")
# env = gym.make("inter-v1")
# env = gym.make("Troad-v1")
# env.reset()
# env = record_videos(env)

pprint.pprint(env.config)
# Make agent
agent_config = {
    "__class__": "<class 'RobustPlanner.agents.tree_search.deterministic.DeterministicPlannerAgent'>",
    "env_preprocessors": [{"method":"simplify"}],
    "budget": 50,
    "gamma": 0.7,
}
agent = agent_factory(env, agent_config)

done = False
obs = env.reset()

# Run episode
for step in trange(env.unwrapped.config["duration"], desc="Running..."):
    action = agent.act(obs)
    obs, reward, done, info = env.step(action)
    # obs, reward, done, truncated, info = env.step(action)
    env.render()
    
# while True:
#     done = False
#     obs = env.reset()
#     while not done:
#         # Test environment
#         action = env.action_type.actions_indexes["IDLE"] 
#         obs, reward, done, info = env.step(action)
        
#         #print(obs)
#         env.render()


