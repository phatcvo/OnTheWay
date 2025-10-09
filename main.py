import gymnasium as gym
import OTW
import pprint

# Create environment
env = gym.make("street-v1")
# env = gym.make("inter-v1")
# env = gym.make("Troad-v1")
env.reset()
pprint.pprint(env.unwrapped.config)


while True:
    done = False
    obs = env.reset()
    while not done:
        # Test environment
        action = env.unwrapped.action_type.actions_indexes["IDLE"]
        obs, reward, done, info = env.step(action)
        
        #print(obs)
        env.render()


