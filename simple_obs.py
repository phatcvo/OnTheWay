from __future__ import print_function, division
from gym import wrappers
from gym.wrappers import Monitor
import gym

import OTW


def run(episodes=1):
    env = gym.make('obstacle-v1')
    env = Monitor(env, '/home/rml-phat/Documents/OTW-out', force=True)

    for _ in range(episodes):
        env.reset()
        # env.unwrapped.set_monitor(env)  # to capture in-between frames
        done = False
        while not done:
            action = env.unwrapped.dynamics.desired_action
            observation, reward, done, info = env.step(action)
            env.render()
    env.close()


if __name__ == '__main__':
    run()