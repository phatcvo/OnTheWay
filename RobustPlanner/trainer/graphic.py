from __future__ import division, print_function
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
sns.set()


class RewardViewer(object):
    def __init__(self):
        self.rewards = []

    def display_trajectory(self, reward):
        self.rewards.append(reward)
        self.display()

    def display(self):
        plt.figure(num='Rewards')
        plt.clf()
        plt.title('Total reward')
        plt.xlabel('Episode')
        plt.ylabel('Reward')

        rewards = pd.Series(self.rewards)
        means = rewards.rolling(window=100).mean()
        plt.plot(rewards)
        plt.plot(means)
        plt.pause(0.001)
        plt.plot(block=False)

        # show Reward =========
        # self.rewards.append(reward)
        # print('Reward', self.rewards)
        # plt.figure(num='Rewards')
        # plt.clf()
        # plt.title('Total reward')
        # plt.xlabel('Episode')
        # plt.ylabel('Reward')

        # rewards = pd.Series(self.rewards)
        # means = rewards.rolling(window=100).mean()
        # plt.plot(rewards)
        # plt.plot(means)
        # plt.pause(0.001)
        # plt.plot(block=False)
        # RewardViewer.display()
        # print('env/crashed', self.config["collision_reward"] * self.vehicle.crashed, 'env/lane:', self.config["right_lane_reward"] * lane / max(len(neighbors) - 1, 1),'env/speed', self.config["high_speed_reward"] * np.clip(scaled_speed, 0, 1), 'Total reward', reward)
