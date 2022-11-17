'''
Description: 
code: 
Author: Li Jiaxin
Date: 2022-11-11 17:12:12
LastEditors: Li Jiaxin
LastEditTime: 2022-11-17 17:28:23
'''
import numpy as np
import pandas as pd
import torch
from dataAndModel.Narx import Net
from math import sqrt
from sklearn.preprocessing import StandardScaler


class Env:
    def __init__(self):
        self.setpoint = 3.25
        self.model = torch.load(
            'dataAndModel/condition_1_model.pkl')
        self.data_path = 'dataAndModel/Condition 1.xlsx'

        self.state_auxiliary = pd.read_excel(
            self.data_path, header=0, usecols=[0, 1, 2, 3, 4, 5]).values
        self.x = pd.read_excel(self.data_path, usecols=[6]).values
        self.y = pd.read_excel(self.data_path, usecols=[7]).values

        self.state_auxiliary_scaler = StandardScaler()
        self.x_scaler = StandardScaler()
        self.y_scaler = StandardScaler()

        self.state_auxiliary = self.state_auxiliary_scaler.fit_transform(
            self.state_auxiliary)
        self.x = self.x_scaler.fit_transform(self.x)
        self.y = self.y_scaler.fit_transform(self.y)

        self.state_auxiliary = torch.from_numpy(
            self.state_auxiliary).type(torch.float32)
        self.x = torch.from_numpy(self.x).type(torch.float32)  # 人工的废酸
        self.y = torch.from_numpy(self.y).type(torch.float32)  # 人工控制下的pH

        self.y_true = self.y  # RL pH
        self.x_true = self.x  # RL 废酸

        self.noise = OUNoise()  # 初始化一个噪音

    def reset(self):
        # state_init = self.inverse(self.y[3])-self.setpoint
        state_init = torch.cat([self.state_auxiliary[3], self.y_true[3]], 0)
        action_init = self.x_true[2]
        return state_init, action_init

    def inverse(self, y):
        return y * sqrt(self.y_scaler.var_) + self.y_scaler.mean_ 

    def step(self, action, state_idx, is_train):
        if is_train:
            action = self.noise.get_action(
                (action.detach().numpy() + 1) * 45, state_idx)
        else:
            action = (action + 1) * 45
        self.x_true[state_idx] = (
            action-self.x_scaler.mean_)/sqrt(self.x_scaler.var_)

        current_input = torch.cat([self.state_auxiliary[state_idx-3], self.x_true[state_idx-3], self.y_true[state_idx-3], self.state_auxiliary[state_idx-2], self.x_true[state_idx-2], self.y_true[state_idx-2],
                                   self.state_auxiliary[state_idx-1], self.x_true[state_idx-1], self.y_true[state_idx-1], self.state_auxiliary[state_idx], self.x_true[state_idx], self.y_true[state_idx]], 0)

        # 下一时刻输出预测
        state_idx += 1
        self.y[state_idx] = self.model(current_input).detach()
        # next_state = self.inverse(self.y[state_idx]) - self.setpoint
        next_state = torch.cat(
            [self.state_auxiliary[state_idx], self.y_true[state_idx]], 0)
        reward = self.calculate_reward(state_idx).type(torch.float32)

        return next_state, reward

    def get_result(self):
        return self.inverse(self.y_true), self.x_true * sqrt(self.x_scaler.var_) + self.x_scaler.mean_ 

    def calculate_reward(self, state_idx):
        dy = self.y_true[state_idx] * \
            sqrt(self.y_scaler.var_) + self.y_scaler.mean_ - self.setpoint
        return -dy*dy


class NormalizedActions():
    ''' 将action范围重定在[0.1]之间
    '''

    def action(self, action):

        low_bound = torch.tensor([0.0])
        upper_bound = torch.tensor([90.0])
        action = low_bound + (action + 1.0) * 0.5 * (upper_bound - low_bound)
        action = torch.clamp(action, low_bound, upper_bound)

        return action

    def reverse_action(self, action):
        low_bound = torch.tensor([0.0])
        upper_bound = torch.tensor([1.0])
        action = 2 * (action - low_bound) / (upper_bound - low_bound) - 1
        action = torch.clamp(action, low_bound, upper_bound)
        return action


class OUNoise(object):
    '''Ornstein–Uhlenbeck噪声
    '''

    def __init__(self, mu=0.0, theta=0.15, max_sigma=0.3, min_sigma=0.3, decay_period=100000):
        self.mu = mu  # OU噪声的参数
        self.theta = theta  # OU噪声的参数
        self.sigma = max_sigma  # OU噪声的参数
        self.max_sigma = max_sigma
        self.min_sigma = min_sigma
        self.decay_period = decay_period
        self.action_dim = 1
        self.low = 0
        self.high = 90
        self.reset()

    def reset(self):
        self.obs = np.ones(self.action_dim) * self.mu

    def evolve_obs(self):  # 迭代，上一时刻的噪声和这一时刻的噪声有关系
        x = self.obs  # np.random.randn(self.action_dim) 标准正太分布N(0,1)
        dx = self.theta * (self.mu - x) + self.sigma * \
            np.random.randn(self.action_dim)
        self.obs = x + dx
        return self.obs

    def get_action(self, action, t=0):
        ou_obs = self.evolve_obs()
        self.sigma = self.max_sigma - \
            (self.max_sigma - self.min_sigma) * \
            min(1.0, t / self.decay_period)  # sigma会逐渐衰减
        # 动作加上噪声后进行剪切
        return torch.from_numpy(np.clip(action + ou_obs, self.low, self.high)).type(torch.float32)


if __name__ == "__main__":
    env = Env()
