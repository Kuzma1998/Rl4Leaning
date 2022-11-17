'''
Description: 
code: 
Author: Li Jiaxin
Date: 2022-11-11 16:59:47
LastEditors: Li Jiaxin
LastEditTime: 2022-11-17 11:03:03
'''
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from common.model import Actor, Critic
from common.memory import ReplayBuffer


class DDPG:
    def __init__(self, state_dim, action_dim, cfg):
        self.device = cfg.device
        self.critic = Critic(state_dim, action_dim, cfg.hidden_dim).to(cfg.device)
        self.actor = Actor(state_dim, action_dim, cfg.hidden_dim).to(cfg.device)
        self.target_critic = Critic(state_dim, action_dim, cfg.hidden_dim).to(cfg.device)
        self.target_actor = Actor(state_dim, action_dim, cfg.hidden_dim).to(cfg.device)

        # 复制参数到目标网络
        for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_param.data.copy_(param.data)
        for target_param, param in zip(self.target_actor.parameters(), self.actor.parameters()):
            target_param.data.copy_(param.data)

        self.critic_optimizer = optim.Adam(
            self.critic.parameters(),  lr=cfg.critic_lr)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=cfg.actor_lr)
        self.memory = ReplayBuffer(cfg.memory_capacity)
        self.batch_size = cfg.batch_size
        self.soft_tau = cfg.soft_tau # 软更新参数
        self.gamma = cfg.gamma

    def choose_action(self, state,last_action):
        # state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        action = self.actor(state,last_action)
        return action

    def update(self):
        if len(self.memory) < self.batch_size: # 当 memory 中不满足一个批量时，不更新策略
            return
        # 从经验回放中(replay memory)中随机采样一个批量的转移(transition)
        state, action, reward, next_state = self.memory.sample(self.batch_size)
        # 转变为张量
        state = torch.tensor(np.array([item.detach().numpy() for item in list(state)])).to(self.device)

        next_state = torch.tensor(np.array([item.detach().numpy() for item in list(next_state)])).to(self.device)
        reward = torch.tensor(np.array([item.detach().numpy() for item in list(reward)])).to(self.device)
        action = torch.tensor(np.array([item.detach().numpy() for item in list(action)])).to(self.device)

       
        policy_loss = self.critic(state, self.actor(state,action)) # policy的优化只看Q函数，目的是要使Q(s,a)最大
        policy_loss = -policy_loss.mean()  
        # Q_func 优化目标，最小化TD error
        next_action = self.target_actor(next_state,action)
        target_value = self.target_critic(next_state, next_action.detach())
        expected_value = reward + self.gamma * target_value # 期望的Q值等于此刻reward+下一个状态的value
        expected_value = torch.clamp(expected_value, -torch.inf, torch.inf)

        value = self.critic(state, action) # 计算的value
        value_loss = nn.MSELoss()(value, expected_value.detach())
        # print(policy_loss,value_loss)
        
        self.actor_optimizer.zero_grad()
        policy_loss.backward()
        self.actor_optimizer.step()
        self.critic_optimizer.zero_grad()
        value_loss.backward()
        self.critic_optimizer.step()
        # 软更新
        for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - self.soft_tau) +
                param.data * self.soft_tau
            )
        for target_param, param in zip(self.target_actor.parameters(), self.actor.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - self.soft_tau) +
                param.data * self.soft_tau
            )
    def save(self,path):
        torch.save(self.actor.state_dict(), path+'checkpoint.pt')

    def load(self,path):
        self.actor.load_state_dict(torch.load(path+'checkpoint.pt')) 