#!/usr/bin/env python
# coding=utf-8
'''
@Author: John
@Email: johnjim0816@gmail.com
@Date: 2020-06-10 15:27:16
@LastEditor: John
LastEditTime: 2022-11-14 19:46:03
@Discription: 
@Environment: python 3.7.7
'''
import random
import numpy as np

class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity # 经验回放的容量
        self.buffer = [] # 缓冲区
        self.position = 0 
    
    def push(self, state, action, reward, next_state):
        ''' 缓冲区是一个队列，容量超出时去掉开始存入的转移(transition)
        '''
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state)
        self.position = (self.position + 1) % self.capacity 
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size) # 随机采出小批量转移
        # batch 是一个长度为64的列表，列表的每个元素为四元组(state, action, reward, next_state, done)
        # state 为长度为64的列表，每个元素为array([s1,s2,s3,s4])概率
        # action 为长度为64的列表，每个元素为array([a1,a2])概率
        # reward 64x1的奖励
        # done 64个bool值
        state, action, reward, next_state =  zip(*batch) # 解压成状态，动作等
        return state, action, reward, next_state
    
    def __len__(self):
        ''' 返回当前存储的量
        '''
        return len(self.buffer)



