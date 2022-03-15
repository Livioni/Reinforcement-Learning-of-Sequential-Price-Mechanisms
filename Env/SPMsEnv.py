import math
from turtle import done
import numpy as np
import gym,random
from gym import spaces
from gym.utils import seeding
from torch import rand

class SPMsEnv(gym.Env):
    metadata = {"render.modes": ["human", "rgb_array"], "video.frames_per_second": 30}

    def __init__(self):
        ## 初始化动作和状态的上下限
        self.items_num = 5     #假设item有n个
        self.agent_num = 3     #假设agent有m个
        self.min_agent = 0
        self.max_agent = 4
        self.min_price = [0]*self.items_num
        self.max_price = [1]*self.items_num
        self.delta = 0

        self.min_price.insert(0,self.min_agent)
        self.max_price.insert(0,self.max_agent)

        self.low_action = np.array(
            self.min_price, dtype=np.float32
        )
        self.max_action = np.array(
            self.max_price, dtype=np.float32
        )

        self.min_item = [0]*self.items_num
        self.max_item = [1]*self.items_num
        self.min_price.extend(self.min_item)
        self.max_price.extend(self.max_item)

        self.low_state = np.array(
            self.min_price, dtype=np.float32
        )
        self.high_state = np.array(
           self.max_price, dtype=np.float32
        )

        self.rou_agents = np.array([i for i in range(self.agent_num)],dtype=np.int16)
        self.rou_items = np.array([i for i in range(self.items_num)],dtype=np.int16)
        self.X = np.zeros([self.agent_num,self.items_num],dtype=np.int16)
        self.tao = np.zeros([1,self.agent_num],dtype=np.float32)
        self.agent = -1  #初始化观察agent
        self.price = np.ones([1,self.items_num],dtype=np.float32) 
        self.socialwelfare = 0    
        self.viewer = None

    def return_dim_info(self):
        self.state_dim = len(self.max_price)
        self.action_dim = self.agent_num+self.items_num
        return self.state_dim, self.action_dim

    def check_done(self):
        return True if len(self.rou_agents) == 0 or len(self.rou_items) == 0 else False

    def step(self, action):
        self.agent = int(action[0])
        self.price = action[1:]
        for i in range(self.items_num):
            if self.valuationF[self.agent][i] > self.price[i]:
                self.X[self.agent][i] = 1
                index = np.argwhere(self.rou_items == i)
                self.rou_items = np.delete(self.rou_items,index)
        price = np.array([self.price],dtype=np.float32)
        self.socialwelfare += self.X[self.agent].dot(self.valuationF[self.agent].T)
        self.agent = np.array([self.agent],dtype=np.int16)

        index = np.argwhere(self.rou_agents == self.agent)
        self.rou_agents = np.delete(self.rou_agents,index)
        done = self.check_done()
        if done:
            reward = self.socialwelfare 
        else: 
            reward = 0
        self.state1 = np.array([action],dtype=np.float32)
        self.state = np.hstack((self.state1.flatten(),self.X.flatten()))
        return self.state, reward, done, {}

    def reset(self):
        ##初始化信息
        self.rou_agents = np.array([i for i in range(self.agent_num)],dtype=np.int16)
        self.rou_items = np.array([i for i in range(self.items_num)],dtype=np.int16)
        self.X = np.zeros([self.agent_num,self.items_num],dtype=np.int16)
        self.tao = np.zeros([1,self.agent_num],dtype=np.float32)
        self.socialwelfare = 0 
        ##初始化效用函数
        self.delta = 0
        self.Z = random.uniform((1-self.delta)/2, (1+self.delta)/2)
        self.valuationF = np.zeros([self.agent_num,self.items_num],dtype=np.float32)
        for i in range(self.agent_num):
            for j in range(self.items_num):
                self.valuationF[i][j] = random.uniform(self.Z-(1-self.delta)/2,self.Z+(1-self.delta)/2)

        #初始化状态
        self.agent = np.array([-1],dtype=np.int16)  #初始化观察agent
        self.price = np.ones([1,self.items_num],dtype=np.float32) #初始化报价
        self.X = np.zeros([self.agent_num,self.items_num],dtype=np.int16) #初始化allocation
        self.state = np.hstack((self.agent,self.price.flatten()))
        self.state = np.hstack((self.state,self.X.flatten()))
        return self.state

    def render(self):
        pass
        return 

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None
