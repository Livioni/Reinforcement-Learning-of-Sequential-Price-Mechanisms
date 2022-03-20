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
        self.agent_num = 20     #假设agent有m个
        self.delta = 0         #调整生成agent的valuation function的correlations

        self.rou_agents = np.ones(self.agent_num,dtype=np.int16)  #剩余的agent bool编码，固定self.agent_num个状态，1为还在，0为不在
        self.rou_items = np.ones(self.items_num,dtype=np.int16)   #剩余的item bool编码，固定self.item_num个状态，1为还在，0为不在  
        self.allocation_matrix = np.zeros([self.agent_num,self.items_num],dtype=np.int16) #分配矩阵 bool编码，固定（self.agent_num * self.item_num)维，1为还在，0为不在  
        self.tau = np.zeros(self.agent_num,dtype=np.float32) #转账矩阵 
        self.agent = np.array(-1,dtype=np.int16)  #初始化观察agent #初始化观察agent
        self.price = np.ones(self.items_num,dtype=np.float32) #要价矩阵
        self.Z = random.uniform((1-self.delta)/2, (1+self.delta)/2)
        self.socialwelfare = 0    
        self.viewer = None

    def return_dim_info(self):
        #总的状态数量为：当前选择的agent（1维）+物品报价（self.item_num维）+当前选择agent的分配矩阵（self.item_num维）+剩余的agent编码（self.agent_num维）+剩余的item编码（self.item_num维）
        self.state_dim = 1 + self.items_num + self.items_num + self.agent_num + self.items_num
        self.action_dim = self.agent_num+self.items_num
        return self.state_dim, self.action_dim

    def check_done(self):
        return True if np.all(self.rou_agents == 0) or np.all(self.rou_items == 0) else False

    def return_socialwelfare(self):
        return self.socialwelfare

    def return_tau_agent(self):
        return self.rou_agents

    def return_allocation_matrix(self):
        return self.allocation_matrix       

    def return_item_num(self):
        return  self.items_num
    
    def return_agent_num(self):
        return  self.agent_num

    def return_valuation_function(self):
        return  self.valuationF

    def step(self, action):
        self.agent = int(action[0])
        self.price = action[1:]
        for ele in range(len(self.price)):
            if self.rou_items[ele] == 1:
                continue
            else:
                self.price[ele] = 99 #如果商品已经售出则标价99 agent买不起。

        for i in range(self.items_num):
            if self.valuationF[self.agent][i] > self.price[i]:
                self.allocation_matrix[self.agent][i] = 1
                self.rou_items[i] = 0

        self.socialwelfare += self.allocation_matrix[self.agent].dot(self.valuationF[self.agent].T)
        self.tau[self.agent] = self.allocation_matrix[self.agent].dot(self.price.T)
        self.agent = np.array([self.agent],dtype=np.int16)

        self.rou_agents[self.agent] = 0
        done = self.check_done()
        #收益函数：如果是最后一步，则计算整个socialwelfare，如果不是最后一步，则为0；
        if done:
            reward = self.socialwelfare 
        else: 
            reward = 0

        self.state1 = np.array([action],dtype=np.float32)
        self.state = np.hstack((self.state1.flatten(),self.allocation_matrix[self.agent].flatten()))  
        self.state = np.hstack((self.state,self.rou_agents.flatten()))    
        self.state = np.hstack((self.state,self.rou_items.flatten()))    
        return self.state, reward, done, {}

    def reset(self):
        ##初始化信息
        self.rou_agents = np.ones(self.agent_num, dtype=np.int16)
        self.rou_items = np.ones(self.items_num ,dtype=np.int16)
        self.allocation_matrix = np.zeros([self.agent_num,self.items_num],dtype=np.int16)
        self.tau = np.zeros(self.agent_num,dtype=np.float32)
        self.socialwelfare = 0 
        ##初始化效用函数
        self.delta = 0
        self.valuationF = np.zeros([self.agent_num,self.items_num],dtype=np.float32)
        for i in range(self.agent_num):
            for j in range(self.items_num):
                self.valuationF[i][j] = random.uniform(self.Z-(1-self.delta)/2,self.Z+(1-self.delta)/2)
        #初始化状态
        self.agent = np.array(-1,dtype=np.int16)  #初始化观察agent
        self.price = np.ones(self.items_num,dtype=np.float32) #初始化报价
        self.allocation_matrix = np.zeros([self.agent_num,self.items_num],dtype=np.int16) #初始化allocation
        #状态表示：【当前选择的agent，报价1，报价2，报价3，报价4，报价5，当前agent是否选择物品1，
        #当前agent是否选择物品2，当前agent是否选择物品3，当前agent是否选择物品4，当前agent是否选择物品5，
        #agent1是否未访问，agent2是否未访问，agent3是否未访问，item1是否还在，item2是否还在，item3是否还在，
        #item4是否还在，item5是否还在】
        self.state = np.hstack((self.agent,self.price.flatten()))
        self.state = np.hstack((self.state,self.allocation_matrix[self.agent].flatten()))    
        self.state = np.hstack((self.state,self.rou_agents.flatten()))    
        self.state = np.hstack((self.state,self.rou_items.flatten()))           
        return self.state

    def render(self):
        pass
        return 

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None
