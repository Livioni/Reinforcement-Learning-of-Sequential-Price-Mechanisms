import numpy as np
import gym,random
from gym import spaces
from gym.utils import seeding
from torch import rand
import torch

class InventoryEnv(gym.Env):
    metadata = {"render.modes": ["human", "rgb_array"], "video.frames_per_second": 30}

    def __init__(self):
        ## 初始化动作和状态的上下限
        self.items_num = 10     #假设item有n个
        self.agent_num = 20     #假设agent有m个

        self.rou_agents = np.ones(self.agent_num,dtype=np.int16)  #剩余的agent bool编码，固定self.agent_num个状态，1为还在，0为不在
        self.rou_items = np.ones(self.items_num,dtype=np.int16)   #剩余的item bool编码，固定self.item_num个状态，1为还在，0为不在  
        self.allocation_matrix = np.zeros([self.agent_num,self.items_num],dtype=np.int16) #分配矩阵 bool编码，固定（self.agent_num * self.item_num)维，1为还在，0为不在  
        self.price_allocation_matrix = np.zeros([self.agent_num,self.items_num],dtype=np.float64) #价格矩阵（self.agent_num * self.item_num)维
        self.tau = np.zeros(self.agent_num,dtype=np.float64) #转账矩阵 
        self.agent = np.array(-1,dtype=np.int16)  #初始化观察agent #初始化观察agent
        self.price = np.ones(self.items_num,dtype=np.float64) #要价矩阵
        self.socialwelfare = 0  #初始化social  welfare  
        self.valuation_set = [0.5,1]
        ##初始化效用函数
        # self.valuationF = np.zeros([self.agent_num,self.items_num],dtype=np.float64)
        # for i in range(self.agent_num):
        #     for j in range(self.items_num):
        #         self.valuationF[i][j] =  random.sample(self.valuation_set, 1)[0]

        self.valuationF = np.array([[1.,1.,0.5,0.5,0.5,1.,0.5,0.5,1.,1.],\
                                    [0.5,1.,1.,1.,0.5,0.5,1.,1.,0.5,0.5],\
                                    [1.,1.,1.,1.,1.,0.5,1.,1.,1.,0.5],\
                                    [0.5,0.5,0.5,0.5,0.5,0.5,0.5,1.,0.5,0.5],\
                                    [1.,1.,1.,1.,0.5,1.,0.5,1.,1.,0.5],\
                                    [0.5,1.,0.5,0.5,0.5,0.5,1.,0.5,0.5,1.],\
                                    [0.5,1.,1.,0.5,1.,1.,1.,1.,1.,1.,],\
                                    [0.5,0.5,0.5,1.,0.5,0.5,0.5,1.,0.5,1.],\
                                    [0.5,1.,0.5,0.5,0.5,1.,0.5,0.5,1.,0.5],\
                                    [1.,1.,1.,0.5,0.5,0.5,1.,0.5,1.,1.],\
                                    [0.5,0.5,0.5,0.5,1.,1.,1.,0.5,1.,1.],\
                                    [1.,0.5,0.5,1.,1.,0.5,1.,0.5,1.,1.],\
                                    [0.5,0.5,0.5,0.5,0.5,0.5,1.,1.,1.,1.],\
                                    [0.5,1.,0.5,0.5,1.,0.5,1.,1.,0.5,1.],\
                                    [0.5,0.5,0.5,1.,0.5,1.,0.5,1.,0.5,1.],\
                                    [0.5,0.5,0.5,0.5,1.,1.,1.,1.,1.,0.5],\
                                    [1.,0.5,1.,0.5,0.5,0.5,0.5,0.5,1.,0.5],\
                                    [1.,0.5,0.5,1.,0.5,1.,0.5,0.5,0.5,1.],\
                                    [0.5,0.5,1.,1.,1.,0.5,1.,0.5,1.,1.],\
                                    [0.5,1.,1.,1.,1.,0.5,0.5,1.,1.,0.5]], dtype= np.float32)

        self.viewer = None

        # #Items/agents left
        self.low_observation = np.zeros(self.agent_num + self.items_num,dtype=np.float32)
        self.high_observation = np.ones(self.agent_num + self.items_num,dtype=np.float32)
        self.low_action = np.zeros(self.agent_num+self.items_num,dtype=np.float32)
        self.high_action = np.ones(self.agent_num+self.items_num,dtype=np.float32)       

        # Allocation matrix
        # self.low_observation = np.zeros(self.agent_num + self.items_num + self.items_num,dtype=np.float32)
        # self.high_observation = np.ones(self.agent_num + self.items_num + self.items_num,dtype=np.float32)
        # self.low_action = np.zeros(self.agent_num+self.items_num,dtype=np.float32)
        # self.high_action = np.ones(self.agent_num+self.items_num,dtype=np.float32)

        # Price-allocation matrix
        # self.low_observation = np.zeros(self.agent_num + self.items_num + self.items_num + self.items_num*self.agent_num,dtype=np.float32)
        # self.high_observation = np.ones(self.agent_num + self.items_num + self.items_num + self.items_num*self.agent_num,dtype=np.float32)
        # self.low_action = np.zeros(self.agent_num+self.items_num,dtype=np.float32)
        # self.high_action = np.ones(self.agent_num+self.items_num,dtype=np.float32)

        self.observation_space = spaces.Box(low=self.low_observation, high=self.high_observation,dtype=np.float32)
        self.action_space = spaces.Box(low=self.low_action,high=self.high_action,dtype=np.float32)

    def check_done(self):
        return True if np.all(self.rou_agents == 0) or np.all(self.rou_items == 0) else False
    
    def set_valuationF(self,valuationF):
        self.valuationF = valuationF

    @property
    def return_allocation_matrix(self):
        return self.allocation_matrix       

    @property
    def return_item_num(self):
        return  self.items_num

    @property
    def return_agent_num(self):
        return  self.agent_num
        
    @property
    def return_valuation_function(self):
        return  self.valuationF
        
    def softmax(self,X):
        X_exp = X.exp() 
        partition = X_exp.sum(dim=0, keepdim=True) 
        return X_exp / partition # 这⾥应⽤了⼴播机制

    def step(self, action):
        output_agent = action[0:self.agent_num]
        for j in range(len(output_agent)):
            output_agent[j] = (output_agent[j]-min(output_agent))/(max(output_agent)-min(output_agent))
        agent_probs = self.softmax(torch.tensor(output_agent))
        value,indices = agent_probs.sort(descending=True)
        agent_ = 0
        while self.rou_agents[indices[agent_].item()] == 0:
            agent_ += 1
        self.agent = indices[agent_].item()
        self.price = action[self.agent_num:]
        utility = []
        for ele in range(len(self.price)):
            # self.price_allocation_matrix[self.agent][ele] = self.price[ele]
            if self.rou_items[ele] == 1:
                continue
            else:
                self.price[ele] = 9 #如果商品已经售出则标价99 agent买不起。

        # for i in range(self.items_num):
        #     utility.append(self.valuationF[self.agent][i] - self.price[i])

        # max_utility = max(utility)
        # if max_utility > 0:
        #     max_index = utility.index(max_utility)
        #     self.allocation_matrix[self.agent][max_index] = 1
        #     self.rou_items[max_index] = 0
        #     self.price_allocation_matrix[self.agent][max_index] = self.price[max_index]


        # unit_demand
        for i in range(self.items_num):
            if self.valuationF[self.agent][i] > self.price[i]:
                self.allocation_matrix[self.agent][i] = 1
                self.rou_items[i] = 0

        self.socialwelfare += max(self.allocation_matrix[self.agent]*(self.valuationF[self.agent].T))
        self.tau[self.agent] = max(self.allocation_matrix[self.agent]*(self.price.T))
        self.agent = np.array([self.agent],dtype=np.int16)

        self.rou_agents[self.agent] = 0
        done = self.check_done()
        #收益函数：如果是最后一步，则计算整个socialwelfare，如果不是最后一步，则为0；
        if done:
            reward = self.socialwelfare 
        else: 
            reward = 0

        # #Items/agents left
        self.state = self.rou_agents
        self.state = np.hstack((self.state,self.rou_items))

        # Allocation matrix
        # self.state = self.rou_agents
        # self.state = np.hstack((self.state,self.rou_items))
        # self.state = np.hstack((self.state,self.allocation_matrix[self.agent].flatten()))

        # # Price-allocation matrix        
        # self.state = self.rou_agents
        # self.state = np.hstack((self.state,self.rou_items))
        # self.state = np.hstack((self.state,self.allocation_matrix[self.agent].flatten()))
        # self.state = np.hstack((self.state,self.price_allocation_matrix.flatten()))
        return self.state, reward, done, {}

    def reset(self):
        ##初始化信息
        self.rou_agents = np.ones(self.agent_num, dtype=np.int16)
        self.rou_items = np.ones(self.items_num ,dtype=np.int16)
        self.allocation_matrix = np.zeros([self.agent_num,self.items_num],dtype=np.int16)
        self.tau = np.zeros(self.agent_num,dtype=np.float64)
        self.socialwelfare = 0 

        #初始化状态
        self.agent = np.array(0,dtype=np.int16)  #初始化观察agent
        self.price = np.ones(self.items_num,dtype=np.float64) #初始化报价
        self.allocation_matrix = np.zeros([self.agent_num,self.items_num],dtype=np.int16) #初始化allocation 20*5维
        self.price_allocation_matrix = np.zeros([self.agent_num,self.items_num],dtype=np.float64)

        # #Items/agents left
        self.state = self.rou_agents
        self.state = np.hstack((self.state,self.rou_items))

        # # Allocation matrix
        # self.state = self.rou_agents
        # self.state = np.hstack((self.state,self.rou_items))
        # self.state = np.hstack((self.state,self.allocation_matrix[self.agent].flatten()))

        # Price-allocation matrix
        # self.state = self.rou_agents
        # self.state = np.hstack((self.state,self.rou_items))
        # self.state = np.hstack((self.state,self.allocation_matrix[self.agent].flatten()))
        # self.state = np.hstack((self.state,self.price_allocation_matrix.flatten()))
        return self.state

    def render(self):
        pass
        return 

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None
