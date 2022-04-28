import numpy as np
import gym,random
from torch import rand
import torch
from gym import spaces
from torch.distributions import Categorical

class Stackelberg(gym.Env):
    metadata = {"render.modes": ["human", "rgb_array"], "video.frames_per_second": 30}

    def __init__(self):
        ## 初始化动作和状态的上下限
        self.items_num = 1     #假设item有n个
        self.agent_num = 2     #假设agent有m个

        self.option = [0,1,2,3,4]
        self.E = 0.2
        self.e = 0.8

        self.rou_agents = np.ones(self.agent_num,dtype=np.int16)  #剩余的agent bool编码，固定self.agent_num个状态，1为还在，0为不在
        self.rou_items = np.ones(self.items_num,dtype=np.int16)   #剩余的item bool编码，固定self.item_num个状态，1为还在，0为不在  

        self.allocation_matrix = np.zeros([self.agent_num,self.items_num],dtype=np.int16) #分配矩阵 bool编码，固定（self.agent_num * self.item_num)维，1为还在，0为不在  
        self.tau = np.zeros(self.agent_num,dtype=np.float64) #转账矩阵 
        self.price = np.ones(self.items_num,dtype=np.float64) #要价矩阵
        self.utility = np.ones(self.agent_num,dtype=np.float32) #效用值
        self.weights_1 = torch.ones(len(self.option),dtype=torch.float64)  #multiplicative weights (MW) algorithm
        self.weights_2 = torch.ones(len(self.option),dtype=torch.float64)
        self.socialwelfare = 0  #初始化social  welfare  

        #估值函数概率
        self.probabilities_1 = [1-self.E,self.E]
        self.sample_list_1 = [0.5,1/(2*self.E)]
        self.probabilities_2 = [0.5,0.5]
        self.sample_list_2 = [0,1]

        #Items/agents left
        self.low_observation = np.zeros(self.agent_num,dtype=np.float32)
        self.high_observation = 4 * np.ones(self.agent_num,dtype=np.float32)
        self.low_action = np.zeros(self.agent_num+self.items_num,dtype=np.float32)
        self.high_action = np.array([1.,1.,2.5],dtype=np.float32)       

        self.observation_space = spaces.Box(low=self.low_observation, high=self.high_observation,dtype=np.float32)
        self.action_space = spaces.Box(low=self.low_action,high=self.high_action,dtype=np.float32)

        
    def return_dim_info(self):
        #总的状态数量为：当前选择的agent（2维）
        self.state_dim = self.agent_num
        #总的动作数量为：当前agent数量+items数量
        self.action_dim = self.agent_num + self.items_num
        return self.state_dim, self.action_dim

    def check_done(self):
        return True if np.all(self.rou_agents == 0) or np.all(self.rou_items == 0) else False
    
    def reset_weights(self):
        self.weights_1 = torch.ones(len(self.option),dtype=torch.float64)  #multiplicative weights (MW) algorithm
        self.weights_2 = torch.ones(len(self.option),dtype=torch.float64)

    @property
    def return_socialwelfare(self):
        return self.socialwelfare

    @property
    def return_tau_agent(self):
        return self.rou_agents

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

        for ele in range(len(self.price)):
            if self.rou_items[ele] == 1:
                continue
            else:
                self.price[ele] = 9 #如果商品已经售出则标价99 agent买不起。

        for i in range(self.items_num):
            if self.valuationF[self.agent][i] > self.price[i]:
                self.allocation_matrix[self.agent][i] = 1
                self.rou_items[i] = 0
                self.utility[self.agent] = self.valuationF[self.agent][i]-self.price[i]

        self.socialwelfare += self.allocation_matrix[self.agent]*self.valuationF[self.agent][0]
        self.tau[self.agent] = self.allocation_matrix[self.agent]*self.price.item()
        self.agent = np.array([self.agent],dtype=np.int16)
        self.rou_agents[self.agent] = 0
        done = self.check_done()
        #收益函数：如果是最后一步，则计算整个socialwelfare，如果不是最后一步，则为0；
        if done:
            reward = self.socialwelfare[0]
            self.weights_1[self.message1.item()] = self.weights_1[self.message1.item()]*(1-self.e)**-self.utility[0]
            self.weights_2[self.message2.item()] = self.weights_2[self.message2.item()]*(1-self.e)**-self.utility[1]
        else: 
            reward = 0

        self.observation = self.allocation_matrix.flatten()
        return self.observation, reward, done, {}

    def reset(self):
        ##初始化信息
        self.rou_agents = np.ones(self.agent_num, dtype=np.int16)
        self.rou_items = np.ones(self.items_num ,dtype=np.int16)
        self.allocation_matrix = np.zeros([self.agent_num,self.items_num],dtype=np.int16)
        self.tau = np.zeros(self.agent_num,dtype=np.float64)
        self.utility = np.ones(self.agent_num,dtype=np.float32)
        self.socialwelfare = 0 
        ##初始化效用函数
        self.valuationF = np.zeros([self.agent_num,self.items_num],dtype=np.float64)
        self.valuationF[0] = random.choices(self.sample_list_1, weights=self.probabilities_1, k=1)[0]
        self.valuationF[1] = random.choices(self.sample_list_2, weights=self.probabilities_2, k=1)[0]
        self.viewer = None
        #初始化状态
        dist_1 = Categorical(torch.softmax(self.weights_1,dim=0))
        dist_2 = Categorical(torch.softmax(self.weights_2,dim=0))
        self.message1 = dist_1.sample()
        self.message2 = dist_2.sample()
        self.observation = np.hstack([self.message1,self.message2])                           #1维      
        return self.observation

    def render(self):
        pass
        return 

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None
