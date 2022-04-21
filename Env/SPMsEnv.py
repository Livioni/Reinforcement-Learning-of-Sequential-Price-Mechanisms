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
        self.price_allocation_matrix = np.zeros([self.agent_num,self.items_num],dtype=np.float64) #价格矩阵（self.agent_num * self.item_num)维
        self.tau = np.zeros(self.agent_num,dtype=np.float64) #转账矩阵 
        self.agent = np.array(-1,dtype=np.int16)  #初始化观察agent #初始化观察agent
        self.price = np.ones(self.items_num,dtype=np.float64) #要价矩阵
        self.Z = random.uniform((1-self.delta)/2, (1+self.delta)/2) 
        self.socialwelfare = 0  #初始化social  welfare  
        ##初始化效用函数
        # self.valuationF = np.zeros([self.agent_num,self.items_num],dtype=np.float64)
        # for i in range(self.agent_num):
        #     for j in range(self.items_num):
        #         self.valuationF[i][j] = random.uniform(self.Z-(1-self.delta)/2,self.Z+(1-self.delta)/2)

        #delta = 0
        self.valuationF =np.array([[0.78900805,0.96012449,0.99099806,0.58527462,0.63666145],\
                        [0.98648185,0.55739215,0.19698906,0.68369219,0.27437320],\
                        [0.86374709,0.85091796,0.43573782,0.13482168,0.40099636],\
                        [0.58141219,0.22629741,0.66612841,0.97642836,0.79005999],\
                        [0.30114841,0.11199923,0.01076650,0.66018063,0.51939904],\
                        [0.83135732,0.50467929,0.34803428,0.23014417,0.93165713],\
                        [0.90753162,0.45139716,0.12398481,0.87917376,0.95310834],\
                        [0.15536485,0.47051726,0.36178991,0.84614371,0.27937186],\
                        [0.46667823,0.16453699,0.61319562,0.41454692,0.11260570],\
                        [0.89602795,0.06285511,0.93314658,0.97294757,0.86253819],\
                        [0.35162777,0.27674798,0.92889346,0.25404701,0.06598934],\
                        [0.20304112,0.12649533,0.10892991,0.84067924,0.33471859],\
                        [0.41421655,0.78001907,0.19546347,0.03083713,0.24251268],\
                        [0.83174977,0.05870072,0.54456963,0.35504824,0.57398383],\
                        [0.04114803,0.28719724,0.76151723,0.68865910,0.15022888],\
                        [0.39452686,0.16493265,0.86196355,0.13994046,0.35771739],\
                        [0.90833496,0.83428713,0.75482767,0.29083134,0.06442374],\
                        [0.33674271,0.28909863,0.67971812,0.01846276,0.81958546],\
                        [0.49674642,0.72062413,0.07787972,0.24753036,0.55676578],\
                        [0.73727425,0.13167262,0.73926587,0.41809112,0.55647347]],dtype=np.float32)

        self.viewer = None
        
        self.low_observation = np.zeros(self.items_num+self.items_num*self.agent_num,dtype=np.float32)
        self.high_observation = np.ones(self.items_num+self.items_num*self.agent_num,dtype=np.float32)
        self.low_action = np.zeros(self.agent_num+self.items_num,dtype=np.float32)
        self.high_action = np.ones(self.agent_num+self.items_num,dtype=np.float32)

        self.observation_space = spaces.Box(low=self.low_observation, high=self.high_observation,dtype=np.float32)
        self.action_space = spaces.Box(low=self.low_action,high=self.high_action,dtype=np.float32)

    def check_done(self):
        return True if np.all(self.rou_agents == 0) or np.all(self.rou_items == 0) else False
    
    def set_valuationF(self,valuationF):
        self.valuationF = valuationF

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

    def step(self, action):
        self.agent = int(action[0])
        self.price = action[1:]
        utility = []
        for ele in range(len(self.price)):
            self.price_allocation_matrix[self.agent][ele] = self.price[ele]
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

        for i in range(self.items_num):
            if self.valuationF[self.agent][i] > self.price[i]:
                self.allocation_matrix[self.agent][i] = 1
                self.rou_items[i] = 0
                break

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

        self.state = self.allocation_matrix[self.agent].flatten()
        self.state = np.hstack((self.state,self.price_allocation_matrix.flatten()))  
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

        self.state = self.allocation_matrix[self.agent].flatten()
        self.state = np.hstack((self.state,self.price_allocation_matrix.flatten()))
        return self.state

    def render(self):
        pass
        return 

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None
