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
        self.valuationF = np.array([[0.78900805,0.96012449,0.99099806,0.58527462,0.63666145],\
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

        #delta = 0.34
        # self.valuationF=np.array([[0.28746766,0.11773536,0.06417870,0.21074364,0.47814521],\
        #                 [0.52800661,0.19562626,0.24214131,0.63061773,0.58011432],\
        #                 [0.25686938,0.27776415,0.07730791,0.40525655,0.43227341],\
        #                 [0.60449626,0.36269815,0.21132956,0.47090524,0.40805888],\
        #                 [0.31917121,0.17862318,0.22754560,0.36751298,0.19221779],\
        #                 [0.19203603,0.24428465,0.16879044,0.36700307,0.08487778],\
        #                 [0.59771820,0.26470922,0.52927054,0.41799680,0.20547174],\
        #                 [0.38280490,0.47394948,0.17736245,0.63987204,0.45280828],\
        #                 [0.22674475,0.17646293,0.20397242,0.67082954,0.05140794],\
        #                 [0.45652455,0.50693406,0.59523298,0.07084946,0.13145058],\
        #                 [0.43195991,0.05722680,0.31895462,0.32064159,0.33700103],\
        #                 [0.66470028,0.33526021,0.60525721,0.69022206,0.56940958],\
        #                 [0.04651746,0.20159853,0.70205233,0.09177878,0.63128829],\
        #                 [0.52686478,0.40227233,0.36102621,0.67907867,0.37154088],\
        #                 [0.20104286,0.33263745,0.55911495,0.48018483,0.16943506],\
        #                 [0.62039231,0.41561842,0.09501664,0.16722161,0.57961700],\
        #                 [0.69621466,0.37048358,0.38992990,0.65110436,0.66278520],\
        #                 [0.31267322,0.66777534,0.15991525,0.37678061,0.68928265],\
        #                 [0.61405565,0.07927069,0.37009645,0.28577439,0.63793179],\
        #                 [0.54774761,0.36121875,0.46010207,0.22939186,0.46555167]],dtype=np.float32)

        self.viewer = None

        #Items/agents left
        self.low_observation = np.zeros(self.agent_num + self.items_num,dtype=np.float32)
        self.high_observation = np.ones(self.agent_num + self.items_num,dtype=np.float32)
        self.low_action = np.zeros(self.agent_num+self.items_num,dtype=np.float32)
        self.high_action = np.ones(self.agent_num+self.items_num,dtype=np.float32)       

        # Allocation matrix
        # self.low_observation = np.zeros(self.agent_num + self.items_num + self.items_num,dtype=np.float32)
        # self.high_observation = np.ones(self.agent_num + self.items_num + self.items_num,dtype=np.float32)
        # self.low_action = np.zeros(self.agent_num+self.items_num,dtype=np.float32)
        # self.high_action = np.ones(self.agent_num+self.items_num,dtype=np.float32)

        # # Price-allocation matrix
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

        for i in range(self.items_num):
            utility.append(self.valuationF[self.agent][i] - self.price[i])

        max_utility = max(utility)
        if max_utility > 0:
            max_index = utility.index(max_utility)
            self.allocation_matrix[self.agent][max_index] = 1
            self.rou_items[max_index] = 0
            # self.price_allocation_matrix[self.agent][max_index] = self.price[max_index]

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

        #Items/agents left
        self.state = self.rou_agents
        self.state = np.hstack((self.state,self.rou_items))

        # # Allocation matrix
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

        #Items/agents left
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
        return self.state

    def render(self):
        pass
        return 

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None
