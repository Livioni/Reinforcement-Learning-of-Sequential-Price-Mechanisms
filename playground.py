import math,random
import numpy as np
items_num = 5     #假设item有n个
agent_num = 3     #假设agent有m个
min_agent = 0
max_agent = 4
min_price = [0]*items_num
max_price = [1]*items_num

min_price.insert(0,min_agent)
max_price.insert(0,max_agent)

low_action = np.array(
    min_price, dtype=np.float32
)
max_action = np.array(
    max_price, dtype=np.float32
)

min_item = [0]*items_num
max_item = [1]*items_num
min_price.extend(min_item)
max_price.extend(max_item)

low_state = np.array(
    min_price, dtype=np.float32
)
high_state = np.array(
    max_price, dtype=np.float32
)

rou_agents = [i for i in range(agent_num)]
rou_items = [i for i in range(items_num)]
X = np.zeros([agent_num,items_num],dtype=np.int16)

delta = 0
Z = random.uniform((1-delta)/2, (1+delta)/2)
valuationF = np.zeros([agent_num,items_num],dtype=np.float32)
for i in range(agent_num):
    for j in range(items_num):
        valuationF[i][j] = random.uniform(Z-(1-delta)/2,Z+(1-delta)/2)
print("valuationF=",valuationF)
print(valuationF[0][0])
price = np.zeros([1,items_num],dtype=np.float32)
print(price[0])