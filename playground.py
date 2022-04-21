import random
import numpy as np
delta = 0
agent_num = 20
items_num = 5
Z = random.uniform((1-delta)/2, (1+delta)/2) 
socialwelfare = 0  #初始化social  welfare  
##初始化效用函数
valuationF = np.zeros([agent_num,items_num],dtype=np.float64)
for i in range(agent_num):
    for j in range(items_num):
        valuationF[i][j] = random.uniform(Z-(1-delta)/2,Z+(1-delta)/2)

print(valuationF)