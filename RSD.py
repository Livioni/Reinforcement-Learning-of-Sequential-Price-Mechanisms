import gym,random
from datetime import datetime
import numpy as np
from itertools import count
from torch.utils.tensorboard import SummaryWriter
start_time = datetime.now().replace(microsecond=0)
log_dir = "runs/stable_baseline/RSD1_delta=0_"+str(start_time)
writer = SummaryWriter(log_dir=log_dir,comment='SPMs Reward Record')
env = gym.make("SPMsEnv-v0")
#Random serial dictatorship, where the agents’ order is determined randomly, and prices are set to zero.
max_timesteps = int(3e6)  
state_dim,action_dim = env.observation_space.shape[0], env.action_space.shape[0]
agent_order = np.linspace(0,19,20)
price = np.zeros(5,dtype=np.float32)
timesteps = 1
i_episode = 1
socialwelfare_pool = []
while timesteps < max_timesteps:
    timesteps_in_episode = 0
    state = env.reset()
    random.shuffle(agent_order)
    for t in count():
        action_array = np.insert(price,0,agent_order[timesteps_in_episode])
        state,reward,done,info = env.step(action_array)
        timesteps += 1
        timesteps_in_episode += 1
        if done:
            socialwelfare = env.socialwelfare
            socialwelfare_pool.append(socialwelfare)
            if i_episode % 100 == 0:
                socialwelfare_average = np.mean(socialwelfare)
                writer.add_scalar('rollout/ep_rew_mean', socialwelfare_average, global_step=i_episode)
                print("Episode: ",i_episode, "social welfare : ",socialwelfare_average)
            i_episode += 1

            break
env.close()