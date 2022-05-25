
import gym
import numpy as np
import time
import random
# import pybullet_envs  # noqa
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.normal import Normal
from torch.utils.tensorboard import SummaryWriter
from torch.distributions.categorical import Categorical

class CategoricalMasked(Categorical):
    def __init__(self, probs=None, logits=None, validate_args=None, masks=[]):
        self.masks = masks
        if len(self.masks) == 0:
            super(CategoricalMasked, self).__init__(probs, logits, validate_args)
        else:
            self.masks = masks.type(torch.BoolTensor)
            logits = torch.where(self.masks, logits, torch.tensor(-1e8))
            super(CategoricalMasked, self).__init__(probs, logits, validate_args)

    def entropy(self):
        if len(self.masks) == 0:
            return super(CategoricalMasked, self).entropy()
        p_log_p = self.logits * self.probs
        p_log_p = torch.where(self.masks, p_log_p, torch.tensor(0.0))
        return -p_log_p.sum(-1)

class Agent(nn.Module):
    def __init__(self,env,state_size):
        super().__init__()
        self.critic = nn.Sequential(
            nn.Linear(state_size, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )
        self.actor_mean = nn.Sequential(
            nn.Linear(state_size, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 25)
        )

        self.actor_logstd = nn.Parameter(torch.zeros(env.items_num))
        self.checkpoint_path_1 = "runs/PPO_preTrained/" + "PPO_critic.pth"
        self.checkpoint_path_2 = "runs/PPO_preTrained/" + "PPO_actor.pth"

    def get_value(self, x):
        return self.critic(x)
    
    def save(self):
        torch.save(self.critic.state_dict(), self.checkpoint_path_1)
        torch.save(self.actor_mean.state_dict(), self.checkpoint_path_2)

    def load(self):
        self.critic.load_state_dict(torch.load(self.checkpoint_path_1, map_location=lambda storage, loc: storage))
        self.actor_mean.load_state_dict(torch.load(self.checkpoint_path_2, map_location=lambda storage, loc: storage))

    def get_action(self, x, action_mask):
        action_mean = self.actor_mean(x)
        #第一部分：选择agent
        logits = action_mean[0:env.agent_num]
        categorical = CategoricalMasked(logits=logits,masks=torch.tensor(action_mask))
        agent = categorical.sample()
        #第二部分：出价
        price_mean = action_mean[env.agent_num:]
        price_std = torch.exp(self.actor_logstd)
        probs = Normal(price_mean, price_std)
        price = probs.sample()
        action = torch.cat((agent.unsqueeze(0),price),dim=0)
        return action

if __name__ == "__main__":
    run_name = f"SPMs_Inference_{int(time.time())}"
    writer = SummaryWriter(f"runs/{run_name}")
    env_name = 'SPMsEnv-v0'
    max_episode = 1000
    env = gym.make(env_name)
    env = gym.wrappers.RecordEpisodeStatistics(env)
    env = gym.wrappers.ClipAction(env)
    env = gym.wrappers.NormalizeObservation(env)
    env = gym.wrappers.TransformObservation(env, lambda obs: np.clip(obs, -10, 10))
    env = gym.wrappers.NormalizeReward(env)
    env = gym.wrappers.TransformReward(env, lambda reward: np.clip(reward, -10, 10))
    state_size = env.observation_space.shape[0]
    agent = Agent(env,state_size)
    agent.load()
    state = env.reset()
    offer_time = 0
    for episode  in range(1,max_episode+1):
        for i in range(100):
            state = torch.tensor(state,dtype=torch.float32)
            action = agent.get_action(state,env.rou_agents)
            nextstate, reward, done, info = env.step(action.numpy())
            state = nextstate
            if offer_time < 24:
                writer.add_scalar("Prices/prices 1",env.record_price[0],offer_time)
                writer.add_scalar("Prices/prices 2",env.record_price[1],offer_time)
                writer.add_scalar("Prices/prices 3",env.record_price[2],offer_time)
                writer.add_scalar("Prices/prices 4",env.record_price[3],offer_time)
                writer.add_scalar("Prices/prices 5",env.record_price[4],offer_time)
                offer_time += 1
            if done:
                socialwelfare = env.socialwelfare
                print("Episode:",episode,"Socialwelfare:",socialwelfare)
                state = env.reset()
                break


                        