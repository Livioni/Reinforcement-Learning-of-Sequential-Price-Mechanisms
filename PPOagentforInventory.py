from datetime import datetime
import gym,os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical, MultivariateNormal
from torch.utils.tensorboard import SummaryWriter
start_time = datetime.now().replace(microsecond=0)
log_dir = "runs/InventoryEnv_"+str(start_time)
writer = SummaryWriter(log_dir=log_dir,comment='InventoryEnv Reward Record')
print("============================================================================================")
####### initialize environment hyperparameters ######
env_name = "InventoryEnv-v0"  # 定义自己的环境名称
max_ep_len = 100  # max timesteps in one episode
max_training_timesteps = int(3e6)  # break training loop if timeteps > max_training_timesteps
print_freq = max_ep_len * 2  # print avg reward in the interval (in num timesteps)
save_model_freq = int(1e4)  # save model frequency (in num timesteps)

#####################################################

## Note : print frequencies should be > than max_ep_len

################ PPO hyperparameters ################
update_timestep = max_ep_len * 10  # update policy every n timesteps
K_epochs = 80  # update policy for K epochs in one PPO update
eps_clip = 0.2  # clip parameter for PPO
gamma = 0.99  # discount factor
lr_actor = 0.0003  # learning rate for actor network
lr_critic = 0.001  # learning rate for critic network
#####################################################
print("training environment name : " + env_name)
env = gym.make(env_name).unwrapped
state_dim,action_dim = env.observation_space.shape[0],env.action_space.shape[0]
################### checkpointing ###################
run_num_pretrained = 'Inventory20-10#test_2'  #### change this to prevent overwriting weights in same env_name folder
directory = "runs/PPO_preTrained"
if not os.path.exists(directory):
    os.makedirs(directory)
directory = directory + '/' + env_name + '/'
if not os.path.exists(directory):
    os.makedirs(directory)
checkpoint_path = directory + "PPO_{}_{}.pth".format(env_name, run_num_pretrained)
print("save checkpoint path : " + checkpoint_path)
#####################################################
############# print all hyperparameters #############

print("--------------------------------------------------------------------------------------------")
print("最大步数 : ", max_training_timesteps)
print("每一幕的最大步数 : ", max_ep_len)
print("模型保存频率 : " + str(save_model_freq) + " timesteps")
print("printing average reward over episodes in last : " + str(print_freq) + " timesteps")
print("--------------------------------------------------------------------------------------------")
print("状态空间维数 : ", state_dim)
print("动作空间维数 : ", action_dim)
print("--------------------------------------------------------------------------------------------")
print("初始化离散动作空间策略")
print("--------------------------------------------------------------------------------------------")
print("PPO 更新频率 : " + str(update_timestep) + " timesteps")
print("PPO K epochs : ", K_epochs)
print("PPO epsilon clip : ", eps_clip)
print("discount factor (gamma) : ", gamma)
print("--------------------------------------------------------------------------------------------")
print("optimizer learning rate actor : ", lr_actor)
print("optimizer learning rate critic : ", lr_critic)
#####################################################

print("============================================================================================")
################################## set device ##################################
print("============================================================================================")
# set device to cpu or cuda
device = torch.device('cpu')
if (torch.cuda.is_available()):
    device = torch.device('cuda:0')
    torch.cuda.empty_cache()
    print("Device set to : " + str(torch.cuda.get_device_name(device)))
else:
    print("Device set to : cpu")
print("============================================================================================")
#########################5######### PPO Policy ##################################
class RolloutBuffer:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []

    def clear(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]


class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, agent_num, item_num):
        super(ActorCritic, self).__init__()

        self.actor = nn.Sequential(
            nn.Linear(state_dim, 64*4),
            nn.Tanh(),
            nn.Linear(64*4, 64*4),
            nn.Tanh(),
            nn.Linear(64*4, action_dim),
        )

        # critic
        self.critic = nn.Sequential(
            nn.Linear(state_dim, 64*4),
            nn.Tanh(),
            nn.Linear(64*4, 64*4),
            nn.Tanh(),
            nn.Linear(64*4, 1)
        )

        self.agent_num = agent_num
        self.item_num =  item_num

    def forward(self):
        raise NotImplementedError

    def softmax(self,X):
        X_exp = X.exp() 
        partition = X_exp.sum(dim=0, keepdim=True) 
        return X_exp / partition # 这⾥应⽤了⼴播机制

    def act(self, state):
        output = self.actor(state)        
        ##################################前self.agent个状态################################## 
        output_agent = output[0:self.agent_num]
        for j in range(len(output_agent)):
            output_agent[j] = (output_agent[j]-min(output_agent))/(max(output_agent)-min(output_agent))
        rou_agents = env.rou_agents
        box = torch.zeros(self.agent_num,dtype=torch.int16)
        for i in range(len(rou_agents)):
            if rou_agents[i] == 1:
                box[i] = 1
        output_agent *= box
        agent_probs = self.softmax(output_agent)
        agent = np.argmax(agent_probs)
        action_logprob1 = agent_probs[agent.item()]
        ##################################后self.item个状态###################################
        action_mean = torch.sigmoid(output[self.agent_num:])
        self.action_var = torch.full((self.item_num,), 0.01).to(device)
        cov_mat = torch.diag(self.action_var).unsqueeze(dim=0)
        dist = MultivariateNormal(action_mean, cov_mat)
        price = dist.sample().squeeze(dim=0)
        action_logprob2 = dist.log_prob(price)
        action = torch.cat([agent.unsqueeze(0),price],dim = 0)
        action_logprob = action_logprob1 * action_logprob2
        return action.detach(), action_logprob.detach()

    def evaluate(self, state, action):
        #不能使用广播机制
        output = self.actor(state)
        ##################################前self.agent个状态################################## 
        indices = torch.tensor([i for i in range(self.agent_num)])
        action_part1 = torch.index_select(output, dim=1,index = indices) 
        outputagent = torch.sigmoid(action_part1)
        action_probs1 = torch.softmax(outputagent,dim=0)
        dist1 = Categorical(action_probs1)
        indices = torch.tensor(0)
        action_pool1 = torch.index_select(action, dim = 1, index = indices)
        action_logprob1 = dist1.log_prob(action_pool1)
        dist1_entropy = dist1.entropy()
        ##################################后self.item个状态###################################
        indices = torch.tensor([i for i in range(self.item_num)])
        action_part2 = torch.index_select(output, dim=1,index = indices) 
        action_mean = torch.sigmoid(action_part2)
        self.action_var = torch.full((self.item_num,), 0.01).to(device)
        cov_mat = torch.diag(self.action_var).unsqueeze(dim=0)
        dist2 = MultivariateNormal(action_mean, cov_mat)
        indices = torch.tensor([i for i in range(1,self.item_num+1)])
        action_pool2 = torch.index_select(action, dim = 1, index = indices)
        action_logprob2 = dist2.log_prob(action_pool2)
        dist2_entropy = dist2.entropy()
        ##################################聚合信息###################################
        action_logprobs = action_logprob1 * action_logprob2
        dist_entropy = (dist1_entropy + dist2_entropy)/2
        state_values = self.critic(state)

        return action_logprobs, state_values, dist_entropy


class PPO:
    def __init__(self, state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip,agent_num, item_num):

        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs

        self.buffer = RolloutBuffer()  # 经验池

        self.policy = ActorCritic(state_dim, action_dim,agent_num=agent_num,item_num=item_num).to(device)  # AC策略
        self.optimizer = torch.optim.Adam([
            {'params': self.policy.actor.parameters(), 'lr': lr_actor},
            {'params': self.policy.critic.parameters(), 'lr': lr_critic}
        ])

        self.policy_old = ActorCritic(state_dim, action_dim,agent_num=agent_num,item_num=item_num).to(device)  # AC策略old网络
        self.policy_old.load_state_dict(self.policy.state_dict())

        self.MseLoss = nn.MSELoss()
        self.record = 0
        
    def select_action(self, state):

        with torch.no_grad():
            state = torch.FloatTensor(state).to(device)
            action, action_logprob = self.policy_old.act(state)

        self.buffer.states.append(state)
        self.buffer.actions.append(action)
        self.buffer.logprobs.append(action_logprob)

        return action.detach().cpu().numpy().flatten()

    def update(self):

        # Monte Carlo estimate of returns
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(self.buffer.rewards), reversed(self.buffer.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)

        # Normalizing the rewards
        rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)

        # convert list to tensor
        old_states = torch.squeeze(torch.stack(self.buffer.states, dim=0)).detach().to(device)
        old_actions = torch.squeeze(torch.stack(self.buffer.actions, dim=0)).detach().to(device)
        old_logprobs = torch.squeeze(torch.stack(self.buffer.logprobs, dim=0)).detach().to(device)

        # Optimize policy for K epochs
        for _ in range(self.K_epochs):
            # Evaluating old actions and values
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)

            # match state_values tensor dimensions with rewards tensor
            state_values = torch.squeeze(state_values)

            # Finding the ratio (pi_theta / pi_theta__old)
            ratios = torch.exp(logprobs - old_logprobs.detach())

            # Finding Surrogate Loss
            advantages = rewards - state_values.detach()
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages

            # final loss of clipped objective PPO
            loss = -torch.min(surr1, surr2) + 0.5 * self.MseLoss(state_values, rewards) - 0.01 * dist_entropy

            # take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

        # Copy new weights into old policy
        self.policy_old.load_state_dict(self.policy.state_dict())

        # clear buffer
        self.buffer.clear()

    def save(self, checkpoint_path):
        torch.save(self.policy_old.state_dict(), checkpoint_path)

    def load(self, checkpoint_path):
        self.policy_old.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))
        self.policy.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))


def train():
    ################# training procedure ################
    agent_num = env.return_agent_num
    item_num = env.return_item_num
    # initialize a PPO agent
    ppo_agent = PPO(state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip, agent_num, item_num)
    # track total training time
    start_time = datetime.now().replace(microsecond=0)
    print("Started training at (GMT) : ", start_time)

    print("============================================================================================")

    if not os.path.exists(checkpoint_path):
        print('Network Initilized.')
    else:
        ppo_agent.load(checkpoint_path)
        print("PPO model has been loaded!")

    # printing and logging variables
    print_running_reward = 0
    print_running_episodes = 0

    time_step = 0
    i_episode = 0

    # training loop
    while time_step <= max_training_timesteps:

        state = env.reset()
        current_ep_reward = 0

        for t in range(1, max_ep_len + 1):

            # select action with policy
            action = ppo_agent.select_action(state)
            state, reward, done, _ = env.step(action)

            # saving reward and is_terminals
            ppo_agent.buffer.rewards.append(reward)  # 保存收益
            ppo_agent.buffer.is_terminals.append(done)  # 保存是否完成一幕

            time_step += 1
            current_ep_reward += reward

            # update PPO agent
            if time_step % update_timestep == 0:
                print('Network updating.')
                ppo_agent.update()

            # printing average reward
            if time_step % print_freq == 0:
                # print average reward till last episode
                print_avg_reward = print_running_reward / print_running_episodes
                print_avg_reward = round(print_avg_reward, 2)
                print("Episode : {} \t\t Timestep : {} \t\t Average Reward : {}".format(i_episode, time_step,
                                                                                        print_avg_reward))
                print_running_reward = 0
                print_running_episodes = 0

            # save model weights
            if time_step % save_model_freq == 0:
                print("--------------------------------------------------------------------------------------------")
                print("saving model at : " + checkpoint_path)
                ppo_agent.save(checkpoint_path)
                print("model saved")
                print("Elapsed Time  : ", datetime.now().replace(microsecond=0) - start_time)
                print("--------------------------------------------------------------------------------------------")

            # break; if the episode is over
            if done:
                socialwelfare = env.socialwelfare
                writer.add_scalar('info/PPO_SW', socialwelfare, global_step=i_episode)
                break

        print_running_reward += current_ep_reward
        print_running_episodes += 1

        i_episode += 1
    env.close()

    # print total training time
    print("============================================================================================")
    end_time = datetime.now().replace(microsecond=0)
    print("Started training at (GMT) : ", start_time)
    print("Finished training at (GMT) : ", end_time)
    print("Total training time  : ", end_time - start_time)
    print("============================================================================================")


if __name__ == '__main__':
    train()
