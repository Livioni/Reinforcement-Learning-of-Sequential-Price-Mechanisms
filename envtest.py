import gym
import numpy as np
env = gym.make("SPMsEnv-v0")
observation = env.reset()
valuation_function = env.return_valuation_function()
print(valuation_function)
# print(observation)
# action = np.array([0,0.5,0.5,0.5,0.5,0.5],dtype=np.float32)
# observation, reward, done, info = env.step(action)
# print(observation)
# print(reward)
# print(done)
# action = np.array([1,0.5,0.5,0.5,0.5,0.5],dtype=np.float32)
# observation, reward, done, info = env.step(action)
# print(observation)
# print(done)
# action = np.array([2,0.5,0.5,0.5,0.5,0.5],dtype=np.float32)
# observation, reward, done, info = env.step(action)
# print(observation)
# print(done)
# print(env.return_allocation_matrix())
# for _ in range(1000):
#   env.render()
#   action = env.action_space.sample() # your agent here (this takes random actions)
#   observation, reward, done, info = env.step(action)

#   if done:
#     observation = env.reset()
env.close()