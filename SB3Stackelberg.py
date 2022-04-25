import gym
from datetime import datetime
from stable_baselines3 import PPO

start_time = datetime.now().replace(microsecond=0)
log_dir = "runs/stable_baseline/test/SPMs"+str(start_time)

env = gym.make('Stackelberg-v0')
model = PPO('MlpPolicy', env, verbose=1)
model.learn(total_timesteps=3e6)

obs = env.reset()
for i in range(100):
    action, _state = model.predict(obs, deterministic=False)
    obs, reward, done, info = env.step(action)
    if done:
        socialwelfare = env.socialwelfare
        print("Episode:",i,"Socialwelfare:",socialwelfare)
        obs = env.reset()