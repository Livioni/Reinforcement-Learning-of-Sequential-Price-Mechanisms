import gym
from datetime import datetime
from stable_baselines3 import PPO

start_time = datetime.now().replace(microsecond=0)
log_dir = "runs/stable_baseline/Inventory/Inventory"+str(start_time)

env = gym.make('InventoryEnv-v0')
model = PPO('MlpPolicy', env, verbose=1,tensorboard_log=log_dir)
model.learn(total_timesteps=1e6)

obs = env.reset()
for i in range(100):
    action, _state = model.predict(obs, deterministic=False)
    obs, reward, done, info = env.step(action)
    if done:
        socialwelfare = env.socialwelfare
        print("Episode:",i,"Socialwelfare:",socialwelfare)
        obs = env.reset()