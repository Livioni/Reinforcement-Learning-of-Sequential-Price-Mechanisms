import gym
from datetime import datetime
from stable_baselines3 import PPO
start_time = datetime.now().replace(microsecond=0)
log_dir = "runs/stable_baseline/unitdemand/SPMs"+str(start_time)
env = gym.make('SPMsEnv-v0')
from stable_baselines3.common.callbacks import BaseCallback

class TensorboardCallback(BaseCallback):
    """
    Custom callback for plotting additional values in tensorboard.
    """

    def __init__(self, verbose=0):
        super(TensorboardCallback, self).__init__(verbose)

    def _on_step(self) -> bool:
        # Log scalar value (here a random variable)
        price1 = env.record_price[0]
        price2 = env.record_price[1]
        price3 = env.record_price[2]
        price4 = env.record_price[3]
        price5 = env.record_price[4]
        self.logger.record('Prices/item pirce1', price1)
        self.logger.record('Prices/item pirce2', price2)
        self.logger.record('Prices/item pirce3', price3)
        self.logger.record('Prices/item pirce4', price4)
        self.logger.record('Prices/item pirce5', price5)
        return True


model = PPO('MlpPolicy', env, verbose=1,tensorboard_log=log_dir)
model.learn(total_timesteps=1e6,callback=TensorboardCallback())
obs = env.reset()



for i in range(100):
    action, _state = model.predict(obs, deterministic=False)
    obs, reward, done, info = env.step(action)
    if done:
        socialwelfare = env.socialwelfare
        print("Episode:",i,"Socialwelfare:",socialwelfare)
        obs = env.reset()

