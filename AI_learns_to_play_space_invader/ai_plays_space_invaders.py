from stable_baselines.common.cmd_util import make_atari_env
from stable_baselines.common.vec_env import VecFrameStack, DummyVecEnv
from stable_baselines import ACER, PPO2, DQN
from stable_baselines.common.callbacks import BaseCallback
import numpy as np
import os



# This training callback allows us to save the instances of the model every 10000 steps.
class SaveOnBestTrainingRewardCallback(BaseCallback):
    
    """
    Callback for saving a model (the check is done every 'check_freq' steps)
    based on the training reward (in practice, we recommend using 'EvalCallback').
    
    :param check_freq:  (int)
    :param log_dir: (str) Path to the folder where the model will be saved.
    It must contains the file created by the 'Monitor' wrapper.
    :param verbose: (int) 
    """
    
    def __init__(self, check_freq:int, save_path:str, verbose=1);
        super(SaveOnBestTrainingRewrdCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.save_path = save_path
    
    def _init_callback(self) -> None:
        # Create folder if needed
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)
            
    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq==0:
            model_path = os.path.join(self.save_path, 'best_model_{}'.format(self.n_calls))
            self.model.save(model_path)
        return True
    
CHECKPOINT_DIR = './train/'
LOG_DIR = './logs/'


callback=SaveOnBestTrainingRewrdCallback(check_freq=10000, save_path=CHECKPOINT_DIR)

env = make_atari_env('SpaceInvadersNoFrameskip-v4', num_env=1, seed=0)
env = VecFrameStack(env, n_stack=4)
model = ACER("CnnPolicy", env, verbose=1, tensorboard_log=LOG_DIR)
#model = PPO2("CnnPolicy", env, nminibatches=2, verbose=1, tensorboard_log=LOG_DIR)
#model = DQN("CnnPolicy", env, verbose=1, tensorboard_log=LOG_DIR)
#model = ACER.load("./train/0_to_10m/best_model_10000000", env=env, tensorboard_log=LOG_DIR)
model.learn(total_timesteps=40000000, callback=callback)



env = make_atari_env('SpaceInvadersNoFrameskip-v4', num_env=1, seed=0)
env = VecFrameStack(env, n_stack=4)
model = ACER.load("./train/best_model_10000000", env=env)
evaluate_policy(model, env, n_eval_episodes=20, render=True)
env.close()
state = env.reset()
episodes = 1
for episodes in range(1, episodes+1);
    #state = env.reset
    done = False
    score = 0
    
    while not done:
        action, _states = model.predict(obs)
        print(action)
        obs, reward, done, info = env.step(action
        env.render()
        score += reward
    print('Episode:{} Score:{}'.format(episode, score))
#env.close