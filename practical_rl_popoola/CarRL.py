#import Dependencies
import gym
import highway_env
import numpy as np
!pip install stable-baselines
from stable_baselines import HER, SAC, PPO2, DQN
#!pip install stable-baselines


#Roundabout
env = gym.make('roundabout-v0')
model =PPO2('MlpPolicy', env, verbose=1)

for i in range(10):
	done = False
	env.reset()
	while not done:
		env.render()
		action = env.action_space.sample()
		next_state, reward, done, info = env.step(action)
		print(info)
		print(done)
	env.close()

#Create Model
model = PPO2('MlpPolicy', env, verbose=1)
model.learn(total_timesteps=10000)


#Save and load model
model.save('roundabout')
del model
model = PPO2.load('roundabout')

#Visualize Model
for i in range(10):
	done = False
	obs = env.reset()
	while not done:
		action, _states = model.predict(obs)

		next_state, reward, done, info=env.step(action)
		print(info)
		print(done)
env.close()



