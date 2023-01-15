# This project is about training a reinforcement learning agent to drive a car
# around racing track.

# Here we will be using racing car environment,
# To leverage this racing car environment, we will need to install swig.
# The installation of swig varies depending on the operating system been used.
# Check out the tutorials.

# For Windws OS users you will have to download the swig files, extract it
# Then add the extracted files to "path" 

# Also we will need to install two new dependencies; Box2D and pyglet
!pip install gym[box2d] pyglet

# importing open-ai gym
import gym

# importing PPO algorithm from stable_baselines3
from stable_baselines3 import PPO
from stable_baselnies3.common.vec_env import DummyVecEnv

# importing 'evaulate_policy' function from stable_baselines3
from stable_baselines3.common.evaluation import evaluate_policy
import os




# 2. 	Test Environment

# The environment we are using is 'CarRacing-v0'
environment_name = 'CarRacing-v0'
env = gym.make(environment_name)

# Resetting the environment 
env.reset()

# taking a look at the action space
env.action_space

# taking a look at the observation space
env.observation_space

# to render the environment
# This is optional because it slows down the training
# But rendering the environment gives the visual representation to see the agent in action.
env.render()

# To close the environment
env.close()


# Testing the environment by taking random actions.
episodes = 5
for episode in range(1, episodes + 1):
	state = env.reset()
	done = False
	score = 0

	while not done:
		env.render()
		action = env.action_space.sample()
		obs, reward, done, info=env.step(action)
		score += reward
	print('Episode:{} Score:{}'.format(episode, score))

# To close the environment	
env.close()

# To close the environment	
env.close()




# 3.	Train Model 

# instantiating the environment
env = gym.make(environment_name)

# Wrapping the environment in "DummyVecEnv" wrapper
env = DummyVecEnv([lambda: env])

# logpath is where we save our tensorboard log
# We can take a look at the tensorboard log to check how our model is performing
# The tutor created a folder named 'Training' in his project folder
# Inside the "Training" folder he created two folders and name them as "Logs" and "Saved Models"
# Tensorboard log is been saved inside the "Logs" folder.
# Trained Models are been saved inside the "Saved Models" folder.

log_path = os.path.join('Training', 'logs')

# Defining our model which uses PPO algorithm
# We are using PPO algorithm
# CnnPolicy(Convolutional Neural Network policy) is the policy we are using; 
# since image is the input into this model it will be better to use "CnnPolicy"
# Policy is the rule which tells an agent how to behave in an environment
# "env" that is the environment is passed as the second parameter
# "verbose = 1" because we want to log out the result of the particular model
# Then we specify our tensorboard log folder path
model = PPO('CnnPolicy', env, verbose=1, tensorboard_log = log_path)

# To train our model
# This model will train for 100000 steps in this case.
model.learn(total_timesteps=100000)



# 4. Save Model
# "ppo_path" is the path variable.
# The next line of code specify that 
# we want to save our model as 'PPO_Driving_Model' inside the 'Saved Models' folder 
# which is located inside the 'Training' folder
ppo_path = os.path.join('Training', 'Saved Models', 'PPO_Driving_Model')
model.save(ppo_path)

# Deleting the model
del model

# To reload the environment
model = PPO.load(ppo_path, env)

# Incase we have a better model we want to use we will pass in the name of the model as
# ppo_path = os.path.join('Training', 'Saved Models', 'name_of_model here')

# Using a model pretrained for a longer time. 
# Training a model for longer period of time helps in reinforcement learning.
# Incase we have a better model we want to use, named 'PPO_428k_Driving_model'
# ppo_path = os.path.join('Training', 'Saved Models', 'PPO_428k_Driving_model')
# model = PPO.load(ppo_path, env)



# 5. Evaluate and Test the environment
evaluate_policy(model, env, n_eval_episodes=10, render=True)

# To close the environment
env.close()


# 6. Alternative approach to test the environment 
# instead of using 'evaluate_policy' method we can also use the approach below.
episodes = 5
for episode in range(1, episodes + 1):
	obs = env.reset()
	done = False
	score = 0

	while not done:
		env.render()
		action = model.predict() # Now using model here
		obs, reward, done, info=env.step(action)
		score += reward
	print('Episode:{} Score:{}'.format(episode, score))
env.close()






