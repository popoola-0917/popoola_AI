# This project is about training a reinforcement learning agent to play the breakout atari game.

# Note it will be better to create a separate virtual environment for every coding project
# Installing dependencies
!pip install stable-baselines3[extra]
!pip install gym


# If one choose to use cuda accelerator, one will have to go to pytorch website to install it
# One might need to check out some tutorials on how to do that.
# After installation you will also need to restart your kernel.

# 1. Import Dependencies
import os
import gym
# importing A2C algorithm from stable_baselines3
from stable_baselines3 import A2C

# Vectorizing the environment
# we vectorize the environment because we will be using four different environments
# in this project and this will speed up our training
from stable_baselines3.common.vec_env import VecFrameStack

# Importing atari game environment
from stable_baselines3.common.env_util import make_atari_env

# importing evaluate_policy; this will be useful while evaluating our model
from stable_baselines3.common.evaluation import evaluate_policy




# 2.	Test Environment
# Recently to use atari game environment one need to download its raw files.
# Downloading all the necessary files for 
# you to work with atari environment
# http://www.atarimania.com/roms/Roms.rar
# after downloading the ROM, one have to copy it into the directory one is working with.
# and then extract it, it gives two files: "HC ROMS" AND "ROM".
# Also extract both "HC ROMS" AND "ROM". 

# Then run the following line of code to install the ROMS.
# Note: one have to pass in the path to the ROM in place of "\ROMS\ROMS" below
!python -m atari_py.import_roms .\ROMS\ROMS


# setting up the environment
 environment_name = 'Breakout-v0'
 env = gym.make(environment_name)

 env.reset()

 # To know the action space of the environment
 env.action_space()

 # To know the observation space of the environment
 env.observation_space()


# Testing our model
# Going through a number of episode and playing "breakout" game

# setting up the number of episode we want to play
episodes = 5

# looping through each one of those episodes
for episode in range(1, episodes + 1):
	state = env.reset()
	done = False
	score = 0

	while not done:
		env.render()

		# taking random actions on the environment; playing randomly in this scenario.
		action = env.action_space.sample()
		obs, reward, done, info=env.step(action)
		score += reward
	print('Episode:{} Score:{}'.format(episode, score))


#To close the environment	
env.close()





# 3.	Vectorise Environment and Train Model
# Here we will vectorise our environment and train four different environment at thesame time.
# The aim of vectorising our environment is to speed up the training process.

# In this line of code we pass in the envinronment 'Breakout-v0' we are working with.
# Note that in the open-ai gym; there are two types of breakout environment; "Breakout-ram-v0" and "Breakout-v0"
# "Breakout-ram-v0" uses RAM as input while "Breakout-v0" uses image as input.

# Here we are using "Breakout-v0" version
env = make_atari_env('Breakout-v0', n_envs=4, seed=0)

# To stacks the environment together
env = VecFrameStack(env, n_stack=4)

# To reset the environment
env.reset()

# To render the environment
env.close()



# logpath is where we save our tensorboard log
# We can take a look at the tensorboard log to check how our model is performing
# The tutor created a folder named 'Training' in his project folder
# Inside the "Training" folder he created two folders and name them as "Logs" and "Saved Models"
# Tensorboard log is been saved inside the "Logs" folder.
# Trained Models are been saved inside the "Saved Models" folder.
log_path = os.path.join('Training', 'Logs')


# Defining our model 
# We are using A2C algorithm
# CnnPolicy(Convolutional Neural Network policy) is the policy we are using; 
# since image is the input into this model it will be better to use "CnnPolicy"
# Policy is the rule which tells an agent how to behave in an environment
# "env" that is the environment is passed as the second parameter
# "verbose = 1" because we want to log out the result of the particular model
# Then we specify our tensorboard log folder path
model = A2C('CnnPolicy', env, verbose=1, tensorboard_log=log_path)

# To train our model
# This model will train for 100000 steps in this case.
model.lean(total_timesteps=100000)



# 4.	Save and Reload Model
a2c_path = os.path.join('Training', 'Saved Models', 'A2C_Breakout_Model')
model.save(a2c_path)

# delete the model
del model

# if one has a model that performs better one can just pass in the model name as below
# a2c_path = os.path.join('Training', 'Saved Models', 'model name')

# The tutor has a better model which was trained for around 2 million times.
# He passed the model as a parameter as shown below;
# a2c_path = os.path.join('Training', 'Saved Models', 'A2C_2M_model')

# to reload the model
model = A2C.load(a2c_path, env)


# 5.	Evaluate and Test
# Remember that we train four environment together 
# But we can only evaluate only one environment at a time.

# Therefore, we will pass only one environment
env = make_atari_env('Breakout-v0', n_envs=1, seed=0)

# But we still stack the environments together.
env = VecFrameStack(env, n_stack=4)

# To evaluate the policy
evaluate_policy(model, env, n_eval_episodes=10, render=True)


# In case the where the environment is freezing, one will have to firstly save your code
# then restart the kernel.
# One will the re-import the necessary dependencies, and then re-run the code.

# One might also likes to try our RNNpolicy (Recurrent Neural Network policy) but as of times of this tutorial
# it has not been implemented in stable_baselines 3.


# To close the environment
env.close()







