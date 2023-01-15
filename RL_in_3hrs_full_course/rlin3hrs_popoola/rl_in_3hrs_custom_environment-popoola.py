# This project is about building a shower environment to get the right temperature every time.
# Note it will be better to create a separate virtual environment for every coding project

# Installing dependencies
!pip install stable-baselines3[extra]
!pip install gym


#1. Import open ai gym dependencies
import gym
from gym import Env  


# Discrete, Box, Dict, Tuple, MultiBinary, MultiDiscrete
# represent different types of spaces available in open ai gym
from gym.spaces import Discrete, Box, Dict, Tuple, MultiBinary, MultiDiscrete

# Import helpers
import numpy as np
import random
import os

# Import Stable baselines stuffs
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy



# 2. Types of Spaces
# Here we will check out the workings of different spaces
# available in open ai gym.

# We use "Discrete" space when we have discrete values
# We want three different discrete actions
Discrete(3)

# To take a look at different values in the discrete space.
Discrete(3).sample()

# Box(lower value, upper value, "shape of the output that is 3x3 array in this case")
# We use "Box" space when we have continous values
Box(0,1, shape=(3,3))

# To take a look at different values in the Box space.
Box(0,1, shape=(3,3)).sample()


# Tuple space allows us to combine different spaces.
Tuple((Discrete(3),Box(0,1, shape=(3,3))
Tuple((Discrete(3),Box(0,1, shape=(3,3))).sample()
# one can also add multiple space to a tuple
Tuple((Discrete(3),Box(0,1, shape=(3,3)), MultiBinary(4)).sample()



# Dict space allows us to combine different spaces as in dictionary.
# This dictionary contains two keys; height and speed
# 'height' is set to Discrete(2),
# 'speed'  is set to Box(0,100,shape=(1,))
# In Box(0,100,shape=(1,)) we pass three argument in the form of 
# Box(lower value, upper value, "shape of the output")
Dict({'height':Discrete(2), "speed":Box(0,100,shape=(1,))}).sample()
# one can also add multiple space to a tuple
Dict({'height':Discrete(2), "speed":Box(0,100,shape=(1,)), "color":MultiBinary(4)}).sample()

# MultiBinary space giving us four positions.
MultiBinary(4).sample()

# MultiDiscrete space
MultiDiscrete([5,2,2]).sample()






# 3. Building an Environment

# Building an agent to give us the best shower possible
# randomly temperature
# Human body optimal temperature is between 37 and 39 degrees
# Therefore we will like to build an agent that automatically respond to different temperature
# Then based on the temperature it will keep the shower temperature between 37 and 39 degrees.



# The four key functions we will need in our shower environment class;  
# __init__ function, step function, render function, reset function


class ShowerEnv(Env):

	# __init__ function triggers when we create our class

	def __init__(self):

		# Defining our action space to be Discrete(3)
		# The three actions that we are having are;
		# Turning the tap up
		# Turning the tap down
		# Leaving the tap unchanged

		# We can also define our action space to be in Box() space
		# But to keep it simple, we use Discrete  space.
		self.action_space = Discrete(3)

		# We set our observation space is in form of Box() space
		self.observation_space = Box(low=np.array([0])), high=np.array([100])

		# Initial state will be set to 38
		# Note that the aim of our agent is to keep the showers
		# temperature between 37 and 39 degrees based on temperature changes
		self.state = 38 + random.randint(-3,3)

		# shower length is 60 seconds
		self.shower_length= 60
		pass

	# step function; here we pass in 'action' as an argument 
	# which makes our agent to take some action
	# Inside this step function we will decrease shower length by 1 seconds
	# everytime we take an action.
	def step(self, action):
		# Apply temperature adjustment
		# '0' will represent decreasing our shower by 1 degree
		# '1' will represent no change
		# '2' will represent increase in the shower temperature by 1 degree.
		self.state += action - 1

		# Decrease shower time
		self.shower_length -= 1

		#Calculate Reward
		if self.state >= 37 and self.state <= 39:
			reward = 1
		else:
			reward = -1

		# Check if the shower is done
		# In the case where the shower is done, stop the episode.
		# Note that if we haven't consume the shower's 60 seconds, the episode is not done.
		if self.shower_length <= 0:
			done = True
		else: 
			done = False
		info = {}

		
		# returning the self.state that is the temperature 
		# returning the reward of the episode
		# returning whether the episode is done or not
		# returning "info"
		return self.state, reward, done, info

		pass


	# render function 
	def render(self):
		# I don't want to implement anything here 
		pass

	
	# reset function 
	# here we need to reset the shower temperature
	# and also reset the shower time to 60 seconds.
	def reset(self):

		# We set the state to random value between "38+3" and "38-3"
		# We then specify its type as type float
		self.state = np.array([38 + random.randint(-3,3)]).astype(float)
		
		# Resetting the shower length to 60 seconds.
		self.shower_length = 60

		# Returning the 'self.state', that is the temperature.
		return self.state



env = ShowerEnv()


env.observation_space
env.observation_space.sample()


env.action_space
env.action_space.sample()




# 4. Test Environment

# Note that our score will increase be 1 if the agent get the temperature between 37 and 39.
# Otherwise it the temperature will decrease by 1.  
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
env.close()



# 5. Train Model

# logpath is where we save our tensorboard log
# We can take a look at the tensorboard log to check how our model is performing
# The tutor created a folder named 'Training' in his project folder
# Inside the "Training" folder he created two folders and name them as "Logs" and "Saved Models"
# Tensorboard log is been saved inside the "Logs" folder.
# Trained Models are been saved inside the "Saved Models" folder.
log_path = os.path.join('Training', 'Logs')

# Defining our model 
# We are using PPO algorithm
# MlpPolicy(Multilayer perceptron policy) is the policy we are using; 
# MlpPolicy good for tabular datas as inputs.
# Policy is the rule which tells an agent how to behave in an environment
# "env" that is the environment is passed as the second parameter
# "verbose = 1" because we want to log out the result of the particular model
# Then we specify our tensorboard log folder path
model = PPO('MlpPolicy', env, verbose=1, tensorboard_log=log_path)

# To train our model
# This model will train for 400000 steps in this case.
model.learn(total_timesteps=40000)

# 6. Save Model
shower_path = os.path.join('Training','Saved Models', 'Shower_Model_PPO')
model.save(shower_path)

# delete the model
del model

# reloading the model
model = PPO.load(shower_path, env)

# Evaluating the policy
evaluate_policy(model, env, n_eval_episodes=10, render=True)
