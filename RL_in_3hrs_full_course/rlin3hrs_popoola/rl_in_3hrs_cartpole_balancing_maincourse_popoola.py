# Note it will be better to create a separate virtual environment for every coding project
# Installing dependencies
!pip install stable-baselines3[extra]
!pip install gym

# importing the dependencies
import os
import gym
# importing PPO algorithm from stable baseline
from stable_baselines3 import PPO
# stable baseline allow one to vectorize the environment.
# This means training an agent on multiple environment at thesame time
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy


# 2. Load Environment

# Open ai gym allow us to build simulated environment more easily.
# The environment we are using here is CartPole-v0

# Defining the environment
environment_name = 'CartPole-v0'
env = gym.make(environment_name)
# To check the environment name
environment_name


# Testing the environment
# We want our agent to take the right action in the environment to maximize the reward
# Therefore we will test the environment

# Testing the cart-pole environment 5 times, that is we will loop through each episodes 5 times
# Think of an episode as one full game within the environment
episodes = 5

# looping through one of each episodes
for episode in range(1, episodes + 1):

	# Here we are resetting our environment
    # By resetting our environment we are getting our previous observations
	state = env.reset()

	# Setting whether or not the episode is done, Note we have a maximum amount of steps in this environment
	done = False

	# Setting up running score counter across the episode
	score = 0

	# While the episode is not done, we will render the environment, that is viewing the graphical
    # representation of the environment.
    # If one is using google colab as the notebook using the rendering the environment is slightly different
    # After episode is done this loop is going to stop
	while not done:
		env.render()
		# generating a random action
		action = env.action_space.sample()
		# Unpacking the values that we get from env.step()
        #  That is, passing random actions to our environment
		n_state, reward, done, info=env.step(action)
		# Accumulating the reward
		score += reward
	# Printing out the result that we get from taking this particular step
	print('Episode:{} Score:{}'.format(episode, score))


# To close the environment	
env.close()
# To reset the environment
env.reset()




# Taking random action
# To check the action space in our environment
# env.action_space.sample()
# To check the observation space in our environment
# env.observation_space.sample()
# episodes = 5
# for episode in range(1, episodes +1):
# 	print(episode)

# Passing through our action
# env.step(1)



# Understanding the Environment
# There are two part in the environment: action space and observaction space

# This will show the type of actionspace in our environment like Discrete(2)
env.action_space
env.action_space.sample()

env.observation_space
env.observation_space.sample()


#3. Train an RL Model
# Model free RL uses the curent state-value to make prediction
# Model based RL try to make prediction about the future state of the model 
# to try to generate the best possible action 
# as at present of this tutorial stable-baseline only deals with Model free RL
# While choosing algorithm one should make sure to choose an algorithm that
# matches appropriately to ones particular type of action-space


# Understanding the training metrics
# Evaluation metrics: Ep_len_mean(this describes the length of the time an episode use), ep_raw_mean
# Time metrics: Fps, iterations, time_elapsed, total_timesteps
# Loss Metrics: Entropy_loss, policy_loss, value_loss
# Other Metrics: Explained_variance, Learning_rate, n_updates

# logpath is where we save our tensorboard log
# We can take a look at the tensorboard log to check how our model is performing
# The tutor created a folder named 'Training' in his project folder
# Inside the "Training" folder he created two folders and name them as "Logs" and "Saved Models"
# Tensorboard log is been saved inside the "Logs" folder.
# Trained Models are been saved inside the "Saved Models" folder.

log_path = os.path.join('Training', 'Logs')
log_path

# Instantiating the environment, note that we are using PPO algorithm
env = gym.make(environment_name)

# Wrapping our environment inside the DummyVecEnv wrapper
env = DummyVecEnv([lambda:env])

# Defining our model 
# MlpPolicy(MultilayerPerceptron) is the policy we are using; it indicates that we are using neural network
# Policy is the rule which tells an agent how to behave in an environment
# "env" that is the environment is passed as the second parameter
# "verbose = 1" because we want to log out the result of the particular model
# Then we specify our tensorboard log path
model = PPO('MlpPolicy', env, verbose=1, tensorboard_log=log_path)

#To check the PPO algorithm
PPO??

# This code trains our model. We will pass in the total timetsteps that we want to use in training our model
# Here the total timesteps is 20000
# For a simple environment one can use a lower total timesteps for a complicated environment one might need a larger total timesteps
# We can train the model longer by running this line of code.
model.learn(total_timesteps=20000)

# 4. Save and Reload Model
# One might need to save the model
# We will have to define a path "PPO_Path" in this case.
# Note that we have already created 'Saved Models' folder inside our 'Training' folder
# Therefore our model will be saved inside the 'Saved Models'
PPO_Path = os.path.join('Training', 'Saved Models', 'PPO_Model_Cartpole')

# This line of code save the model
model.save(PPO_Path)

# We can delete our model and reload it to simulate deployment
del model

# Just to check the path we define to save our model
PPO_Path

# Ths line of code is reloading the model
# We will pass in the path we define above as the first parameter, 
# and "env" our environment as the second parameter.
model = PPO.load(PPO_Path, env=env)


# 5. Evaluation
# To evaluate our model we will be using "evaluate_policy" method we imported 
# at the top of the program
# Note: the PPO model in this particular case is considered solved if we get a score of 200 on average or higher.
# Other environment might also have a point where they might be considered solved.
# Some environment may noy have caps; they are continuous, the higher the score the better.

# We passed in our model, environment, the amount of episode we are testing it for, rendering the environment
evaluate_policy(model, env, n_eval_episodes=10, render = True)

# If our output is shown as below it means total reward of 200 and standard deviation of 0.0
# Output : (200, 0.0)

# In google colab we have to set "render=False" because rendering is not working with default evaluation in google colab
#evaluate_policy(model, env, n_eval_episodes=10, render = False)

# This line of code closes the environment
env.close()



# 6. Test Model

#action, _ = model.predict(obs)
#action


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


# Breakdown of the above code
# Resetting our environment and setting it to our observation
#obs = env.reset()

# Taking the observation and passing it into the model
# Using "model.predict" function on the observation to generate action
# model.predict(obs)
# output : (array([1], dtype=int64), None) : it means to get the best result we have to take action [1]


# env.action_space.sample()

# Reward is gotten from the one of the output of this
#env.step(action)



# 7. Viewing Logs in Tensorboard
# If one is training a complicated environment,
# we can view the training log in the Tensorboard
# Ideally it should be run from the command prompt
# Try to install tensorboard before using it
# !tensorboard --logdir=(training_log_path)



# First thing get the log directories we will be using.
training_log_path = os.path.join(log_path, 'PPO_2')

# To check the training log path
training_log_path


# Try to install tensorboard before using it
# The exclamation mark "!" is what is known as magic command
# It enable programmer to install right from the jupyter notebook
!tensorboard --logdir=(training_log_path)


# Core Metrics to look at:
#1. Average reward
#2. Average episode length

# Training Strategies:
#1. Train for longer 
#2. Hyparameter tuning
#3. Try different algorithms; take a look at the best algorith people are using for the problem.



# 8. Adding a callback to the training Stage
# This section deals with the ability to stop training
# after reaching some reward threshold.
# This is useful when one have a large model and 
# we have to stop it before it get unstable
# This will also automatically save the best models.

# importing callback helpers
from stable_baelines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold

# path to save the best model
save_path = os.path.join('Training', 'Savedd Models')


# setup callback
# Timestamps 1:30:02

# This is the line of code stopping our training once we pass the reward threshold 
stop_callback = StopTrainingonRewardThreshold(reward_threshold=200, verbose = 1)

# Any time there is a new best model this callback will be rerun
# After 10000 steps it will check whether it has pass the reward threshold
# If it has pass the reward threshold, stop the training and save the best model into save_path
eval_callback = EvalCallback(env, callback_on_new_best = stop_callback, 
	eval_freq = 10000, best_model_save_path = save_path, verbose=1)

# The model we are using.
model = PPO('MlpPolicy', env, verbose=1, tensorboard_log=log_path)

# applying callback to train step
model.learn(total_timesteps=20000,  callback=eval_callback)



# 9. Changing Policies

# Defining new Multi layer perceptron architecture
net_arch = [dict(pi=[128,128,128,128], vf=[128,128,128,128])]

# Applying the new neural network to the model/chosen algorithm
model = PPO('MlpPolicy', env, verbose=1, 
	tensorboard_log=log_path, policy_kwargs=('net_arch':net_arch))
model.learn(total_timesteps=20000, callback=eval_callback)


# 10. Using an Alternative Algorithm
# Assuming we want to check out DQN algorithm on our model

# importing DQN algorithm
from stable_baselines3 import DQN

# setting up DQN algorithm
model = DQN('MlpPolicy', env, verbose=1, tensorboard_log=log_path)
model.learn(total_timesteps=20000)

# saving the model
model.save()

# loading the DQN
DQN.load()