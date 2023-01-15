# In this project we will create an agent whose goal is to pickup 
# passengger at one location and drop them at another location.
# The agent is been traned using Q-learning.


# 1. Import Required Packages
import gym
import numpy as np
import matplotlib.pyplot as plt
from Ipython.display import clear_output

# We can install matplotlib using this line of code
# !pip install matplotlib


# Creating our environment
env = gym.make('Taxi-v3')


# Let start our environment by taking random action
episodes = 10

for episode in range(1, episodes):
	state = env.reset()
	done = False #when done equals true then our agent has completed the level
	score = 0

	while not done:
		env.render() # we want to render our environment

		# passing random action
		state, reward, done, info = env.step(env.action_space.sample())

		# Increasing the reward.
		score += reward
		clear_output(wait=True) # Now our output will be cleared each time

	# Print episode and score
	print('Episode: {}\nScore: {}'.format(episode, score))

# To close the environment.
env.close()


# After running the above code we discover that the output
# are stacking on each other. 
# To clear our output we use "from Ipython.display import clear_output"







# 2. Creating Q-Table
actions = env.action_space.n
state = env.observation_space.n

# initialize each step with zeroes
q_table = np.zeros((state, actions)) 

# PARAMETERS FOR Q-LEARNING
#number of times we are re-iterating our algorithm
num_episodes = 1000

# amount of steps we take per episode
max_steps_per_episode = 100

learing_rate = 0.1
discount_rate = 0.99

exploration_rate = 1
max_exploration_rate = 1
min_exploration_rate = 0.01
exploration_decay_rate = 0.001

#A empty list for storing the reward
rewards_all_episodes = [] 



# 3. Training Q-Learning Agent 

# Q-Learning Algorithm
for episode in range(num_episodes):

	# resetting the environment
	state = env.reset() 
	
 	# setting the done flag to false
	done = False  
	rewards_current_episode = 0 

	# iterating the max step per episode
	for step in range(max_steps_per_episode): 
		
		#Coding the exploration threshold versus exploration rate
		exploration_threshold = random.uniform(0,1)

		#if exploration threshold is greater than exploration rate
		#we want to look into our q table and take the 
		#associated action. if not we will take a random action.
		if exploration_threshold > exploration_rate:
			action = np.argmax(q_table[state, :])
		else:
			action = env.action_space.sample()
		new_state, reward, done, info = env.step(action)
		

		# Updating Q-Table
		q_table[state, action] =q_table[state, action] *(1-learning_rate) + learning_rate * (reward + discount_rate * np.max(q_table[new_state, :]))

		state = new_state
		rewards_current_episode += reward

		if done == True:
			break

	# To decays exploration rate overtime		
	exploration_rate = min_exploration_rate + \
						(max_exploration_rate - min_exploration_rate)* np.exp(-exploration_decay_rate * episode)

	rewards_all_episodes.append(rewards_current_episode)

print("****************Training Finished ********************")

# To view the q-table
q_table




# 4. Analyzing our Trained Agent

# Calculate and print average reward per thousand episodes
rewards_per_thousand_episodes = np.split(np.array(rewards_all_episodes), num_episodes/1000)
count = 1000

print("Average per thousand episodes")
for r in rewards_per_thousand_episodes:
	print(count, " : ", str(sum(r/1000)))
	count += 1000



# 5. Visualize our Trained Agent

# Visualize Agent
import time

# We are telling our agent to run for 3 episodes.
for episode in range(3):

	# Reset environment after each episode
	state = env.reset()

	# Setting the "done" flag to False.
	# Which means our episode has not finished yet.
	done = False
	print("Episode is: " + str(episode))
	time.sleep(1)

	# Take action in the environment
	for step in range(max_steps_per_episode):
		clear_output(wait=True)
		env.render()

		# To slow down the program.
		time.sleep(0.4)

		# Extracting specific action from the "q_table".
		action = np.argmax(q_table[state, :])

		new_state, reward, done, info = env.step(action)

		# If the episode is done
		# Then render the environment
		if done:
		 	clear_output(wait=True)
		 	env.render()

		 	# If return equals 1 then we have reach our goal.
		 	if reward == 1:
		 		print("*****Reached Goal*****")

		 		# To slow down the program
		 		time.sleep(2)
		 		clear_output(wait=True)
		 	else:
		 		print("*****Failed*****")

		 		# To slow down the program
		 		time.sleep(2)
		 		clear_output(wait=True)
		 	break
		state = new_state

env.close()







