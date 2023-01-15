# In this project we are creating a Flappy Bird agent
# Using DQN algorithm.

# First and Foremost we have to install "flappy-bird-gym"
# "flappy-bird-gym" is an environment of Flappy Bird game.
!pip install flappy-bird-gym


# 1. Import Dependencies
import random
import numpy as np
import flappy_bird_gym
from collections import deque
from keras.layers import Input, Dense
from keras.models import load_model, save_model, Sequential
from keras.optimizers import RMSprop



# 2. Neural Network for Agent
def NeuralNetwork(input_shape, output_shape):
	model = Sequential()
	# model.add(Input(input_shape))

	model.add(Dense(512, input_shape=input_shape, 
		activation='relu, kernel_initializer='he_uniform))
	model.add(Dense(256, activation='relu', kernel_initializer='he_uniform'))
	model.add(Dense(64, activation='relu', kernel_initializer='he_uniform'))
	model.add(Dense(output_shape, activation='linear', kernel_initializer='he_uniform'))
	model.compile(loss='mse', optimizer=RMSprop(lr=0.0001, rho=0.95, epsilon=0.01), metrics=['accuracy'])


# 3. Building the class of  flappy bird agent
class DQNAgent:
	def __init__(self):

		# Environment Variables
		# Creating the environment
		self.env = flappy_bird_gym.make("FlappyBird-v0")

		# The number of episode
		self.episodes = 1000
		self.state_space = self.env.observation_space.shape[0]
		self.action_space = self.env.action_space.n

		# The memory will act as the database that the model will train on.
		self.memory = deque(maxlen=2000)

		# Hyperparameters(variable that you can tune to get different result)
		self.gamma = 0.95

		# We can think of it as probability of taking random actions
		self.epsilon = 1
		self.epsilon_decay=0.9999
		self.epsilon_min = 0.01

		# amount of data to be inputed into our environment
		# 16,32,64,128,256
		self.batch_number = 64 

		self.train_start = 1000
		self.jump_prob = 0.01
		self.model=NeuralNet(input_shape=(self.state_space,), output_shape=self.action_space)


	# 4. Acting Function
	def act(self, state):
		if np.random.random() > self.epsilon:
			return np.argmax(self.model.predict(state))
		return 1 if np.random.random() < self.jump_prob else 0 



	# 5. Learn Function
	def learn(self):
		# To Make sure we have enough data
		if len(self.memory) < self.train_start:
			return

		# Take a batch of our memory
		# Create minibatch
		minibatch = random.sample(self.memory, min(len(self.memory), self.batch_number))

		# Variables to store minibatch info
		state = np.zeros((self.batch_number, self.state_space))
		next_state = np.zeros((self.batch_number, self.state_space))
		action, reward, done = [], [], []

		for i in range(self.batch_number):
			state[i] = minibatch[i][0]
			action.append(minibatch[i][1])
			reward.append(minibatch[i][2])
			next_state[i] = minibatch[i][3]
			done.append(minibatch[i][4])

		# Predict y label
		target = self.model.predict(state)
		target_next = self.model.predict(next_state)

		for i in range(self.batch_number):
			if done[i]:
				target[i][action[i]] = reward[i]
			else:
				target[i][action[i]] = reward[i] + self.gamma * (np.amax(target_next[i]))

		self.model.fit(state, target, batch_size=self.batch_number, verbose=0)


	# 6. Train function
	def train(self):

		# Episode Iterations for training
		for i in range(self.episodes):

			# Environment variables for training
			state = self.env.reset()

			# we reshape state so that our neural network will understand it
			state = np.reshape(state, [1, self.state_space])
			done = False
			score = 0 
			self.epsilon = self.epsilon * self.epsilon_decay if self.epsilon *self.epsilon_decay > self.min_epsilon else self.min_epsilon

			
			while not done:
				self.env.render()
				action = self.act(state)
				next_state, reward, done info = self.env.step(action)

				# reshape next state
				next_state = np.reshape(next_state, [1, self.state_space])
				score += 1

				# Punish it everytime it dies
				if done:
					reward -= 100

				self.memory.append((state, action, reward, next_state, done))
				state = next_state

				
				if done:
					print('Episode: {}\nScore: {}\nEpsilon: {:.2}'.format(i, score, self.epsilon))

					# Save model if the score is greater than 1000 scores.
					if score >= 1000:
						# save the model
						self.model.save('flappybrain.h5')
						return

				self.learn()

				# save the model after each iteration.
				self.model.save('flappybrain.h5')


	# 7. Visualize our model
	def perform(self):
		self.model = load_model('flappybrain.h5')
		while 1:
			state = self.env.reset()
			state = np.reshape(state, [1, self.state_space])
			done = False
			score = 0 

			while not done:
				self.env.render()
				action = np.argmax(self.model.predict(state))
				next_state, reward, done, info = self.env.step(action)
				state = np.reshape(next_state, [1, self.state_space])
				score += 1

				print("Current Score: {}".format(score))

				if done:
					print('DEAD')
				break


if __name__ == '__main__':
	agent = DQNAgent()

	# Calling the agent.train() function
	agent.train()

	# Calling the agent.perform() function
	agent.perform()





