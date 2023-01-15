# In this project a space invaders agent will be created using DQN algorithm



# 1. Importing Packages and setting up the environment dendencies
import gym
#! wget http://www.atarimania.com/roms/Roms.rar && unrar x Roms.rar && unzip Roms/ROMS.zip
#! pip3 install gym-retro
#! python3 -m retro.import ROMS/

env = gym.make('SpaceInvaders-v0')


#the amount of time we are trying the environment
episodes = 10


for episode in range (1, episodes):
  # reset the environment
  state = env.reset()

  # This is telling the code whether the episode has been completed or not
  done = False
  score = 0

  # While the episode has not end.
  while not done:

    # To render the environment
    env.render()

    # Taking random actions
    state, reward, done, info = env.step(env.action_space.sample())
    score += reward

  # To print episode and score
  print('Episode: {}\n Score: {}'.format(episode, score))

# To close the environment.
env.close()




# 2. Building the Neural network
#Import Neural Network Packages
import numpy as np
from tensorflow.keras.models import Sequential

# Importing the layers of neural network
# Dense Layer-Fully connected layers, 
# Flatten layer- flattens the previous output, 
# Conv2D - A convolutional layer
from tensorflow.keras.layers import Dense, Flatten, Conv2D

# Adam optimizer will help us to find optimal weight.
from tensorflow.keras.optimizers import Adam


# Defining our model

# The first three parameters; height, width, channels
# are the pixel of the screen that the model will learn from.
def build_model(height, width, channels, actions):

  # Creating a sequential layers
  model = Sequential()

  # Adding a convolutional layer
  # "relu" activations allows non-linearity in the model
  # this will helps to learn more complex patterns.
  # "input_shape" will take in the input of the screen and perform some computation 
  # and an action will be output based on the specific screen information
  model.add(Conv2D(32, (8,8), strides=(4,4), activation='relu', 
                   input_shape=(3, height, width, channels)))

  # To add a convolutional layer
  model.add(Conv2D(64, (4,4), strides=(2,2), activation='relu'))

  # To add a convolutional layer
  model.add(Conv2D(64, (4,4), strides=(2,2), activation='relu'))

  # To add a flattened layer
  # To flatten the output value to a one-dimensional array
  model.add(Flatten())

  # To add a fully connected layer with "relu" activation
  model.add(Dense(512, activation='relu'))
  model.add(Dense(256, activation='relu'))
  model.add(Dense(64, activation='relu'))

  # To add a fully connected layer with "linear" activation
  model.add(Dense(actions, activation='linear'))

  # To return the model
  return model


# Setting "env.observation_space.shape" to "height, width, channels" variables
height, width, channels = env.observation_space.shape

# Setting "env.action_space.n" to "actions" variable
actions = env.action_space.n


# To delete the model
del model

# Building the model
model = build_model(height, width, channels, actions)




# 3. Building our Reinforcement Learning Agent

# Import "DQNAgent" to create the Deep Q Network
from rl.agents import DQNAgent

# The memory can be referred to as dataset that the model will be train on.
from rl.memory import SequentialMemory

# Importing the Policies 
from rl.policy import LinearAnnealedPolicy, EpsGreedyQPolicy

# We start building our reinforcement learning agent here
def build_agent(model, actions):
  policy = LinearAnnealedPolicy(EpsGreedyQPolicy(), attr='eps', 
                                value_max=.1, value_min=.1, 
                                value_test=.2, nb_steps=10000)

  memory = SequentialMemory(limit=2000, window_length=3)

  dqn = DQNAgent(model=model, memory=memory, policy=policy,
                 enable_dueling_network=True, dueling_type='avg',
                 nb_actions=actions, nb_steps_warmup=1000)
  return dqn

# Create a variable that stores the "build_agent" functions
dqn = build_agent(model, actions)



# 4. Training Our Agent

# We first need to compile the model
dqn.compile(Adam(lr=0.001))

# We start training our environment here
dqn.fit(env, nb_steps=40000, visualize=True,
        verbose=1)

# if you want to use tensorflow you can install it using the following command
# note that you need CUDA installed to your path on your computer before installing tensorflow
# Alternatively one can run the code on "Google colab" to use their gpu for free.
#!pip install tensorflow-gpu==2.3.1


# 5. Visualizing the agents

# Things to do to improve the model
# (1) Train the model for longer period of time
# (2) Add more convolutional layers

# Test the model
scores = dqn.test(env, nb_episodes=10, visualize=True)

# Primt the mean episode
print(np.mean(scores.history['episode_reward']))

# Save the trained model in this folder
dqn.save_weights('models/dqn.h5f')

# Load the model from this folder
dqn.load_weights('models/dqn.h5f')