# CREATE CUSTOM CRYPTO TRADING ENVIRONMENT FROM SCRATCH

# In this project, we will write a step-by-step foundation for our 
# custom Bitcoin trading environment where we could do further 
# development, tests, and automated trading experiments.
# Note that it is better to create a separate virtual environment for each project.

# Introduction to the environment
# The environment contains all the necessary functionality to run an agent and allow it to learn. 
# Each environment "must-have" implemented the following interface:

# class CustomEnv:
#     def __init__(self, arg1, arg2, ...):
#         # Define action space and state size
        
#         # Example when using discrete actions of 0,1,2:
#         self.action_space = np.array([0, 1, 2])
        
#         # Example for using image as input for custom environment:
#         self.state_size = np.empty((HEIGHT, WIDTH, CHANNELS)), dtype=np.uint8)
    
#     # Execute one time step within the environment
#     def step(self, action):
#         ...
        
#     # Reset the state of the environment to an initial state
#     def reset(self):
#         ...
#     # render environment visualization
#     def render(self):
#         ...

# In the constructor above, we first define the type and shape of our action_space,
# which will contain all of the actions possible for an agent to take in the environment. 
# Similarly, we'll define the state_size (above is an example with an image as input), 
# which contains all of the environment's data to be observed by the agent.
# Our reset method will be called periodically to reset the environment to an initial state.
# Many steps follow this through the environment, in which an action will be provided by the model 
# and must be executed, and the next observation returned. 
# This is also the place where rewards are calculated.
# Finally, the render method may be called periodically to print a rendition of the environment. 
# This could be as simple as a print statement or as complicated as rendering a 3D environment using OpenGL.



# Bitcoin Trading Environment
# To demonstrate how this all works, I am going to create a cryptocurrency trading environment. 
# I will then train our agent to beat the market and become a profitable trader within the environment. 
# This shouldn't be Bitcoin; we'll be able to choose any market we want.
# The first thing that we need to consider is how we, humans, decide what trade we would like to make. 
# What observations do we make before deciding to make a trade?
# Usually, a professional trader would most likely look at some charts of price action, 
# perhaps overlaid with a couple of technical indicators. 
# From there, they would combine this visual information with their prior knowledge of similar price actions 
# to make an informed decision in what direction the price is likely to move.
# So, we need to translate these human actions into code so that our custom-created agent can understand price action similarly.
# We want state_size to contain all of the input variables that we need our agent to consider before taking action.
#  In this project, I want that my agent could "see" the main market data points (open price, high, low, close, and daily volume) 
# for the last 50 days, as well as a couple of other data points like its account balance, current open positions, and current profit.
# We want our agent for each timestep to consider the price action leading up to the current price and 
# their own portfolio's status to make an informed decision for the next action.
# Talking about actions, our agent will have action_space that will consist of three possibilities: buy, sell, or hold in the current time step.
# But this is not enough to know what amount of Bitcoin to buy or sell each time. 
# So we will need to create an action space with a discrete number of action types (buy, sell, and hold)
# and a continuous spectrum of amounts to buy/sell (0-100% of the account balance/position size, respectively)
# Right now we won't consider the amount for simplicity reasons. 
# But after we get some working environment with a trained agent, 
# I will try to integrate the risk of our agent by considering the amount.
# The last thing to consider before implementing a custom environment is the reward.
# We want to promote long-term profit to calculate the account balance difference between the previous step and the current step for each step. 
# Anyway, we want our agents to maintain a higher balance for longer, rather than those who rapidly gain money using unsustainable strategies.


# Implementation
# The environment expects a pandas data frame to be passed that contains the market data to be learned from. 
# Adding to this, we must know our dataset length, starting trading balance, and how many steps of market memory 
# we want our agent to "see", we define all of these parameters in our __init__ part.
# I describe them as a deque list; this means that our list has a limited size of 50 steps. 
# When we append a new item into the list, the last one is removed. 
# The market raw data was downloaded from; https://bitcoincharts.com/ page.

import pandas as pd
import numpy as np
import random
from collections import deque

class CustomEnv:
    # A custom Bitcoin trading environment
    def __init__(self, df, initial_balance=1000, lookback_window_size=50):
        # Define action space and state size and other custom parameters
        self.df = df.dropna().reset_index()
        self.df_total_steps = len(self.df)-1
        self.initial_balance = initial_balance
        self.lookback_window_size = lookback_window_size

        # Action space from 0 to 3, 0 is hold, 1 is buy, 2 is sell
        self.action_space = np.array([0, 1, 2])

        # Orders history contains the balance, net_worth, crypto_bought, crypto_sold, crypto_held values for the last lookback_window_size steps
        self.orders_history = deque(maxlen=self.lookback_window_size)
        
        # Market history contains the OHCL values for the last lookback_window_size prices
        self.market_history = deque(maxlen=self.lookback_window_size)

        # State size contains Market+Orders history for the last lookback_window_size steps
        self.state_size = (self.lookback_window_size, 10)

 # Next, we'll write the reset method, which should be called every time a new environment 
 # is created or to reset an existing environment's state to the primary. 
 # Here we'll set the starting balance to our initial balance, define our 
 # start and finish steps in the dataset (used to separate training and testing data)
 # As you can see, we create our primary state by concatenating our orders and the market history of our 50 history steps.


# Reset the state of the environment to an initial state
def reset(self, env_steps_size = 0):
    self.balance = self.initial_balance
    self.net_worth = self.initial_balance
    self.prev_net_worth = self.initial_balance
    self.crypto_held = 0
    self.crypto_sold = 0
    self.crypto_bought = 0
    if env_steps_size > 0: # used for training dataset
        self.start_step = random.randint(self.lookback_window_size, self.df_total_steps - env_steps_size)
        self.end_step = self.start_step + env_steps_size
    else: # used for testing dataset
        self.start_step = self.lookback_window_size
        self.end_step = self.df_total_steps

    self.current_step = self.start_step

    for i in reversed(range(self.lookback_window_size)):
        current_step = self.current_step - i
        self.orders_history.append([self.balance, self.net_worth, self.crypto_bought, self.crypto_sold, self.crypto_held])
        self.market_history.append([self.df.loc[current_step, 'Open'],
                                    self.df.loc[current_step, 'High'],
                                    self.df.loc[current_step, 'Low'],
                                    self.df.loc[current_step, 'Close'],
                                    self.df.loc[current_step, 'Volume']
                                    ])

    state = np.concatenate((self.market_history, self.orders_history), axis=1)
    return state

# Next, our environment needs to be able to take a step. Our agent will choose and take either buy, 
# sell, or hold action, calculate the reward, and return the next observation at each step.
# In a real situation, usually, our price fluctuates in a 1h timeframe 
# (I chose 1 hour for this project) up and down until it closes.
# In historical data, we can't see these movements, so we need to create them. 
# I do this by taking a random price between open and close prices. 
# Because I am using my total balance, I can easily calculate how much Bitcoin amount I will buy/sell and represent that in balance, 
# crypto_bought, crypto_held, crypto_sold, and net_worth parameters (I add them to my orders_history, so I will send these values to my agent).
# Also, I can calculate rewards by subtracting net worth from the previous step and current step. 
# And lastly, we must pick a new state by calling _next_observation().

# Execute one time step within the environment
def step(self, action):
    self.crypto_bought = 0
    self.crypto_sold = 0
    self.current_step += 1

    # Set the current price to a random price between open and close
    current_price = random.uniform(
        self.df.loc[self.current_step, 'Open'],
        self.df.loc[self.current_step, 'Close'])

    if action == 0: # Hold
        pass

    elif action == 1 and self.balance > 0:
        # Buy with 100% of current balance
        self.crypto_bought = self.balance / current_price
        self.balance -= self.crypto_bought * current_price
        self.crypto_held += self.crypto_bought

    elif action == 2 and self.crypto_held>0:
        # Sell 100% of current crypto held
        self.crypto_sold = self.crypto_held
        self.balance += self.crypto_sold * current_price
        self.crypto_held -= self.crypto_sold

    self.prev_net_worth = self.net_worth
    self.net_worth = self.balance + self.crypto_held * current_price

    self.orders_history.append([self.balance, self.net_worth, self.crypto_bought, self.crypto_sold, self.crypto_held])

    # Calculate reward
    reward = self.net_worth - self.prev_net_worth

    if self.net_worth <= self.initial_balance/2:
        done = True
    else:
        done = False

    obs = self._next_observation()

    return obs, reward, done

  # I could have merged the _next_observation() function with a step function, 
  # but I separated them to make my code a little simpler. Here I take a new step 
  # from history and concatenate it with the orders_history list taken from the step function.



# Get the data points for the given current_step
def _next_observation(self):
    self.market_history.append([self.df.loc[self.current_step, 'Open'],
                                self.df.loc[self.current_step, 'High'],
                                self.df.loc[self.current_step, 'Low'],
                                self.df.loc[self.current_step, 'Close'],
                                self.df.loc[self.current_step, 'Volume']
                                ])
    obs = np.concatenate((self.market_history, self.orders_history), axis=1)
    return obs

# Usually, we want to see how our agent learns, performs, etc. 
# So we need to create a render function. For simplicity's sake, 
# we will render the current step of our environment and the net worth so far.

# render environment
def render(self):
    print(f'Step: {self.current_step}, Net Worth: {self.net_worth}')

# Now  I'll create a function that would simulate our agent actions, 
# but instead of doing these actions with the AI agent, we'll randomly do them.

def Random_games(env, train_episodes = 50, training_batch_size=500):
    average_net_worth = 0
    for episode in range(train_episodes):
        state = env.reset(env_steps_size = training_batch_size)

        while True:
            env.render()

            action = np.random.randint(3, size=1)[0]

            state, reward, done = env.step(action)

            if env.current_step == env.end_step:
                average_net_worth += env.net_worth
                print("net_worth:", env.net_worth)
                break

    print("average_net_worth:", average_net_worth/train_episodes)

# We can instantiate a CustomEnv environment with a data frame and test it with our Random_games agent. 
# I set lookback_window_size=50, and this will be history steps held in the state. 
# Then I separate data frame to train and test data frames, which will be used to train our RL agent. 
# Also, we'll need separate test and train environments for evaluation, so now I will create them just for fun. 
# And finally, let's run our Random_games function within one of our built environments:

df = pd.read_csv('./pricedata.csv')
df = df.sort_values('Date')

lookback_window_size = 50
train_df = df[:-720-lookback_window_size]
test_df = df[-720-lookback_window_size:] # 30 days

train_env = CustomEnv(train_df, lookback_window_size=lookback_window_size)
test_env = CustomEnv(test_df, lookback_window_size=lookback_window_size)

Random_games(train_env, train_episodes = 10, training_batch_size=500)

# After ten episodes of random games, we should see similar results as shown below. 
# For ten episodes, my average net worth was around 981$, which means our accidental agent 
# lost 19$ on average through 10 episodes. Every time we run this random agent, 
# we'll receive different results; it also might be positive.
# Step: 1771, Net Worth: 973.1386787365324
# Step: 1772, Net Worth: 973.1386787365324
# Step: 1773, Net Worth: 973.1386787365324
# Step: 1774, Net Worth: 973.1386787365324
# Step: 1775, Net Worth: 973.1386787365324
# Step: 1776, Net Worth: 973.1386787365324
# Step: 1777, Net Worth: 973.1386787365324
# net_worth: 973.1386787365324
# average_net_worth: 981.7271660925813
# >>>


# Of course, this whole tutorial was just an introduction to all future tutorials. 
# This was just for fun to test creating an interesting, custom Reinforcement Learning 
# environment with some actions, observations, and reward spaces. 
# It will take a lot more time and effort if we want to create an agent that could beat the market and make some profit.