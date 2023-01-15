
import gym
import gym_anytrading

from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import A2C

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


df = pd.read_csv('S&P.csv')
df
df.dtypes


#Preprocessing our Data
#we need to preprocess our data so
#that gym_anytrading environment understands
#therefore we have to convert our date to date-time object

df['Date'] = pd.to_datetime(df['Date'])
df.dtypes

#	We need to sort our data in ascending order
df.sort_values('Date', ascending=True, inplace=True)
#df = df.sort_values('Date', ascending=True)
df.set_index('Date', inplace=True)
df.dtypes

# To replace the commas with an empty space
# and convert its datatype into float type
df['Open'] = df['Open'].apply(lambda x: float(x.replace(',', '')))
df['High'] = df['High'].apply(lambda x: float(x.replace(',', '')))
df['Close'] = df['Close'].apply(lambda x:float(x.replace(',', '')))
df['Low'] = df['Low'].apply(lambda x: float(x.replace(',', '')))

df
df.dtypes


#	Creating our Environment
#	we will take day 5 to day 200 into consideration 
env = gym.make('stocks-v0', df=df, frame_bound=(5,200), window_size=5)

env.signal_features
env.action_space


#	Taking Random actions in our Environment
state = env.reset()
while True:
	action = env.action_space.sample()
	n_state, reward, done, info = env.step(action)

	if done:
		print(info)
		break

plt.figure(figsize=(15,6))
plt.cla()
env.render_all()
plt.show()


env_training = lambda: gym.make('stocks-v0', df=df, frame_bound=(5,200), window_size=5)

# Vectorizing our environment
# You can pass in multiple environment in the parameter position below
env = DummyVecEnv([env_training])


#Create our model
model = A2C('MlpLstmPolicy', env, verbose=1)
model.learn(total_timesteps=100000)


#Visualizing our agent
env = gym.make('stocks-v0', df=df, frame_bound=(200,253), window_size=5)
obs = env.reset()

while True:
	# Reshaping it into format that we can predict
	obs = obs[np.newaxis, ...]
	action, states = model.predict(obs)
	obs, rewards, done, info = env.step(action)

	if done:
		print(info)
		break

plt.figure(figsize=(15,6))
plt.cla()
env.render_all()
plt.show()


#	Implementing 98 Technical Indicators
#	Add Technical analysis to data
!pip install yfinance ta

import yfinance as yf 
from pandas_datareader import data as pdr 
data = pdr.get_data_yahoo('SPY', start='2017-01-01', end='2021-01-01')

data

# Starting adding technical indicator from here
from ta import add_all_ta_features

df2 = add_all_ta_features(data, open='Open', high='High', low='Low', 
	close='Close', volume='Volume', fillna=True)

df2

pd.set_option('display.max_columns', None)


#	Creating Environment with technical Indicator
from gym_trading.envs import StocksEnv

def my_processed_data(env):
	start = env.frame_bound[0] - env.window_size
	end = env.frame_bound[1]
	prices = env.df.loc[:, 'Low'].to_numpy([start:end])

	# Extracting some important columns from the technical indicator
	signal_features = env.df.loc[:, ['Close', 'Volume','momentum_rsi', 'volume_obv', 'trend_macd_diff']].to_numpy()[start:end]
	return prices, signal_features

# Connecting the data to our custom environment with technical indicator.
# We will inherit from StocksEnv which is part of anygymtrading class.
class MyCustomEnv(StocksEnv):
	_process_data = my_processed_data

# Creating Variable that will hold our environment
env2 = MyCustomEnv(df=df2, window_size=5, frame_bound=(5,700))
env.signal_features


#	Visualizing our Agent in TA environment
training_env = lambda:env2
env = DummyVecEnv([training_env])

# Creating the model
model = A2C('MlpLstmPolicy', env, verbose=1)
model.learn(total_timesteps=100000)


env = MyCustomEnv(df=df2, window_size=5, frame_bound=(700,1000))
obs = env.reset()

while True:
	obs=obs[np.newaxis, ...]
	action, states=model.predict(obs)
	obs, rewards, done, info=env.step(action)

	if done:
		print(info)
		break
plt.figure(figsize=(15,6))
plt.cla()
env.render_all()
plt.show()




