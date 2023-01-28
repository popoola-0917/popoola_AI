# Integrating Technical Indicators to Automated Bitcoin Trading Bot #5
# We'll integrate a few most popular technical indicators into our Bitcoin trading 
# bot to make it learn even better decisions while making automated trades


# This tutorial will integrate a few most popular technical indicators 
# into our Bitcoin trading bot to learn even better decisions 
# while making automated trades in the market.

# In my previous tutorials, we already created a python custom Environment 
# to trade Bitcoin; we wrote a Reinforcement Learning agent to do so. 
# Also, we tested three different architectures (Dense, CNN, LSTM) and 
# compared their performance, training durations, and tested their profitability. 
# So I thought, if we can create a trading bot making some profitable trades 
# just from Price Action, maybe we can use indicators to improve our bot accuracy 
# and profitability by integrating indicators? Let's do this! I thought that probably 
# there are no traders or investors who would be making blind trades without doing 
# some technical or fundamental analysis; more or less, everyone uses technical indicators.

# First of all, we will be adding five widely known and used technical indicators to our data set. 
# The technical indicators should add some relevant information to our data set, which can be 
# complimented well by the forecasted information from our prediction model. 
# This combination of indicators ought to give a pleasant balance of practical observations for our model to find out from:

# I am going to cover each of the above given technical indicators shortly. 
# To implement them, we'll use an already prepared ta Python library used to 
# calculate a batch of indicators. If we succeed with these indicators with our 
# RL Bitcoin trading agent, maybe we'll try more of them in the future.


# Moving Average (MA)
# The MA - or 'simple moving average' (SMA) - is an indicator accustomed 
# to determining the direction of a current market trend while not interfering 
# with shorter-term market spikes. The moving average indicator combines market 
# points of a selected instrument over a particular timeframe. It divides it by 
# the number of timeframe points to present us the direction of a trend line.

# The data used depends on the length of the MA. For instance, a two 
# hundred MA needs 200 days of historical information. 
# By exploiting the MA indicator, you'll be able to study support and resistance levels 
# and see previous price action (the history of the market). 
# This implies you'll be able to determine possible future price patterns.

# I wrote the following function that we'll use to plot our indicators with OHCL bars and Matplotlib:




import matplotlib.pyplot as plt
from mplfinance.original_flavor import candlestick_ohlc
import matplotlib.dates as mpl_dates

def Plot_OHCL(df):
    df_original = df.copy()
    # necessary convert to datetime
    df["Date"] = pd.to_datetime(df.Date)
    df["Date"] = df["Date"].apply(mpl_dates.date2num)

    df = df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]
    
    # We are using the style ‘ggplot’
    plt.style.use('ggplot')
    
    # figsize attribute allows us to specify the width and height of a figure in unit inches
    fig = plt.figure(figsize=(16,8)) 

    # Create top subplot for price axis
    ax1 = plt.subplot2grid((6,1), (0,0), rowspan=5, colspan=1)

    # Create bottom subplot for volume which shares its x-axis
    ax2 = plt.subplot2grid((6,1), (5,0), rowspan=1, colspan=1, sharex=ax1)

    candlestick_ohlc(ax1, df.values, width=0.8/24, colorup='green', colordown='red', alpha=0.8)
    ax1.set_ylabel('Price', fontsize=12)
    plt.xlabel('Date')
    plt.xticks(rotation=45)

    # Add Simple Moving Average
    ax1.plot(df["Date"], df_original['sma7'],'-')
    ax1.plot(df["Date"], df_original['sma25'],'-')
    ax1.plot(df["Date"], df_original['sma99'],'-')

    # Add Bollinger Bands
    ax1.plot(df["Date"], df_original['bb_bbm'],'-')
    ax1.plot(df["Date"], df_original['bb_bbh'],'-')
    ax1.plot(df["Date"], df_original['bb_bbl'],'-')

    # Add Parabolic Stop and Reverse
    ax1.plot(df["Date"], df_original['psar'],'.')

    # # Add Moving Average Convergence Divergence
    ax2.plot(df["Date"], df_original['MACD'],'-')

    # # Add Relative Strength Index
    ax2.plot(df["Date"], df_original['RSI'],'-')

    # beautify the x-labels (Our Date format)
    ax1.xaxis.set_major_formatter(mpl_dates.DateFormatter('%y-%m-%d'))# %H:%M:%S'))
    fig.autofmt_xdate()
    fig.tight_layout()
    
    plt.show()







# I'll not explain this code line by line because I already wrote a similar function 
# in my second tutorial, where I explained everything step-by-step.

# We can add all of our 3 SMA indicators into our data frame and plot 
# it with the following simple piece of code:

import pandas as pd
from ta.trend import SMAIndicator

def AddIndicators(df):
    # Add Simple Moving Average (SMA) indicators
    df["sma7"] = SMAIndicator(close=df["Close"], window=7, fillna=True).sma_indicator()
    df["sma25"] = SMAIndicator(close=df["Close"], window=25, fillna=True).sma_indicator()
    df["sma99"] = SMAIndicator(close=df["Close"], window=99, fillna=True).sma_indicator()
    
    return df

if __name__ == "__main__":   
    df = pd.read_csv('./pricedata.csv')
    df = df.sort_values('Date')
    df = AddIndicators(df)

    test_df = df[-400:]

    # Add Simple Moving Average
    Plot_OHCL(test_df, ax1_indicators=["sma7", "sma25", "sma99"])




# After indicators calculation for our entire dataset and when we plot it for the last 720 bars, it looks following:









# Bollinger Bands

# A Bollinger Band is a technical analysis tool outlined by a group of trend lines 
# with calculated two standard deviations (positively and negatively) far from a 
# straightforward moving average (SMA) of a market's value, which may be adjusted 
# to user preferences. Bollinger Bands were developed and copyrighted by notable 
# technical day trader John Bollinger and designed to get opportunities that could 
# offer investors a better likelihood of correctly identifying market conditions (oversold or overbought). 
# Bollinger Bands are a modern technique. Many traders believe the closer the prices move to the upper band, 
# the more overbought the market is, and the closer the prices move to the lower band, the more oversold 
# the market is. Let's add this to our code for the same data set as we did with SMA:



import pandas as pd
from ta.volatility import BollingerBands

def AddIndicators(df):
    # Add Bollinger Bands indicator
    indicator_bb = BollingerBands(close=df["Close"], window=20, window_dev=2)
    df['bb_bbm'] = indicator_bb.bollinger_mavg()
    df['bb_bbh'] = indicator_bb.bollinger_hband()
    df['bb_bbl'] = indicator_bb.bollinger_lband()
    
    return df

if __name__ == "__main__":   
    df = pd.read_csv('./pricedata.csv')
    df = df.sort_values('Date')
    df = AddIndicators(df)

    test_df = df[-400:]

    # Add Bollinger Bands
    Plot_OHCL(test_df, ax1_indicators=["bb_bbm", "bb_bbh", "bb_bbl"])


# Parabolic Stop and Reverse (Parabolic SAR)
# The parabolic SAR is a widely used technical indicator to determine market direction, 
# but it draws attention to it at the exact moment once the market direction is changing. 
# This indicator also can be called the "stop and reversal system," 
# the parabolic SAR was developed by J. Welles Wilder Junior. - the creator of the relative strength index (RSI).

# The indicator seems like a series of dots placed either higher than or
#  below the candlestick bars on a chart. When the dots flip, 
#  it indicates that a possible change in asset direction is possible. 
#  For example, if the dots are above the candlestick price, and then 
#  they appear below the price, it could signal a change in market trend. 
#  A drop below the candlestick is deemed to be an optimistic bullish signal. 
#  Conversely, a dot above the fee illustrates that the bears are in control 
#  and that the momentum is likely to remain downward.


# The SAR dots start to move a little quicker as the market direction goes up 
# until the dots catch up to the market price. As the market price rises, the dots 
# will rise as well, first slowly and then picking up speed and accelerating with the trend. 
# We can add PSAR to our chart with the following code:

import pandas as pd
from ta.trend import PSARIndicator

def AddIndicators(df):
    # Add Parabolic Stop and Reverse (Parabolic SAR) indicator
    indicator_psar = PSARIndicator(high=df["High"], low=df["Low"], close=df["Close"], step=0.02, max_step=2, fillna=True)
    df['psar'] = indicator_psar.psar()
    
    return df

if __name__ == "__main__":   
    df = pd.read_csv('./pricedata.csv')
    df = df.sort_values('Date')
    df = AddIndicators(df)

    test_df = df[-400:]

    # Add Parabolic Stop and Reverse
    Plot_OHCL(test_df, ax1_indicators=["psar"])