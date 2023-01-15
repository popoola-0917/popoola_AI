#import the libraries
import numpy as np
import pandas as pd

#Load the data
from google.colab import files
files.upload()

#Get the stock data
df = pd.read_csv('DOGECOIN.csv')
#set the date as index
df = df.set_index(pd.DatetimeIndex(df['Date'].values))
#show the data
df



#Ask the user the amount of money that they invested and the date it was been invested
amount_invested = input('Enter Amount invested in Dollars: ')
print(amount_invested)
invest_date = input('Enter the Date that you invested: ')
print(invest_date)


#Get the low and the high price for the day that the user invested to get a price range


#Create variable to store the low and high price column names
col1 = 'Low'
col2 = 'High'



#Get the low and high price of the asset for the specific day that the user invested
price1 = df[col1][invest_date]
price2 = df[col2][invest_date]
#Compute the quantity of asset the user would have
quantity1 = int(amount_invested) /price1
quantity2 = int(amount_invested) / price2
#Compute the profit range
profit1 = (quantity1 * df[col1][-1] - int(amount_invested))
profit2 = (quantity2 * df[col1][-1] - int(amount_invested))

#Print the range the amount would have made or lost
print('You would have made between $', round(profit1, 2), 'and $', round(profit2, 2), 'as of', df['Date'][-1] )


#Compute the Return on Investment (ROI) range
ROI1 = profit1 / int(amount_invested) *100
ROI2 = profit2 / int(amount_invested) *100

#Print the ROI range
print('Your return on investment(ROI) would  be between', round(ROI1, 2), '%and', round(ROI2, 2),  '% as of', df['Date'][-1])