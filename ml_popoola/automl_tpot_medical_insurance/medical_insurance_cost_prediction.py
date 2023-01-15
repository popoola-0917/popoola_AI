# MEDICAL INSURANCE COST PREDICTION USING TPOT


#import the necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


from google.colab import drive
drive.mount('/content/drive')


path='/content/drive/MyDrive/AutoML datasets/insurance.csv'
df = pd.read_csv(path)


# DATA PREPROCESSING AND VISUALIZATION
df.head()


#To check the datatype of the columns
df.info()


#To describe our dataset
df.describe()

sns.countplot('children', data=df)

sns.countplot('smoker', data=df)

sns.countplot('region', data=df)

plt.scatter('smoker', 'charges', data=df)


#correlation heatmap
corrmat = df.corr()

top_corr_features = corrmat.index
plt.figure(figsize = (10,10))
g = sns.heatmap(df[top_corr_features].corr(), annot=True, cmap='RdYlGn')

df.head()


#Convert categorical variable to numbers 

def smoker(yes):
  if yes == 'yes':
    return 1
  else:
    return 0


df['smoker']=df['smoker'].apply(smoker) #we are are apply data function over dataframe

def sex(s):
  if s =='male':
    return 1
  else:
    return 0
df['sex'] = df['sex'].apply(sex)


#separating the X and Y from the dataset
X = df.drop(['charges', 'region'], axis=1)
Y = df['charges']
print(X.shape)
print(Y.shape)


#using sklearn to split training and testing dataset
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)



#install TPOT
!pip install tpot


from tpot import TPOTRegressor
from sklearn import metrics


tpot = TPOTRegressor(
    generations = 5,
    population_size = 50,
    verbosity = 2,
    random_state = 42,
    max_time_mins = 5,
    max_eval_time_mins = 2
    
)

tpot.fit(X_train, Y_train)


y_pred = tpot.predict(X_test)

print(metrics.mean_squared_error(Y_test, y_pred))

print(np.sqrt(metrics.mean_squared_error(Y_test, y_pred)))