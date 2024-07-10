
#Importing the libraries


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn.datasets
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn import metrics

#Importing the Boston House dataset

house_price_dataset = sklearn.datasets.fetch_california_housing()
print(house_price_dataset)

#Loading the dataset to pandas data frame
house_price_dataframe = pd.DataFrame(house_price_dataset.data, columns=house_price_dataset.feature_names)

house_price_dataframe.head()

house_price_dataframe['price'] = house_price_dataset.target
house_price_dataframe.head()

house_price_dataframe.shape

house_price_dataframe.isnull().sum()

# statistical measures of the dtaset 
house_price_dataframe.describe()

# Correlation between two columns or values
correlation=house_price_dataframe.corr()

#constructing a heatmap
plt.figure(figsize=(10,10))
sns.heatmap(correlation, cbar=True , square=True, fmt='.1f', annot=True, annot_kws={'size':8}, cmap='Blues')

#Splitting the data and target
x=house_price_dataframe.drop(['price'] , axis=1)
y=house_price_dataframe['price']

print(x)
print(y)

#Splitting the data into training data and test data
x_train, x_test, y_train, y_test= train_test_split(x, y, test_size=0.2, random_state=2)

print(x.shape , x_test.shape, x_train.shape)

# Model Training
model = XGBRegressor()

model.fit(x_train, y_train)

# accuracy for prediction on training data
training_data_prediction= model.predict(x_train)

# R Squared error
score1= metrics.r2_score(y_train , training_data_prediction)

# Mean absolute error
score2=metrics.mean_absolute_error(y_train , training_data_prediction)

print("r squared error", score1)
print("mean absolute error", score2)


#visualizing the actual prices and predicted prices
plt.scatter(y_train, training_data_prediction)
plt.xlabel("Actual prices")
plt.ylabel("predicted prices")
plt.title("actual prices vs predicted prices")
plt.show()


test_data_prediction= model.predict(x_test)
score1= metrics.r2_score(y_test , test_data_prediction)

# Mean absolute error
score2=metrics.mean_absolute_error(y_test , test_data_prediction)

print("r squared error", score1)
print("mean absolute error", score2)


plt.scatter(y_test, test_data_prediction)
plt.xlabel("Actual prices")
plt.ylabel("predicted prices")
plt.title("actual prices vs predicted prices")
plt.show()