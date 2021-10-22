import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.tree import DecisionTreeRegressor

model_dataset = 'billionaires.csv'  # Set the file name
billionaire_data = pd.read_csv(model_dataset)  # read csv into data
billionaire_data = billionaire_data.dropna()  # drops N/A data
y = billionaire_data['wealth.worth in billions']  # sets the value to predict
features = ['company.founded', 'demographics.age', 'location.gdp']
X = billionaire_data[features]  # features
billionaire_model = DecisionTreeRegressor()  # define the model type to use
billionaire_model.fit(X, y)  # fit model using un-split data X and y
predictions = billionaire_model.predict(X)  # predicts target value using X
mae = mean_absolute_error(y, predictions)  # validate mae using y and predictions w/o splitting data
print('Data MAE: ', mae)  # print mae
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=0)  # split the dataset
billionaire_model.fit(train_X, train_y)  # fit features and value to predict to model type
val_predictions = billionaire_model.predict(val_X)  # predict value using validation data
mae = mean_absolute_error(val_y, val_predictions)  # validate model with mae using val_y and val_predictions
print('Split Data MAE: ', mae)  # print mae
