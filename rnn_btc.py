### DATA PREPROCESSING

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the training set
dataset_train = pd.read_csv('BTC_Histoday_Prices_Train.csv')
training_set = dataset_train.iloc[:, 1:2].values

# Feature Scaling using normalization 
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0, 1))
training_set_scaled = sc.fit_transform(training_set)

# Creating a data structure with 90 timesteps and 1 output
X_train = []
y_train = []
for i in range(90, 1777):
    X_train.append(training_set_scaled[i-90:i, 0])
    y_train.append(training_set_scaled[i, 0])
X_train, y_train = np.array(X_train), np.array(y_train)

# Reshaping (adding another dimension)
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))



### BUILDING THE RNN MODEL

# Importing the Keras libraries 
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

# Initializing the RNN and adding 4 LSTM layers:
regressor = Sequential()

regressor.add(LSTM(units = 80, return_sequences = True, input_shape = (X_train.shape[1], 1)))
regressor.add(Dropout(0.2))

regressor.add(LSTM(units = 80, return_sequences = True))
regressor.add(Dropout(0.2))

regressor.add(LSTM(units = 80, return_sequences = True))
regressor.add(Dropout(0.2))

regressor.add(LSTM(units = 80))
regressor.add(Dropout(0.2))

# Adding the output layer
regressor.add(Dense(units = 1))

# Compiling the RNN model
regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')

# Fitting the RNN to the Training set
regressor.fit(X_train, y_train, epochs = 100, batch_size = 32)



### MAKING THE BTC PREDICTIONS AND VISUALIZING THE DATA

# Getting the real BTC  price for 2018
dataset_test = pd.read_csv('BTC_Histoday_Prices_Test.csv')
real_btc_price = dataset_test.iloc[:, 1:2].values

# Getting the predicted BTC price for 2018
dataset_total = pd.concat((dataset_train['Open'], dataset_test['Open']), axis = 0)
inputs = dataset_total[len(dataset_total) - len(dataset_test) - 90:].values
inputs = inputs.reshape(-1,1)
inputs = sc.transform(inputs)
X_test = []
for i in range(90, 120):
    X_test.append(inputs[i-90:i, 0])
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
predicted_btc_price = regressor.predict(X_test)
predicted_btc_price = sc.inverse_transform(predicted_btc_price)

# Visualising the results
plt.plot(real_btc_price, color = 'red', label = 'Real BTC Price')
plt.plot(predicted_btc_price, color = 'blue', label = 'Predicted BTC Price')
plt.title('Bitcoin Price Prediction in US Dollars')
plt.xlabel('Time')
plt.ylabel('Bitcoin Price')
plt.legend()
plt.show()
