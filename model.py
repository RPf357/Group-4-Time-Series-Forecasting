import numpy as np
import pandas as pd
from datetime import datetime
from keras.models import Sequential
from keras.layers import Dense, LSTM
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Define the hyperparameters
# Number of layers
n_layers = 10

# Number of units in each layer
units = 64

# Number of epochs
epochs = 10

# Load the datasets
train_df = pd.read_csv('train.csv')
stores_df = pd.read_csv('stores.csv')
oil_df = pd.read_csv('oil.csv')
holidays_events_df = pd.read_csv('holidays_events.csv')

# Convert the date strings to datetime objects
train_df['date'] = pd.to_datetime(train_df['date'])

# Create a new DataFrame that contains all of the columns that you need for your model
merged_df = pd.DataFrame()
merged_df['date'] = train_df['date'].dt.dayofyear
merged_df['store_nbr'] = train_df['store_nbr']
merged_df['family'] = train_df['family']
merged_df['onpromotion'] = train_df['onpromotion']
merged_df['oil_price'] = oil_df['dcoilwtico']
merged_df['type'] = stores_df['type']
merged_df['sales'] = train_df['sales']

# Encode the categorical features
categorical_features = ['family', 'onpromotion', 'type']
for feature in categorical_features:
    encoder = LabelEncoder()
    merged_df[feature] = encoder.fit_transform(merged_df[feature])

# Preprocess the data
# Create a new column for the target variable
merged_df['target'] = merged_df['sales'].shift(-1)

# Drop any rows with missing values
merged_df.dropna(inplace=True)

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(merged_df[['date', 'store_nbr', 'family', 'onpromotion', 'oil_price', 'type']], merged_df['target'], test_size=0.25, shuffle=True, random_state = 42)

# Normalize the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Reshape the data for the LSTM model
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

# Create the LSTM model
model = Sequential()
for i in range(n_layers):
  model.add(LSTM(units, return_sequences=True, input_shape=(X_train.shape[1], 1) if i == 0 else (units,)))
model.add(Dense(1))

# Compile the model
model.compile(loss='mse', optimizer='adam')

# Train the model
model.fit(X_train, y_train, epochs=epochs)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
# Flatten the y_pred and y_test arrays before passing them to the np.mean() function
y_pred_flat = y_pred.ravel()

# Pad the shorter array with zeros
y_test_flat = np.pad(y_test.ravel(), (0, len(y_pred_flat) - len(y_test.ravel())), 'constant')

mse = np.mean((y_pred_flat - y_test_flat)**2)
mae = np.mean(np.abs(y_pred_flat - y_test_flat))
rmse = np.sqrt(mse)  # Calculate RMSE

print('MSE:', mse)
print("MAE:", mae)
print("RMSE:", rmse)

# Plot the training and validation loss curves
plt.figure(figsize=(10, 6))
plt.plot(train_loss, label='Train Loss')
plt.plot(val_loss, label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title('Training and Validation Loss Curves')
plt.show()

model.save('my_model.h5')

import pickle
pickle.dump(model, open('model.pkl', 'wb'))

model = pickle.load(open('model.pkl', 'rb'))

