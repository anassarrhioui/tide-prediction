# https://datascience.stackexchange.com/questions/19365/predict-sinus-with-keras-feed-forward-neural-network

from keras.models import Sequential
from keras.layers import Dense
import numpy as np
import matplotlib.pylab as plt

# Create dataset
x = np.arange(0, np.pi * 2, 0.1)
y = np.sin(x)

# Some parameters
ACTIVE_FUN = 'tanh'
BATCH_SIZE = 1
VERBOSE=0

# Create the model
model = Sequential()
model.add(Dense(5, input_shape=(1,), activation=ACTIVE_FUN))
model.add(Dense(5, activation=ACTIVE_FUN))
model.add(Dense(1, activation='linear'))

# Compile the model
model.compile(loss='mean_squared_error', optimizer='sgd', metrics=['mean_squared_error'])

# Fit the model
model.fit(x, y, epochs=1000, batch_size=BATCH_SIZE, verbose=1)

# Evaluate the model
scores = model.evaluate(x, y, verbose=VERBOSE)
print('%s: %.2f%%' % (model.metrics_names[1], scores[1] * 100))

# Make predictions
y_pred = model.predict(x)

# Plot
plt.plot(x, y, color='blue', linewidth=1, markersize='1')
plt.plot(x, y_pred, color='green', linewidth=1, markersize='1')
plt.xlabel('Angle [rad]')
plt.ylabel('sin(x)')
plt.axis('tight')
plt.show()