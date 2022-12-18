import tensorflow as tf
from tensorflow import keras
from keras import activations, initializers
import csv
import numpy as np

# month_model = keras.models.Sequential([
#     keras.layers.Dense(5, input_shape=(1,), activation=activations.tanh),
#     keras.layers.Dense(5, activation=activations.tanh),
#     keras.layers.Dense(1, activation=activations.linear)
#     ])

month_model = keras.models.Sequential([
    keras.layers.Input(shape=(8,1)),
    keras.layers.LSTM(6),
    keras.layers.Dense(1, activation=activations.linear)
    ])

week_model = keras.models.Sequential([
    keras.layers.Input(shape=(1,)),
    keras.layers.Dense(6, activation=activations.tanh),
    keras.layers.Dense(1, activation=activations.linear)
    ])

loss = keras.losses.MeanSquaredError()
# optim = keras.optimizers.Adam(learning_rate=0.01)
optim = keras.optimizers.SGD(learning_rate=0.01)
metrics = [ keras.metrics.MeanSquaredError()]

month_model.compile(loss=loss, optimizer=optim, metrics=metrics)
week_model.compile(loss=loss, optimizer=optim, metrics=metrics)




def main():
    with open("dataset\\Achill_Island_MODELLED-1-2017.csv", 'r') as f:
        data = list(csv.reader(f))

    data = np.array(data, dtype=np.float)
    x_train, y_train = data[:,0], data[:,1]

    with open("dataset\Achill_Island_MODELLED-2-2017.csv", 'r') as f:
        data = list(csv.reader(f))
        data = np.array(data, dtype=np.float)

    x_test, y_test = data[:,0], data[:,1]

    # x_train, y_train = x_train[:20], y_train[:20]

    # x_train = (x_train-1483228800)/259200
    # x_test = (x_test-1483228800)/259200

    x_train = (x_train-1483228800)/36000
    x_test = (x_test-1483228800)/36000

    # y_train = (y_train-2)/4
    # y_test = (y_test-2)/4

    batch_size = None
    epochs = 1000
    # epochs = 100

    x_train=x_train.reshape(x_train.shape[0],1)

    month_model.evaluate( x_train, y_train )
    month_model.fit( x_train[:240], y_train[:240], batch_size=batch_size, epochs=epochs, verbose=1)
    month_model.evaluate( x_train, y_train )

    import matplotlib.pyplot as plt

    y_train_predicted = month_model.predict(x_train) 
    plt.plot( x_train[:300], y_train[:300], 'b' )
    plt.plot( [x_train[240]]*2, [0,4] )
    plt.plot( x_train[:300], y_train_predicted[:300], 'r' )

    plt.show()

if __name__ == "__main__":
    main()