import tensorflow as tf
from tensorflow import keras
from keras import activations
import csv
import numpy as np

month_model = keras.models.Sequential([
    keras.layers.Input(shape=(1,)),
    keras.layers.Dense(1, activation=activations.tanh),
    keras.layers.Dense(1, activation=activations.linear)
    ])

week_model = keras.models.Sequential([
    keras.layers.Input(shape=(1,)),
    keras.layers.Dense(6, activation=activations.tanh),
    keras.layers.Dense(1, activation=activations.linear)
    ])

loss = keras.losses.MeanSquaredError()
optim = keras.optimizers.Adam(learning_rate=0.0001)
metrics = [ keras.metrics.MeanSquaredError()]

month_model.compile(loss=loss, optimizer=optim, metrics=metrics)
week_model.compile(loss=loss, optimizer=optim, metrics=metrics)




def main():
    with open("dataset\Achill_Island_MODELLED-1-2017.csv", 'r') as f:
        data = list(csv.reader(f))

    data = np.array(data, dtype=np.float)
    x_train, y_train = data[:,0], data[:,1]

    with open("dataset\Achill_Island_MODELLED-2-2017.csv", 'r') as f:
        data = list(csv.reader(f))
        data = np.array(data, dtype=np.float)

    x_test, y_test = data[:,0], data[:,1]

    batch_size = 50
    epochs = 200
    # epochs = 100

    month_model.evaluate( x_train, y_train )
    # print(month_model.predict([1483268400]))
    month_model.fit( x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1)
    month_model.evaluate( x_train, y_train )
    # print(month_model.predict([1483268400]))

    import matplotlib.pyplot as plt

    y_test_predicted = month_model.predict(x_test) 
    plt.plot( x_test[:70], y_test[:70], 'b' )
    plt.plot( x_test[:70], y_test_predicted[:70], 'r' )

    plt.show()

if __name__ == "__main__":
    main()