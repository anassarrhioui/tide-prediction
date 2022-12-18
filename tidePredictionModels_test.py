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
    keras.layers.Input(shape=(20,)),
    keras.layers.Dense(5, activation=activations.tanh),
    keras.layers.Dense(1, activation=activations.linear)
    ])

loss = keras.losses.MeanSquaredError()
optim = keras.optimizers.Adam(learning_rate=0.01)
# optim = keras.optimizers.SGD(learning_rate=0.01)
metrics = [ keras.metrics.MeanSquaredError()]

month_model.compile(loss=loss, optimizer=optim, metrics=metrics)




def main():
    with open("dataset\\Achill_Island_MODELLED-1-2017.csv", 'r') as f:
        data = list(csv.reader(f))

    data = np.array(data, dtype=np.float)
    x_train, y_train = data[:,0], data[:,1]

    x_train = (x_train-1483228800)/3600

    

    X_data = []
    Y_data = []

    for i in range( 20, len(x_train) ):
        X_data.append( y_train[ i-20:i ] )
        Y_data.append( y_train[ i ] )

    X_data = np.array(X_data)
    Y_data = np.array(Y_data)

    X_train, Y_train = X_data[:200], Y_data[:200]
    X_test, Y_test = X_data[ 200 : 400], Y_data[200 : 400]


    # x_train=x_train.reshape(x_train.shape[0],1)

    batch_size = None
    epochs = 1000

    print(">>>>>>>>>")
    print( X_train.shape )
    month_model.evaluate( X_train, Y_train )
    month_model.fit( X_train, Y_train, batch_size=batch_size, epochs=epochs, verbose=1)
    month_model.evaluate( X_train, Y_train )

    import matplotlib.pyplot as plt

    y_train_predicted = month_model.predict(X_train) 
    plt.plot( x_train, y_train, 'y' )
    plt.plot( x_train[20:220], Y_train, 'b' )
    # plt.plot( [x_train[240]]*2, [0,4] )
    plt.plot( x_train[20:220], y_train_predicted, 'g' )

    y_test_predicted = month_model.predict(X_test) 
    plt.plot( x_train[220:420], y_test_predicted, 'r' )

    plt.show()

if __name__ == "__main__":
    main()