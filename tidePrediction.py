# note !
# this is the best configuration yet for 1 and 2 months of traning
# 740 and 1400 hours

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

def get_model( sequence_length : int = 20,  ):
    model = keras.models.Sequential([
        keras.layers.Input(shape=(sequence_length,)),
        # keras.layers.Dense(6, activation=activations.tanh),
        keras.layers.Dense(3, activation=activations.tanh),
        keras.layers.Dense(1, activation=activations.linear)
        ])
    loss = keras.losses.MeanSquaredError()
    # optim = keras.optimizers.Adam(learning_rate=0.01)
    optim = keras.optimizers.SGD(learning_rate=0.01)
    metrics = [ keras.metrics.MeanSquaredError()]

    model.compile(loss=loss, optimizer=optim, metrics=metrics)

    return model

def reshape_data( time_data, height_data, sequence_length=20 ):
    # -- convert time to number of hours
    # init_time = time_data[0]
    # time_data = ( time_data-init_time )/3600

    x_train = []
    y_train = []

    for i in range( sequence_length, len(time_data) ):
        x_train.append( height_data[ i-sequence_length:i ] )
        y_train.append( height_data[ i ] )

    x_train = np.array(x_train)
    y_train = np.array(y_train)

    return time_data[sequence_length:], x_train, y_train


def train_model(model, x_train, y_train, epochs = 100, batch_size=None):
    # model.evaluate( x_train, y_train )
    model.fit( x_train, y_train, epochs=epochs, batch_size=batch_size, verbose=1)
    # model.evaluate( x_train, y_train )

def predict(model, previous_data, sequence_length=20, prediction_duration=100):
    y_predicted = list(previous_data[-sequence_length:])
    
    for i in range(prediction_duration):
        previous_data = [ y_predicted[-sequence_length:] ]
        y_predicted.append( float( model.predict(previous_data)[0][0]) )
        

    # y_predicted = np.array(y_predicted)
    return y_predicted[sequence_length:]

    


def main():
    SEQ_LEN = 200
    PRED_DUR = 600

    with open("dataset\\Achill_Island_MODELLED-1-3.csv", 'r') as f:
        data = list(csv.reader(f))

    data = np.array(data, dtype=np.float)
    time_data, height_data = data[:,0], data[:,1]

    # convert time to number of hours
    time_data = (time_data-time_data[0])/3600
    height_data = (height_data-2.25)/4.5

    time_data_train, time_data_test = time_data[:740], time_data[740:]
    height_data_train, height_data_test = height_data[:740], height_data[740:]


    time_data_train, x_train, y_train = reshape_data(time_data_train, height_data_train, sequence_length=SEQ_LEN)

    model = get_model(SEQ_LEN)
    train_model(model, x_train, y_train, epochs=2000, batch_size=None)
    
    y_predicted = predict(model, height_data_train, sequence_length=SEQ_LEN, prediction_duration=PRED_DUR)
    y_predicted = np.array(y_predicted)
    

    import matplotlib.pyplot as plt

    # -- scale back
    height_data = (height_data*4.5)+2.25
    y_train = (y_train*4.5)+2.25
    y_predicted = (y_predicted*4.5)+2.25
    # ------
    plt.plot( time_data, height_data, 'y' )
    plt.plot( time_data_train, y_train, 'b' )
    plt.plot( time_data_test[:PRED_DUR], y_predicted, 'r' )

    plt.show()

if __name__ == "__main__":
    main()