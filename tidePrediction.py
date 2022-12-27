# note !
# this is the best configuration yet for 1 and 2 months of traning
# 740 and 1400 hours

import tensorflow as tf
from tensorflow import keras
from keras import activations, initializers
import csv
import numpy as np
import matplotlib.pyplot as plt

# month_model = keras.models.Sequential([
#     keras.layers.Dense(5, input_shape=(1,), activation=activations.tanh),
#     keras.layers.Dense(5, activation=activations.tanh),
#     keras.layers.Dense(1, activation=activations.linear)
#     ])

def get_model( sequence_length : int = 20, hidden_neurons : int = 3  ):
    model = keras.models.Sequential([
        keras.layers.Input(shape=(sequence_length,)),
        # keras.layers.Dense(6, activation=activations.tanh),
        keras.layers.Dense(hidden_neurons, activation=activations.tanh),
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

def print_analysis(x, y):
    corr_indx = np.corrcoef(x , y)
    diff = x - y
    mse = sum( diff**2 ) / len( diff )
    print("mse : ", mse)
    print("corr : ", corr_indx[0][1])
    print("max err : ", max(abs(diff)))
    print("min err : ", min(abs(diff)))


def analyse_data( real_heights, predicted_heights ):
    print("-------------")
    print_analysis(real_heights, predicted_heights)
    print("-------------")

    for i in range( 0, len(real_heights), 24 ):
        x, y = real_heights[i:i+24], predicted_heights[i:i+24]
        print("---- day ",i//24 + 1)
        print_analysis(x, y)
        print("-from start-")
        x, y = real_heights[:i+24], predicted_heights[:i+24]
        print_analysis(x, y)
        print("----")


    plt.subplot(212)
    # plot ( real_heights, predicted_heights )
    a, b = np.polyfit(real_heights, predicted_heights, 1)
    plt.scatter(real_heights, predicted_heights)
    plt.plot(real_heights, a*real_heights+b)
    


def main():
    SEQ_LEN = 100
    PRED_DUR = 30
    HIDDEN_NEURONS = 1

    with open("dataset\\Achill_Island_MODELLED-1-4.csv", 'r') as f:
        data = list(csv.reader(f))

    data = np.array(data, dtype=np.float)
    time_data, height_data = data[:,0], data[:,1]

    # convert time to number of hours
    time_data = (time_data-time_data[0])/3600
    height_data = (height_data-2.25)/4.5

    time_data_train, time_data_test = time_data[:740], time_data[740:]
    height_data_train, height_data_test = height_data[:740], height_data[740:]


    time_data_train, x_train, y_train = reshape_data(time_data_train, height_data_train, sequence_length=SEQ_LEN)

    model = get_model(SEQ_LEN, HIDDEN_NEURONS)
    train_model(model, x_train, y_train, epochs=1, batch_size=None)
    
    y_predicted = predict(model, height_data_train, sequence_length=SEQ_LEN, prediction_duration=PRED_DUR)
    y_predicted = np.array(y_predicted)
    

    # -- scale back
    height_data = (height_data*4.5)+2.25
    y_train = (y_train*4.5)+2.25
    height_data_test = (height_data_test*4.5)+2.25
    y_predicted = (y_predicted*4.5)+2.25
    # ------

    analyse_data(height_data_test[:PRED_DUR], y_predicted)


    plt.subplot(211)
    plt.plot( time_data, height_data, 'y' )
    plt.plot( time_data_train, y_train, 'b' )
    plt.plot( time_data_test[:PRED_DUR], y_predicted, 'r' )

    plt.show()

if __name__ == "__main__":
    main()