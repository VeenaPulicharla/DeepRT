from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense, Dropout, LeakyReLU
from utils.predictions import predictions
import tensorflow as tf
import keras.backend.tensorflow_backend as k


from keras.callbacks import EarlyStopping
early_stopping_monitor = EarlyStopping(monitor='mean_absolute_error',
                                       patience=15,
                                       mode='auto')


def build_model(input_shape):
    # with k.tf.device('/device:CPU:0'):
    #     k.set_session(k.tf.Session(config=k.tf.ConfigProto(allow_soft_placement=True, log_devuce_placement=True)))
    rnn_model = Sequential()
    rnn_model.add(LSTM(50, input_shape=input_shape, return_sequences=True))
    rnn_model.add(LSTM(50, return_sequences=True))
    rnn_model.add(LSTM(50))
    rnn_model.add(Dense(1))
    rnn_model.compile(loss='mse', optimizer='Adam', metrics=['mse', 'mae'])
    return rnn_model


def rnn(x_train, y_train):
    input_shape = (x_train.shape[1], x_train.shape[2])
    rnn_model = build_model(input_shape)
    print(".......................RNN Training started............................")
    rnn_model.fit(x_train,
                  y_train,
                  batch_size=30,
                  epochs=1,
                  verbose=2,
                  callbacks=[early_stopping_monitor],
                  validation_split=0.1)
    print("RNN Training Finished")
    # print("...............................Running RNN Predictions...................................")
    # preds = predictions(rnn_model, x_test, y_test)
    # print("...............................RNN  Predictions Finished......................................")
    # return preds
    return rnn_model
