from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dense, Dropout, LeakyReLU
from keras.optimizers import Adam


from keras.callbacks import EarlyStopping
early_stopping_monitor = EarlyStopping(monitor='mean_absolute_error',
                                       patience=15,
                                       mode='auto')

def build_model(input_shape):
    rnn_model = Sequential()
    rnn_model.add(LSTM(50, input_shape=input_shape, return_sequences=True))
    rnn_model.add(LSTM(50, return_sequences=True))
    rnn_model.add(LSTM(50))
    rnn_model.add(Dense(1))
    rnn_model.compile(loss='mse', optimizer='Adam', metrics=['mse', 'mae'])
    return rnn_model

def rnn(train, test):
    input_shape = (train.shape[1], train.shape[2])
    print(input_shape)
    rnn_model = build_model(input_shape)
    print(".......................RNN Training started............................")
    rnn_model.fit(train,
                  test,
                  batch_size=30,
                  epochs=2,
                  verbose=1,
                  callbacks=[early_stopping_monitor],
                  validation_split=0.1)

    return rnn_model