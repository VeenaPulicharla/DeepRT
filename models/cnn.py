from keras.models import Sequential
from keras.layers import Dense, Dropout, LeakyReLU
from keras.layers import Conv2D, GlobalAveragePooling2D, MaxPooling2D
from keras.optimizers import Adam

from keras.callbacks import EarlyStopping
early_stopping_monitor = EarlyStopping(monitor='mean_absolute_error',
                                       patience=15,
                                       mode='auto')


def cnn_pre_process(data):
    # seq_data = one_hot.reshape([one_hot.shape[0], one_hot.shape[1], one_hot.shape[2], 1])
    train_data = data.reshape([data.shape[0], data.shape[1], data.shape[2], 1])
    return train_data


def build_model(input_shape):
    model = Sequential()
    model.add(Conv2D(20, (3, 3), input_shape=input_shape))
    model.add(LeakyReLU())
    model.add(Conv2D(20, (3, 3)))
    model.add(LeakyReLU())
    #  model.add(MaxPooling2D(3))
    model.add(Conv2D(20, (3, 3)))
    model.add(LeakyReLU())
    model.add(Conv2D(20, (1, 1)))
    model.add(LeakyReLU())
    model.add(GlobalAveragePooling2D())
    model.add(Dropout(0.2))
    model.add(Dense(1))
    model.compile(loss='mse', optimizer='Adam', metrics=['mse', 'mae'])
    return model


def cnn(x_train, x_test):
    x_train = cnn_pre_process(x_train)
    input_shape = (x_train.shape[1], x_train.shape[2], 1)
    # Train
    model = build_model(input_shape)
    print("......................CNN Training Started.........................")
    model.fit(x_train,
              x_test.values,
              batch_size=10,
              epochs=2,
              callbacks=[early_stopping_monitor],
              verbose=1,
              validation_split=0.1)
    return model
