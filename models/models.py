from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense, Dropout, LeakyReLU
from keras.layers import Conv2D, GlobalAveragePooling2D
from utils.io import save_nn_model, intermediate_output, save_sk_model
from sklearn.ensemble import RandomForestRegressor
from keras.callbacks import EarlyStopping


class Models(object):

    def __init__(self):
        self.early_stopping_monitor = EarlyStopping(monitor='mean_absolute_error', patience=10, mode='auto')

    def build_cnn_model(self, input_shape):
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
        model.add(Dense(1, name='dense_l1'))
        model.compile(loss='mse', optimizer='Adam', metrics=['mse', 'mae'])
        return model

    def build_rnn_model(self, input_shape):
        # with k.tf.device('/device:CPU:0'):
        #     k.set_session(k.tf.Session(config=k.tf.ConfigProto(allow_soft_placement=True, log_devuce_placement=True)))
        rnn_model = Sequential()
        rnn_model.add(LSTM(50, input_shape=input_shape, return_sequences=True))
        rnn_model.add(LSTM(50, return_sequences=True))
        rnn_model.add(LSTM(50))
        rnn_model.add(Dense(1, name='dense_last_layer'))
        rnn_model.compile(loss='mse', optimizer='Adam', metrics=['mse', 'mae'])
        return rnn_model

    def random_forest(self, x_train, y_train, filename):
        print("Trainig Random Forest")
        rf = RandomForestRegressor()
        rf.fit(x_train, y_train)
        save_sk_model(rf, filename)

    def cnn(self, x_train, y_train, filename, model_return_list):
        input_shape = (x_train.shape[1], x_train.shape[2], 1)
        # Train
        model = self.build_cnn_model(input_shape)

        print("......................CNN Training Started.........................")
        model.fit(x_train,
                  y_train.values,
                  batch_size= 30,
                  epochs=500,
                  callbacks=[self.early_stopping_monitor],
                  verbose=2,
                  validation_split=0.1)
        print("CNN Training Finished")
        save_nn_model(model, filename)
        # model.save('/home/veena/Documents/DeepRT/cnn_model.h5')
        output = intermediate_output('dense_l1', model, x_train)
        model_return_list.append(output)

    def rnn(self, x_train, y_train, filename, model_return_list):

        input_shape = (x_train.shape[1], x_train.shape[2])

        rnn_model = self.build_rnn_model(input_shape)
        print(".......................RNN Training started......."
              ".....................")
        rnn_model.fit(x_train,
                      y_train,
                      batch_size=30,
                      epochs=500,
                      verbose=2,
                      callbacks=[self.early_stopping_monitor],
                      validation_split=0.1)
        print("RNN Training Finished")
        save_nn_model(rnn_model, filename)
        # rnn_model.save('/home/veena/Documents/DeepRT/rnn_model.h5')
        output = intermediate_output('dense_last_layer', rnn_model, x_train)
        model_return_list.append(output)
