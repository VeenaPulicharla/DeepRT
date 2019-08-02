from keras.models import load_model
import pandas as pd
from keras.models import Model
import pickle
import joblib


def load_nn_model(filename):
    model = load_model(filename)
    return model


def save_nn_model(model, filename):
    model.save(filename)
    print("Model saved successfully")


def load_sk_model(filename):
    with open(filename, 'rb') as fl:
        model = pickle.load(fl)
    return model


def save_sk_model(model, filename):
    with open(filename, 'wb') as fs:
        pickle.dump(model, fs)
    print("Saved RF model successfully")


def intermediate_output(layer_name, model, x_train):
    intermediate_layer_model = Model(inputs=model.input,
                                     outputs=model.get_layer(layer_name).output)
    intermediate_op = intermediate_layer_model.predict(x_train)
    return intermediate_op


def load_data(path):
    path = pd.read_csv(path)
    return path
