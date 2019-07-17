from multiprocessing import Process, Pool, Queue
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout, LeakyReLU, LSTM, Conv2D, GlobalAveragePooling2D, MaxPooling2D



class NNProcess(Process):
    def __init__(self, process_id, nr_nets, ret_queue:Queue):
        super(NNProcess, self).__init__()
        self.process_id = process_id
        self.neural_nets = []
        self.nr_nets = nr_nets
        self.ret_queue = ret_queue

    def get_session_config(self):
        num_cores = 1
        num_cpu =1
        num_gpu = 0

        config = tf.ConfigProto(intra_op_parallelism_threads = num_cores, inter_op_parallelism_threads=num_cores,
                                allow_soft_placement = False, device_count = {'CPU': num_cpu, 'GPU':num_gpu})
        return config


    def build_cnn_model(input_shape):
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
        # model.compile(loss='mse', optimizer='Adam', metrics=['mse', 'mae'])
        return model

    def build_rnn_model(input_shape):
        rnn_model = Sequential()
        rnn_model.add(LSTM(50, input_shape=input_shape, return_sequences=True))
        rnn_model.add(LSTM(50, return_sequences=True))
        rnn_model.add(LSTM(50))
        rnn_model.add(Dense(1))
        # rnn_model.compile(loss='mse', optimizer='Adam', metrics=['mse', 'mae'])
        return rnn_model

    def build_model(self):
        neural_nets = list()
        neural_nets.append(self.build_cnn_model())
        neural_nets.append(self.build_rnn_model())

    def run(self):
        print("process " + str(self.process_id) + " starting...")

        with tf.Session(graph=tf.Graph(), config=self.get_session_config()) as session:
            self.build_model()
            self.compile()
            self.fit_nets()
            for i in range(0, self.nr_nets):
                file_name = self.neural_nets[i].name + "_" + str(i) + ".pickle"
                self.neural_nets[i].save(file_name)
                self.ret_queue.put(file_name)
        print("process " + str(self.process_id) + " finished.")

    def compile(self):
        for neural_net in self.neural_nets:
            neural_net.compile(loss='mse', optimizer='Adam', metrics=['mse', 'mae'])

    def fit_nets(self):
        for i in range(self.nr_nets):
            self.neural_nets[i].fit()
