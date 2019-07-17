import pandas as pd
import argparse
from models.pre_processing import pre_process
from models.cnn import cnn
from utils.predictions import predictions
from models.rnn import rnn
from threading import Thread
import multiprocessing
import time


def load_data(train_path, test_path):
    trainset = pd.read_csv(train_path)
    testset = pd.read_csv(test_path)
    return trainset, testset


def main():
    parser = argparse.ArgumentParser(description='Input Parameters ')
    parser.add_argument(
        "-train_path",
        "--train_data_path",
        type=str,
        help='Location of train dataset',
        default="/home/veena/thermofisher_cv/trainset.csv"
    )
    parser.add_argument(
        "-test_path",
        "--test_data_path",
        type=str,
        help='Location of test dataset',
        default="/home/veena/thermofisher_cv/testset.csv"
    )

    args = parser.parse_args()
    train_path = args.train_data_path
    test_path = args.test_data_path

    trainset, testset = load_data(train_path, test_path)
    train_sequences = pre_process(trainset['peptide_seq'])
    test_sequences = pre_process(testset['peptide_seq'])

    # Multi-processing
    manager = multiprocessing.Manager()
    model_return_dict = manager.dict()
    jobs = []

    p1 = multiprocessing.Process(target= cnn, args = (train_sequences, trainset['CV'], model_return_dict))
    p1.start()
    jobs.append(p1)

    p2 = multiprocessing.Process(target= rnn, args = (train_sequences, trainset['CV'], model_return_dict))
    p2.start()
    jobs.append((p2))

    for proc in jobs:
        proc.join()

    cnn_model = model_return_dict['cnn']
    rnn_model = model_return_dict['rnn']

    # CNN Model
    # cnn_model = cnn(train_sequences, trainset['CV'])
    # RNN Model
    # rnn_model = rnn(train_sequences, trainset['CV'])

    # get predictions
    print("...............................CNN Results...................................")
    predictions(cnn_model, test_sequences, testset['CV'], is_cnn=True)

    print("...............................RNN Results...................................")
    predictions(rnn_model, test_sequences, testset['CV'])


if __name__ == '__main__':
    main()