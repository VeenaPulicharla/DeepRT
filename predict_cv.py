import pandas as pd
import argparse
from models.pre_processing import pre_process
from models.cnn import cnn
from utils.predictions import predictions
from models.rnn import rnn
from threading import Thread
from multiprocessing.pool import ThreadPool
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

    pool = ThreadPool(processes=1)
    try:
        # CNN
        cnn_model = Thread(cnn(train_sequences, trainset['CV'])).start()
        # RNN
        rnn_model = Thread(rnn(train_sequences, trainset['CV'])).start()

    except:
        print("Error: unable to start thread")

    # Test
    print("...............................CNN Results...................................")
    predictions(cnn_model, test_sequences, testset['CV'])

    print("...............................RNN Results...................................")
    predictions(rnn_model, test_sequences, testset['CV'])


if __name__ == '__main__':
    main()