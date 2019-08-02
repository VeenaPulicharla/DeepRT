import pandas as pd
import argparse
import multiprocessing
import time
from utils.io import load_data
from utils.pre_processing import PreProcessing
from models.models import Models
from itertools import chain


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
        "-cnn",
        "--cnn_model_path",
        type=str,
        help='Location to save cnn model',
        default="/home/veena/thermofisher_cv/cnn_model.h5"
    )
    parser.add_argument(
        "-rnn",
        "--rnn_model_path",
        type=str,
        help='Location to save rnn model',
        default="/home/veena/thermofisher_cv/rnn_model.h5"
    )
    parser.add_argument(
        "-rf",
        "--rf_model_path",
        type=str,
        help='Location to save random forest model',
        default="/home/veena/thermofisher_cv/rf_model.pickle"
    )

    args = parser.parse_args()
    train_path = args.train_data_path
    cnn_filename = args.cnn_model_path
    rnn_filename = args.rnn_model_path
    rf_filename = args.rf_model_path

    trainset = load_data(train_path)

    pre_obj = PreProcessing()

    train_sequences = pre_obj.pre_process(trainset['peptide_seq'])
    cnn_train_sequences = pre_obj.cnn_pre_process(train_sequences)

    models = Models()

    start_time = time.time()

    # Multi-processing
    manager = multiprocessing.Manager()
    model_return_list = manager.list()
    jobs = []

    p1 = multiprocessing.Process(target=models.cnn,
                                 args=(cnn_train_sequences, trainset['CV'], cnn_filename, model_return_list))
    p1.start()
    jobs.append(p1)

    p2 = multiprocessing.Process(target=models.rnn,
                                 args=(train_sequences, trainset['CV'], rnn_filename, model_return_list))
    p2.start()

    jobs.append(p2)
    for proc in jobs:
        proc.join()

    features_df = pd.DataFrame({'cnn_preds': list(chain.from_iterable(model_return_list[0])),
                                # 'rnn_preds':  list(chain.from_iterable(model_return_list[1]))
                                })
    models.random_forest(trainset['CS'].values.reshape(-1,1), trainset['CV'], rf_filename)

    end_time = time.time() - start_time
    print("Model Training total time :", end_time)


if __name__ == '__main__':
    main()
