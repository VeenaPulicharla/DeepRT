from utils.io import load_nn_model, load_sk_model, load_data
from utils.predictions import predictions, predict_rf
from utils.pre_processing import PreProcessing
import argparse
import pandas as pd
from itertools import chain


def run_predictions():
    parser = argparse.ArgumentParser(description='Input Parameters ')
    parser.add_argument(
        "-cnn",
        "--cnn_model_path",
        type=str,
        help='Location of saved cnn model',
        default="/home/veena/thermofisher_cv/cnn_model.h5"
    )
    parser.add_argument(
        "-rnn",
        "--rnn_model_path",
        type=str,
        help='Location of saved rnn model',
        default="/home/veena/thermofisher_cv/rnn_model.h5"
    )
    parser.add_argument(
        "-rf",
        "--rf_model_path",
        type=str,
        help='Location of saved random forest model',
        default="/home/veena/thermofisher_cv/rf_model.pickle"
    )
    parser.add_argument(
        "-test_path",
        "--test_data_path",
        type=str,
        help='Location of test dataset',
        default="/home/veena/thermofisher_cv/testset.csv"
    )

    args = parser.parse_args()
    cnn_filename = args.cnn_model_path
    rnn_filename = args.rnn_model_path
    rf_filename = args.rf_model_path
    data_path = args.test_data_path

    # load models
    cnn_model = load_nn_model(cnn_filename)
    rnn_model = load_nn_model(rnn_filename)
    rf_model = load_sk_model(rf_filename)


    preprocess_obj = PreProcessing()
    testset = load_data(data_path)
    test_sequences = preprocess_obj.pre_process(testset['peptide_seq'])
    cnn_x_test = preprocess_obj.cnn_pre_process(test_sequences)

    # get predictions
    cnn_preds = predictions(cnn_model, cnn_x_test)
    rnn_preds = predictions(rnn_model, test_sequences)

    features_df = pd.DataFrame({'cnn_preds': list(chain.from_iterable(cnn_preds)),
                                'rnn_preds': list(chain.from_iterable(rnn_preds))})

    print("................................ Results ...................................")
    predict_rf(rf_model, testset['CS'].values.reshape(-1,1), testset['CV'])


if __name__ == '__main__':
    run_predictions()