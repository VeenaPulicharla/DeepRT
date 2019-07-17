from models import cnn

def accuracy(y_test, y_preds, diff):
    # y_true = y_test.tolist()
    un_normalized_pred = [i * 100 for i in y_preds]
    un_normalized_true = [i * 100 for i in y_test]
    predicted_correct = 0
    predicted_incorrect = 0

    for i, j in zip(un_normalized_true, un_normalized_pred):
        # print(i,j)
        if (abs(abs(i) - abs(j))) < diff:
            # print(i,j)
            predicted_correct += 1
        else:
            predicted_incorrect += 1
    print("predicted correct:", predicted_correct)
    print("predicted_incorrect", predicted_incorrect)
    print("with error differrence {}".format(diff), ":",
          (predicted_correct /
           (predicted_correct + predicted_incorrect)) * 100)


def predict(model, x_test):
    preds = model.predict(x_test)
    return preds


def predictions(model, x_test, y_test, is_cnn = False):
    if is_cnn:
        x_test = cnn.cnn_pre_process(x_test)

    preds = predict(model, x_test)
    accuracy(y_test, preds, 5)
    accuracy(y_test, preds, 10)
    accuracy(y_test, preds, 15)
    return preds
