from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
from keras.utils import to_categorical


def tokenization(data):
    tokenizer = Tokenizer(char_level=True)
    tokenizer.fit_on_texts(data)
    interger_encoded_seq = tokenizer.texts_to_sequences(data)
    padded_sequences = sequence.pad_sequences(interger_encoded_seq,
                                              padding='post',
                                              maxlen=50,
                                              value=22)
    # vocab_size = len(
    #     tokenizer.word_index) + 1  # Adding 1 because of reserved 0 index
    # print("Vocabulary size(no.of unique chars present including padded digit)",
    #       vocab_size)
    return padded_sequences


def convert_onehot(data):
    one_hot = to_categorical(data)
    return one_hot


def pre_process(data):
    sequences = tokenization(data)
    one_hot_seq = convert_onehot(sequences)
    return one_hot_seq