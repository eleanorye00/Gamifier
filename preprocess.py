import tensorflow as tf
import numpy as np
from functools import reduce


def get_data(train_file, test_file):
    """
    Read and parse the train and test file line by line, then tokenize the sentences to build the train and test data separately.
    Create a vocabulary dictionary that maps all the unique tokens from your train and test data as keys to a unique integer value.
    Then vectorize your train and test data based on your vocabulary dictionary.

    :param train_file: Path to the training file.
    :param test_file: Path to the test file.
    :return: Tuple of train (1-d list or array with training words in vectorized/id form), test (1-d list or array with testing words in vectorized/id form), vocabulary (Dict containg index->word mapping)
    """
    # TODO: load and concatenate training data from training file.
    with open(train_file, 'r') as f:
        opened_train_file = f.read()
        # Create a vocabulary dictionary for the model
        training = opened_train_file.split()

    # TODO: load and concatenate testing data from testing file.
    with open(test_file, 'r') as f:
        opened_test_file = f.read()
        testing = opened_test_file.split()

    vocabulary = set(training)
    dictionary = {word: i for i, word in enumerate(vocabulary)}

    # TODO: read in and tokenize training data
    train_data = []
    for sentence in training:
        specific_sentence = dictionary.get(sentence)
        train_data.append(specific_sentence)
    train_data = np.array(train_data, dtype=np.int32)
    # print(train_data.shape)

    # TODO: read in and tokenize testing data
    test_data = []
    for sentence in testing:
        specific_sentence = dictionary.get(sentence)
        test_data.append(specific_sentence)
    test_data = np.array(test_data, dtype=np.int32)
    # print(test_data.shape)

    # BONUS: Ensure that all words appearing in test also appear in train
    # for word in test_data:
    #    if word not in train_data:
    #        print("Oop! There is a word that I don't know!", word)
    #    else:
    #       continue

    # TODO: return tuple of training tokens, testing tokens, and the vocab dictionary.
    return train_data, test_data, dictionary


# Not too sure if we need this or not
def get_word_id(vocab):
    id2word = {i: w for w, i in vocab.items()}
    return id2word


def ngram_vectorize_labels(data):
    STEP_SIZE = 2
    data_vector = np.array(data.shape[0], dtype=np.int32)
    labels = np.array(np.zeros((data_vector, 1)))  # .astype(dtype=np.int32)
    label_vector_size = data_vector - STEP_SIZE
    for i in range(label_vector_size):
        labels[i] = data[i + STEP_SIZE]
    labels = labels.astype(dtype=np.int32)
    return labels


def ngram_vectorize_inputs(data):
    STEP_SIZE = 2
    data_vector = np.array(data.shape[0], dtype=np.int32)
    inputs = np.zeros((data_vector, 2))  # .astype(dtype=np.int32)
    input_vector_size = data_vector - STEP_SIZE
    for i in range(input_vector_size):
        inputs[i, :] = data[i: i + STEP_SIZE]
    inputs = inputs.astype(dtype=np.int32)
    return inputs


# RIGHT - Keeps track of the time dimension (nested dimensionality) - needed by LSTM
def rnn_vectorize_labels(data, window_size):
    labels = [[data[j] for j in range(i + 1, (window_size + i) + 1)] for i in
              range(0, len(data) - window_size, window_size)]
    return labels


# RIGHT - Keeps track of the time dimension (nested dimensionality) - needed by LSTM
def rnn_vectorize_inputs(data, window_size):
    inputs = [[data[j] for j in range(i, window_size + i)] for i in range(0, len(data) - window_size, window_size)]
    return inputs


def batch_getter(obj, batch_size, index):
    start = index - batch_size
    end = index
    obj = obj[start:end]
    return obj


# Getter method to get the batch of the inputs
# Param: batch = inputs, size = size of batch, index = index of iteration
def batch_getter_inputs(batch, size, index):
    return batch[index * size: (index * size) + size][:]


# Getter method to get the batch of the labels
# Param: batch = labels, size = size of batch, index = index of iteration
def batch_getter_labels(batch, size, index):
    return batch[index * size: (index * size) + size]


# WRONG - Does not keep track of the time dimension; needs to be nested (keeping here to learn from it).
def rnn_vectorize_labels_wrong(data, window_size):
    for i in range(0, len(data) - window_size, window_size):
        labels = [data[j] for j in range(i + 1, (window_size + i) + 1)]
    # labels = np.array(labels, dtype=np.int32)
    return labels


# WRONG - Does not keep track of the time dimension; needs to be nested (keeping here to learn from it).
def rnn_vectorize_inputs_wrong(data, window_size):
    for i in range(0, len(data) - window_size, window_size):
        inputs = [data[j] for j in range(i, window_size + i)]
    # inputs = np.array(inputs, dtype=np.int32)
    return inputs
