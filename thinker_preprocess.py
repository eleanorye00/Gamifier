import numpy as np


def get_data(train_file, test_file):

    with open(train_file, 'r') as f:
        opened_train_file = f.read()
        training = opened_train_file.split()

    with open(test_file, 'r') as f:
        opened_test_file = f.read()
        testing = opened_test_file.split()

    vocabulary = set(training)
    dictionary = {word: i for i, word in enumerate(vocabulary)}

    train_data = []
    for sentence in training:
        specific_sentence = dictionary.get(sentence)
        train_data.append(specific_sentence)
    train_data = np.array(train_data, dtype=np.int32)

    test_data = []
    for sentence in testing:
        specific_sentence = dictionary.get(sentence)
        test_data.append(specific_sentence)
    test_data = np.array(test_data, dtype=np.int32)

    return train_data, test_data, dictionary


def get_object_data(object_data):
    with open(object_data, 'r') as f:
        opened_object_data = f.read()
        object_dictionary = opened_object_data.split()
    vocabulary = set(object_dictionary)
    dictionary = {word: i for i, word in enumerate(vocabulary)}
    return dictionary


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


def rnn_vectorize_labels(data, window_size):
    labels = [[data[j] for j in range(i + 1, (window_size + i) + 1)] for i in
              range(0, len(data) - window_size, window_size)]
    return labels


def rnn_vectorize_inputs(data, window_size):
    inputs = [[data[j] for j in range(i, window_size + i)] for i in range(0, len(data) - window_size, window_size)]
    return inputs


def batch_getter(obj, batch_size, index):
    start = index - batch_size
    end = index
    obj = obj[start:end]
    return obj


def batch_getter_inputs(batch, size, index):
    return batch[index * size: (index * size) + size][:]


def batch_getter_labels(batch, size, index):
    return batch[index * size: (index * size) + size]


def rnn_vectorize_labels_wrong(data, window_size):
    for i in range(0, len(data) - window_size, window_size):
        labels = [data[j] for j in range(i + 1, (window_size + i) + 1)]
    # labels = np.array(labels, dtype=np.int32)
    return labels


def rnn_vectorize_inputs_wrong(data, window_size):
    for i in range(0, len(data) - window_size, window_size):
        inputs = [data[j] for j in range(i, window_size + i)]
    # inputs = np.array(inputs, dtype=np.int32)
    return inputs
