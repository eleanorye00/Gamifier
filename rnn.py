import tensorflow as tf
import numpy as np
from tensorflow.keras import Model
from preprocess import get_data
from preprocess import batch_getter
from preprocess import rnn_vectorize_inputs
from preprocess import rnn_vectorize_labels


class Model(tf.keras.Model):

    def __init__(self, vocab_size):

        super(Model, self).__init__()

        self.vocab_size = vocab_size
        self.window_size = 20  # DO NOT CHANGE!
        self.embedding_size = 128  # TODO
        self.batch_size = 100  # TODO
        self.lstm_rnn_size = 256
        self.learning_rate = 0.008
        self.optimizer = tf.keras.optimizers.Adam(self.learning_rate)

        self.E = tf.Variable(
            tf.random.normal([self.vocab_size, self.embedding_size], stddev=.1, dtype=tf.float32))
        self.LSTM = tf.keras.layers.LSTM(self.lstm_rnn_size, return_sequences=True, return_state=True)
        self.DENSE = tf.keras.layers.Dense(self.vocab_size, activation='softmax')

    def call(self, inputs, initial_state):
        if initial_state is None:
            embedding = tf.nn.embedding_lookup(self.E, inputs)
            # FOR GRU:
            # whole_seq_output, final_state = self.GRU(embedding, initial_state)
            # FOR LSTM:
            whole_seq_output, final_memory_state, final_carry_state = self.LSTM(embedding, initial_state)
            final_state = (final_memory_state, final_carry_state)
            fully_connected_layer = self.DENSE(whole_seq_output)
        else:
            assert False, "Initial state detected. I.S. used in GRU; this is a LSTM network"
        return fully_connected_layer, final_state

    def loss(self, probs, labels):
        loss_wo_mean = tf.keras.losses.sparse_categorical_crossentropy(labels, probs, from_logits=False)
        loss = tf.reduce_mean(loss_wo_mean)
        return loss


def train(model, train_inputs):
    inputs_training = rnn_vectorize_inputs(train_inputs, model.window_size)
    labels_training = rnn_vectorize_labels(train_inputs, model.window_size)

    initial_state = None
    curr_loss = 0
    step = 0
    for i in range(model.batch_size, len(labels_training) + model.batch_size, model.batch_size):

        # Get the batch for the inputs
        tr_inputs = batch_getter(inputs_training, model.batch_size, i)
        # Get the batch for the labels
        tr_labels = batch_getter(labels_training, model.batch_size, i)

        # Use back propagation to calculate the gradients for the weights and biases (perform gradient descent)
        with tf.GradientTape() as tape:
            probabilities, _ = model.call(tr_inputs, initial_state)
            loss = model.loss(probabilities, tr_labels)

        curr_loss += loss
        step += 1
        if i % 100 == 0:
            print('Batch %d\tLoss: %.3f' % (i, curr_loss / step))

        gradient_descent = tape.gradient(loss, model.trainable_variables)
        model.optimizer.apply_gradients(zip(gradient_descent, model.trainable_variables))


def test(model, test_inputs):
    inputs_testing = rnn_vectorize_inputs(test_inputs, model.window_size)
    labels_testing = rnn_vectorize_labels(test_inputs, model.window_size)

    loss_list = []
    curr_loss = 0
    step = 0
    for i in range(model.batch_size, len(labels_testing) + model.batch_size, model.batch_size):

        te_inputs = batch_getter(inputs_testing, model.batch_size, i)
        te_labels = batch_getter(labels_testing, model.batch_size, i)

        probabilities, _ = model.call(te_inputs, None)
        loss = model.loss(probabilities, te_labels)
        loss_list.append(loss)

        curr_loss += loss
        step += 1
        if i % 100 == 0:
            print('Test Batch %d\tLoss: %.3f' % (i, curr_loss / step))

    perplexity = np.exp(tf.reduce_mean(loss_list))
    return perplexity


def euclidean_distance(vec1, vec2):
    return np.sqrt(np.sum((vec1 - vec2) ** 2))


def find_closest(word, vocab, model):
    # vocab = 54 words
    # model already has trained weights
    reverse_vocab = {idx: word for word, idx in vocab.items()}

    first_string = word
    word_index = vocab[word]
    next_input = [[word_index]]
    word_list = [first_string]

    # logits, previous_state = model.call(next_input, None)

    min_dist = 10000  # to act like positive infinity
    min_index = -1
    # for index, vector in enumerate(logits):

    for i in range(10):
        logits, previous_state = model.call(next_input, None)
        query_vector = logits[word_index]

        if euclidean_distance(logits[i], query_vector) < min_dist and not np.array_equal(logits[i], query_vector):
            min_dist = euclidean_distance(logits, query_vector)
            min_index = query_vector
            closest_word = reverse_vocab[min_index]
            word_list.append(closest_word)

            # top = np.argsort(logits)[-i:]
            top_10 = np.argsort(logits)[-10:]
            n_logits = np.exp(logits[top_10]) / np.exp(logits[top_10]).sum()
            out_index = np.random.choice(top_10, p=n_logits)
            next_input = [[out_index]]

    return closest_word


def order_polygen(word, vocab, model):


def generate_sentence(word1, length, vocab, model, sample_n=10):
    """
    Takes a model, vocab, selects from the most likely next word from the model's distribution

    :param model: trained RNN model
    :param vocab: dictionary, word to id mapping
    :return: None
    """

    # NOTE: Feel free to play around with different sample_n values

    reverse_vocab = {idx: word for word, idx in vocab.items()}
    previous_state = None

    first_string = word1
    first_word_index = vocab[word1]
    next_input = [[first_word_index]]
    text = [first_string]

    for i in range(length):
        logits, previous_state = model.call(next_input, previous_state)
        logits = np.array(logits[0, 0, :])
        top_n = np.argsort(logits)[-sample_n:]
        n_logits = np.exp(logits[top_n]) / np.exp(logits[top_n]).sum()
        out_index = np.random.choice(top_n, p=n_logits)

        text.append(reverse_vocab[out_index])
        next_input = [[out_index]]

    print(" ".join(text))


def main():
    train_data, test_data, dictionary = get_data(
        'data/train.txt',
        'data/test.txt')

    # TODO: initialize model
    vocab_size = len(dictionary)
    model = Model(vocab_size)

    # TODO: Set-up the training step
    train(model, train_data, train_data)

    # TODO: Set up the testing steps
    perplexity = test(model, test_data, test_data)
    print("Perplexity score = ", perplexity)


if __name__ == '__main__':
    main()
