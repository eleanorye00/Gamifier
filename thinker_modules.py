import numpy as np
import tensorflow as tf
from tensorflow.keras import Model

from thinker_preprocess import batch_getter
from thinker_preprocess import get_data
from thinker_preprocess import get_object_data
from thinker_preprocess import rnn_vectorize_inputs
from thinker_preprocess import rnn_vectorize_labels


class Model(tf.keras.Model):

    def __init__(self, vocab_size):

        super(Model, self).__init__()

        self.vocab_size = vocab_size
        self.window_size = 20
        self.embedding_size = 128
        self.batch_size = 100
        self.lstm_rnn_size = 256
        self.initial_learning_rate = 0.01
        self.learning_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=self.initial_learning_rate, decay_steps=100, decay_rate=0.96, staircase=True)
        self.optimizer = tf.keras.optimizers.Adam(self.initial_learning_rate)

        self.E = tf.Variable(
            tf.random.normal([self.vocab_size, self.embedding_size], stddev=.1, dtype=tf.float32))
        self.LSTM = tf.keras.layers.LSTM(self.lstm_rnn_size, return_sequences=True, return_state=True)
        self.DENSE = tf.keras.layers.Dense(self.vocab_size, activation='softmax')

    def call(self, inputs, initial_state):
        if initial_state is None:
            embedding = tf.nn.embedding_lookup(self.E, inputs)
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

        tr_inputs = batch_getter(inputs_training, model.batch_size, i)
        tr_labels = batch_getter(labels_training, model.batch_size, i)

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
    if word not in vocab:
        print('Sorry! I can create 54 3-D objects, but this is not one of them yet! Please select your object from '
              'the list of available 3D models')
    word_to_id = {w: i for i, w in enumerate(list(vocab))}
    id_to_word = {i: w for w, i in vocab.items()}
    vectors = model.E.read_value()
    word_index = word_to_id[word]
    min_dist = 1000
    query_vector = vectors[word_index]
    word_list = []
    for index, vector in enumerate(vectors):
        if euclidean_distance(vector, query_vector) < min_dist and vector is not query_vector:
            min_dist = euclidean_distance(vector, query_vector)
            word_list.append((id_to_word[index], min_dist))
    word_list.sort(key=lambda x: x[-1])
    return word_list


def main():
    train_data, test_data, dictionary = get_data(
        '/Users/matthewstephens/PycharmProjects/RNN-Gamifier/data/train.txt',
        '/Users/matthewstephens/PycharmProjects/RNN-Gamifier/data/train.txt')
    vocab_size = len(dictionary)
    model = Model(vocab_size)
    train(model, train_data)
    perplexity = test(model, test_data)
    print("Perplexity score for this model is: ", perplexity)
    return model


def check(model):
    object_dictionary = get_object_data('/Users/matthewstephens/PycharmProjects/RNN-Gamifier/data/objects.txt')
    print("bag:", find_closest('bag', object_dictionary, model))
    print("basket:", find_closest('basket', object_dictionary, model))
    print("bathtub:", find_closest('bath', object_dictionary, model))
    print("bottle:", find_closest('bottle', object_dictionary, model))
    print("bowl:", find_closest('bowl', object_dictionary, model))
    print("can:", find_closest('can', object_dictionary, model))
    print("tin:", find_closest('tin', object_dictionary, model))
    print("car:", find_closest('car', object_dictionary, model))
    print("machine:", find_closest('machine', object_dictionary, model))
    print("chair:", find_closest('chair', object_dictionary, model))
    print("jar:", find_closest('jar', object_dictionary, model))
    print("knife:", find_closest('knife', object_dictionary, model))
    print("pillow:", find_closest('pillow', object_dictionary, model))
    print("pot:", find_closest('pot', object_dictionary, model))
    print("remote:", find_closest('remote', object_dictionary, model))
    print("rocket:", find_closest('rocket', object_dictionary, model))
    print("lounge:", find_closest('lounge', object_dictionary, model))
    print("tower:", find_closest('tower', object_dictionary, model))
    print("train:", find_closest('train', object_dictionary, model))
    print("vessel:", find_closest('vessel', object_dictionary, model))
