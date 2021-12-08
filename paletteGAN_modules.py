import tensorflow as tf
import numpy as np
tf.keras.backend.set_floatx('float64')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from paletteGAN_data_loader import *
from paletteGAN_utils import *


class Generator(tf.keras.Model):
    def __init__(self, c_dim):
        super(Generator, self).__init__()
        #self.leaky_relu = tf.keras.layers.LeakyReLU(0.01)
        self.model = tf.keras.Sequential()
        self.model.add(tf.keras.layers.Dense(300, input_shape=(c_dim,), activation="relu"))
        self.model.add(tf.keras.layers.Dense(300, activation="relu"))
        self.model.add(tf.keras.layers.Dense(75, activation="relu"))
        self.model.add = tf.keras.layers.Dense(15) # , activation='tanh')

    @tf.function
    def call(self, c: tf.Tensor) -> tf.Tensor:
        """Generates a batch of palettes given a tensor of class conditioning vectors.
        Inputs:
        - x: A [batch_size, input_sz] tensor of noise vectors
        Returns:
        TensorFlow Tensor with shape [batch_size, 15], containing the generated colors.
        """
        return self.model(c)


class Discriminator(tf.keras.Model):

    def __init__(self, palette_dim, c_dim):
        super(Discriminator, self).__init__()
        self.leaky_relu = tf.keras.layers.LeakyReLU(0.01)
        self.model = tf.keras.Sequential()
        self.model.add(tf.keras.layers.Dense(256, input_shape=(palette_dim + c_dim,), activation=self.leaky_relu))
        self.model.add(tf.keras.layers.Dense(256, activation=self.leaky_relu))
        self.model.add(tf.keras.layers.Dense(64, activation=self.leaky_relu))
        self.model.add(tf.keras.layers.Dense(1))

    @tf.function
    def call(self, palette: tf.Tensor, c: tf.Tensor) -> tf.Tensor:
        """Compute discriminator score for a batch of input palette and class condition.
        Inputs:
        - palette: TensorFlow Tensor of shape [batch_size, palette_dim], where palette_dim defaults to 15
        - c: TensorFlow Tensor of class condition of shape [batch_size, c_dim], where c_dim defaults to 300
        Returns:
        TensorFlow Tensor with shape [batch_size, 1], containing the score for a palette being real for
        each input (palette + class condition)."""
        assert palette.shape[0] == c.shape[0], "Palette and class condition must be of the same batch size."
        x = tf.concat([palette, c], axis=1)
        return self.model(x)


class PaletteGAN():
    def __init__(self, args):
        self.args = args
        self.generator = Generator(args["c_dim"])
        self.discriminator = Discriminator(args["palette_dim"], args["c_dim"])
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=args["lr"], beta_1=args["beta_1"])

    def generator_loss(self, logits_fake: tf.Tensor, logits_real: tf.Tensor) -> tf.Tensor:
        """Compute the discriminator loss.
        Inputs:
        - logits_real: Tensor, shape [batch_size, 1], output of discriminator for each real image.
        - logits_fake: Tensor, shape[batch_size, 1], output of discriminator for each fake image."""
        G_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(logits_fake),
                                                                        logits=logits_fake))
        return G_loss

    def discriminator_loss(self, logits_fake: tf.Tensor, logits_real: tf.Tensor) -> tf.Tensor:
        """Compute the discriminator loss."""
        D_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                                                    labels=tf.zeros_like(logits_fake),
                                                    logits=logits_fake))
        D_loss += tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                                                    labels=tf.ones_like(logits_real),
                                                    logits=logits_real))
        return D_loss

    def optimize(self, tape: tf.GradientTape, component: tf.keras.Model, loss: tf.Tensor):
        """ This optimizes a component (generator or discriminator) with respect to its loss.
        Inputs:
        - tape: the Gradient Tape
        - model: the model to be trained
        - loss: the model's loss."""
        with tape as tape:
            gradients = tape.gradient(loss, component.trainable_variables)
            self.optimizer.apply_gradients(zip(gradients, component.trainable_variables))




def train(model, train_loader, num_epochs=1000):

    print('Start training...')
    g_loss_history = np.zeros(num_epochs)
    d_loss_history = np.zeros(num_epochs)
    for epoch in range(num_epochs):

        # Keep track of per palette (i.e. example) loss
        epoch_g_loss = []
        epoch_d_loss = []
        for batch_idx, (txt_embeddings, real_palettes) in enumerate(train_loader):
            batch_size = txt_embeddings.size(0)

            with tf.GradientTape(persistent=True) as tape:
                # call, get outputs
                fake_palettes = model.generator(txt_embeddings)
                logits_real = model.discriminator(real_palettes, txt_embeddings)
                logits_fake = model.discriminator(fake_palettes, txt_embeddings)

                g_loss = model.generator_loss(logits_fake, logits_real)
                d_loss = model.discriminator_loss(logits_fake, logits_real)

            model.optimize(tape, model.generator, g_loss)
            model.optimize(tape, model.discriminator, d_loss)

            epoch_g_loss.append(g_loss/batch_size)
            epoch_d_loss.append(d_loss/batch_size)

        avg_g_loss = tf.reduce_mean(epoch_g_loss)
        avg_d_loss = tf.reduce_mean(epoch_d_loss)
        g_loss_history[epoch] = avg_g_loss
        d_loss_history[epoch] = avg_d_loss

        if epoch % 10 == 0 or epoch == num_epochs-1:
            print("Epoch", epoch,
                  "generator loss:", avg_g_loss, ", discriminator loss:", avg_d_loss)

    return model, g_loss_history, d_loss_history


if __name__ == "main":
    args = {
        "palette_dim": 15,
        "c_dim": 300,
        "lr": 1e-4,
        "beta_1": 0.5
        "batch_size": 32,

    }
    model = PaletteGAN(args=args)
    train_loader =

