import tensorflow as tf
import numpy as np
import os
import time
import matplotlib.pyplot as plt
plt.switch_backend('agg')
from skimage.color import lab2rgb

from paletteGAN_modules import *
from paletteGAN_utils import *
from paletteGAN_data_loader import *


class Solver(object):
    def __init__(self, args):
        self.args = args
        self.device = tf.device('cuda' if tf.test.is_gpu_available() else 'cpu')
        # Build the model.
        self.build_model(args.mode)


    def prepare_dict(self):
        input_dict = Dictionary()
        src_path = os.path.join('./data/hexcolor_vf/all_names.pkl')
        with open(src_path, 'rb') as f:
            text_data = pickle.load(f)
            f.close()

        print("Loading %s palette names..." % len(text_data))
        print("Making text dictionary...")

        for i in range(len(text_data)):
            input_dict.index_elements(text_data[i])

        return input_dict


    def build_model(self, mode):

        if mode == "TRAIN":
            self.input_dict = self.prepare_dict()
            self.train_loader, test_loader = t2p_loader(batch_size=32, input_dict=self.input_dict)

            # Load pre-trained GloVe embeddings.
            embeddings_filepath = os.path.join('./data', 'Color-Hex-vf.pth')
            if os.path.isfile(embeddings_filepath):
                W_emb = tf.data.experimental.load(embeddings_filepath)
            else:
                W_emb = load_pretrained_embedding(self.input_dict.word2index, '../data/glove.840B.300d.txt', 300)
                W_emb = tf.convert_to_tensor(W_emb)
                tf.data.experimental.save(W_emb, embeddings_filepath)
            W_emb = W_emb.to(self.device)

            # Instantiate generator and discriminator.
            self.encoder = EncoderRNN(self.input_dict.n_words, self.args.hidden_size,
                                      self.args.n_layers, self.args.dropout_p, W_emb).to(self.device)
            self.G = AttnDecoderRNN(self.input_dict, self.args.hidden_size,
                                    self.args.n_layers, self.args.dropout_p).to(self.device)
            self.D = Discriminator(15, self.args.hidden_size).to(self.device)

            # Initialize weights.
            self.encoder.apply(init_weights_normal)
            self.G.apply(init_weights_normal)
            self.D.apply(init_weights_normal)

            # Optimizer.
            self.G_parameters = list(self.encoder.parameters()) + list(self.G.parameters())
            # self.g_optimizer = torch.optim.Adam(self.G_parameters, lr=self.args.lr, weight_decay=self.args.weight_decay)
            # self.d_optimizer = torch.optim.Adam(self.D.parameters(), lr=self.args.lr, betas=(self.args.beta1, self.args.beta2))
            self.g_optimizer = tf.keras.optimizers.Adam(learning_rate=self.args.lr)
            self.d_optimizer = tf.keras.optimizers.Adam(learning_rate=self.args.lr,
                                                        beta_1=self.args.beta1, beta_2=self.args.beta2)

        elif mode == 'TEST':
            self.input_dict = self.prepare_dict()
            # Load pre-trained GloVe embeddings.
            emb_filepath = os.path.join('./data', 'Color-Hex-vf.pth')
            if os.path.isfile(emb_filepath):
                W_emb = tf.data.experimental.load(emb_filepath)
            else:
                W_emb = load_pretrained_embedding(self.input_dict.word2index,'../data/glove.840B.300d.txt',300)
                W_emb = tf.convert_to_tensor(W_emb)
                tf.data.experimental.save.save(W_emb, emb_filepath)
            W_emb = W_emb.to(self.device)
            # Data loader.
            # self.test_loader, self.imsize = test_loader(self.args.dataset, self.args.batch_size, self.input_dict)
            # Load the trained generators.
            self.encoder = EncoderRNN(self.input_dict.n_words, self.args.hidden_size,
                                          self.args.n_layers, self.args.dropout_p, W_emb).to(self.device)
            self.G_TPN = AttnDecoderRNN(self.input_dict, self.args.hidden_size,
                                        self.args.n_layers, self.args.dropout_p).to(self.device)


    def train(self):
        # Loss function.
        criterion_GAN = tf.keras.losses.BinaryCrossentropy()
        # criterion_smoothL1 = nn.SmoothL1Loss() [solution: directly use tf.compat.v1.losses.huber_loss()]

        # Start training from scratch.
        start_epoch = 0
        self.encoder.train()
        self.G.train()
        self.D.train()

        print('Start training...')
        start_time = time.time()
        for epoch in range(start_epoch, self.args.num_epochs):
            for batch_idx, (txt_embeddings, real_palettes) in enumerate(self.train_loader):

                # Compute text input size (without zero padding).
                batch_size = txt_embeddings.size(0)
                where = tf.not_equal(txt_embeddings, 0)
                nonzero_indices = list(tf.where(where)[:, 0])
                each_input_size = [nonzero_indices.count(j) for j in range(batch_size)]

                # Prepare training data.
                txt_embeddings = txt_embeddings
                real_palettes = real_palettes.float()

                # Prepare labels for the BCE loss.
                real_labels = tf.ones(batch_size).to(self.device)
                fake_labels = tf.zeros(batch_size).to(self.device)

                # Prepare input and output variables.

                # palette = torch.FloatTensor(batch_size, 3).zero_().to(self.device)
                palette = tf.zeros((batch_size, 3))
                # fake_palettes = torch.FloatTensor(batch_size, 15).zero_().to(self.device)
                with tf.device(self.device):
                    palette = tf.zeros((batch_size, 3))
                    fake_palettes = tf.zeros((batch_size, 15))

                    # Condition for the generator.
                    encoder_hidden = self.encoder.init_hidden(batch_size)
                    encoder_outputs, decoder_hidden, mu, logvar = self.encoder(txt_embeddings, encoder_hidden)

                # Generate color palette.
                for i in range(5):
                    palette, decoder_context, decoder_hidden, _ = self.G(palette,
                                                                         decoder_hidden.squeeze(0),
                                                                         encoder_outputs,
                                                                         each_input_size,
                                                                         i)
                    fake_palettes[:, 3 * i:3 * (i+1)] = palette

                # Condition for the discriminator.
                with tf.device(self.device):
                    each_input_size = tf.convert_to_tensor(each_input_size).to(self.device)
                each_input_size = each_input_size.unsqueeze(1).expand(batch_size, self.G.hidden_size)
                encoder_outputs = tf.reduce_sum(encoder_outputs, 0)
                encoder_outputs = tf.math.divide(encoder_outputs, each_input_size)

                # =============================== Train the discriminator =============================== #
                # Compute BCE loss using real palettes.
                real = self.D(real_palettes, encoder_outputs)
                d_loss_real = criterion_GAN(real, real_labels)

                # Compute BCE loss using fake palettes.
                fake = self.D(fake_palettes, encoder_outputs)
                d_loss_fake = criterion_GAN(fake, fake_labels)

                d_loss = d_loss_real + d_loss_fake

                # Backprop and optimize.
                self.d_optimizer.zero_grad()
                d_loss.backward(retain_graph=True)
                self.d_optimizer.step()

                # ================================ Train the generator ================================= #
                # Compute BCE loss (fool the discriminator).
                fake = self.D(fake_palettes, encoder_outputs)
                g_loss_GAN = criterion_GAN(fake, real_labels)

                # Compute smooth L1 loss.
                g_loss_smoothL1 = criterion_smoothL1(fake_palettes, real_palettes)

                # Compute KL loss.
                kl_loss = KL_loss(mu, logvar)

                g_loss = g_loss_GAN + g_loss_smoothL1 * self.args.lambda_sL1 + kl_loss * self.args.lambda_KL

                # Backprop and optimize.
                self.g_optimizer.zero_grad()
                g_loss.backward()
                self.g_optimizer.step()

            # For debugging. Save training output.
            if (epoch+1) % self.args.sample_interval == 0:
                for x in range(5):  # saving 5 samples
                    fig1, axs1 = plt.subplots(nrows=1, ncols=5)
                    input_text = ''
                    for idx in txt_embeddings[x]:
                        if idx.item() == 0: break
                        input_text += self.input_dict.index2word[idx.item()] + " "
                    axs1[0].set_title(input_text)
                    for k in range(5):
                        lab = np.array([fake_palettes.data[x][3*k],
                                        fake_palettes.data[x][3*k+1],
                                        fake_palettes.data[x][3*k+2]], dtype='float64')
                        rgb = lab2rgb_1d(lab)
                        axs1[k].imshow([[rgb]])
                        axs1[k].axis('off')

                    fig1.savefig(os.path.join(self.args.train_sample_dir,
                                              'epoch{}_sample{}.jpg'.format(epoch+1, x+1)))
                    plt.close()
                print('Saved train sample...')

            if (epoch+1) % self.args.log_interval == 0:
                elapsed_time = time.time() - start_time
                print('Elapsed time [{:.4f}], Iteration [{}/{}], '
                      'd_loss: {:.6f}, g_loss: {:.6f}'.format(
                       elapsed_time, (epoch+1), self.args.num_epochs,
                       d_loss.item(), g_loss.item()))

            if (epoch+1) % self.args.save_interval == 0:
                torch.save(self.encoder.state_dict(),
                           os.path.join(self.args.text2pal_dir, '{}_G_encoder.ckpt'.format(epoch+1)))
                torch.save(self.G.state_dict(),
                           os.path.join(self.args.text2pal_dir, '{}_G_decoder.ckpt'.format(epoch+1)))
                torch.save(self.D.state_dict(),
                           os.path.join(self.args.text2pal_dir, '{}_D.ckpt'.format(epoch+1)))
                print('Saved model checkpoints...')


