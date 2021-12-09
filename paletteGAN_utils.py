import numpy as np
import warnings
from skimage.color import lab2rgb, rgb2lab
import os
import pickle
import matplotlib.pyplot as plt

# ======================== For text embeddings ======================== #
SOS_token = 0
EOS_token = 1


class Dictionary:
    def __init__(self):
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "SOS", 1: "EOS"}
        self.n_words = 2
        self.max_len = 0

    def index_elements(self, data):
        for element in data:
            self.max_len = len(data) if self.max_len < len(data) else self.max_len
            self.index_element(element)

    def index_element(self, element):
        if element not in self.word2index:
            self.word2index[element] = self.n_words
            self.word2count[element] = 1
            self.index2word[self.n_words] = element
            self.n_words += 1
        else:
            self.word2count[element] += 1


def load_pretrained_embedding(dictionary, embed_file, embed_dim):
    if embed_file is None: return None

    pretrained_embed = {}
    with open(embed_file, 'r', encoding='utf-8') as f:
        for line in f:
            tokens = line.split(' ')
            word = tokens[0]
            entries = tokens[1:]
            if word == '<unk>':
                continue
            pretrained_embed[word] = entries
        f.close()

    vocab_size = len(dictionary) + 2
    W_emb = np.random.randn(vocab_size, embed_dim).astype('float32')
    n = 0
    for word, index in dictionary.items():
        if word in pretrained_embed:
            W_emb[index, :] = pretrained_embed[word]
            n += 1

    print("%d/%d vocabs are initialized with GloVe embeddings." % (n, vocab_size))
    return W_emb


def lab2rgb_1d(in_lab, clip=True):
    warnings.filterwarnings("ignore")
    tmp_rgb = lab2rgb(in_lab[np.newaxis, np.newaxis, :], illuminant='D50').flatten()
    if clip:
        tmp_rgb = np.clip(tmp_rgb, 0, 1)
    return tmp_rgb


def display_palette(GAN_output, word: str, save_to_local=True, set_title=True):
    """
    Display the generated palette along with the input word.
    :param GAN_output: an Numpy array of 1x15, containing rgb values for the 5 colors but normalized to 0-1.
    :param word: the input to generator. Used here as the title for generated plot.
    :return: none.
    """
    GAN_output *= 255
    GAN_output = GAN_output.reshape(5, 3)
    I = np.array([[0, 1, 2, 3, 4]])
    RGB = GAN_output[I]
    plt.imshow(RGB)
    if set_title:
        plt.title(word)
    if save_to_local:
        plt.savefig(os.path.join('./paletteGAN_outputs', word+'.png'))
