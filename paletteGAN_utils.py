import numpy as np
import warnings
from skimage.color import lab2rgb, rgb2lab
import os
import pickle


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

    print ("%d/%d vocabs are initialized with GloVe embeddings." % (n, vocab_size))
    return W_emb


class Embed(nn.Module):
    def __init__(self, vocab_size, embed_dim, W_emb, train_emb):
        super(Embed, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)

        if W_emb is not None:
            print ("Using pre-trained word embeddings...")
            self.embed.weight = nn.Parameter(W_emb)

        if train_emb == False:
            print ("Not training word embeddings...")
            self.embed.requires_grad = False

    def forward(self, doc):
        doc = self.embed(doc)
        return doc


# ======================== For processing data ======================== #

def process_palette_ab(pal_data, batch_size):
    img_a_scale = (pal_data[:, :, 1:2] + 88) / 185
    img_b_scale = (pal_data[:, :, 2:3] + 127) / 212
    img_ab_scale = np.concatenate((img_a_scale, img_b_scale), axis=2)
    ab_for_global = torch.from_numpy(img_ab_scale).float()
    ab_for_global = ab_for_global.view(batch_size, 10).unsqueeze(2).unsqueeze(2)
    return ab_for_global

def process_palette_lab(pal_data, batch_size):
    img_l = pal_data[:, :, 0:1] / 100
    img_a_scale = (pal_data[:, :, 1:2] + 88) / 185
    img_b_scale = (pal_data[:, :, 2:3] + 127) / 212
    img_lab_scale = np.concatenate((img_l, img_a_scale, img_b_scale), axis=2)
    lab_for_global = torch.from_numpy(img_lab_scale).float()
    lab_for_global = lab_for_global.view(batch_size, 15).unsqueeze(2).unsqueeze(2)
    return lab_for_global

def process_global_ab(input_ab, batch_size, always_give_global_hint):
    X_hist = input_ab

    if always_give_global_hint:
        B_hist = torch.ones(batch_size, 1, 1, 1)
    else:
        B_hist = torch.round(torch.rand(batch_size, 1, 1, 1))
        for l in range(batch_size):
            if B_hist[l].numpy() == 0:
                X_hist[l] = torch.rand(10)

    global_input = torch.cat([X_hist, B_hist], 1)
    return global_input

def process_global_lab(input_lab, batch_size, always_give_global_hint):
    X_hist = input_lab

    if always_give_global_hint:
        B_hist = torch.ones(batch_size, 1, 1, 1)
    else:
        B_hist = torch.round(torch.rand(batch_size, 1, 1, 1))
        for l in range(batch_size):
            if B_hist[l].numpy() == 0:
                X_hist[l] = torch.rand(15)

    global_input = torch.cat([X_hist, B_hist], 1)
    return global_input





# ============================= Etc. ============================= #
def KL_loss(mu, logvar):
    KLD_element = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
    KLD = torch.mean(KLD_element).mul_(-0.5)
    return KLD


def lab2rgb_1d(in_lab, clip=True):
    warnings.filterwarnings("ignore")
    tmp_rgb = lab2rgb(in_lab[np.newaxis, np.newaxis, :], illuminant='D50').flatten()
    if clip:
        tmp_rgb = np.clip(tmp_rgb, 0, 1)
    return tmp_rgb

def init_weights_normal(m):
    if type(m) == nn.Conv1d:
        m.weight.data.normal_(0.0, 0.05)
    if type(m) == nn.Linear:
        m.weight.data.normal_(0.0, 0.05)
