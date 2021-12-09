import os
import pickle
import warnings

import numpy as np
import tensorflow as tf
from skimage.color import rgb2lab

from paletteGAN_utils import Dictionary


def prepare_dict():
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


def t2p_loader(batch_size):
    input_dict = prepare_dict()
    train_src_path = os.path.join('./data/hexcolor_vf/train_names.pkl')
    train_trg_path = os.path.join('./data/hexcolor_vf/train_palettes_rgb.pkl')

    with open(train_src_path, 'rb') as fin:
        src_seqs = pickle.load(fin)
    with open(train_trg_path, 'rb') as fin:
        trg_seqs = pickle.load(fin)

    words_index = []  # Going to be a list of 1 to 4xxx.
    word_id_to_palette_id = {}
    for index, palette_name in enumerate(src_seqs):
        for word in palette_name:
            word_id = input_dict.word2index[word]
            if word_id not in word_id_to_palette_id.keys():
                word_id_to_palette_id[word_id] = index
                words_index.append(word_id)
    src_seqs = tf.convert_to_tensor(words_index, dtype=tf.float32)

    palette_list = []
    for i in words_index:
        palette_id = word_id_to_palette_id[i]
        palette = trg_seqs[palette_id]
        temp = []
        for color in palette:
            rgb = np.array([color[0], color[1], color[2]]) / 255.0
            warnings.filterwarnings("ignore")
            lab = rgb2lab(rgb[np.newaxis, np.newaxis, :], illuminant='D50').flatten()
            temp.append(lab[0])
            temp.append(lab[1])
            temp.append(lab[2])
        palette_list.append(temp)
    trg_seqs = tf.convert_to_tensor(palette_list, dtype=tf.float32)

    data_tuple = (src_seqs, trg_seqs)
    dataset = tf.data.Dataset.from_tensor_slices(data_tuple)
    dataset_batched = dataset.batch(batch_size=batch_size)

    return dataset_batched, input_dict


def go_through_everything_test():
    batch_idx_list = []
    a_shape_list = []
    b_shape_list = []
    for batch_idx, (txt_embeddings, real_palettes) in enumerate(t2p_loader(32)):
        batch_idx_list.append(batch_idx)
        a_shape_list.append(txt_embeddings)
        b_shape_list.append(real_palettes)

    return batch_idx_list, a_shape_list, b_shape_list
