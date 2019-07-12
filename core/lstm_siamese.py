import argparse, random, math, json
import numpy as np

import tensorflow as tf

import keras
import keras.backend as K
from keras.models import Model
from keras.layers import Activation
from keras.layers import Embedding, Input, TimeDistributed, Dropout
from keras.layers import LSTM, Lambda, concatenate, Dense, Bidirectional
from keras import regularizers
from keras.callbacks import EarlyStopping
from keras.utils import to_categorical
from keras.wrappers.scikit_learn import KerasClassifier

from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold

def pad_vec_sequence(vec_seq, max_seq_len=100):
    seq_len = vec_seq.shape[0]
    vec_len = vec_seq.shape[1]
    if seq_len > max_seq_len:

def get_xy(pair_data, embeddings, vec_len):
    x = []
    y = []
    for i in range(len(pair_data)):
        p = pair_data[i]
        p1 = p[0].split("_")[0]
        p2 = p[0].split("_")[1]
        y.append(p[1])
        p1_vec_seq = embeddings[()][p1]
        p2_vec_seq = embeddings[()][p2]
        x.append(np.hstack((embeddings[()][p1], embeddings[()][p2])))
    return np.array(x), np.array(y)

def get_samples(parapair_data):
    pairs = parapair_data["parapairs"]
    labels = parapair_data["labels"]
    pos_pairs = []
    neg_pairs = []
    for i in range(len(labels)):
        if labels[i] == 0:
            neg_pairs.append(pairs[i])
        else:
            pos_pairs.append(pairs[i])
    return pos_pairs, neg_pairs

def get_discriminative_samples(parapair_data, hier_qrels_reverse_dict):
    pairs = parapair_data["parapairs"]
    labels = parapair_data["labels"]
    pos_pairs = []
    neg_pairs = []
    for i in range(len(labels)):
        if labels[i] == 0:
            neg_pairs.append(pairs[i])
        else:
            p1 = pairs[i].split("_")[0]
            p2 = pairs[i].split("_")[1]
            if hier_qrels_reverse_dict[p1] == hier_qrels_reverse_dict[p2]:
                pos_pairs.append(pairs[i])
    return pos_pairs, neg_pairs

def prepare_train_data(parapair_dict, embeddings, hier_qrels_reverse, train_val_split=0.8):
    pages = parapair_dict.keys()
    train_pages = set(random.sample(pages, math.floor(len(pages) * train_val_split)))
    val_pages = pages - train_pages
    train_pairs = []
    val_pairs = []
    for page in train_pages:
        train_pos, train_neg = get_discriminative_samples(parapair_dict[page], hier_qrels_reverse)
        train_neg = random.sample(train_neg, len(train_pos))
        for p in train_pos:
            train_pairs.append([p, 1])
        for p in train_neg:
            train_pairs.append([p, 0])
    for page in val_pages:
        val_pos, val_neg = get_discriminative_samples(parapair_dict[page], hier_qrels_reverse)
        val_neg = random.sample(val_neg, len(val_pos))
        for p in val_pos:
            val_pairs.append([p, 1])
        for p in val_neg:
            val_pairs.append([p, 0])
    random.shuffle(train_pairs)
    random.shuffle(val_pairs)

    Xtrain, ytrain = get_xy(train_pairs, embeddings)
    Xval, yval = get_xy(val_pairs, embeddings)

    return Xtrain, ytrain, Xval, yval