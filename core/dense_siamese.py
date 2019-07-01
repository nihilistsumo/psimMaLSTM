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

def get_xy(pair_data, embeddings):
    x = []
    y = []
    for p in pair_data:
        p1 = p[0].split("_")[0]
        p2 = p[0].split("_")[1]
        y.append(p[1])
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

def prepare_train_data(parapair_dict, embeddings, train_val_split=0.8):
    pages = parapair_dict.keys()
    train_pages = set(random.sample(pages, math.floor(len(pages) * train_val_split)))
    val_pages = pages - train_pages
    train_pairs = []
    val_pairs = []
    for page in train_pages:
        train_pos, train_neg = get_samples(parapair_dict[page])
        train_neg = random.sample(train_neg, len(train_pos))
        for p in train_pos:
            train_pairs.append([p, 1])
        for p in train_neg:
            train_pairs.append([p, 0])
    for page in val_pages:
        val_pos, val_neg = get_samples(parapair_dict[page])
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

def prepare_test_data(parapair_dict, embeddings):
    test_pairs = []
    for page in parapair_dict.keys():
        test_pos, test_neg = get_samples(parapair_dict[page])
        test_neg = random.sample(test_neg, len(test_pos))
        for p in test_pos:
            test_pairs.append([p, 1])
        for p in test_neg:
            test_pairs.append([p, 0])
    random.shuffle(test_pairs)

    Xtest, ytest = get_xy(test_pairs, embeddings)

    return Xtest, ytest


def dense_siamese(Xtrain, ytrain, Xval, yval, Xtest, ytest, embed_vec_len, optim,
           layer_size=10, out_layer_size=10, learning_rate=0.01, num_epochs=3, num_bacthes=1, pat=10):
    para_vec1 = Input(shape=(embed_vec_len,), dtype='float32', name='vec1')
    para_vec2 = Input(shape=(embed_vec_len,), dtype='float32', name='vec2')

    drop = Dropout(0.5)
    dense_layer = Dense(layer_size, activation='relu', input_shape=(embed_vec_len,), kernel_regularizer=regularizers.l2(0.001))
    p1_out = dense_layer(drop(para_vec1))
    p2_out = dense_layer(drop(para_vec2))

    concats = concatenate([p1_out, p2_out], axis=-1)
    concats_out = drop(concats)

    distance_out_layer = Dense(out_layer_size, activation='relu', input_shape=(2*layer_size,), kernel_regularizer=regularizers.l2(0.001))(concats_out)

    main_output = Dense(1, activation='relu')(distance_out_layer)

    model = Model(inputs=[para_vec1, para_vec2], outputs=[main_output])
    #
    if optim == 'adadelta':
        opt = keras.optimizers.Adadelta(lr=learning_rate, clipnorm=1.25)
    else:
        opt = keras.optimizers.Adam(lr=learning_rate)

    model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=pat)
    model.summary()

    history = model.fit([Xtrain[:, :embed_vec_len], Xtrain[:, embed_vec_len:]], ytrain,
                        validation_data=([Xval[:, :embed_vec_len], Xval[:, embed_vec_len:]], yval), epochs=num_epochs,
                        batch_size=num_bacthes, verbose=1, callbacks=[es])

def main():
    parser = argparse.ArgumentParser(description="Train and evaluate Dense-Siamese network for paragraph similarity task")
    parser.add_argument("-rd", "--train_parapair", required=True, help="Path to train parapair file")
    parser.add_argument("-sd", "--test_parapair", required=True, help="Path to test parapair file")
    parser.add_argument("-rem", "--train_embedding", required=True, help="Path to train embedding file")
    parser.add_argument("-sem", "--test_embedding", required=True, help="Path to test embedding file")
    parser.add_argument("-v", "--vec", required=True, type=int, help="Length of each paragraph vector")
    parser.add_argument("-dl", "--dense_layer_size", type=int, help="Size of each dense layer")
    parser.add_argument("-ol", "--out_layer_size", type=int, help="Size of output dense layer")
    parser.add_argument("-lr", "--learning_rate", type=float, help="Learning rate")
    parser.add_argument("-e", "--epochs", type=int, help="No. of epochs")
    parser.add_argument("-b", "--batches", type=int, required=True, help="No. of batches")
    parser.add_argument("-p", "--patience", type=int, required=True, help="Patience value; No of epochs to execute before early stopping")
    parser.add_argument("-opt", "--optimizer", help="Choose optimizer (adam/adadelta)")
    parser.add_argument("-o", "--out", required=True, help="Path to save trained keras model")

    args = vars(parser.parse_args())
    train_pp_file = args["train_parapair"]
    test_pp_file = args["test_parapair"]
    train_embedding_file = args["train_embedding"]
    test_embedding_file = args["test_embedding"]
    vec_len = args["vec"]
    dense_size = args["dense_layer_size"]
    out_size = args["out_layer_size"]
    learning_rate = args["learning_rate"]
    epochs = args["epochs"]
    batches = args["batches"]
    pat = args["patience"]
    optim = args["optimizer"]
    out_file = args["out"]

    with open(train_pp_file, 'r') as trpp:
        train_parapair = json.load(trpp)
    with open(test_pp_file, 'r') as tpp:
        test_parapair = json.load(tpp)
    train_emb = np.load(train_embedding_file, allow_pickle=True)
    test_emb = np.load(test_embedding_file, allow_pickle=True)

    Xtrain, ytrain, Xval, yval = prepare_train_data(train_parapair, train_emb)
    Xtest, ytest = prepare_test_data(test_parapair, test_emb)

    dense_siamese(Xtrain, ytrain, Xval, yval, Xtest, ytest, vec_len, optim, dense_size, out_size, learning_rate, epochs, batches, pat)

if __name__ == '__main__':
    main()