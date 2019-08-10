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
from sklearn.metrics import roc_auc_score

def get_xy(pair_data, elmo_embeddings, bbc_cat_embeddings, vartn):
    x = []
    y = []
    for i in range(len(pair_data)):
        p = pair_data[i]
        p1 = p[0].split("_")[0]
        p2 = p[0].split("_")[1]
        y.append(p[1])
        if vartn == 1:
            p1vec = np.hstack((bbc_cat_embeddings[()][p1], elmo_embeddings[()][p1]))
            p2vec = np.hstack((bbc_cat_embeddings[()][p2], elmo_embeddings[()][p2]))
        elif vartn == 2:
            p1vec = elmo_embeddings[()][p1]
            p2vec = elmo_embeddings[()][p2]
        else:
            p1vec = bbc_cat_embeddings[()][p1]
            p2vec = bbc_cat_embeddings[()][p2]
        x.append(np.hstack((p1vec, p2vec)))
    return np.array(x), np.array(y)

def prepare_data(parapair_dict, elmo_embeddings, bbc_cat_embeddings, vartn):
    pages = parapair_dict.keys()
    pairs = []
    for page in pages:
        page_pairs = parapair_dict[page]['parapairs']
        page_labels = parapair_dict[page]['labels']
        assert len(page_pairs) == len(page_labels)
        for i in range(len(page_pairs)):
            if page_labels[i] == 1:
                pairs.append([page_pairs[i], 1])
            else:
                pairs.append([page_pairs[i], 0])
    random.shuffle(pairs)

    X, y = get_xy(pairs, elmo_embeddings, bbc_cat_embeddings, vartn)

    return X, y

def precision(ytrue, yhat):
    true_pos = K.sum(K.round(K.clip(ytrue * yhat, 0, 1)))
    predicted_pos = K.sum(K.round(K.clip(yhat, 0, 1)))
    precision = true_pos / (predicted_pos + K.epsilon())
    return precision

def recall(ytrue, yhat):
    true_pos = K.sum(K.round(K.clip(ytrue * yhat, 0, 1)))
    possible_pos = K.sum(K.round(K.clip(ytrue, 0, 1)))
    recall = true_pos / (possible_pos + K.epsilon())
    return recall

def fbeta_score(ytrue, yhat, beta=1):
    """Computes the F score.
    The F score is the weighted harmonic mean of precision and recall.
    Here it is only computed as a batch-wise average, not globally.
    This is useful for multi-label classification, where input samples can be
    classified as sets of labels. By only using accuracy (precision) a model
    would achieve a perfect score by simply assigning every class to every
    input. In order to avoid this, a metric should penalize incorrect class
    assignments as well (recall). The F-beta score (ranged from 0.0 to 1.0)
    computes this, as a weighted mean of the proportion of correct class
    assignments vs. the proportion of incorrect class assignments.
    With beta = 1, this is equivalent to a F-measure. With beta < 1, assigning
    correct classes becomes more important, and with beta > 1 the metric is
    instead weighted towards penalizing incorrect class assignments.
    """
    if beta < 0:
        raise ValueError('The lowest choosable beta is zero (only precision).')

    # If there are no true positives, fix the F score at 0 like sklearn.
    if K.sum(K.round(K.clip(ytrue, 0, 1))) == 0:
        return 0

    p = precision(ytrue, yhat)
    r = recall(ytrue, yhat)
    bb = beta ** 2
    fbeta_score = (1 + bb) * (p * r) / (bb * p + r + K.epsilon())
    return fbeta_score


def fmeasure(ytrue, yhat):
    # also known as f1 measure
    return fbeta_score(ytrue, yhat, beta=1)

def dense_siamese_bbc_cat(Xtrain, ytrain, Xval, yval, Xtest, ytest, embed_vec_len, optim,
           layer_size=10, reg=0.0001, learning_rate=0.01, num_epochs=3, num_bacthes=1, pat=10):
    para_vec1 = Input(shape=(embed_vec_len,), dtype='float32', name='vec1')
    para_vec2 = Input(shape=(embed_vec_len,), dtype='float32', name='vec2')

    drop = Dropout(0.5)
    dense_layer1 = Dense(layer_size, activation='relu', input_shape=(embed_vec_len,), kernel_regularizer=regularizers.l2(reg))
    p1_d1_out = dense_layer1(drop(para_vec1))
    p2_d1_out = dense_layer1(drop(para_vec2))

    # dense_layer2 = Dense(layer_size, activation='relu', input_shape=(layer_size,), kernel_regularizer=regularizers.l2(0.001))
    dense_layer2 = Dense(layer_size, activation='relu', input_shape=(layer_size,), kernel_regularizer=regularizers.l2(reg))
    p1_d2_out = dense_layer2(p1_d1_out)
    p2_d2_out = dense_layer2(p2_d1_out)

    concats = concatenate([p1_d2_out, p2_d2_out], axis=-1)

    dense_layer_3 = Dense(layer_size, activation='relu', input_shape=(2 * layer_size,), kernel_regularizer=regularizers.l2(reg))
    d3_out = dense_layer_3(concats)

    distance_out_layer = Dense(1, activation='sigmoid', input_shape=(layer_size,),
                               kernel_regularizer=regularizers.l2(reg), name='distance')(d3_out)

    model = Model(inputs=[para_vec1, para_vec2], outputs=[distance_out_layer])

    if optim == 'adadelta':
        opt = keras.optimizers.Adadelta(lr=learning_rate, clipnorm=1.25)
    else:
        opt = keras.optimizers.Adam(lr=learning_rate)

    model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy', precision, recall, fmeasure])
    es = EarlyStopping(monitor='val_fmeasure', mode='max', verbose=1, patience=pat)
    model.summary()

    history = model.fit([Xtrain[:, :embed_vec_len], Xtrain[:, embed_vec_len:]], ytrain,
                        validation_data=([Xval[:, :embed_vec_len], Xval[:, embed_vec_len:]], yval), epochs=num_epochs,
                        batch_size=num_bacthes, verbose=1, callbacks=[es])

    return model

def main():
    parser = argparse.ArgumentParser(description="Train and evaluate Dense-Siamese network for paragraph similarity task")
    parser.add_argument("-vr", "--variation", type=int, help="1: bbc+elmo embed, 2: elmo embed, 3: bbc embed")
    parser.add_argument("-rd", "--train_parapair", required=True, help="Path to train parapair file")
    parser.add_argument("-sd", "--test_parapair", required=True, help="Path to test parapair file")
    parser.add_argument("-rem", "--train_elmo", required=True, help="Path to train elmo embedding file")
    parser.add_argument("-sem", "--test_elmo", required=True, help="Path to test elmo embedding file")
    parser.add_argument("-rbe", "--train_bbc", required=True, help="Path to train bbc cat embedding file")
    parser.add_argument("-sbe", "--test_bbc", required=True, help="Path to test bbc cat embedding file")
    parser.add_argument("-v", "--vec", required=True, type=int, help="Length of each paragraph vector (elmo+bbc)")
    parser.add_argument("-dl", "--dense_layer_size", type=int, help="Size of each dense layer")
    parser.add_argument("-reg", "--regularization", type=float, help="Regularization value")
    parser.add_argument("-lr", "--learning_rate", type=float, help="Learning rate")
    parser.add_argument("-e", "--epochs", type=int, help="No. of epochs")
    parser.add_argument("-b", "--batches", type=int, required=True, help="No. of batches")
    parser.add_argument("-p", "--patience", type=int, required=True, help="Patience value; No of epochs to execute before early stopping")
    parser.add_argument("-opt", "--optimizer", help="Choose optimizer (adam/adadelta)")
    parser.add_argument("-o", "--out", required=True, help="Path to save trained keras model")

    args = vars(parser.parse_args())
    variation = args["variation"]
    train_pp_file = args["train_parapair"]
    test_pp_file = args["test_parapair"]
    train_elmo_file = args["train_elmo"]
    test_elmo_file = args["test_elmo"]
    train_bbc_file = args["train_bbc"]
    test_bbc_file = args["test_bbc"]
    vec_len = args["vec"]
    dense_size = args["dense_layer_size"]
    reg = args["regularization"]
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
    train_elmo_emb = np.load(train_elmo_file, allow_pickle=True)
    test_elmo_emb = np.load(test_elmo_file, allow_pickle=True)
    train_bbc_emb = np.load(train_bbc_file, allow_pickle=True)
    test_bbc_emb = np.load(test_bbc_file, allow_pickle=True)

    val_parapair = dict()
    val_pages = random.sample(train_parapair.keys(), round(len(train_parapair.keys()) * 0.2))
    for vpage in val_pages:
        val_parapair[vpage] = train_parapair[vpage]
        del train_parapair[vpage]

    Xtrain, ytrain = prepare_data(train_parapair, train_elmo_emb, train_bbc_emb, variation)
    Xval, yval = prepare_data(val_parapair, train_elmo_emb, train_bbc_emb, variation)
    Xtest, ytest = prepare_data(test_parapair, test_elmo_emb, test_bbc_emb, variation)

    m = dense_siamese_bbc_cat(Xtrain, ytrain, Xval, yval, Xtest, ytest, vec_len, optim, dense_size, reg, learning_rate, epochs, batches, pat)
    m.save(out_file)

    yhat = m.predict([Xtest[:, :vec_len], Xtest[:, vec_len:]], verbose=0)
    test_eval = m.evaluate([Xtest[:, :vec_len], Xtest[:, vec_len:]], ytest)
    test_auc_score = roc_auc_score(ytest, yhat)
    print("Evaluation on test set: " + str(test_eval))
    print("Test auc score: "+ str(test_auc_score))

if __name__ == '__main__':
    main()