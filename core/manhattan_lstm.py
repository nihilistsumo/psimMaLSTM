#!/usr/bin/python3

import os, argparse, json
from sys import exit

from time import time
import datetime
import argparse
from math import exp

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

import numpy as np

def exponent_neg_manhattan_distance(x, layer_size):
    ''' Helper function for the similarity estimate of the LSTMs outputs '''
    return K.exp(-K.sum(K.abs(x[:,:layer_size] - x[:,layer_size:]), axis=1, keepdims=True))

def exponent_neg_cosine_distance(x, lstm_layer_size=10):
    ''' Helper function for the similarity estimate of the LSTMs outputs '''
    leftNorm = K.l2_normalize(x[:,:lstm_layer_size], axis=-1)
    rightNorm = K.l2_normalize(x[:,lstm_layer_size:], axis=-1)
    return K.exp(K.sum(K.prod([leftNorm, rightNorm], axis=0), axis=1, keepdims=True))

def prepare_data(train_data, val_data, test_data, seq_len, vec_len):
    Xtrain = train_data[:, :-1]
    ytrain = train_data[:, len(train_data[0]) - 1]
    Xval = val_data[:, :-1]
    yval = val_data[:, len(val_data[0]) - 1]
    Xtest = test_data[:, :-1]
    ytest = test_data[:, len(test_data[0]) - 1]
    train_samples = Xtrain.shape[0]
    val_samples = Xval.shape[0]
    test_samples = Xtest.shape[0]
    Xtrain = Xtrain.reshape((train_samples, 2 * seq_len, vec_len))
    Xval = Xval.reshape((val_samples, 2 * seq_len, vec_len))
    Xtest = Xtest.reshape((test_samples, 2 * seq_len, vec_len))
    return Xtrain, ytrain, Xval, yval, Xtest, ytest

def malstm(Xtrain, ytrain, train_pairs, Xval, yval, val_pairs, Xtest, ytest, test_pairs, seq_len, vec_len, optim,
           lstm_layer_size=10, learning_rate=0.01, num_epochs=3, num_bacthes=1, pat=10):
    Xtrain1 = Xtrain[:, :seq_len, :]
    Xtrain2 = Xtrain[:, seq_len:, :]
    Xval1 = Xval[:, :seq_len, :]
    Xval2 = Xval[:, seq_len:, :]
    Xtest1 = Xtest[:, :seq_len, :]
    Xtest2 = Xtest[:, seq_len:, :]

    Xtrain_comb = [Xtrain1, Xtrain2]
    Xval_comb = [Xval1, Xval2]
    Xtest_comb = [Xtest1, Xtest2]

    # ytrain = to_categorical(ytrain)
    # yval = to_categorical(yval)
    # ytest = to_categorical(ytest)

    para_seq1 = Input(shape=(seq_len, vec_len, ), dtype='float32', name='sequence1')
    para_seq2 = Input(shape=(seq_len, vec_len, ), dtype='float32', name='sequence2')

    drop = Dropout(0.5)
    lstm = LSTM(lstm_layer_size, kernel_regularizer=regularizers.l2(0.001))
    bilstm = Bidirectional(lstm)
    l1_out = bilstm(drop(para_seq1))
    print(l1_out.shape)
    l2_out = bilstm(drop(para_seq2))
    print(l2_out.shape)
    #
    concats = concatenate([l1_out, l2_out], axis=-1)
    concats_out = drop(concats)

    dist_output = Lambda(exponent_neg_manhattan_distance, output_shape=(1,), arguments={'layer_size':lstm_layer_size * 2}, name='distance')(concats_out)
    main_output = Dense(1, activation='relu')(dist_output)

    model = Model(inputs=[para_seq1, para_seq2], outputs=[main_output])
    #
    if optim == 'adadelta':
        opt = keras.optimizers.Adadelta(lr=learning_rate, clipnorm=1.25)
    else:
        opt = keras.optimizers.Adam(lr=learning_rate)
    #
    # opt = keras.optimizers.SGD(lr=learning_rate, decay=1e-6, momentum=0.9, nesterov=True)

    model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=pat)
    model.summary()

    history = model.fit(Xtrain_comb, ytrain, validation_data=(Xval_comb, yval), epochs=num_epochs, batch_size=num_bacthes, verbose=1, callbacks=[es])

    intermediate_layer_model = Model(inputs=model.input, outputs=model.get_layer('distance').output)
    intermediate_output_train = intermediate_layer_model.predict(Xtrain_comb)
    intermediate_output_val = intermediate_layer_model.predict(Xval_comb)
    intermediate_output_test = intermediate_layer_model.predict(Xtest_comb)
    parapair_score_dict = dict()

    for i in range(intermediate_output_train.shape[0]):
        parapair_score_dict[train_pairs[i]] = float(intermediate_output_train[i][0])

    for i in range(intermediate_output_val.shape[0]):
        parapair_score_dict[val_pairs[i]] = float(intermediate_output_val[i][0])

    num_test_sample = ytest.shape[0]
    yhat = model.predict(Xtest_comb, verbose=0)
    test_eval = model.evaluate(Xtest_comb, ytest)
    for i in range(num_test_sample):
        print('Expected:', ytest[i], 'Predicted', yhat[i][0], 'Similarity', intermediate_output_test[i][0])
        parapair_score_dict[test_pairs[i]] = float(intermediate_output_test[i][0])
    print(test_eval)

    return model, parapair_score_dict

def main():
    parser = argparse.ArgumentParser(description="Train and evaluate MaLSTM for paragraph similarity task")
    parser.add_argument("-dn", "--data_nature", required=True, help="Nature of the input data, f: single file, d: multiple files in directory")
    parser.add_argument("-d", "--data", required=True, help="Path to data dict file")
    parser.add_argument("-s", "--seq", required=True, type=int, help="Maximum length of each paragraph in terms of sentence count")
    parser.add_argument("-v", "--vec", required=True, type=int, help="Length of each vector")
    parser.add_argument("-lstm", "--lstm_layer_size", type=int, help="Size of each LSTM layer")
    parser.add_argument("-lr", "--learning_rate", type=float, help="Learning rate")
    parser.add_argument("-e", "--epochs", type=int, help="No. of epochs")
    parser.add_argument("-b", "--batches", type=int, required=True, help="No. of batches")
    parser.add_argument("-p", "--patience", type=int, required=True, help="Patience value; No of epochs to execute before early stopping")
    parser.add_argument("-opt", "--optimizer", help="Choose optimizer (adam/adadelta)")
    parser.add_argument("-o", "--out", required=True, help="Path to save trained keras model")
    parser.add_argument("-po", "--parapair_score_out", required=True, help="Path to save the parapair scores")

    args = vars(parser.parse_args())
    data_nature = args["data_nature"]
    path_to_data = args["data"]
    seq_len = args["seq"]
    vec_len = args["vec"]
    lstm_size = args["lstm_layer_size"]
    learning_rate = args["learning_rate"]
    epochs = args["epochs"]
    batches = args["batches"]
    pat = args["patience"]
    optim = args["optimizer"]
    out_file = args["out"]
    parapair_out = args["parapair_score_out"]

    LAMBDA_LAYER_SIZE = lstm_size

    if data_nature == 'f':
        data = np.load(path_to_data)
        train_data = data[()]["train_data"]
        train_pairs = data[()]["train_parapairs"]
        val_data = data[()]["val_data"]
        val_pairs = data[()]["val_parapairs"]
        test_data = data[()]["test_data"]
        test_pairs = data[()]["test_parapairs"]
    else:
        train_data = np.load(path_to_data + "/train_data")
        train_pairs = np.load(path_to_data + "train_parapair_list")
        val_data = np.load(path_to_data + "val_data")
        val_pairs = np.load(path_to_data + "val_parapair_list")
        test_data = np.load(path_to_data + "test_data")
        test_pairs = np.load(path_to_data + "test_parapair_list")
    Xtrain, ytrain, Xval, yval, Xtest, ytest = prepare_data(train_data, val_data, test_data, seq_len, vec_len)
    model, parapair_scores = malstm(Xtrain, ytrain, train_pairs, Xval, yval, val_pairs, Xtest, ytest, test_pairs,
                                    seq_len, vec_len, optim, lstm_size, learning_rate, epochs, batches, pat)

    model.save(out_file)
    with open(parapair_out, 'w') as pout:
        json.dump(parapair_scores, pout)

    print("Finished! Trained model saved at: "+out_file)

if __name__ == '__main__':
    main()