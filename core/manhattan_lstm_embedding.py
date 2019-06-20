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

def prepare_data(train_data, val_data, test_data, seq_len):
    assert seq_len == (len(train_data[0]) - 1) / 2
    assert seq_len == (len(test_data[0]) - 1) / 2
    assert seq_len == (len(val_data[0]) - 1) / 2
    Xtrain = train_data[:, :-1]
    ytrain = train_data[:, 2 * seq_len]
    Xval = val_data[:, :-1]
    yval = val_data[:, 2 * seq_len]
    Xtest = test_data[:, :-1]
    ytest = test_data[:, 2 * seq_len]
    return Xtrain, ytrain, Xval, yval, Xtest, ytest

def malstm(Xtrain, ytrain, Xval, yval, Xtest, ytest, embeddings, seq_len, embed_vec_len, optim,
           lstm_layer_size=10, learning_rate=0.01, num_epochs=3, num_bacthes=1, pat=10):

    # ytrain = to_categorical(ytrain)
    # yval = to_categorical(yval)
    # ytest = to_categorical(ytest)

    para_seq1 = Input(shape=(seq_len, ), dtype='int32', name='sequence1')
    para_seq2 = Input(shape=(seq_len, ), dtype='int32', name='sequence2')

    embedding_layer = Embedding(len(embeddings), embed_vec_len, weights=[embeddings], input_length=seq_len, trainable=False)

    encoded_left = embedding_layer(para_seq1)
    encoded_right = embedding_layer(para_seq2)

    drop = Dropout(0.5)
    lstm = LSTM(lstm_layer_size, kernel_regularizer=regularizers.l2(0.001))
    l1_out = lstm(drop(encoded_left))
    print(l1_out.shape)
    l2_out = lstm(drop(encoded_right))
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