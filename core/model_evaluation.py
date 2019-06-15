#!/usr/bin/python3

import manhattan_lstm
import numpy as np
from keras.models import load_model, Model

def get_parapair_scores(model_file, data_file):
    data = np.load(data_file)
    model = load_model(model_file)
    print("Model loaded")
    model.summary()
    train_data = data[()]["train_data"]
    train_pairs = data[()]["train_parapairs"]
    val_data = data[()]["val_data"]
    val_pairs = data[()]["val_parapairs"]
    test_data = data[()]["test_data"]
    test_pairs = data[()]["test_parapairs"]
    Xtrain, ytrain, Xval, yval, Xtest, ytest = manhattan_lstm.prepare_data(train_data, val_data, test_data, seq_len, vec_len)

    intermediate_layer_model = Model(inputs=model.input, outputs=model.get_layer('distance').output)
