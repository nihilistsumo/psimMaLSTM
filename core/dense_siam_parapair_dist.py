#!/usr/bin/python3

import dense_siamese, lstm_siamese
import numpy as np
from keras.models import load_model, Model
import random, argparse, json
from sklearn.metrics import roc_auc_score

def get_Xtest(parapairs, embeddings):
    x = []
    for i in range(len(parapairs)):
        p1 = parapairs[i].split("_")[0]
        p2 = parapairs[i].split("_")[1]
        x.append(np.hstack((embeddings[()][p1], embeddings[()][p2])))
    Xtest = np.array(x)
    return Xtest

def get_parapair_dist(model, parapairs, embeddings, vec_len):
    Xtest = get_Xtest(parapairs, embeddings)
    yhat = model.predict([Xtest[:, :vec_len], Xtest[:, vec_len:]], verbose=0)