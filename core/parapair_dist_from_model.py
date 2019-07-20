#!/usr/bin/python3

import manhattan_lstm
import dense_siamese
import numpy as np
from keras.models import load_model, Model
import random, argparse, json

def calculate_parapair_scores_from_model(m, Xtest, ytest, pairlist, vec_len, outfile):
    num_test_sample = ytest.shape[0]
    yhat = m.predict([Xtest[:, :vec_len], Xtest[:, vec_len:]], verbose=0)
    assert len(yhat) == len(pairlist)
    parapair_dist = dict()
    for i in range(len(pairlist)):
        parapair_dist[pairlist[i][0]] = float(yhat[i][0])
    with open(outfile, 'w') as out:
        json.dump(parapair_dist, out)
    test_eval = m.evaluate([Xtest[:, :vec_len], Xtest[:, vec_len:]], ytest)
    print("Showing predictions from 100 random samples in test data")
    for i in random.sample(range(num_test_sample), 100):
        print('Expected: ', ytest[i], 'Predicted: ', yhat[i][0], 'Similarity/Distance: ', yhat[i][0])
    print("Evaluation on test set: " + str(test_eval))