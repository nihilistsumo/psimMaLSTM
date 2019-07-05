#!/usr/bin/python3

import manhattan_lstm
import dense_siamese
import numpy as np
from keras.models import load_model, Model
import random, argparse, json

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

def evaluate_dense_siamese(m, Xtest, ytest, Xtest_rand, ytest_rand, vec_len):
    num_test_sample = ytest.shape[0]
    yhat = m.predict([Xtest[:, :vec_len], Xtest[:, vec_len:]], verbose=0)
    test_eval = m.evaluate([Xtest[:, :vec_len], Xtest[:, vec_len:]], ytest)
    print("Showing predictions from 100 random samples in test data")
    for i in random.sample(range(num_test_sample), 100):
        print('Expected: ', ytest[i], 'Predicted: ', yhat[i][0], 'Similarity/Distance: ', yhat[i][0])
    print("Evaluation on test set: " + str(test_eval))

    num_test_sample = ytest_rand.shape[0]
    yhat_rand = m.predict([Xtest_rand[:, :vec_len], Xtest_rand[:, vec_len:]], verbose=0)
    test_eval = m.evaluate([Xtest_rand[:, :vec_len], Xtest_rand[:, vec_len:]], ytest_rand)
    print("Showing predictions from 100 random samples in rand test data")
    for i in random.sample(range(num_test_sample), 100):
        print('Expected: ', ytest_rand[i], 'Predicted: ', yhat_rand[i][0], 'Similarity/Distance: ', yhat_rand[i][0])
    print("Evaluation on randomized and balanced test set: " + str(test_eval))

def main():
    parser = argparse.ArgumentParser(description="Evaluate Dense-Siamese model for paragraph similarity task")
    parser.add_argument("-m", "--model_file", required=True, help="Path to model file to load")
    parser.add_argument("-pp", "--parapair", required=True, help="Path to parapair file")
    parser.add_argument("-hq", "--hier_qrels", required=True, help="Path to hierarchical qrels file")
    parser.add_argument("-em", "--embedding", required=True, help="Path to embedding file")
    parser.add_argument("-v", "--vec", required=True, type=int, help="Length of each paragraph vector")
    args = vars(parser.parse_args())
    model_file = args["model_file"]
    parapair_file = args["parapair"]
    hier_qrels_file = args["hier_qrels"]
    emb_file = args["embedding"]
    vec_len = args["vec"]

    model = load_model(model_file, custom_objects={'precision':dense_siamese.precision, 'recall':dense_siamese.recall,
                                                   'fmeasure':dense_siamese.fmeasure, 'fbeta_score':dense_siamese.fbeta_score})
    with open(parapair_file, 'r') as tpp:
        parapair = json.load(tpp)
    emb = np.load(emb_file, allow_pickle=True)
    hier_qrels_reverse = dict()
    with open(hier_qrels_file, 'r') as hq:
        for l in hq:
            hier_qrels_reverse[l.split(" ")[2]] = l.split(" ")[0]

    Xtest, ytest = dense_siamese.prepare_test_data(parapair, emb, hier_qrels_reverse, False)
    Xtest_rand, ytest_rand = dense_siamese.prepare_test_data(parapair, emb, hier_qrels_reverse, True)

    evaluate_dense_siamese(model, Xtest, ytest, Xtest_rand, ytest_rand, vec_len)

if __name__ == '__main__':
    main()