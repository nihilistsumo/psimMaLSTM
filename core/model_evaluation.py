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

def prepare_test_data(parapair_dict, embeddings):
    test_pairs = []
    for page in parapair_dict.keys():
        test_pos, test_neg = dense_siamese.get_samples(parapair_dict[page])
        for p in test_pos:
            test_pairs.append([p, 1])
        for p in test_neg:
            test_pairs.append([p, 0])

    Xtest, ytest = dense_siamese.get_xy(test_pairs, embeddings)

    return Xtest, ytest, test_pairs

def evaluate_dense_siamese(m, Xtest, ytest, pairlist, vec_len, outfile):
    num_test_sample = ytest.shape[0]
    yhat = m.predict([Xtest[:, :vec_len], Xtest[:, vec_len:]], verbose=0)
    assert len(yhat) == len(pairlist)
    parapair_dist = dict()
    for i in range(len(pairlist)):
        parapair_dist[pairlist[i][0]] = yhat[i][0]
    with open(outfile, 'w') as out:
        json.dump(parapair_dist, out)
    test_eval = m.evaluate([Xtest[:, :vec_len], Xtest[:, vec_len:]], ytest)
    print("Showing predictions from 100 random samples in test data")
    for i in random.sample(range(num_test_sample), 100):
        print('Expected: ', ytest[i], 'Predicted: ', yhat[i][0], 'Similarity/Distance: ', yhat[i][0])
    print("Evaluation on test set: " + str(test_eval))

def main():
    parser = argparse.ArgumentParser(description="Evaluate Dense-Siamese model for paragraph similarity task")
    parser.add_argument("-m", "--model_file", required=True, help="Path to model file to load")
    parser.add_argument("-pp", "--parapair", required=True, help="Path to test parapair file")
    # parser.add_argument("-hq", "--hier_qrels", required=True, help="Path to hierarchical qrels file")
    parser.add_argument("-em", "--embedding", required=True, help="Path to test embedding file")
    parser.add_argument("-v", "--vec", required=True, type=int, help="Length of each paragraph vector")
    parser.add_argument("-o", "--out", required=True, help="Path to parapair score output file")
    args = vars(parser.parse_args())
    model_file = args["model_file"]
    parapair_file = args["parapair"]
    # hier_qrels_file = args["hier_qrels"]
    emb_file = args["embedding"]
    vec_len = args["vec"]
    outfile = args["out"]

    model = load_model(model_file, custom_objects={'precision':dense_siamese.precision, 'recall':dense_siamese.recall,
                                                   'fmeasure':dense_siamese.fmeasure, 'fbeta_score':dense_siamese.fbeta_score})
    with open(parapair_file, 'r') as tpp:
        parapair = json.load(tpp)
    emb = np.load(emb_file, allow_pickle=True)

    # hier_qrels_reverse = dict()
    # with open(hier_qrels_file, 'r') as hq:
    #     for l in hq:
    #         hier_qrels_reverse[l.split(" ")[2]] = l.split(" ")[0]

    Xtest, ytest, parapair_list = prepare_test_data(parapair, emb)

    evaluate_dense_siamese(model, Xtest, ytest, parapair_list, vec_len, outfile)

if __name__ == '__main__':
    main()