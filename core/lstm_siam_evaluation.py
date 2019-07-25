#!/usr/bin/python3

import manhattan_lstm
import dense_siamese, lstm_siamese
import numpy as np
from keras.models import load_model, Model
import random, argparse, json
from sklearn.metrics import roc_auc_score

def prepare_test_data(parapair_dict, embeddings, vec_len, max_seq_len):
    test_pairs = []
    for page in parapair_dict.keys():
        test_pos, test_neg = lstm_siamese.get_samples(parapair_dict[page])
        for p in test_pos:
            test_pairs.append([p, 1])
        for p in test_neg:
            test_pairs.append([p, 0])

    Xtest, ytest = lstm_siamese.get_xy(test_pairs, embeddings, vec_len, max_seq_len)

    return Xtest, ytest, test_pairs

def evaluate_lstm_siamese(m, Xtest, ytest, pairlist, vec_len, outfile):
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
    print("ROC AUC score: " + str(roc_auc_score(ytest, yhat)))

def main():
    parser = argparse.ArgumentParser(description="Evaluate Dense-Siamese model for paragraph similarity task")
    parser.add_argument("-m", "--model_file", required=True, help="Path to model file to load")
    parser.add_argument("-pp", "--parapair", required=True, help="Path to test parapair file")
    # parser.add_argument("-hq", "--hier_qrels", required=True, help="Path to hierarchical qrels file")
    parser.add_argument("-em", "--embedding", required=True, help="Path to test embedding file")
    parser.add_argument("-s", "--max_seq_len", required=True, help="Max sequence length for which the model is trained for")
    parser.add_argument("-v", "--vec", required=True, type=int, help="Length of each paragraph vector")
    parser.add_argument("-o", "--out", required=True, help="Path to parapair score output file")
    args = vars(parser.parse_args())
    model_file = args["model_file"]
    parapair_file = args["parapair"]
    # hier_qrels_file = args["hier_qrels"]
    emb_file = args["embedding"]
    seq_len = args["max_seq_len"]
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

    Xtest, ytest, parapair_list = prepare_test_data(parapair, emb, vec_len, seq_len)

    evaluate_lstm_siamese(model, Xtest, ytest, parapair_list, vec_len, outfile)

if __name__ == '__main__':
    main()