#!/usr/bin/python3

import dense_siamese, lstm_siamese
import numpy as np
from keras.models import load_model, Model
import random, argparse, json
from sklearn.metrics import roc_auc_score

def get_Xtest(parapairs, embeddings, paralist):
    x = []
    for i in range(len(parapairs)):
        p1 = parapairs[i].split("_")[0]
        p2 = parapairs[i].split("_")[1]
        p1_emb = embeddings[paralist.index(p1)]
        p2_emb = embeddings[paralist.index(p2)]
        # x.append(np.hstack((embeddings[()][p1], embeddings[()][p2])))
        x.append(np.hstack((p1_emb, p2_emb)))
    Xtest = np.array(x)
    return Xtest

def get_parapair_dist(model, parapairs, embeddings, paralist, vec_len):
    Xtest = get_Xtest(parapairs, embeddings, paralist)
    yhat = model.predict([Xtest[:, :vec_len], Xtest[:, vec_len:]], verbose=0)
    parapair_dist = dict()
    for i in range(len(parapairs)):
        parapair_dist[parapairs[i]] = float(yhat[i][0])
    return parapair_dist

def main():
    parser = argparse.ArgumentParser(description="Produce parapair distance dict using pretrained Dense-Siamese model")
    parser.add_argument("-m", "--model_file", required=True, help="Path to model file to load")
    parser.add_argument("-pp", "--parapair", required=True, help="Path to test parapair file")
    parser.add_argument("-em", "--embedding", required=True, help="Path to all jordan embedding file")
    parser.add_argument("-emp", "--emb_para_list", required=True, help="Ordered list of paraids corresponding to emb file")
    parser.add_argument("-v", "--vec", required=True, type=int, help="Length of each paragraph vector")
    parser.add_argument("-o", "--out", required=True, help="Path to parapair score output file")
    args = vars(parser.parse_args())
    model_file = args["model_file"]
    parapair_file = args["parapair"]
    emb_file = args["embedding"]
    emb_para_file = args["emb_para_list"]
    vec_len = args["vec"]
    outfile = args["out"]

    with open(parapair_file, 'r') as pp:
        parapair_dict = json.load(pp)
    parapair_list = []
    for page in parapair_dict.keys():
        parapair_list.extend(parapair_dict[page]["parapairs"])
    emb = np.load(emb_file, allow_pickle=True)
    paraids_ordered_list = []
    with open(emb_para_file, 'r') as ek:
        for p in ek:
            paraids_ordered_list.append(p.rstrip('\n'))
    model = load_model(model_file, custom_objects={'precision': dense_siamese.precision, 'recall': dense_siamese.recall,
                                                   'fmeasure': dense_siamese.fmeasure,
                                                   'fbeta_score': dense_siamese.fbeta_score})
    parapair_dist = get_parapair_dist(model, parapair_list, emb, paraids_ordered_list, vec_len)
    with open(outfile, 'w') as out:
        json.dump(parapair_dist, out)

if __name__ == '__main__':
    main()