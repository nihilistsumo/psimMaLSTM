#!/usr/bin/python3

import json, random, math, csv
from collections import Counter
import argparse
import glove_embedding
import numpy as np
import pandas as pd

def get_para_seq(para_token_seq, vocab, seq_len):
    seq = []
    if len(para_token_seq) < seq_len:
        padding = [0] * (seq_len - len(para_token_seq))
        seq = [vocab.index(t) + 1 for t in para_token_seq]
        seq = padding + seq
    else:
        seq = [vocab.index(t) + 1 for t in para_token_seq][:seq_len]
    return seq

def prepare_test_data(parapair_data_dict, para_token_dict, vocab, seq_len):
    test_seq_data = []
    test_pages = parapair_data_dict.keys()

    for page in test_pages:
        parapairs = parapair_data_dict[page]["parapairs"]
        labels = parapair_data_dict[page]["labels"]
        assert len(parapairs) == len(labels)
        for i in range(len(parapairs)):
            p1 = parapairs[i].split("_")[0]
            p2 = parapairs[i].split("_")[1]
            p1_seq = get_para_seq(para_token_dict[()][p1], vocab, seq_len)
            p2_seq = get_para_seq(para_token_dict[()][p2], vocab, seq_len)
            test_seq_data.append(p1_seq + p2_seq + [labels[i]])
        print(page)
    random.shuffle(test_seq_data)
    test_seq_data = np.array(test_seq_data).astype(int)

    return test_seq_data

def prepare_train_data(parapair_data_dict, para_token_dict, vocab, seq_len, train_val_split_ratio):
    train_seq_data = []
    val_seq_data = []
    total_pages_no = len(parapair_data_dict.keys())
    train_pages_no = math.floor(total_pages_no * train_val_split_ratio)
    val_pages_no = total_pages_no - train_pages_no
    train_pages = random.sample(parapair_data_dict.keys(), train_pages_no)
    val_pages = [p for p in parapair_data_dict.keys() if p not in train_pages]

    for page in train_pages:
        parapairs = parapair_data_dict[page]["parapairs"]
        labels = parapair_data_dict[page]["labels"]
        no_pos_samples = Counter(labels)[1]
        print(page+" positive samples: {}".format(no_pos_samples))
        assert len(parapairs) == len(labels)
        count_neg_samples = 0
        for i in range(len(parapairs)):
            p1 = parapairs[i].split("_")[0]
            p2 = parapairs[i].split("_")[1]
            p1_seq = get_para_seq(para_token_dict[()][p1], vocab, seq_len)
            p2_seq = get_para_seq(para_token_dict[()][p2], vocab, seq_len)

            # This part will make sure to create a balanced dataset assuming postive samples < negative samples
            if labels[i] == 0:
                if count_neg_samples < no_pos_samples:
                    train_seq_data.append(p1_seq + p2_seq + [labels[i]])
                    count_neg_samples += 1
            else:
                train_seq_data.append(p1_seq + p2_seq + [labels[i]])
    random.shuffle(train_seq_data)

    for page in val_pages:
        parapairs = parapair_data_dict[page]["parapairs"]
        labels = parapair_data_dict[page]["labels"]
        no_pos_samples = Counter(labels)[1]
        print(page + " positive samples: {}".format(no_pos_samples))
        assert len(parapairs) == len(labels)
        count_neg_samples = 0
        for i in range(len(parapairs)):
            p1 = parapairs[i].split("_")[0]
            p2 = parapairs[i].split("_")[1]
            p1_seq = get_para_seq(para_token_dict[()][p1], vocab, seq_len)
            p2_seq = get_para_seq(para_token_dict[()][p2], vocab, seq_len)

            if labels[i] == 0:
                if count_neg_samples < no_pos_samples:
                    val_seq_data.append(p1_seq + p2_seq + [labels[i]])
                    count_neg_samples += 1
            else:
                val_seq_data.append(p1_seq + p2_seq + [labels[i]])
    random.shuffle(val_seq_data)
    train_seq_data = np.array(train_seq_data).astype(int)
    val_seq_data = np.array(val_seq_data).astype(int)

    return train_seq_data, val_seq_data

def main():
    parser = argparse.ArgumentParser(description="Generate parapair data suitable for MaLSTM model with glove embedding")
    parser.add_argument("-trp", "--train_parapair", required=True, help="Path to train parapair file")
    parser.add_argument("-trpt", "--train_para_token", required=True, help="Path to train para token file")
    parser.add_argument("-tp", "--test_parapair", required=True, help="Path to test parapair file")
    parser.add_argument("-tpt", "--test_para_token", required=True, help="Path to test para token file")
    parser.add_argument("-vo", "--vocab_list", required=True, help="Path to vocabulary list file")
    parser.add_argument("-l", "--seq_len", type=int, required=True, help="Maximum length of paragraph sequence")
    parser.add_argument("-vs", "--train_split_ratio", type=float, required=True, help="Fraction of train/validation split")
    parser.add_argument("-o", "--out_dir", required=True, help="Path to output directory")
    args = vars(parser.parse_args())
    train_pp_file = args["train_parapair"]
    train_para_token_file = args["train_para_token"]
    test_pp_file = args["test_parapair"]
    test_para_token_file = args["test_para_token"]
    vocab_file = args["vocab_list"]
    seq_len = args["seq_len"]
    tv_ratio = args["train_split_ratio"]
    outdir = args["out_dir"]

    with open(train_pp_file, 'r') as trpp:
        train_pp = json.load(trpp)
    with open(test_pp_file, 'r') as tpp:
        test_pp = json.load(tpp)
    train_pt = np.load(train_para_token_file, allow_pickle=True)
    test_pt = np.load(test_para_token_file, allow_pickle=True)

    vocab_list = np.load(vocab_file).tolist()
    print("Data loaded")

    train_seq_data, val_seq_data = prepare_train_data(train_pp, train_pt, vocab_list, seq_len, tv_ratio)
    test_seq_data = prepare_test_data(test_pp, test_pt, vocab_list, seq_len)

    np.save(outdir+"/train_data", train_seq_data)
    np.save(outdir+"/val_data", val_seq_data)
    np.save(outdir+"/test_data", test_seq_data)

    print("Done")

if __name__ == '__main__':
    main()
