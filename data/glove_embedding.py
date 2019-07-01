#!/usr/bin/python3

import argparse, os, json, csv
import numpy as np
import pandas as pd

def get_glove_embedding_df(glove_file):
    glove = pd.read_table(glove_file, sep=" ", index_col=0, header=None, quoting=csv.QUOTE_NONE)
    return glove

def get_vocab_list(tokenized_para_dict):
    vocab = set()
    for para in tokenized_para_dict.keys():
        vocab = vocab.union(set(tokenized_para_dict[para]))
    vocab = list(vocab)
    return vocab

def create_embed_matrix(glove, vocab):
    vec_len = glove.loc['the'].shape[0]
    vocab_size = len(vocab)
    embed_matrix = np.random.uniform(-1.0, 1.0, (vocab_size+1, vec_len))
    embed_matrix[0] = np.zeros(vec_len)
    for i in range(vocab_size):
        word = vocab[i]
        if word in glove.index:
            embed_matrix[i+1] = np.array(glove.loc[word])
    return embed_matrix

def get_embed_matrix(g, tokenized_para):
    vocab = get_vocab_list(tokenized_para)
    embed = create_embed_matrix(g, vocab)
    return embed, vocab

def main():
    parser = argparse.ArgumentParser(
        description="Generate parapair data suitable for MaLSTM model with glove embedding")
    parser.add_argument("-trpt", "--train_para_token", required=True, help="Path to train para token file")
    parser.add_argument("-tpt", "--test_para_token", required=True, help="Path to test para token file")
    parser.add_argument("-g", "--glove_file", required=True, help="Path to glove file")
    parser.add_argument("-o", "--out_dir", required=True, help="Path to output directory")

    args = vars(parser.parse_args())
    train_para_token_file = args["train_para_token"]
    test_para_token_file = args["test_para_token"]
    glove_file = args["glove_file"]
    outdir = args["out_dir"]

    g = get_glove_embedding_df(glove_file)
    train_pt = np.load(train_para_token_file, allow_pickle=True)[()]
    test_pt = np.load(test_para_token_file, allow_pickle=True)[()]
    combined_tokenized_para = train_pt
    combined_tokenized_para.update(test_pt)

    embedding_matrix, vocab_list = get_embed_matrix(g, combined_tokenized_para)
    np.save(outdir+"/embedding_matrix", embedding_matrix)
    np.save(outdir+"/vocab_list", vocab_list)

    print("Done")

if __name__ == '__main__':
    main()