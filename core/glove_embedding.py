#!/usr/bin/python3

import argparse, os, json, csv
import numpy as np
import pandas as pd

def get_glove_embedding_df(glove_file):
    glove = pd.read_table(glove_file, sep=" ", index_col=0, header=None, quoting=csv.QUOTE_NONE)
    return glove

def get_vocab_list(tokenized_para_dict):
    vocab = set()
    for para in tokenized_para_dict[()].keys():
        vocab = vocab.union(set(tokenized_para_dict[()][para]))
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

def get_embed_matrix(glove_file, tokenized_para_file):
    g = get_glove_embedding_df(glove_file)
    tokenized_para = np.load(tokenized_para_file)
    vocab = get_vocab_list(tokenized_para)
    embed = create_embed_matrix(g, vocab)
    return embed, vocab