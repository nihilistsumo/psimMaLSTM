import argparse, random, math, json
import numpy as np

import tensorflow as tf

import keras
import keras.backend as K
from keras.models import Model
from keras.layers import Activation
from keras.layers import Embedding, Input, TimeDistributed, Dropout
from keras.layers import LSTM, Lambda, concatenate, Dense, Bidirectional
from keras import regularizers
from keras.callbacks import EarlyStopping
from keras.utils import to_categorical
from keras.wrappers.scikit_learn import KerasClassifier

from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold

def get_xy(pair_data, embeddings):
    x = []
    y = []
    for i in range(len(pair_data)):
        p = pair_data[i]
        p1 = p[0].split("_")[0]
        p2 = p[0].split("_")[1]
        y.append(p[1])
        x.append(np.hstack((embeddings[()][p1], embeddings[()][p2])))
    return np.array(x), np.array(y)

def get_samples(parapair_data):
    pairs = parapair_data["parapairs"]
    labels = parapair_data["labels"]
    pos_pairs = []
    neg_pairs = []
    for i in range(len(labels)):
        if labels[i] == 0:
            neg_pairs.append(pairs[i])
        else:
            pos_pairs.append(pairs[i])
    return pos_pairs, neg_pairs

def get_discriminative_samples(parapair_data, hier_qrels_reverse_dict):
    pairs = parapair_data["parapairs"]
    labels = parapair_data["labels"]
    pos_pairs = []
    neg_pairs = []
    for i in range(len(labels)):
        if labels[i] == 0:
            neg_pairs.append(pairs[i])
        else:
            p1 = pairs[i].split("_")[0]
            p2 = pairs[i].split("_")[1]
            if hier_qrels_reverse_dict[p1] == hier_qrels_reverse_dict[p2]:
                pos_pairs.append(pairs[i])
    return pos_pairs, neg_pairs

def prepare_train_data(parapair_dict, embeddings, hier_qrels_reverse, train_val_split=0.8):
    pages = parapair_dict.keys()
    train_pages = set(random.sample(pages, math.floor(len(pages) * train_val_split)))
    val_pages = pages - train_pages
    train_pairs = []
    val_pairs = []
    for page in train_pages:
        train_pos, train_neg = get_discriminative_samples(parapair_dict[page], hier_qrels_reverse)
        train_neg = random.sample(train_neg, len(train_pos))
        for p in train_pos:
            train_pairs.append([p, 1])
        for p in train_neg:
            train_pairs.append([p, 0])
    for page in val_pages:
        val_pos, val_neg = get_discriminative_samples(parapair_dict[page], hier_qrels_reverse)
        val_neg = random.sample(val_neg, len(val_pos))
        for p in val_pos:
            val_pairs.append([p, 1])
        for p in val_neg:
            val_pairs.append([p, 0])
    random.shuffle(train_pairs)
    random.shuffle(val_pairs)

    Xtrain, ytrain = get_xy(train_pairs, embeddings)
    Xval, yval = get_xy(val_pairs, embeddings)

    return Xtrain, ytrain, Xval, yval

def prepare_test_data(parapair_dict, embeddings, hier_qrels_reverse, take_random):
    test_pairs = []
    for page in parapair_dict.keys():
        test_pos, test_neg = get_discriminative_samples(parapair_dict[page], hier_qrels_reverse)
        # test_pos, test_neg = get_samples(parapair_dict[page])
        if take_random:
            test_neg = random.sample(test_neg, len(test_pos))
        for p in test_pos:
            test_pairs.append([p, 1])
        for p in test_neg:
            test_pairs.append([p, 0])
    random.shuffle(test_pairs)

    Xtest, ytest = get_xy(test_pairs, embeddings)

    return Xtest, ytest

def manhattan_distance(x, layer_size):
    return K.abs(x[:,:layer_size] - x[:,layer_size:])

def exponent_neg_manhattan_distance(x, layer_size):
    return K.exp(-K.sum(K.abs(x[:,:layer_size] - x[:,layer_size:]), axis=1, keepdims=True))

def precision(ytrue, yhat):
    true_pos = K.sum(K.round(K.clip(ytrue * yhat, 0, 1)))
    predicted_pos = K.sum(K.round(K.clip(yhat, 0, 1)))
    precision = true_pos / (predicted_pos + K.epsilon())
    return precision

def recall(ytrue, yhat):
    true_pos = K.sum(K.round(K.clip(ytrue * yhat, 0, 1)))
    possible_pos = K.sum(K.round(K.clip(ytrue, 0, 1)))
    recall = true_pos / (possible_pos + K.epsilon())
    return recall

def fbeta_score(ytrue, yhat, beta=1):
    """Computes the F score.
    The F score is the weighted harmonic mean of precision and recall.
    Here it is only computed as a batch-wise average, not globally.
    This is useful for multi-label classification, where input samples can be
    classified as sets of labels. By only using accuracy (precision) a model
    would achieve a perfect score by simply assigning every class to every
    input. In order to avoid this, a metric should penalize incorrect class
    assignments as well (recall). The F-beta score (ranged from 0.0 to 1.0)
    computes this, as a weighted mean of the proportion of correct class
    assignments vs. the proportion of incorrect class assignments.
    With beta = 1, this is equivalent to a F-measure. With beta < 1, assigning
    correct classes becomes more important, and with beta > 1 the metric is
    instead weighted towards penalizing incorrect class assignments.
    """
    if beta < 0:
        raise ValueError('The lowest choosable beta is zero (only precision).')

    # If there are no true positives, fix the F score at 0 like sklearn.
    if K.sum(K.round(K.clip(ytrue, 0, 1))) == 0:
        return 0

    p = precision(ytrue, yhat)
    r = recall(ytrue, yhat)
    bb = beta ** 2
    fbeta_score = (1 + bb) * (p * r) / (bb * p + r + K.epsilon())
    return fbeta_score


def fmeasure(ytrue, yhat):
    # also known as f1 measure
    return fbeta_score(ytrue, yhat, beta=1)

def dense_siamese(Xtrain, ytrain, Xval, yval, Xtest, ytest, embed_vec_len, optim,
           layer_size=10, learning_rate=0.01, num_epochs=3, num_bacthes=1, pat=10):
    para_vec1 = Input(shape=(embed_vec_len,), dtype='float32', name='vec1')
    para_vec2 = Input(shape=(embed_vec_len,), dtype='float32', name='vec2')

    drop = Dropout(0.5)
    dense_layer1 = Dense(layer_size, activation='relu', input_shape=(embed_vec_len,), kernel_regularizer=regularizers.l2(0.01))
    p1_d1_out = dense_layer1(drop(para_vec1))
    p2_d1_out = dense_layer1(drop(para_vec2))

    dense_layer2 = Dense(layer_size, activation='relu', input_shape=(layer_size,),
                        kernel_regularizer=regularizers.l2(0.001))
    p1_d2_out = dense_layer2(p1_d1_out)
    p2_d2_out = dense_layer2(p2_d1_out)

    concats = concatenate([p1_d2_out, p2_d2_out], axis=-1)
    # concats_out = drop(concats)

    dense_layer_3 = Dense(layer_size, activation='relu', input_shape=(2 * layer_size,), kernel_regularizer=regularizers.l2(0.001))
    d3_out = dense_layer_3(concats)

    # dist_output = Lambda(manhattan_distance, output_shape=(layer_size,), arguments={'layer_size': layer_size})(concats_out)
    # dist_output = Lambda(exponent_neg_manhattan_distance, output_shape=(1,), arguments={'layer_size': layer_size})(concats_out)

    # distance_out_layer = Dense(1, activation='sigmoid', input_shape=(layer_size,), kernel_regularizer=regularizers.l2(0.001), name='distance')(dist_output)
    # distance_out_layer = Dense(1, activation='sigmoid', input_shape=(2 * layer_size,), kernel_regularizer=regularizers.l2(0.001), name='distance')(concats_out)
    distance_out_layer = Dense(1, activation='sigmoid', input_shape=(2 * layer_size,),
                               kernel_regularizer=regularizers.l2(0.001), name='distance')(d3_out)

    # main_output = Dense(1, activation='sigmoid')(distance_out_layer)

    model = Model(inputs=[para_vec1, para_vec2], outputs=[distance_out_layer])
    #
    if optim == 'adadelta':
        opt = keras.optimizers.Adadelta(lr=learning_rate, clipnorm=1.25)
    else:
        opt = keras.optimizers.Adam(lr=learning_rate)

    model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy', precision, recall, fmeasure])
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=pat)
    model.summary()

    history = model.fit([Xtrain[:, :embed_vec_len], Xtrain[:, embed_vec_len:]], ytrain,
                        validation_data=([Xval[:, :embed_vec_len], Xval[:, embed_vec_len:]], yval), epochs=num_epochs,
                        batch_size=num_bacthes, verbose=1, callbacks=[es])

    # intermediate_layer_model = Model(inputs=model.input, outputs=model.get_layer('distance').output)
    # intermediate_output_train = intermediate_layer_model.predict([Xtrain[:, :embed_vec_len], Xtrain[:, embed_vec_len:]])
    # intermediate_output_val = intermediate_layer_model.predict([Xval[:, :embed_vec_len], Xval[:, embed_vec_len:]])
    # intermediate_output_test = intermediate_layer_model.predict([Xtest[:, :embed_vec_len], Xtest[:, embed_vec_len:]])

    return model

def deep_dense_siamese(Xtrain, ytrain, Xval, yval, Xtest, ytest, embed_vec_len, optim,
           layer_size=10, learning_rate=0.01, num_epochs=3, num_bacthes=1, pat=10):
    para_vec1 = Input(shape=(embed_vec_len,), dtype='float32', name='vec1')
    para_vec2 = Input(shape=(embed_vec_len,), dtype='float32', name='vec2')

    drop = Dropout(0.5)
    dense_layer1 = Dense(layer_size, activation='relu', input_shape=(embed_vec_len,), kernel_regularizer=regularizers.l2(0.01))
    p1_d1_out = dense_layer1(drop(para_vec1))
    p2_d1_out = dense_layer1(drop(para_vec2))

    dense_layer2 = Dense(layer_size, activation='relu', input_shape=(layer_size,),
                        kernel_regularizer=regularizers.l2(0.001))
    p1_d2_out = dense_layer2(p1_d1_out)
    p2_d2_out = dense_layer2(p2_d1_out)

    dense_layer3 = Dense(layer_size, activation='relu', input_shape=(layer_size,),
                         kernel_regularizer=regularizers.l2(0.001))
    p1_d3_out = dense_layer3(p1_d2_out)
    p2_d3_out = dense_layer3(p2_d2_out)

    dense_layer4 = Dense(layer_size, activation='relu', input_shape=(layer_size,),
                         kernel_regularizer=regularizers.l2(0.001))
    p1_d4_out = dense_layer4(p1_d3_out)
    p2_d4_out = dense_layer4(p2_d3_out)

    dense_layer5 = Dense(layer_size, activation='relu', input_shape=(layer_size,),
                         kernel_regularizer=regularizers.l2(0.001))
    p1_d5_out = dense_layer5(p1_d4_out)
    p2_d5_out = dense_layer5(p2_d4_out)

    concats = concatenate([p1_d5_out, p2_d5_out], axis=-1)

    dense_layer_6 = Dense(layer_size, activation='relu', input_shape=(2 * layer_size,), kernel_regularizer=regularizers.l2(0.001))
    d6_out = dense_layer_6(concats)

    distance_out_layer = Dense(1, activation='sigmoid', input_shape=(2 * layer_size,),
                               kernel_regularizer=regularizers.l2(0.001), name='distance')(d6_out)

    model = Model(inputs=[para_vec1, para_vec2], outputs=[distance_out_layer])

    if optim == 'adadelta':
        opt = keras.optimizers.Adadelta(lr=learning_rate, clipnorm=1.25)
    else:
        opt = keras.optimizers.Adam(lr=learning_rate)

    model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy', precision, recall, fmeasure])
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=pat)
    model.summary()

    history = model.fit([Xtrain[:, :embed_vec_len], Xtrain[:, embed_vec_len:]], ytrain,
                        validation_data=([Xval[:, :embed_vec_len], Xval[:, embed_vec_len:]], yval), epochs=num_epochs,
                        batch_size=num_bacthes, verbose=1, callbacks=[es])

    return model

def main():
    parser = argparse.ArgumentParser(description="Train and evaluate Dense-Siamese network for paragraph similarity task")
    parser.add_argument("-v", "--variation", type=int, help="1: dense siamese, 2: deep dense siamese")
    parser.add_argument("-rd", "--train_parapair", required=True, help="Path to train parapair file")
    parser.add_argument("-sd", "--test_parapair", required=True, help="Path to test parapair file")
    parser.add_argument("-rhq", "--train_hier_qrels", required=True, help="Path to train hierarchical qrels")
    parser.add_argument("-shq", "--test_hier_qrels", required=True, help="Path to test hierarchical qrels")
    parser.add_argument("-rem", "--train_embedding", required=True, help="Path to train embedding file")
    parser.add_argument("-sem", "--test_embedding", required=True, help="Path to test embedding file")
    parser.add_argument("-v", "--vec", required=True, type=int, help="Length of each paragraph vector")
    parser.add_argument("-dl", "--dense_layer_size", type=int, help="Size of each dense layer")
    parser.add_argument("-lr", "--learning_rate", type=float, help="Learning rate")
    parser.add_argument("-e", "--epochs", type=int, help="No. of epochs")
    parser.add_argument("-b", "--batches", type=int, required=True, help="No. of batches")
    parser.add_argument("-p", "--patience", type=int, required=True, help="Patience value; No of epochs to execute before early stopping")
    parser.add_argument("-opt", "--optimizer", help="Choose optimizer (adam/adadelta)")
    parser.add_argument("-o", "--out", required=True, help="Path to save trained keras model")

    args = vars(parser.parse_args())
    variation = args["variation"]
    train_pp_file = args["train_parapair"]
    test_pp_file = args["test_parapair"]
    hier_qrels_file = args["train_hier_qrels"]
    test_hier_qrels_file = args["test_hier_qrels"]
    train_embedding_file = args["train_embedding"]
    test_embedding_file = args["test_embedding"]
    vec_len = args["vec"]
    dense_size = args["dense_layer_size"]
    learning_rate = args["learning_rate"]
    epochs = args["epochs"]
    batches = args["batches"]
    pat = args["patience"]
    optim = args["optimizer"]
    out_file = args["out"]

    with open(train_pp_file, 'r') as trpp:
        train_parapair = json.load(trpp)
    with open(test_pp_file, 'r') as tpp:
        test_parapair = json.load(tpp)
    train_emb = np.load(train_embedding_file, allow_pickle=True)
    test_emb = np.load(test_embedding_file, allow_pickle=True)

    hier_qrels_reverse = dict()
    with open(hier_qrels_file, 'r') as hq:
        for l in hq:
            hier_qrels_reverse[l.split(" ")[2]] = l.split(" ")[0]

    test_hier_qrels_reverse = dict()
    with open(test_hier_qrels_file, 'r') as thq:
        for l in thq:
            test_hier_qrels_reverse[l.split(" ")[2]] = l.split(" ")[0]

    Xtrain, ytrain, Xval, yval = prepare_train_data(train_parapair, train_emb, hier_qrels_reverse)
    Xtest, ytest = prepare_test_data(test_parapair, test_emb, test_hier_qrels_reverse, False)
    Xtest_rand, ytest_rand = prepare_test_data(test_parapair, test_emb, test_hier_qrels_reverse, True)

    if variation == 1:
        m = dense_siamese(Xtrain, ytrain, Xval, yval, Xtest, ytest, vec_len, optim, dense_size, learning_rate, epochs, batches, pat)
    else:
        m = deep_dense_siamese(Xtrain, ytrain, Xval, yval, Xtest, ytest, vec_len, optim, dense_size, learning_rate, epochs, batches, pat)

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

    m.save(out_file)

    print("Finished! Trained model saved at: " + out_file)

if __name__ == '__main__':
    main()