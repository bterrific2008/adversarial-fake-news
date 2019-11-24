import numpy as np
from keras.layers import Embedding
from keras.preprocessing import sequence
from collections import Counter
import argparse
import getEmbeddings2
import pickle

top_words = 5000

def lstm_preprocess_text(input):
    cnt = Counter()
    x_train = input.split()
    for word in x_train:
        cnt[word] += 1

    # Storing most common words
    print("Store common words")
    word_bank = pickle.load(open("word_bank.pickle", "rb"))

    # Encode the sentences
    print("Encode sentences")
    i = 0
    other_train = x_train
    while i < len(x_train):
        if other_train[i] in word_bank:
            other_train[i] = word_bank[other_train[i]]
            i += 1
        else:
            del other_train[i]

    # invert word_bank
    inv_word_bank = {v: k for k, v in word_bank.items()}

    return other_train

def lstm_undo_preprocessing(input):
    cnt = Counter()
    x_train = input.split()
    for word in x_train:
        cnt[word] += 1

    # Storing most common words
    print("Store common words")
    most_common = cnt.most_common(top_words + 1)
    word_bank = {}
    id_num = 1
    for word, freq in most_common:
        word_bank[word] = id_num
        id_num += 1

    # Encode the sentences
    print("Econde sentences")
    i = 0
    while i < len(x_train):
        if x_train[i] in word_bank:
            x_train[i] = word_bank[x_train[i]]
            i += 1
        else:
            del x_train[i]

    return x_train

if __name__ == '__main__':

    """if not os.path.isfile('./xtr_shuffled.npy') or \
            not os.path.isfile('./xte_shuffled.npy') or \
            not os.path.isfile('./ytr_shuffled.npy') or \
            not os.path.isfile('./yte_shuffled.npy'):
        getEmbeddings2.clean_data()"""

    input_news = ""
    with open('input.txt', encoding='utf-8') as file:
        for line in file:
            input_news += line + " "

    input_news = getEmbeddings2.cleanup(input_news)

    print(input_news)

    cnt = Counter()
    x_train = lstm_preprocess_text(input_news)

    # Truncate and pad input sequences
    print("Truncate and pad input sequences")
    max_review_length = 500
    embedding_vecor_length = 32
    X_train = sequence.pad_sequences([x_train], maxlen=max_review_length)

    print("Load Model")
    model = pickle.load(open("lstm_model.pickle", "rb"))

    # Draw the confusion matrix
    print("Make predictions")
    y_pred = model.predict_classes(X_train)
    y_pred = y_pred.flatten()[0]
    y_prob = model.predict_proba(X_train)
    y_prob = y_prob.flatten()[0]
    """X_adversary = X_train
    y_prob_adversary = model.predict_proba(X_adversary)
    y_pred_adversary = model.predict_classes(X_adversary)
    while y_pred_adversary == y_pred:
        # Select a word in X_train"""


    text_file = open("LSTM_test_results.txt", "w")
    for idx, (read, pred, prob) in enumerate(zip(X_train, y_pred.flatten(), y_prob.flatten())):
        text_file.write("idx: {}\n{}\npred: {} {}\n\n".format(idx, read, pred, prob))
    text_file.close()
