import numpy as np
from keras.preprocessing import sequence
from collections import Counter
import os
import getEmbeddings2
import pickle

top_words = 5000

if __name__ == '__main__':

    if not os.path.isfile('./xtr_shuffled.npy') or \
            not os.path.isfile('./xte_shuffled.npy') or \
            not os.path.isfile('./ytr_shuffled.npy') or \
            not os.path.isfile('./yte_shuffled.npy'):
        getEmbeddings2.clean_data()

    xtr = np.load('./xtr_shuffled.npy')
    xte = np.load('./xte_shuffled.npy')
    y_train = np.load('./ytr_shuffled.npy')
    y_test = np.load('./yte_shuffled.npy')

    cnt = Counter()
    x_train = []
    for x in xtr:
        x_train.append(x.split())
        for word in x_train[-1]:
            cnt[word] += 1

    # Storing most common words
    most_common = cnt.most_common(top_words + 1)
    word_bank = {}
    id_num = 1
    for word, freq in most_common:
        word_bank[word] = id_num
        id_num += 1

    # Encode the sentences
    for news in x_train:
        i = 0
        while i < len(news):
            if news[i] in word_bank:
                news[i] = word_bank[news[i]]
                i += 1
            else:
                del news[i]

    y_train = list(y_train)
    y_test = list(y_test)

    # Delete the short news
    i = 0
    while i < len(x_train):
        if len(x_train[i]) > 10:
            i += 1
        else:
            del x_train[i]
            del y_train[i]

    # Generating test data
    x_test = []
    for x in xte:
        x_test.append(x.split())

    # Encode the sentences
    for news in x_test:
        i = 0
        while i < len(news):
            if news[i] in word_bank:
                news[i] = word_bank[news[i]]
                i += 1
            else:
                del news[i]

    # Truncate and pad input sequences
    max_review_length = 500
    X_train = sequence.pad_sequences(x_train, maxlen=max_review_length)
    X_test = sequence.pad_sequences(x_test, maxlen=max_review_length)

    # Convert to numpy arrays
    y_train = np.array(y_train)
    y_test = np.array(y_test)

    print("Huh")
    model = pickle.load(open("lstm_model.pickle", "rb"))
    scores = model.evaluate(X_test, y_test, verbose=0)

    print("Accuracy= %.2f%%" % (scores[1] * 100))

    # Draw the confusion matrix
    y_pred = model.predict_classes(X_test)
    y_prob = model.predict_proba(X_test)
    text_file = open("LSTM_results.txt", "w")
    for idx, (read, truth, pred, prob) in enumerate(zip(xte, y_test, y_pred.flatten(), y_prob.flatten())):
        text_file.write("idx: {}\n{}\ntruth: {}\npred: {} {}\n\n".format(idx, read, truth, pred, prob))
    text_file.close()