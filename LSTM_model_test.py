import numpy as np
from keras.preprocessing import sequence
from collections import Counter
import argparse
import getEmbeddings2
import pickle

top_words = 5000

if __name__ == '__main__':
    """parser = argparse.ArgumentParser(description='Use a Fake News Model')
    parser.add_argument('-model', dest='model_name',
                        help='The model you want to test against.')
    parser.add_argument('-data', dest='data_fname',
                        help='The file name of the data that you are testing against')
    args = parser.parse_args()"""


    """if not os.path.isfile('./xtr_shuffled.npy') or \
            not os.path.isfile('./xte_shuffled.npy') or \
            not os.path.isfile('./ytr_shuffled.npy') or \
            not os.path.isfile('./yte_shuffled.npy'):
        getEmbeddings2.clean_data()"""

    print("Get Embeddings")
    getEmbeddings2.clean_test_data('datasets/test.csv')


    xtr = np.load('./test_data.npy')

    cnt = Counter()
    x_train = []
    for x in xtr:
        x_train.append(x.split())
        for word in x_train[-1]:
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
    for news in x_train:
        i = 0
        while i < len(news):
            if news[i] in word_bank:
                news[i] = word_bank[news[i]]
                i += 1
            else:
                del news[i]

    """# Delete the short news
    print("Delete short news")
    i = 0
    while i < len(x_train):
        if len(x_train[i]) > 10:
            i += 1
        else:
            del x_train[i]"""



    # Truncate and pad input sequences
    print("Truncate and pad input sequences")
    max_review_length = 500
    X_train = sequence.pad_sequences(x_train, maxlen=max_review_length)


    print("Load Model")
    model = pickle.load(open("lstm_model.pickle", "rb"))


    # Draw the confusion matrix
    print("Make predictions")
    y_pred = model.predict_classes(X_train)
    y_prob = model.predict_proba(X_train)
    text_file = open("LSTM_test_results.txt", "w")
    for idx, (read, pred, prob) in enumerate(zip(X_train, y_pred.flatten(), y_prob.flatten())):
        text_file.write("idx: {}\n{}\npred: {} {}\n\n".format(idx, read, pred, prob))
    text_file.close()