import numpy as np
from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from collections import Counter
import os
import getEmbeddings2
import matplotlib.pyplot as plt
import scikitplot.plotters as skplt
import pickle
import pandas as pd


top_words = 5000
epoch_num = 5
batch_size = 64

def save_classifier(classifier, classifier_fname):
    classifier_file = open(classifier_fname, 'wb')
    pickle.dump(classifier, classifier_file)
    classifier_file.close()


def plot_cmat(yte, ypred):
    '''Plotting confusion matrix'''
    skplt.plot_confusion_matrix(yte, ypred)
    plt.show()

data_twitter = pd.read_csv('datasets/train.csv')
data = data_twitter[data_twitter.label == 1]

data.dropna(subset=['id', 'label', 'text'], inplace=True)
vector_dimension = 300

missing_rows = []

data['text'] = data['text'].map(
        lambda news_text: getEmbeddings2.cleanup(news_text + " follow me on twitter"))

data = data.sample(frac=1).reset_index(drop=True)

x = data.loc[:, 'text'].values
y = data.loc[:, 'label'].values

y_test = list(y)

# Generating test data
x_test = []
for x_news in x:
    x_test.append(x_news.split())

word_bank = pickle.load(open("word_bank.pickle", "rb"))

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
X_test = sequence.pad_sequences(x_test, maxlen=max_review_length)

# Convert to numpy arrays
y_test = np.array(y_test)

# Create the model
print("Create the model")
model = pickle.load(open("lstm_model.pickle", "rb"))
# Final evaluation of the model
print(x[0])
print(y_test[0])

scores = model.evaluate(X_test, y_test, verbose=0)
print("Accuracy= %.2f%%" % (scores[1]*100))

# Draw the confusion matrix
y_pred = model.predict_classes(X_test)
y_prob = model.predict_proba(X_test)
# plot_cmat(y_test, y_pred)

text_file = open("LSTM_results.txt", "w")
for idx, (read, truth, pred, prob) in enumerate(zip(x, y_test, y_pred.flatten(), y_prob.flatten())):
    text_file.write("idx: {}\n{}\ntruth: {}\npred: {} {}\n\n".format(idx, read, truth, pred, prob))
text_file.close()