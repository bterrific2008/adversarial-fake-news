import argparse
import itertools
import pickle
import pandas as pd

if __name__ == '__main__':

    word_bank = pickle.load(open("word_bank.pickle", "rb"))
    vocab = set(word_bank.keys())

    word_sentiment = {}

    data = pd.read_csv("word_bank_sentiment.csv")

    data["score"] = data.reliable / (data.reliable + data.unreliable)

    top_500_pos_words = data.sort_values(by='score', ascending=False).head(200).word.tolist()
    top_500_neg_words = data.sort_values(by='score', ascending=True).head(200).word.tolist()

    word_bank_file = open("word_bank.txt", "w")
    for word, code in word_bank.items():
        word_bank_file.write("{} {}\n".format(word, code))
    word_bank_file.close()

    word_pairs_file = open("word_pairs.txt", "w")
    for word1, word2 in itertools.combinations(top_500_pos_words, 2):
        word_pairs_file.write("{} {} pos\n".format(word1, word2))
    for word1, word2 in itertools.combinations(top_500_neg_words, 2):
        word_pairs_file.write("{} {} pos\n".format(word1, word2))
    word_pairs_file.close()


    """data = pd.read_csv("datasets/train.csv")
    for index, news in data.iterrows():
        text = news.text
        label = news.label

        if not isinstance(text, str):
            continue

        intersection = set(vocab) & set(text.split())
        for word in intersection:
            if word not in word_sentiment:
                word_sentiment[word] = [0, 0]
            word_sentiment[word][label] += 1

    file = open("word_bank_sentiment.csv", "w")
    file.write("{},{},{},{}\n".format("word", "reliable", "unreliable", "score"))
    for key, value in word_sentiment.items():
        file.write("{},{},{},{}\n".format(key, value[0], value[1], value[1]/(sum(value))))"""

    print("Hello")



    """parser = argparse.ArgumentParser(description='Use a Fake News Model')
    parser.add_argument('-model', dest='model_name',
                        help='The model you want to test against.')
    parser.add_argument('-data', dest='data_fname',
                        help='The file name of the data that you are testing against')
    parser.add_argument('-out', dest='fake',
                        help='The file where you want to save the generated adversarial example')
    args = parser.parse_args()

    if args.model_name == 'lstm':"""
