#!/usr/bin/python3

import nltk
import csv
import sys
import numpy as np
from nltk.tokenize import word_tokenize
from featx import bag_of_words, high_information_words, bag_of_words_in_set
from classification import precision_recall
from random import shuffle
from os import listdir  # to read files
from os.path import isfile, join  # to read files
from sklearn.model_selection import KFold

punct = ['.', ',', '?', ':', ';', '"', "'"]
stop = open('stop.txt', 'r')


# return all the filenames in a folder
def get_filenames_in_folder(folder):
    return [f for f in listdir(folder) if isfile(join(folder, f))]


# make and return stopword-list from stop.txt
def prepare_word_lists(text_file):
    words_list = []
    for line in text_file:
        line = line.strip()
        line = line.lower()
        words_list.append(line)
    return words_list


# reads all the files that correspond to the input list of categories and puts their contents in bags of words
def read_files(categories, stopwords):
    feats = list()

    print("\n##### Reading files...")
    for category in categories:
        files = get_filenames_in_folder(category)
        num_tweets = 0
        for i in files:
            with open("{}/{}".format(category, i), encoding='Latin-1') as csvfile:
                csv_reader = csv.reader(csvfile, delimiter=',', quotechar='|')
                for tweet in csv_reader:
                    tweet_body = tweet[4]
                    clean = []
                    tokens = word_tokenize(tweet_body)

                    for i in tokens:
                        i = i.lower()  # lowering all text
                        if i in stopwords:
                            pass
                        if i in punct:  # remove punctuation
                            pass
                        else:
                            clean.append(i)  # Filter out all stopwords

                    bag = bag_of_words(clean)
                    feats.append((bag, category))
                    num_tweets += 1

        print("  Category {}, amount of tweets ={}".format(category, num_tweets))

    print("  Total, %i files read" % (len(feats)))
    return feats


def highest(feats, high_info_words):
    for i, k in feats:
        for x in list(i):  # compare all the words with the words that are highest info
            if x in high_info_words:  # if given word is in highest info: do nothing
                pass
            else:
                del i[x]  # if word not in highest info: delete word from main dictionary

    return feats


# splits a labelled dataset into two disjoint subsets train and test
def split_train_test(feats, split=0.9):
    train_feats = []
    test_feats = []

    shuffle(feats)  # randomise dataset before splitting into train and test
    cutoff = int(len(feats) * split)
    train_feats, test_feats = feats[:cutoff], feats[cutoff:]

    print("\n##### Splitting datasets...")
    print("  Training set: %i" % len(train_feats))
    print("  Test set: %i" % len(test_feats))
    return train_feats, test_feats


def split_folds(feats, folds=10):
    print("\n##### 10-cross Validation...")
    shuffle(feats)  # randomise dataset before splitting into train and test
    nf = KFold(folds)
    accuracies = []

    for train, test in nf.split(feats):
        train_data = np.array(feats)[train]
        test_data = np.array(feats)[test]
        # based on user choice the corresponding classifier is used
        if choice == 1:
            classifier = nltk.NaiveBayesClassifier.train(train_data)
        else:
            classifier = nltk.classify.MaxentClassifier.train(train_data, max_iter=10)
        acc = nltk.classify.accuracy(classifier, test_data)
        accuracies.append(acc)

    return (accuracies)


# trains a classifier
def train(train_feats):
    if choice == 1:
        # based on user choice the corresponding classifier is used
        from nltk.probability import LaplaceProbDist
        classifier = nltk.classify.NaiveBayesClassifier.train(train_feats, estimator=LaplaceProbDist)
    else:
        algorithm = nltk.classify.MaxentClassifier.ALGORITHMS[0]
        classifier = nltk.classify.MaxentClassifier.train(train_feats, algorithm, max_iter=10)

    return classifier


# function to measure f-score
def calculate_f(precisions, recalls):
    f_measures = {}

    for k, v in precisions.items():
        recall = recalls[k]
        precision = v
        fscore = 2 * ((precision * recall) / (precision + recall))
        fscore = round(fscore, 6)
        f_measures[k] = fscore

    return f_measures


# prints accuracy, precision and recall
def evaluation(classifier, test_feats, categories):
    print("\n##### Evaluation...")
    print("  Accuracy: %f" % nltk.classify.accuracy(classifier, test_feats))
    precisions, recalls = precision_recall(classifier, test_feats)
    try:
        f_measures = calculate_f(precisions, recalls)

        print(" |-----------|-----------|-----------|-----------|")
        print(" |%-11s|%-11s|%-11s|%-11s|" % ("category", "precision", "recall", "F-measure"))
        print(" |-----------|-----------|-----------|-----------|")
        for category in categories:
            if precisions[category] is None:
                print(" |%-11s|%-11s|%-11s|%-11s|" % (category, "NA", "NA", "NA"))
            else:
                print(" |%-11s|%-11f|%-11f|%-11s|" % (
                category, precisions[category], recalls[category], f_measures[category]))
        print(" |-----------|-----------|-----------|-----------|")
    except ValueError:
        return


# show informative features
def analysis(classifier):
    print("\n##### Analysis...")
    print(classifier.show_most_informative_features(10))


# obtain the high information words
def high_information(feats, categories):
    labelled_words = [(category, []) for category in categories]

    # convert the formatting of our features to that required by high_information_words
    from collections import defaultdict
    words = defaultdict(list)
    all_words = list()
    for category in categories:
        words[category] = list()

    for feat in feats:
        category = feat[1]
        bag = feat[0]
        for w in bag.keys():
            words[category].append(w)
            all_words.append(w)

    labelled_words = [(category, words[category]) for category in categories]

    high_info_words = set(high_information_words(labelled_words))

    return high_info_words


def print_menu():
    print(25 * "~", "Classifier-Menu", 25 * "~")
    print("1. Naive Bayes Classifier")
    print("2. Maximum Entropy Classifier")
    print("3. Exit")
    print(67 * "~")


loop = True
while loop:
    print_menu()  # Prints choice menu in the shell
    choice = input()
    choice = int(choice)
    # check if user choice is valid and start main
    if choice in [1, 2]:
        loop = False

    elif choice == 3:
        exit()

    else:
        print("\nERROR: WRONG INPUT :(\n", file=sys.stderr)


def main():
    categories = ['POSITIVE', 'NEUTRAL', 'NEGATIVE']
    stopwords = prepare_word_lists(stop)
    feats = read_files(categories, stopwords)
    high_info_words = high_information(feats, categories)
    highest(feats, high_info_words)

    train_feats, test_feats = split_train_test(feats)

    classifier = train(train_feats)
    evaluation(classifier, test_feats, categories)
    analysis(classifier)
    accuracies = split_folds(feats)
    print('\n')
    x = 0
    for i in accuracies:
        print("X = {} --> {}".format(x, i))
        x += 1
    print("Total = {}".format(sum(accuracies) / len(accuracies)))


if __name__ == '__main__':
    main()
