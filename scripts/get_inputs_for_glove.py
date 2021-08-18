# -*- coding: utf-8 -*-
"""
This script create the inputs (vocabulary, cooccurence and initial theta) for glove
model using some corpus in NLTK package, this inputs will be used for glove model
to optimizate theta. you must run this script before run_glove, taking in to
consideration the path directory for save the inputs.
"""
import os
import json

import nltk

import utils.util
import glove.co_occurrence


# import corpus
texts = nltk.corpus.brown.words()
texts = texts + nltk.corpus.gutenberg.words()
texts = texts + nltk.corpus.webtext.words()
texts = texts + nltk.corpus.reuters.words()
texts = texts + nltk.corpus.inaugural.words()

# clean corpus
corpus = [w.lower() for w in texts]
print("Size of corpus: ", "{:,.0f}".format(len(corpus)))
print(
    "lexical diversity: ", "{}%".format(round(len(set(corpus)) / len(corpus) * 100, 3))
)

# filter vocabulary
frequency = nltk.FreqDist(corpus)
vocabulary = [w for w, freq in frequency.items() if freq > 20]
print("Size of filtered vocabulary: ", "{:,.0f}".format(len(vocabulary)))

# hyperparameters
S_WINDOW = int(input("Enter window size for context words: "))
DIMENSION = int(input("Enter the dimension for vector representation of the words: "))

print("Calculating the co-occurrence matrix ...")
co_occurrences = glove.co_occurrence.cooccurrences(corpus, vocabulary, S_WINDOW)

theta = utils.util.gen_theta(vocabulary, DIMENSION, seed=123)
print("Size of theta: ", "{:,.0f}".format(len(theta)))

# save inputs for glove
base = "files"
print("Saving vocabulary, cooccurrence, theta")
files_name = ["vocabulary", "co_occurrence", "theta"]
files = [
    {"vocabulary": vocabulary},
    {"co_occurrence": co_occurrences},
    {"theta": list(theta)},
]

for file, file_name in zip(files, files_name):
    file_path = os.path.join(base, file_name + ".json")
    with open(file_path, "w") as fp:
        json.dump(file, fp)
