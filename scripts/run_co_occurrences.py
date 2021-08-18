# -*- coding: utf-8 -*-
"""
This model create a vector representation using co-occurrence matrix between central
and context words, this matrix is normalized to improve visualitations, finally
using PCA for dimensionality reduction the vector representation of the words is obtained.
The script creates 6 figures for some samples, where shows a hierarchical
clasification using euclidean distanse.
"""
import nltk
import pandas as pd
import matplotlib.pyplot as plt

import sklearn.preprocessing
import sklearn.decomposition  # for PCA
import sklearn.cluster
import scipy.cluster.hierarchy as sch

import glove.co_occurrence


# import corpus
text = nltk.corpus.brown.words()
text = text + nltk.corpus.gutenberg.words()
print("\n Size imported corpus, tockens: ", len(text))

# clean corpus
corpus = [w.lower() for w in text]

# filter vocabulary
frequency = nltk.FreqDist(corpus)
vocabulary = [w for w, freq in frequency.items() if freq > 50]

print("\n Size of corpus: ", len(corpus))
print("Size of vocabulary: ", len(vocabulary))

# hyperparameters
S_WINDOW = int(input("Enter window size for context words: "))
DIMENSION = int(input("Enter the dimension for vector representation of the words: "))

print("\n Calculating the co-occurrence matrix ...")
coocur_matx = glove.co_occurrence.matrix_frequency(corpus, vocabulary, S_WINDOW)

print("Frequencies obtained: ", coocur_matx.sum())
print(" \n Size of matrix: ", coocur_matx.shape)

segur = coocur_matx.copy()
# normalized matrix
coocur_matx = sklearn.preprocessing.normalize(coocur_matx, axis=1)


pca = sklearn.decomposition.PCA(n_components=DIMENSION)
pca_r = pca.fit_transform(coocur_matx)

data = pd.DataFrame(pca_r, index=vocabulary)

# shows samples:
samples = [
    [
        "man",
        "women",
        "person",
        "human",
        "boy",
        "car",
        "road",
        "motor",
        "speed",
        "city",
        "region",
        "country",
        "art",
        "science",
        "research",
    ],
    [
        "house",
        "building",
        "castle",
        "terrible",
        "bad",
        "poor",
        "king",
        "queen",
        "prince",
        "man",
        "women",
        "person",
        "human",
        "boy",
    ],
    ["king", "queen", "prince", "power", "gold", "water", "land", "air"],
    ["sports", "play", "ball", "war", "weapon", "power"],
    ["city", "region", "country", "art", "science", "research"],
    ["king", "queen", "prince", "man", "women", "person", "human", "boy"],
]

print(
    "This words not exist in the vocabulary: ",
    [word for element in samples for word in element if word not in list(data.index)],
)

tittles = [
    "Dendogram_sample_1",
    "Dendogram_sample_2",
    "Dendogram_sample_3",
    "Dendogram_sample_4",
    "Dendogram_sample_5",
    "Dendogram_sample_6",
]

for tittle, sample in zip(tittles, samples):
    plt.subplots(figsize=(10, 8))
    data_filter = data.loc[sample]
    hierarchy = sch.linkage(data_filter, method="ward")
    dendrogram = sch.dendrogram(hierarchy, labels=list(data_filter.index))
    plt.title(tittle)
    plt.xticks(rotation=45)
    plt.xlabel("Words")
    plt.ylabel("Euclidean distance")
    plt.show()
    plt.savefig(tittle)
