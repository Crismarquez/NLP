# -*- coding: utf-8 -*-
"""
This script create the inputs (vocabulary, cooccurence and initial theta) for glove
model using some corpus in spanish, this inputs will be used for glove model
to optimizate theta. you must run this script before run_glove, taking in to
consideration the path directory for save the inputs.
The co_occurence function allows to load a previus co-occurrence dictionary.
"""
import os
import json
import time


import utils.util
import glove.co_occurrence


# import corpus
base = "C:/Users/Cristian Marquez/Documents/Cristian/Academico/Projects/NLP/word2vec_V2/corpus/spanish/"
file_names = ["MercadoL"]

file_path = os.path.join(base, file_names[0] + ".json")
with open(file_path, "r") as f:
    corpus = json.load(f)["MercadoL_inter"]

# clean corpus
corpus = [utils.util.clean_word(word) for word in corpus]
print("Size of corpus: ", "{:,.0f}".format(len(corpus)))
print(
    "lexical diversity: ", "{}%".format(round(len(set(corpus)) / len(corpus) * 100, 3))
)

# import previuos co_occurrence
base = "C:/Users/Cristian Marquez/Documents/Cristian/Academico/Projects/NLP/word2vec_V2/files/spanish/300d_5w/"
file_names = ["co_occurrence"]

file_path = os.path.join(base, file_names[0] + ".json")
with open(file_path, "r") as f:
    prev_cooccurrences = json.load(f)


# hyperparameters
S_WINDOW = 5
DIMENSION = 300

inicio = time.time()
print("Calculating the co-occurrence matrix ...")
co_occurrences = glove.co_occurrence.cooccurrences(corpus, S_WINDOW)
print((time.time() - inicio) / 60)

del_key = "9999"
co_occurrences = utils.util.del_key(co_occurrences, del_key)

# theta = utils.util.gen_theta(vocabulary, DIMENSION, seed=123)
# print("Size of theta: ", "{:,.0f}".format(len(theta)))

# save inputs for glove
base = "C:/Users/Cristian Marquez/Documents/Cristian/Academico/Projects/NLP/word2vec_V2/files/spanish/300d_5w/"
print("Saving vocabulary, cooccurrence, theta")
file_name = "co_occurrence"
file = co_occurrences

file_path = os.path.join(base, file_name + ".json")
with open(file_path, "w") as fp:
    json.dump(file, fp)
