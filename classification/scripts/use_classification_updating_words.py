# -*- coding: utf-8 -*-
"""
Use a clasification model for words using vector representation from glove
"""
import os
import csv

import pandas as pd
import matplotlib.pyplot as plt

import utils.util
import classification.cost_function
import classification.gradient
import classification.fit
import classification.predict


base = "C:/"

file_name = "words_topic"
file_path = os.path.join(base, file_name + ".csv")
df = pd.read_csv(file_path, index_col=0, na_values=list)

word_label = utils.util.df_to_dict(df)
word_label = {key: value[0] for key, value in word_label.items()}

f_vocabulary = [str(word) for word in list(word_label.keys())]
f_vocabulary = list(set(f_vocabulary))

file_name = "glove_5g"
file_path = os.path.join(base, file_name + ".txt")
df = pd.read_csv(
    file_path,
    index_col=0,
    header=None,
    sep=r"\s+",
    quoting=csv.QUOTE_NONE,
    chunksize=100000,
)


df_glove = [utils.util.filter_vocabulary(chunk, f_vocabulary) for chunk in df]
df_glove = pd.concat(df_glove)

# filter finally words exist in corpus glove
filter_words = df_glove.index
word_label = {key: value for key, value in word_label.items() if key in filter_words}

word_vector = utils.util.df_to_dict(df_glove)
gradient_w2v = utils.util.gen_grandient(word_vector)

labels = utils.util.get_labels(word_label)
dimension = len(list(word_vector.values())[0])

theta = utils.util.gen_theta_class_words(labels, dimension)
gradient_theta = utils.util.gen_grandient(theta)

learning_rate = 4

cost = classification.cost_function.cost_classification_words(
    word_label, word_vector, theta
)
hist_cost = [cost]
count = 0
while count < 60:
    print(f"iteration {count}")

    gradient_theta = classification.gradient.gradient_classification_words(
        word_label, word_vector, theta
    )
    theta = classification.fit.update_theta(theta, gradient_theta, learning_rate)

    gradient_w2v = classification.gradient.gradient_word_vector(
        word_label, word_vector, theta
    )
    word_vector = classification.fit.update_theta(
        word_vector, gradient_w2v, learning_rate
    )

    hist_cost.append(
        classification.cost_function.cost_classification_words(
            word_label, word_vector, theta
        )
    )
    count += 1


plt.plot(range(len(hist_cost)), hist_cost)
plt.title("optimizing theta - using glove word2vec")
plt.xlabel("iteration")
plt.ylabel("cost")
plt.show()

summary_prediction = classification.predict.summary_predict_classwords(
    word_label, word_vector, theta
)

accuracy = classification.predict.accuracy_classwords(word_label, summary_prediction)
print("Accuracy for classification model: ", round(accuracy, 2))
print("amount of words:", len(word_label.keys()))
print("amount of predicted words:", int(accuracy * len(word_label.keys())))
print("amount of labels: ", len(set(list(word_label.values()))))
