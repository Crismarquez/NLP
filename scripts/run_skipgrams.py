"""
Show a simple example how to use the functions for skip-gram.
"""

import utils.util
import skip_grams.functions


# input parameters
corpus = [
    "i",
    "like",
    "NLP",
    ",",
    "i",
    "like",
    "machine",
    "learning",
    ",",
    "i",
    "like",
    "NLP",
    ",",
    "i",
    "like",
    "machine",
    "learning",
    ".",
]
DIMENSION = 3  # dimention for each word vector
S_WINDOW = 3  # width for windows

# initial components
vocabulary = utils.util.gen_vocabulary(corpus)
theta = utils.util.gen_theta(vocabulary, DIMENSION, seed=1234)

matrix_frequency = skip_grams.functions.matrix_frequency(corpus, vocabulary, S_WINDOW)

matrix_soft = skip_grams.functions.matrix_softmax(vocabulary, theta)

cost_function = skip_grams.functions.cost_function(
    corpus, matrix_soft, matrix_frequency
)
df_actual = skip_grams.functions.derivative(
    corpus, vocabulary, theta, matrix_soft, matrix_frequency
)
