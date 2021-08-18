# -*- coding: utf-8 -*-
"""
This module compute the cost function for skip-gram model and its derivative
function, the logic behind the algorithms is to calculate the cost and its
derivative using the softmax matrix and frequency matrix where the columns
refers to the central words and rows refers to context words.
"""
from typing import List
import numpy as np
import utils.util


def denom_softmax(
    vocabulary: List[str], theta: np.ndarray, dimension: int
) -> List[float]:
    """
    Calculates the denominator corresponding to the soft-max function, that is
    the sum for exponential function corresponding for the dot product between
    a central word and each context word.
    The function will compute this value for each word in the vocabulary.

    Parameters
    ----------
    vocabulary : list
        List of unique words in the corpus. initially use gen_vocabulary function.
    theta : numpy.array
        Array that contains the vector representation of words, initially use
        gen_theta to get this parameter.
    dimension : int
        Size of dimension that have each vector representation of words.

    Returns
    -------
    denom_vec : List
        Each element contains the value of denominator to the soft-max
        function in the same order to vocabulary list.
    """
    denom_vec = list()

    for central_w in enumerate(vocabulary):
        central_v = utils.util.find_vector(central_w[0], theta, dimension)
        acum = 0
        for context_w in enumerate(vocabulary):
            context_v = utils.util.find_vector(
                context_w[0], theta, dimension, central=False
            )
            acum += np.exp(np.dot(context_v, central_v))
        denom_vec.append(acum)

    return denom_vec


def matrix_softmax(vocabulary: List[str], theta: np.ndarray) -> np.ndarray:
    """
    This function creates a matrix with the softmax calculation for all
    convination central word and context words, creates a matrix where the
    columns refers to central words and the rows refers to context word.

    Parameters
    ----------
    vocabulary : list
        List of unique words in the corpus. use gen_vocabulary function.
    theta : numpy.array
        Array that contains the vector representation of words.

    Returns
    -------
    matrix : numpy.ndarray
        Matrix with a row and column size equal to the number of words, each.

    """
    dimension = len(theta) // 2 // len(vocabulary)
    denom = denom_softmax(vocabulary, theta, dimension)
    matrix = np.zeros((len(vocabulary), len(vocabulary)), "float")
    for central_w in enumerate(vocabulary):
        central_v = utils.util.find_vector(central_w[0], theta, dimension)
        central_denom = denom[central_w[0]]
        for context_w in enumerate(vocabulary):
            context_v = utils.util.find_vector(
                context_w[0], theta, dimension, central=False
            )
            matrix[context_w[0], central_w[0]] = (
                np.exp(np.dot(context_v, central_v)) / central_denom
            )
    return matrix


def matrix_frequency(
    corpus: List[str],
    vocabulary: List[str],
    s_window: int,
) -> np.ndarray:
    """
    Calculates the frequency for each combination between central and context
    word, return a matrix where the columns refers to central words and the rows
    refers to context word. Ignoring the words that not exist in vocabulary.

    Parameters
    ----------
    corpus : list
        This list contains the tokenized corpus.
    vocabulary : list
        List of unique words in the corpus. use gen_vocabulary function.
    s_window : int
        Size of window to take the context words, e.g. s_window = 2, take two
        context words at left and two context words at right.
    Returns
    -------
    numpy.ndarray
        Matrix with row and column size equal to the number of words, each
        element represent the frequency in the corpus for the conexion central
        word (column) and context word(row).
    """
    mtx_frequency = np.zeros((len(vocabulary), len(vocabulary)), "int")

    for pos_central in range(s_window, len(corpus) - s_window):

        if corpus[pos_central] in vocabulary:

            central_index = utils.util.find_index(corpus[pos_central], vocabulary)

            # find conexion in couple (left - right)
            for i in range(s_window):
                take_l = pos_central - (1 + i)  # left
                take_r = pos_central + (1 + i)  # right

                if corpus[take_l] in vocabulary:
                    context_index = utils.util.find_index(corpus[take_l], vocabulary)
                    mtx_frequency[context_index, central_index] = (
                        mtx_frequency[context_index, central_index] + 1
                    )

                if corpus[take_r] in vocabulary:
                    context_index = utils.util.find_index(corpus[take_r], vocabulary)
                    mtx_frequency[context_index, central_index] = (
                        mtx_frequency[context_index, central_index] + 1
                    )

    return mtx_frequency


def cost_function(
    corpus: List[str], mtx_soft: np.ndarray, mtx_frequency: np.ndarray
) -> float:
    """
    Computes the cost function difined by skip-gram model, using the log function
    for the softmax matrix and then apply hadamard product between log softmax
    and frequency. finally sums all elments and multiply by (1/T).
    The term "T" defined by the model would be the size of corpus.

    Parameters
    ----------
    corpus : list
        This list contains the tokenized corpus.
    matrix_soft : numpy.ndarray
        This matrix contains the value of softmax function for all combination
        central and context word, the columns refers to central words and the
        rows refers to context word.
    matrix_frequency: numpy.ndarray
        This matrix contains the frequency for each combination in the corpus
        between central and context word, a matrix where the columns refers to
        central words and the rows refers to context word.

    Returns
    -------
    float

    """

    return -(1 / len(corpus)) * np.sum(np.log(mtx_soft) * mtx_frequency)


def derivative(
    corpus: List[str],
    vocabulary: List[str],
    theta: np.ndarray,
    mtx_soft: np.ndarray,
    mtx_frequency: np.ndarray,
):
    """
    Computes the derivative for the cost function difined by skip-gram model,
    first calculate the derivative for the central words and then for the
    context words.
    The term "T" defined by the model would be the size of corpus.

    Parameters
    ----------
    corpus : list
        This list contains the tokenized corpus.
    vocabulary : list
        List of unique words in the corpus. use gen_vocabulary function.
    theta : numpy.array
        Array that contains the vector representation of words.
    mtx_soft : numpy.ndarray
        This matrix contains the value of softmax function for all combination
        central and context word, the columns refers to central words and the
        rows refers to context word.
    mtx_frequency: numpy.ndarray
        This matrix contains the frequency for each combination in the corpus
        between central and context word, a matrix where the columns refers to
        central words and the rows refers to context word.

    Returns
    -------
    numpy.array
        Vector that contains the derivativa evalated in theta.

    """
    dimension = len(theta) // 2 // len(vocabulary)
    grad_theta = np.zeros_like(theta)

    # derivative for central words
    mtx_deraux = np.zeros((dimension, len(vocabulary)), "float")

    for central_w in enumerate(vocabulary):
        for context_w in enumerate(vocabulary):
            context_v = utils.util.find_vector(
                context_w[0], theta, dimension, central=False
            )
            mtx_deraux[:, central_w[0]] += (
                mtx_soft[context_w[0], central_w[0]] * context_v
            )
    mtx_deraux = np.sum(mtx_frequency, axis=0) * mtx_deraux

    for central_w in enumerate(vocabulary):
        start, end = utils.util.find_location(central_w[0], theta, dimension)
        der = 0
        for context_w in enumerate(vocabulary):
            context_v = utils.util.find_vector(
                context_w[0], theta, dimension, central=False
            )
            der += mtx_frequency[context_w[0], central_w[0]] * context_v
        grad_theta[start:end] = der - mtx_deraux[:, central_w[0]]

    # derivative for context words
    for context_index in enumerate(vocabulary):
        start, end = utils.util.find_location(
            context_index[0], theta, dimension, central=False
        )
        der = 0
        for central_index in range(len(vocabulary)):
            central_v = utils.util.find_vector(central_index, theta, dimension)
            der += mtx_frequency[context_index[0], central_index] * central_v
            der += -(
                mtx_soft[context_index[0], central_index]
                * np.sum(mtx_frequency, axis=0)[central_index]
                * central_v
            )

        grad_theta[start:end] = der
    return -(1 / len(corpus)) *  grad_theta
