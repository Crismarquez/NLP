# -*- coding: utf-8 -*-
"""
This module compute the gradient for the cost function implemented by glove model
"""
from typing import List, Dict, Optional, Union
import random
import numpy as np
import utils.util
import glove.cost_function


def stochastic_gradient_descent(
    vocabulary: List[str],
    theta: Dict[str, Dict[str, np.ndarray]],
    co_occurrences: Dict[str, int],
    stochastic_rate: float = 0.5,
    factor: Optional[Union[int, float]] = None,
) -> np.ndarray:
    """
    Compute the stochastic gradient descent. using cooccurrences from a dictionary
    each iteration will take a random sample of words and filter it in
    the co_occurrences dictionary, and only this words will be update in theta.
    first update central words sample, then context words sample will be updated.

    Parameters
    ----------
    vocabulary : List[str]
        List of unique words in the corpus.
    theta : numpy.array
        Array that contains the vector representation of words, central and context
        representation.
    co_occurrences : Dict[str, int]
        This dictionary contains the co-occurrence for each combination in the corpus
        between central and context word, the keys are tuples of two elements where
        the first element of key refers to central words and the second one refers
        to context word.
    stochastic_rate: float = 0.5
        From the sample dictionary, what dimensions in the vector will be calculated.
    factor: Optional[Union[int, float]] = None
        The constant parameter in the sigma function.

    Returns
    -------
    numpy.array
        Numpy array that contains the gradient for the vector theta.

    """

    grad_theta = utils.util.gen_grandient_dict(vocabulary)
    FACTOR = 10
    if factor is not None:
        FACTOR = factor

    choice = random.choices([True, False])
    # SGD for central words

    dimension = len(theta["central"][list(theta["central"].keys())[0]])
    active = int(dimension * stochastic_rate)
    desactive = dimension - active
    ran_choice = [0] * desactive + [1] * active
    np.random.shuffle(choice)

    if choice is True:
        for central_context in co_occurrences.keys():
            central_word, context_word = central_context.split("<>")
            central_vector = theta["central"][central_word]
            context_vector = theta["context"][context_word]

            P_ij = co_occurrences[central_context]

            dot_product = np.dot(context_vector, central_vector)

            central_gradient = (
                (2 * glove.cost_function.sigmoid(P_ij, FACTOR) - 1)
                * context_vector
                * (dot_product - np.log(P_ij))
            )

            grad_theta["central"][central_word] += central_gradient * np.array(
                ran_choice
            )

    else:
        for central_context in co_occurrences.keys():
            central_word, context_word = central_context.split("<>")
            central_vector = theta["central"][central_word]
            context_vector = theta["context"][context_word]

            P_ij = co_occurrences[central_context]

            dot_product = np.dot(context_vector, central_vector)

            context_gradient = (
                (2 * glove.cost_function.sigmoid(P_ij, FACTOR) - 1)
                * central_vector
                * (dot_product - np.log(P_ij))
            )

            grad_theta["context"][context_word] += context_gradient * np.array(
                ran_choice
            )

    return grad_theta


def gradient_descent_dict(
    vocabulary: List[str],
    theta: Dict[str, Dict[str, np.ndarray]],
    co_occurrences: Dict[str, int],
    factor: Optional[Union[int, float]] = None,
) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Compute the gradient descent. using cooccurrences from a dictionary.

    Parameters
    ----------
    vocabulary : List[str]
        List of unique words in the corpus.
    theta : Dict[str, Dict[str, np.ndarray]]
        Contains the vector representation of words, the first dictionary
        related with central and the second with context words,
    co_occurrences : Dict[str, int]
        This dictionary contains the co-occurrence for each combination in the corpus
        between central and context word, the key is a string that contain the central
        word in the right and context word on the left, this words are separed by
        "<>" character.
    factor: Optional[Union[int, float]] = None
        The constant parameter in the sigma function.

    Returns
    -------
    Dict[str, Dict[str, np.ndarray]]
        Numpy array that contains the gradient for the vector theta.

    """

    grad_theta = utils.util.gen_grandient_dict(vocabulary)
    FACTOR = 10
    if factor is not None:
        FACTOR = factor

    for central_context in co_occurrences.keys():
        central_word, context_word = central_context.split("<>")
        central_vector = theta["central"][central_word]
        context_vector = theta["context"][context_word]

        P_ij = co_occurrences[central_context]

        dot_product = np.dot(context_vector, central_vector)

        central_gradient = (
            (2 * glove.cost_function.sigmoid(P_ij, FACTOR) - 1)
            * context_vector
            * (dot_product - np.log(P_ij))
        )

        context_gradient = (
            (2 * glove.cost_function.sigmoid(P_ij, FACTOR) - 1)
            * central_vector
            * (dot_product - np.log(P_ij))
        )

        grad_theta["central"][central_word] += central_gradient

        grad_theta["context"][context_word] += context_gradient

    return grad_theta


def gradient_descent(
    vocabulary: List[str],
    theta: np.ndarray,
    co_occurrence_mtx: np.ndarray,
) -> np.ndarray:
    """
    Campute the gradient  descent.

    Parameters
    ----------
    vocabulary : List[str]
        List of unique words in the corpus.
    theta : numpy.array
        Array that contains the vector representation of words, central and context
        representation.
    co_occurrence_mtx : numpy.ndarray
        This matrix contains the co-occurrence for each combination in the corpus
        between central and context word, a matrix where the columns refers to
        central words and the rows refers to context word.

    Returns
    -------
    numpy.array
        Numpy array that contains the gradient for the vector theta.

    """

    grad_theta = np.zeros_like(theta)
    dimension = len(theta) // 2 // len(vocabulary)
    FACTOR = 10
    for central_index, _ in enumerate(vocabulary):
        central_vector = utils.util.find_vector(central_index, theta, dimension)
        central_start, central_end = utils.util.find_location(
            central_index, theta, dimension
        )
        for context_index, _ in enumerate(vocabulary):
            context_vector = utils.util.find_vector(
                context_index, theta, dimension, central=False
            )
            context_start, context_end = utils.util.find_location(
                context_index, theta, dimension, central=False
            )

            P_ij = co_occurrence_mtx[context_index, central_index]
            if P_ij != 0:
                dot_product = np.dot(context_vector, central_vector)
                context_gradient = (
                    (2 * glove.cost_function.sigmoid(P_ij, FACTOR) - 1)
                    * central_vector
                    * (dot_product - np.log(P_ij))
                )
                central_gradient = (
                    (2 * glove.cost_function.sigmoid(P_ij, FACTOR) - 1)
                    * context_vector
                    * (dot_product - np.log(P_ij))
                )

                grad_theta[context_start:context_end] = (
                    grad_theta[context_start:context_end] + context_gradient
                )
                grad_theta[central_start:central_end] = (
                    grad_theta[central_start:central_end] + central_gradient
                )

    return grad_theta
