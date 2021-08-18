# -*- coding: utf-8 -*-
"""
Created on Mon May 17 11:33:07 2021

@author: Cristian Marquez
"""
import numpy as np

import portfolio_classification.product_classifier


def test_convert1():
    """
    test convert a phrase to tensor

    Returns
    -------
    None.

    """

    phrase = "the animal and human"
    theta = {
        "human": np.array([0.3, 0.21, 0.4]),
        "animal": np.array([0.31, 0.32, 0.33]),
    }
    dim_word2vec = 3

    result = portfolio_classification.product_classifier.converter_phrase(
        phrase, theta, dim_word2vec, phrase_size=6
    )

    dim = result.shape[0] + result.shape[1] + result.shape[2]
    assert (dim == 10) & (result[0, 1, 0] == 0.31)


def test_convert_large():
    """
    test convert a phrase to tensor

    Returns
    -------
    None.

    """
    phrase = "the animal and human , the animal and human"
    theta = {
        "human": np.array([0.3, 0.21, 0.4]),
        "animal": np.array([0.31, 0.32, 0.33]),
    }
    dim_word2vec = 3

    result = portfolio_classification.product_classifier.converter_phrase(
        phrase, theta, dim_word2vec, phrase_size=4
    )

    dim = result.shape[0] + result.shape[1] + result.shape[2]
    assert (dim == 8) & (result[0, 1, 0] == 0.31)


def test_convert_block():
    """
    test convert a set of phrases to tensor

    Returns
    -------
    None.

    """
    phrase_1 = "the animal and human , the animal and human"
    phrase_2 = "the banana and apple"

    corpus = [phrase_1, phrase_2]
    theta = {
        "human": np.array([0.3, 0.21, 0.4]),
        "animal": np.array([0.31, 0.32, 0.33]),
        "banana": np.array([0.41, 0.42, 0.43]),
        "apple": np.array([0.21, 0.22, 0.23]),
    }

    result = portfolio_classification.product_classifier.converter_block(
        corpus, theta, phrase_size=4
    )

    dim = result.shape[0] + result.shape[1] + result.shape[2]
    assert (dim == 9) & (result[0, 1, 0] == 0.31)


def test_convert_weights_phrase():
    """
    test convert a phrase to vector

    Returns
    -------
    None.

    """
    phrase = "the animal and human , the animal and human"

    theta = {
        "human": np.array([0.3, 0.21, 0.4]),
        "animal": np.array([0.31, 0.32, 0.33]),
        "banana": np.array([0.41, 0.42, 0.43]),
        "apple": np.array([0.21, 0.22, 0.23]),
    }

    result = portfolio_classification.product_classifier.phrase2vector(phrase, theta)

    factor = 20
    weights = [
        np.exp(-0 / factor),
        np.exp(-1 / factor),
        np.exp(-2 / factor),
        np.exp(-3 / factor),
    ]

    first_component = (
        0.31 * weights[0] + 0.3 * weights[1] + 0.31 * weights[2] + 0.3 * weights[3]
    ) / sum(weights)

    assert round(first_component, 5) == round(result[0], 5)


def test_convert_weights_block_1():
    """
    test convert a set of phrases to vector

    Returns
    -------
    None.

    """

    phrase_1 = "the animal and human , the animal and human"
    phrase_2 = "the banana and apple"

    corpus = [phrase_1, phrase_2]
    theta = {
        "human": np.array([0.3, 0.21, 0.4]),
        "animal": np.array([0.31, 0.32, 0.33]),
        "banana": np.array([0.41, 0.42, 0.43]),
        "apple": np.array([0.21, 0.22, 0.23]),
    }

    result = portfolio_classification.product_classifier.phrases2vectors(corpus, theta)

    factor = 20
    weights = [
        np.exp(-0 / factor),
        np.exp(-1 / factor),
        np.exp(-2 / factor),
        np.exp(-3 / factor),
    ]

    first_component = (
        0.31 * weights[0] + 0.3 * weights[1] + 0.31 * weights[2] + 0.3 * weights[3]
    ) / sum(weights)

    assert round(first_component, 5) == round(result[0, 0], 5)


def test_convert_weights_block_2():
    """
    test convert a set of phrases to vector

    Returns
    -------
    None.

    """
    phrase_1 = "the animal and human , the animal and human"
    phrase_2 = "the banana and apple"

    corpus = [phrase_1, phrase_2]
    theta = {
        "human": np.array([0.3, 0.21, 0.4]),
        "animal": np.array([0.31, 0.32, 0.33]),
        "banana": np.array([0.41, 0.42, 0.43]),
        "apple": np.array([0.21, 0.22, 0.23]),
    }

    result = portfolio_classification.product_classifier.phrases2vectors(corpus, theta)

    assert (result.shape[0] == 2) & (result.shape[1] == 3)
