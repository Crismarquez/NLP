# -*- coding: utf-8 -*-
"""
Test util functions
"""
import numpy as np
import utils.util


# Test gen_vocabulary()
def test_size_gen_vocabulary():
    """
    Test the size for vocabulary
    """
    corpus = ["test", "for", "gen_vocabulary", "test", "corpus"]
    vocabulary = utils.util.gen_vocabulary(corpus)

    assert len(vocabulary) == 4


def test_unique_gen_vocabulary():
    """
    Test unique word in vocabulary
    """
    corpus = ["test", "for", "gen_vocabulary", "test", "corpus"]
    vocabulary = utils.util.gen_vocabulary(corpus)

    assert vocabulary.count("test") == 1


# Test gen_theta()
def test_size_gen_theta():
    """
    Test the size for vocabulary
    """
    vocabulary = ["test", "for", "gen_vocabulary", "corpus"]
    dimension = 300
    theta = utils.util.gen_theta(vocabulary, dimension)

    assert len(theta) == 2 * len(vocabulary) * dimension


def test_find_vector_central():
    """
    Test find the vector represntation when the word is central
    """
    dimension = 2
    theta = np.array([1, 2, 3, 4, 5, 6, 7, 8])
    central_vector = utils.util.find_vector(1, theta, dimension)

    assert central_vector.sum() == 7


def test_find_vector_context():
    """
    Test find the vector represntation when the word is ccontext
    """
    dimension = 2
    theta = np.array([1, 2, 3, 4, 5, 6, 7, 8])
    central_vector = utils.util.find_vector(0, theta, dimension, central=False)

    assert central_vector.sum() == 11
