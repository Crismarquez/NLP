# -*- coding: utf-8 -*-
"""
test cost functions for classification module
"""
import numpy as np
import classification.cost_function


def test_cost_classification_words_twocategories():
    """
    test the cost function for a problem with two categories
    """
    TOL = 0.0000001
    word_label = {"i": "human", "person": "human", "cat": "animal", "dog": "animal"}

    word_vector = {
        "i": [0.1, 0.2, 0.3],
        "person": [0.5, 0.52, 0.52],
        "man": [0.1, 0.22, 0.32],
        "women": [0.12, 0.24, 0.34],
        "cat": [0.32, 0.34, 0.44],
        "dog": [0.36, 0.36, 0.46],
        "bird": [0.15, 0.14, 0.12],
        "lion": [0.35, 0.34, 0.32],
        "potato": [0.16, 0.15, 0.14],
        "tomatoes": [0.17, 0.14, 0.13],
        "garlic": [0.15, 0.14, 0.12],
        "onions": [0.15, 0.14, 0.12],
    }

    theta = {
        "human": np.array([0.3, 0.21, 0.4]),
        "animal": np.array([0.31, 0.32, 0.33]),
    }

    cost_calculated = classification.cost_function.cost_classification_words(
        word_label, word_vector, theta
    )
    print(cost_calculated)
    assert abs(cost_calculated - 0.6940498887019539) < TOL
