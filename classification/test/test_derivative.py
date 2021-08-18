# -*- coding: utf-8 -*-
"""
Test gradient
"""
import numpy as np

import classification.cost_function
import classification.gradient


def test_gradient_classification_words():
    """
    test the gradient for classification words

    """
    word_label = {
        "i": "human",
        "person": "human",
        "man": "human",
        "women": "human",
        "cat": "animal",
        "dog": "animal",
        "bird": "animal",
        "lion": "animal",
        "potato": "vegetables",
        "tomatoes": "vegetables",
        "garlic": "vegetables",
        "onions": "vegetables",
    }

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
        "vegetables": np.array([0.11, 0.12, 0.13]),
    }

    cost_z = classification.cost_function.cost_classification_words(
        word_label, word_vector, theta
    )
    gradient = classification.gradient.gradient_classification_words(
        word_label, word_vector, theta
    )

    TOL = 0.0001
    count = 0
    while count < 8:
        copy_theta = theta.copy()
        ramdom_key = np.random.choice(list(theta.keys()))
        increment = np.zeros_like(copy_theta[ramdom_key])
        choice = np.random.choice(np.arange(len(increment)))
        increment[choice] = 0.00001
        copy_theta[ramdom_key] = copy_theta[ramdom_key] + increment
        cost_z_h = classification.cost_function.cost_classification_words(
            word_label, word_vector, copy_theta
        )

        df_approx = (cost_z_h - cost_z) / increment[choice]
        df_actual = gradient[ramdom_key][choice]

        count += 1
        assert abs(df_actual - df_approx) < TOL


def test_gradient_word_vector():
    """
    test gradient that update word vector representation.
    """
    word_label = {
        "i": "human",
        "person": "human",
        "man": "human",
        "women": "human",
        "cat": "animal",
        "dog": "animal",
        "bird": "animal",
        "lion": "animal",
        "potato": "vegetables",
        "tomatoes": "vegetables",
        "garlic": "vegetables",
        "onions": "vegetables",
    }

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
        "vegetables": np.array([0.11, 0.12, 0.13]),
    }
    cost_z = classification.cost_function.cost_classification_words(
        word_label, word_vector, theta
    )
    gradient = classification.gradient.gradient_word_vector(
        word_label, word_vector, theta
    )

    TOL = 0.0001
    count = 0
    while count < 8:
        copy_word_vector = word_vector.copy()
        ramdom_key = np.random.choice(list(word_vector.keys()))
        increment = np.zeros_like(copy_word_vector[ramdom_key])
        choice = np.random.choice(np.arange(len(increment)))
        increment[choice] = 0.00001
        copy_word_vector[ramdom_key] = copy_word_vector[ramdom_key] + increment
        cost_z_h = classification.cost_function.cost_classification_words(
            word_label, copy_word_vector, theta
        )

        df_approx = (cost_z_h - cost_z) / increment[choice]
        df_actual = gradient[ramdom_key][choice]

        count += 1
        assert abs(df_actual - df_approx) < TOL
