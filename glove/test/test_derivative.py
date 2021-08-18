# -*- coding: utf-8 -*-
"""
This module tests the gradient for the cost function in the glove model,
this module uses the numerical derivative as an aproximation for the derivative.
"""
import copy
import numpy as np
import utils.util
import glove.co_occurrence
import glove.cost_function
import glove.gradient
import glove.fit


def test_derivative_numeric_one_dimension_dict():
    """
    Tests the derivative when vector representation of words has only
    one dimension, tests a vocabulary with two words. test in central
    """
    # input parameters
    vocabulary = ["i", "like"]
    theta = {
        "central": {"i": np.array([0.1]), "like": np.array([-0.2])},
        "context": {"i": np.array([-0.2]), "like": np.array([-0.1])},
    }

    cooccurrences = {"i<>like": 5, "like<>i": 4}

    cost_z = glove.cost_function.cost_glove_dict(theta, cooccurrences)

    df_actual = glove.gradient.gradient_descent_dict(vocabulary, theta, cooccurrences)

    # test
    TOL = 0.0001
    count = 0
    while count < 10:
        ramd_pos = np.random.choice(["central", "context"])
        ramd_pos = "context"
        copy_theta = {
            "central": {"i": np.array([0.1]), "like": np.array([-0.2])},
            "context": {"i": np.array([-0.2]), "like": np.array([-0.1])},
        }
        ramd_key = np.random.choice(list(copy_theta[ramd_pos].keys()))
        increment = 0.00001
        copy_theta[ramd_pos][ramd_key] = copy_theta[ramd_pos][ramd_key] + increment
        cost_z_h = glove.cost_function.cost_glove_dict(copy_theta, cooccurrences)

        df_approx = (cost_z_h - cost_z) / increment

        count += 1

        assert abs(df_actual[ramd_pos][ramd_key][0] - df_approx) < TOL


def test_derivative_numeric_two_words_dict():
    """
    Tests the derivative when vector representation of words has two dimension,
    tests a vocabulary with two words.
    """
    # input parameters
    vocabulary = ["i", "like"]
    theta = {
        "central": {"i": np.array([0.1, 0.2]), "like": np.array([-0.2, 0.1])},
        "context": {"i": np.array([-0.2, 0.2]), "like": np.array([-0.1, 0.3])},
    }
    cooccurrences = {"i<>like": 5, "like<>i": 4}

    cost_z = glove.cost_function.cost_glove_dict(theta, cooccurrences)

    df_actual = glove.gradient.gradient_descent_dict(vocabulary, theta, cooccurrences)

    # test
    TOL = 0.0001
    count = 0
    while count < 10:
        copy_theta = copy.deepcopy(theta)
        ramd_pos = np.random.choice(["central", "context"])
        ramd_key = np.random.choice(list(copy_theta[ramd_pos].keys()))
        increment = np.zeros_like(copy_theta[ramd_pos][ramd_key])
        choice = np.random.choice(np.arange(len(increment)))
        increment[choice] = 0.00001
        copy_theta[ramd_pos][ramd_key] = copy_theta[ramd_pos][ramd_key] + increment
        cost_z_h = glove.cost_function.cost_glove_dict(copy_theta, cooccurrences)

        df_approx = (cost_z_h - cost_z) / increment[choice]

        count += 1

        assert abs(df_actual[ramd_pos][ramd_key][choice] - df_approx) < TOL


def test_derivative_numeric_more_words_dict():
    """
    Tests the derivative when vector representation of words has five dimension
    and a vocabulary with more than two words.
    """
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
    DIMENSION = 5  # dimention for each word vector
    S_WINDOW = 3  # width for windows

    # initial components
    vocabulary = utils.util.gen_vocabulary(corpus)
    theta = utils.util.gen_theta_dict(vocabulary, DIMENSION)

    cooccurrences = glove.co_occurrence.cooccurrences(corpus, S_WINDOW)

    cost_z = glove.cost_function.cost_glove_dict(theta, cooccurrences)

    df_actual = glove.gradient.gradient_descent_dict(vocabulary, theta, cooccurrences)

    # test
    TOL = 0.0001
    count = 0
    while count < 10:
        copy_theta = copy.deepcopy(theta)
        ramd_pos = np.random.choice(["central", "context"])
        ramd_key = np.random.choice(list(copy_theta[ramd_pos].keys()))
        increment = np.zeros_like(copy_theta[ramd_pos][ramd_key])
        choice = np.random.choice(np.arange(len(increment)))
        increment[choice] = 0.00001
        copy_theta[ramd_pos][ramd_key] = copy_theta[ramd_pos][ramd_key] + increment
        cost_z_h = glove.cost_function.cost_glove_dict(copy_theta, cooccurrences)

        df_approx = (cost_z_h - cost_z) / increment[choice]

        count += 1
        if not isinstance(df_actual[ramd_pos][ramd_key], int):

            assert abs(df_actual[ramd_pos][ramd_key][choice] - df_approx) < TOL


def test_derivative_numeric_more_words_dict_factor():
    """
    Tests the derivative when vector representation of words has five dimension
    and a vocabulary with more than two words.
    """
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
    DIMENSION = 5  # dimention for each word vector
    S_WINDOW = 3  # width for windows
    factor = 5
    # initial components
    vocabulary = utils.util.gen_vocabulary(corpus)
    theta = utils.util.gen_theta_dict(vocabulary, DIMENSION)

    cooccurrences = glove.co_occurrence.cooccurrences(corpus, S_WINDOW)

    cost_z = glove.cost_function.cost_glove_dict(theta, cooccurrences, factor)

    df_actual = glove.gradient.gradient_descent_dict(
        vocabulary, theta, cooccurrences, factor
    )

    # test
    TOL = 0.0001
    count = 0
    while count < 10:
        copy_theta = copy.deepcopy(theta)
        ramd_pos = np.random.choice(["central", "context"])
        ramd_key = np.random.choice(list(copy_theta[ramd_pos].keys()))
        increment = np.zeros_like(copy_theta[ramd_pos][ramd_key])
        choice = np.random.choice(np.arange(len(increment)))
        increment[choice] = 0.00001
        copy_theta[ramd_pos][ramd_key] = copy_theta[ramd_pos][ramd_key] + increment
        cost_z_h = glove.cost_function.cost_glove_dict(
            copy_theta, cooccurrences, factor
        )

        df_approx = (cost_z_h - cost_z) / increment[choice]

        count += 1
        if not isinstance(df_actual[ramd_pos][ramd_key], int):

            assert abs(df_actual[ramd_pos][ramd_key][choice] - df_approx) < TOL


def test_derivative_numeric_one_dimension():
    """
    Tests the derivative when vector representation of words has only
    one dimension, tests a vocabulary with two words.
    """
    # input parameters
    vocabulary = ["i", "like"]
    theta = np.array([0.1, -0.2, -0.2, -0.1])

    cooccurrence_matx = np.array([[0, 14], [10, 0]])

    f_z = glove.cost_function.cost_glove(vocabulary, theta, cooccurrence_matx)

    df_actual = glove.gradient.gradient_descent(vocabulary, theta, cooccurrence_matx)

    # test
    TOL = 0.0001
    count = 0
    while count < 10:
        h = np.zeros_like(theta)
        choice = np.random.choice(np.arange(len(theta)))
        h[choice] = 0.00001
        f_zh = glove.cost_function.cost_glove(vocabulary, theta + h, cooccurrence_matx)

        df_approx = (f_zh - f_z) / h[choice]

        count += 1

        assert abs(df_actual[choice] - df_approx) < TOL


def test_derivative_numeric_two_words():
    """
    Tests the derivative when vector representation of words has two dimension,
    tests a vocabulary with two words.
    """
    # input parameters
    vocabulary = ["i", "like"]
    theta = np.array([0.1, 0.2, -0.2, 0.1, -0.2, 0.2, -0.1, 0.3])

    cooccurrence_matx = np.array([[0, 14], [10, 0]])

    f_z = glove.cost_function.cost_glove(vocabulary, theta, cooccurrence_matx)

    df_actual = glove.gradient.gradient_descent(vocabulary, theta, cooccurrence_matx)

    # test
    TOL = 0.0001
    count = 0
    while count < 10:
        h = np.zeros_like(theta)
        choice = np.random.choice(np.arange(len(theta)))
        h[choice] = 0.00001
        f_zh = glove.cost_function.cost_glove(vocabulary, theta + h, cooccurrence_matx)

        df_approx = (f_zh - f_z) / h[choice]

        count += 1

        assert abs(df_actual[choice] - df_approx) < TOL


def test_derivative_numeric_more_words():
    """
    Tests the derivative when vector representation of words has five dimension
    and a vocabulary with more than two words.
    """
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
    DIMENSION = 5  # dimention for each word vector
    S_WINDOW = 3  # width for windows

    # initial components
    vocabulary = utils.util.gen_vocabulary(corpus)
    theta = utils.util.gen_theta(vocabulary, DIMENSION)

    cooccurrence_matx = glove.co_occurrence.matrix_frequency(
        corpus, vocabulary, S_WINDOW
    )

    f_z = glove.cost_function.cost_glove(vocabulary, theta, cooccurrence_matx)

    df_actual = glove.gradient.gradient_descent(vocabulary, theta, cooccurrence_matx)

    # test
    TOL = 0.0001
    count = 0
    while count < 10:
        h = np.zeros_like(theta)
        choice = np.random.choice(np.arange(len(theta)))
        h[choice] = 0.00001
        f_zh = glove.cost_function.cost_glove(vocabulary, theta + h, cooccurrence_matx)

        df_approx = (f_zh - f_z) / h[choice]

        count += 1

        assert abs(df_actual[choice] - df_approx) < TOL


def test_stochastic_gradient_descent():
    """
    Test reductions of cost function using the stocastic gradient descent SGD.

    Returns
    -------
    None.

    """
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
    DIMENSION = 5  # dimention for each word vector
    S_WINDOW = 3  # width for windows

    # initial components
    vocabulary = utils.util.gen_vocabulary(corpus)
    theta = utils.util.gen_theta_dict(vocabulary, DIMENSION)

    cooccurrences = glove.co_occurrence.cooccurrences(corpus, S_WINDOW)

    cost_0 = glove.cost_function.cost_glove_dict(theta, cooccurrences)

    learning_rate = 0.008
    count = 0
    while count < 4:
        df_actual = glove.gradient.stochastic_gradient_descent(
            vocabulary, theta, cooccurrences
        )
        theta = glove.fit.update_theta(theta, df_actual, learning_rate)
        count += 1

    cost_n = glove.cost_function.cost_glove_dict(theta, cooccurrences)

    assert cost_0 > cost_n
